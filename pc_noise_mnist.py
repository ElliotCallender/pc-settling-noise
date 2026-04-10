"""
PC noise robustness on MNIST — does predictive coding stay robust
when noise is injected at every layer during every settling step?

Experiment:
1. Find minimum hidden dim for 10-layer SGD net with >95% test accuracy
2. Run PC nets at that architecture with noise_sigma = {0, 0.001, 0.002, 0.01, 0.02, 0.05}
3. Compare test accuracy curves at depth 3 and depth 10

Noise model: at each settling step, each value node v_i gets
    v_i += sigma * torch.randn_like(v_i)
This simulates synaptic + membrane noise (additive Gaussian per-neuron per-step).

Usage:
    python pc_noise_mnist.py --phase sweep_dim
    python pc_noise_mnist.py --phase run_all
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json
import os
import time
import argparse
from dataclasses import dataclass, field
from typing import Optional


# ── Config (single source of truth for all hyperparameters) ───────────────────


@dataclass
class Config:
    # Architecture
    depth: int = 10             # number of hidden layers
    hidden_dim: int = 48        # hidden layer width (set by sweep)

    # PC settling
    settling_steps: int = 20    # iterations per settling phase
    settle_lr: float = 0.1      # value node update rate

    # Training
    weight_lr: float = 5e-4     # Adam lr for weights
    epochs: int = 15
    batch_size: int = 256

    # Noise
    noise_sigma: float = 0.0    # per-layer per-step Gaussian noise std

    # System
    device: str = "cpu"
    num_workers: int = 0
    max_cores: int = 10

    # Paths
    data_dir: str = os.path.join(os.path.dirname(__file__), "data")
    log_dir: str = os.path.join(os.path.dirname(__file__), "pc_noise_logs")


INPUT_DIM: int = 784
OUTPUT_DIM: int = 10


# ── Data ──────────────────────────────────────────────────────────────────────


def get_loaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(cfg.data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(cfg.data_dir, train=False, download=True, transform=transform)
    train_loader: DataLoader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
    )
    test_loader: DataLoader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
    )

    return train_loader, test_loader


# ── SGD Network ───────────────────────────────────────────────────────────────


class DeepMLP(nn.Module):
    """Deep MLP for MNIST classification (SGD baseline)."""

    def __init__(self, depth: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim: int = hidden_dim
        layers: list[nn.Module] = []
        layers.append(nn.Linear(INPUT_DIM, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, OUTPUT_DIM))
        self.net: nn.Sequential = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 784)
        return self.net(x)  # (batch, 10)


def train_sgd_epoch(
    model: DeepMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    model.train()
    total_loss: float = 0.0
    n_batches: int = 0
    for images, labels in loader:
        images = images.view(-1, INPUT_DIM).to(device)  # (batch, 784)
        labels = labels.to(device)                       # (batch,)
        logits: torch.Tensor = model(images)             # (batch, 10)
        loss: torch.Tensor = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def eval_accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct: int = 0
    total: int = 0
    for images, labels in loader:
        images = images.view(-1, INPUT_DIM).to(device)  # (batch, 784)
        labels = labels.to(device)                       # (batch,)
        if isinstance(model, DeepMLP):
            logits: torch.Tensor = model(images)         # (batch, 10)
        else:
            logits = model.forward_pass(images)           # (batch, 10)
        preds: torch.Tensor = logits.argmax(dim=-1)      # (batch,)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


# ── PC Network ────────────────────────────────────────────────────────────────


class PCNet(nn.Module):
    """Predictive coding network for MNIST classification.

    Architecture mirrors DeepMLP. Settling updates value nodes via local
    prediction errors; weight updates are strictly local (each W_i only
    sees its own layer's error).
    """

    def __init__(self, depth: int, hidden_dim: int) -> None:
        super().__init__()
        self.depth: int = depth
        self.hidden_dim: int = hidden_dim

        # W[i] maps layer i -> layer i+1
        # Layer 0 = input (784), layers 1..depth = hidden, layer depth+1 = output (10)
        self.weights: nn.ModuleList = nn.ModuleList()
        self.weights.append(nn.Linear(INPUT_DIM, hidden_dim, bias=True))
        for _ in range(depth - 1):
            self.weights.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
        self.weights.append(nn.Linear(hidden_dim, OUTPUT_DIM, bias=True))

    def forward_pass(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass (no settling). For eval.

        Args:
            x: (batch, 784)

        Returns:
            (batch, 10) logits
        """
        h: torch.Tensor = x
        for i in range(self.depth):
            h = F.relu(self.weights[i](h))       # (batch, hidden_dim)
        h = self.weights[self.depth](h)           # (batch, 10)

        return h

    def settle(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        cfg: Config,
    ) -> list[torch.Tensor]:
        """PC settling: update value nodes to minimize prediction errors.

        Both input (layer 0) and output (layer depth+1) are clamped.
        Noise is injected at every hidden layer at every settling step.

        Energy: E = sum_i 0.5 * ||epsilon_i||^2
        where epsilon_i = v_{i+1} - f_i(v_i)
        dE/dv_i = epsilon_{i-1} - J_i^T @ epsilon_i

        Args:
            x: (batch, 784) input (clamped)
            target: (batch,) class labels
            cfg: experiment config

        Returns:
            values: list of settled value node tensors [v_0, ..., v_{depth+1}]
        """
        # Initialize value nodes via forward pass
        values: list[torch.Tensor] = [x]
        h: torch.Tensor = x
        for i in range(self.depth):
            h = F.relu(self.weights[i](h))        # (batch, hidden_dim)
            values.append(h.detach().clone())

        # Output clamped to target one-hot
        target_onehot: torch.Tensor = F.one_hot(
            target, num_classes=OUTPUT_DIM,
        ).float().to(cfg.device)                   # (batch, 10)
        values.append(target_onehot.clone())

        # Settling loop — update hidden value nodes only (layers 1..depth)
        for step in range(cfg.settling_steps):
            # Compute prediction errors: epsilon_i = v_{i+1} - f_i(v_i)
            errors: list[torch.Tensor] = []
            for i in range(self.depth):
                pred: torch.Tensor = F.relu(self.weights[i](values[i]))
                errors.append(values[i + 1] - pred)  # (batch, hidden_dim)
            pred_out: torch.Tensor = self.weights[self.depth](values[self.depth])
            errors.append(values[self.depth + 1] - pred_out)  # (batch, 10)

            # Update each hidden node
            for i in range(1, self.depth + 1):
                bu_grad: torch.Tensor = errors[i - 1]  # (batch, hidden_dim)

                # Top-down: J_i^T @ epsilon_i via autograd
                v_i_ag: torch.Tensor = values[i].detach().requires_grad_(True)
                if i < self.depth:
                    pred_above: torch.Tensor = F.relu(self.weights[i](v_i_ag))
                else:
                    pred_above = self.weights[i](v_i_ag)

                td_grad: torch.Tensor = torch.autograd.grad(
                    pred_above, v_i_ag, grad_outputs=errors[i],
                )[0]  # (batch, hidden_dim)

                values[i] = values[i] - cfg.settle_lr * (bu_grad - td_grad)

                if cfg.noise_sigma > 0:
                    values[i] = values[i] + cfg.noise_sigma * torch.randn_like(values[i])

                values[i] = values[i].clamp(-10, 10)

        return values

    def local_weight_update(self, values: list[torch.Tensor]) -> float:
        """Local-only weight update from settled values.

        Each W_i minimizes its own prediction error:
            L_i = 0.5 * ||v_{i+1} - f(W_i, v_i)||^2
        No global loss. Biologically local.

        Args:
            values: settled value nodes from settle()

        Returns:
            total local prediction error (for logging)
        """
        self.zero_grad()
        total_loss: torch.Tensor = torch.tensor(0.0, device=values[0].device)
        for i in range(self.depth):
            pred: torch.Tensor = F.relu(self.weights[i](values[i].detach()))
            loss: torch.Tensor = 0.5 * (
                (values[i + 1].detach() - pred) ** 2
            ).sum(dim=-1).mean()
            total_loss = total_loss + loss
        pred_out: torch.Tensor = self.weights[self.depth](values[self.depth].detach())
        loss_out: torch.Tensor = 0.5 * (
            (values[self.depth + 1].detach() - pred_out) ** 2
        ).sum(dim=-1).mean()
        total_loss = total_loss + loss_out

        total_loss.backward()

        return total_loss.item()


def train_pc_epoch(
    model: PCNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
) -> float:
    """Train one epoch: settle, then one local weight update per batch."""
    model.train()
    total_loss: float = 0.0
    n_batches: int = 0
    for images, labels in loader:
        images = images.view(-1, INPUT_DIM).to(cfg.device)  # (batch, 784)
        labels = labels.to(cfg.device)                       # (batch,)

        values: list[torch.Tensor] = model.settle(images, labels, cfg)
        loss: float = model.local_weight_update(values)
        optimizer.step()
        total_loss += loss
        n_batches += 1

    return total_loss / n_batches


# ── Phase 1: Sweep hidden dim for SGD baseline ───────────────────────────────


def sweep_hidden_dim(cfg: Config) -> int:
    """Find minimum hidden dim where 10-layer SGD net gets >95% test accuracy."""
    os.makedirs(cfg.log_dir, exist_ok=True)
    train_loader, test_loader = get_loaders(cfg)

    dims_to_try: list[int] = [16, 32, 48, 64, 96, 128]
    results: dict[int, float] = {}

    for dim in dims_to_try:
        print(f"\n=== SGD dim={dim}, depth={cfg.depth} ===")
        model: DeepMLP = DeepMLP(cfg.depth, dim).to(cfg.device)
        optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_acc: float = 0.0
        for epoch in range(cfg.epochs):
            loss: float = train_sgd_epoch(model, train_loader, optimizer, cfg.device)
            acc: float = eval_accuracy(model, test_loader, cfg.device)
            best_acc = max(best_acc, acc)
            print(f"  epoch {epoch+1}/{cfg.epochs}  loss={loss:.4f}  acc={acc:.4f}", flush=True)

            if acc > 0.95:
                print(f"  -> hit 95% at epoch {epoch+1}")
                break

        results[dim] = best_acc
        print(f"  best_acc={best_acc:.4f}")

        if best_acc > 0.95:
            break

    with open(os.path.join(cfg.log_dir, "sgd_dim_sweep.json"), "w") as f:
        json.dump({"depth": cfg.depth, "results": {str(k): v for k, v in results.items()}}, f, indent=2)

    passing: list[int] = [d for d, a in results.items() if a > 0.95]
    if passing:
        chosen: int = min(passing)
        print(f"\nChosen hidden_dim={chosen} (acc={results[chosen]:.4f})")
    else:
        chosen = dims_to_try[-1]
        print(f"\nNo dim hit 95%; using largest={chosen} (acc={results[chosen]:.4f})")

    return chosen


# ── Phase 2: Run all conditions ───────────────────────────────────────────────


def run_all(cfg: Config) -> None:
    """Run PC at noise levels {0, 0.001, 0.002, 0.01, 0.02, 0.05}."""
    os.makedirs(cfg.log_dir, exist_ok=True)
    train_loader, test_loader = get_loaders(cfg)
    all_results: dict = {
        "hidden_dim": cfg.hidden_dim,
        "depth": cfg.depth,
        "settling_steps": cfg.settling_steps,
        "settle_lr": cfg.settle_lr,
        "weight_lr": cfg.weight_lr,
        "sgd_baseline_acc": 0.9631,
        "runs": {},
    }

    noise_levels: list[float] = [0.0, 0.001, 0.002, 0.01, 0.02, 0.05]

    for sigma in noise_levels:
        cfg.noise_sigma = sigma
        label: str = f"pc_d{cfg.depth}_s{cfg.settling_steps}_noise{sigma}"
        print(f"\n{'='*60}")
        print(f"{label}")
        print(f"{'='*60}")

        torch.manual_seed(42)
        model: PCNet = PCNet(cfg.depth, cfg.hidden_dim).to(cfg.device)
        optimizer: torch.optim.Adam = torch.optim.Adam(
            model.parameters(), lr=cfg.weight_lr,
        )
        log: list[dict] = []

        for epoch in range(cfg.epochs):
            t0: float = time.time()
            loss: float = train_pc_epoch(model, train_loader, optimizer, cfg)
            acc: float = eval_accuracy(model, test_loader, cfg.device)
            dt: float = time.time() - t0
            log.append({"epoch": epoch, "loss": loss, "acc": acc, "time": dt})
            print(
                f"  epoch {epoch+1}/{cfg.epochs}  loss={loss:.4f}  acc={acc:.4f}  ({dt:.1f}s)",
                flush=True,
            )
        all_results["runs"][label] = log

    # Save
    out_path: str = os.path.join(cfg.log_dir, "pc_noise_comparison.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY (dim={cfg.hidden_dim}, depth={cfg.depth}, "
          f"steps={cfg.settling_steps}, settle_lr={cfg.settle_lr})")
    print(f"{'='*60}")
    print(f"{'Condition':<40} {'Final':>8} {'Best':>8}")
    print("-" * 58)
    print(f"{'SGD baseline':<40} {'0.9631':>8} {'0.9631':>8}")
    for name, log in all_results["runs"].items():
        final: float = log[-1]["acc"]
        best: float = max(e["acc"] for e in log)
        print(f"{name:<40} {final:>8.4f} {best:>8.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["sweep_dim", "run_all"], default="run_all")
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--settling-steps", type=int, default=20)
    parser.add_argument("--settle-lr", type=float, default=0.1)
    parser.add_argument("--weight-lr", type=float, default=5e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-cores", type=int, default=10)
    args: argparse.Namespace = parser.parse_args()

    cfg: Config = Config(
        depth=args.depth,
        settling_steps=args.settling_steps,
        settle_lr=args.settle_lr,
        weight_lr=args.weight_lr,
        device=args.device,
        max_cores=args.max_cores,
        num_workers=min(4, args.max_cores),
    )

    torch.set_num_threads(cfg.max_cores)
    torch.set_num_interop_threads(min(4, cfg.max_cores))

    if args.hidden_dim is not None:
        cfg.hidden_dim = args.hidden_dim

    if args.phase == "sweep_dim":
        sweep_hidden_dim(cfg)
    else:
        if args.hidden_dim is None:
            sweep_path: str = os.path.join(cfg.log_dir, "sgd_dim_sweep.json")
            if os.path.exists(sweep_path):
                with open(sweep_path) as f:
                    sweep: dict = json.load(f)
                passing = [int(d) for d, a in sweep["results"].items() if a > 0.95]
                cfg.hidden_dim = min(passing) if passing else 128
            else:
                cfg.hidden_dim = sweep_hidden_dim(cfg)
        run_all(cfg)


if __name__ == "__main__":
    main()
