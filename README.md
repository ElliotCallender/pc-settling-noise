# PC Settling Is Fragile to Internal Noise

Predictive coding (PC) networks are often claimed to be robust to noise. This experiment tests whether that robustness extends to **biologically realistic noise injected at every layer during every settling step** -- as opposed to the input-only noise tested in the literature.

**Finding: it does not.** Even sigma=0.001 additive Gaussian noise at each value node during settling destroys learning at depth 10 and degrades it at depth 3. This is not graceful degradation -- it is a cliff.

## Experiment Design

**Task:** MNIST classification (10-class, 784-dimensional input).

**SGD baseline:** 10-hidden-layer MLP, hidden_dim=48 (minimum width for >95% test accuracy), trained with Adam. Achieves **96.3%** test accuracy.

**PC network:** Same architecture (10 hidden layers, dim 48). Standard supervised predictive coding:
- Input and output layers clamped
- Hidden value nodes updated via energy gradient: `dE/dv_i = epsilon_{i-1} - J_i^T @ epsilon_i`
- Weight updates are **strictly local** -- each W_i only minimizes its own layer's prediction error
- No global loss, no backprop through the network

**Noise model:** At each settling step, each hidden value node receives additive Gaussian noise: `v_i += sigma * randn_like(v_i)`. This simulates synaptic transmission noise and membrane voltage fluctuations -- the kind of noise every biological neuron experiences continuously.

**Settling:** 20 steps at settle_lr=0.1 for depth 10; 10 steps at settle_lr=0.1 for depth 3.

## Results

### Depth 10 (20 settling steps)

| Condition | Final Acc | Best Acc |
|-----------|-----------|----------|
| SGD baseline | 0.963 | 0.963 |
| PC noiseless | 0.921 | 0.923 |
| PC sigma=0.001 | 0.101 | 0.251 |
| PC sigma=0.002 | 0.114 | 0.127 |
| PC sigma=0.01 | 0.114 | 0.114 |
| PC sigma=0.02 | 0.114 | 0.114 |
| PC sigma=0.05 | 0.114 | 0.114 |

### Depth 3 (10 settling steps)

| Condition | Final Acc | Best Acc |
|-----------|-----------|----------|
| PC noiseless | 0.961 | 0.961 |
| PC sigma=0.001 | 0.748 | 0.934 |
| PC sigma=0.002 | 0.388 | 0.921 |

## Key Observations

1. **Noise fragility scales with depth.** Depth-3 PC tolerates sigma=0.001 early in training (93%) but degrades over epochs. Depth-10 PC cannot learn at all at the same noise level. Settling amplifies noise geometrically through the layer stack.

2. **Noise robustness is transient even at shallow depth.** The depth-3 runs initially learn (~93%) then collapse. Persistent noise doesn't just slow learning -- it actively destroys learned representations over time.

3. **The failure is a cliff, not a slope.** There is no gradual degradation from sigma=0 to sigma=0.001. Noiseless PC at depth 10 reaches 92%; sigma=0.001 peaks at 25% then collapses to chance.

4. **PC requires O(depth) settling steps.** 10 settling steps cannot propagate the error signal through 10 layers (each step moves information one layer). This is itself a biological constraint -- 20 recurrent sweeps per inference is already beyond typical estimates for cortical autorecurrence within a gamma cycle.

## Implications

The "noise robustness" claimed in PC literature (e.g. robustness to corrupted inputs) does not extend to the internal noise that biological neurons experience during the settling process itself. Synaptic noise (~1mV std on ~10mV PSPs) corresponds to sigma ~0.1 in normalized units -- orders of magnitude above the sigma=0.001 that already destroys learning here.

This suggests that biological predictive coding, if it exists, either:
- Operates at very shallow depth (2-3 layers), or
- Uses mechanisms not captured by standard PC settling (e.g., dendritic compartments, oscillatory gating, precision weighting), or
- Does not rely on iterative settling at all

## Reproduction

```bash
# Find minimum SGD hidden dim (already done: 48)
python pc_noise_mnist.py --phase sweep_dim --depth 10

# Run all noise conditions at depth 10
python pc_noise_mnist.py --phase run_all --depth 10 --hidden-dim 48 --settling-steps 20

# Run at depth 3
python pc_noise_mnist.py --phase run_all --depth 3 --hidden-dim 48 --settling-steps 10
```

Requires PyTorch and torchvision. MNIST is downloaded automatically. CPU-only, runs in ~30 minutes per depth on 10 cores.

## Files

- `pc_noise_mnist.py` -- experiment code (single file, ~300 lines)
- `results/sgd_dim_sweep.json` -- SGD hidden dim sweep
- `results/pc_noise_comparison.json` -- depth-10 noise sweep
- `results/pc_noise_ablation.json` -- depth-3 vs depth-10 at sigma 0.001/0.002
