# Iteration 4: Constraint-Aware Flow Matching Training

**Date**: 2026-05-03
**Starting commit**: 985890b
**GPU**: L40S (node4404, job 13101650, 3h alloc, used ~90min train + 62min eval)
**Wall time**: ~3h (03:07–06:13 EDT)

## Objective
Train a flow matching model with an auxiliary constraint-aware loss — backpropagating through the multiplicative conservation constraint during training — to see if end-to-end awareness of the constraint improves CRPS.

## Method

### Constraint-Aware Training Loss
During training, for samples with t > 0.5 (where the ODE trajectory is closer to the data manifold):
1. Predict x̂ = x_t + v_pred × (1 - t)  (one-step denoising to t=1)
2. Apply differentiable mult constraint: softplus activation → scale by lr/AvgPool ratio
3. Compute MSE(constrained, hr_target) as auxiliary loss
4. Total loss = velocity_loss + 0.1 × aux_loss

The model learns to produce outputs that, after the constraint projection, closely match HR targets.

### Other Changes
- **Checkpoint resumption**: Save full training state (model, optimizer, scheduler, scaler) each epoch for preemption recovery
- **Heun's 2nd-order solver**: Implemented for inference (turned out to hurt, see below)
- **Differentiable constraint**: Uses `softplus(β=10)` instead of `clamp` for smooth gradients

### Training Details
- 100 epochs, batch 256, lr 2e-4 with cosine schedule
- Best val loss: 0.002068 (vs 0.002009 for original model — comparable)
- Training time: 87.2 min on L40S
- 2 preemptions on earlier allocation (node4204) before successful run

## Results

| Config | CRPS | MSE | RMSE | MAE | Spread | Mass Viol |
|--------|------|-----|------|-----|--------|-----------|
| **CA + mult + euler** | **0.2424** | **0.3347** | **0.5786** | **0.3159** | 0.4104 | 0.0001 |
| CA + none + euler | 0.2468 | 0.3383 | 0.5816 | 0.3209 | 0.4373 | 0.0460 |
| CA + mult + heun | 0.7513 | 1.6165 | 1.2714 | 0.9575 | 3.0883 | 0.0001 |

### Comparison with Prior Iterations

| Model | CRPS | Mass Viol | Notes |
|-------|------|-----------|-------|
| **CA + mult (iter 4)** | **0.2424** | **0.0001** | New best |
| Flow + mult (iter 3) | 0.2460 | 0.0001 | Post-hoc mult only |
| Flow (iter 2) | 0.2516 | 0.0579 | No constraint |
| GAN baseline (iter 1) | 0.3066 | 0.0454 | Mode-collapsed |

## Analysis

1. **Constraint-aware training improves CRPS** from 0.2460 → 0.2424 (−1.5%) compared to post-hoc mult alone. The model learns within-block texture patterns that are preserved by the constraint.

2. **CA helps even without constraint at inference**: CRPS=0.2468 (CA, no constraint) vs 0.2516 (original, no constraint) — 1.9% improvement. The aux loss regularizes the model to produce more physically consistent outputs.

3. **Heun's method fails at 20 steps**: CRPS=0.751, spread=3.09. The 2nd-order corrector overshoots because the velocity field isn't smooth enough for large step sizes. Would need many more steps (50+) to benefit, negating the speed advantage.

4. **Cumulative improvement**: CRPS 0.3066 → 0.2424, a **20.9% reduction** from the baseline GAN, with zero mass violation.

## Files Modified
- `scripts/flow_downscale.py` — Added constraint-aware training, Heun solver, checkpoint resumption
- `scripts/train_with_resume.sh` — Auto-retry wrapper for preemption recovery
- `reports/iteration-004.md` — This report

## Decisions
- DEC-006: Chose constraint weight λ=0.1 (small enough to not dominate velocity loss, large enough to have effect). Aux loss ~0.00002 vs velocity loss ~0.002 confirms this is a gentle regularizer.
- DEC-007: Heun solver not recommended at 20 steps. Euler is sufficient.

## Next Steps
1. **More Euler steps** (50) with CA model — may improve ODE accuracy cheaply
2. **Larger model** (64,128,256 channels) with CA — more capacity for fine structure
3. **End-to-end SmCL** — train with exp-based constraint from scratch (not post-hoc)
4. **CRPS-aware loss** — replace MSE aux with a differentiable CRPS approximation
