# Iteration 3: Post-hoc Conservation Constraints on Flow Model

**Date**: 2026-05-03
**Starting commit**: fb44366
**GPU**: L40S (node3615, job 13098594, 2h alloc, used ~35min eval)
**Wall time**: ~2h (02:20–04:10 EDT)

## Objective
Apply conservation constraints to the flow matching model output at inference time — enforce AvgPool_4x4(HR) = LR exactly — and evaluate effect on CRPS and mass violation.

## Method

Two post-hoc constraint variants tested on the **same trained model** from iteration 2 (100 epochs, no constraint during training):

### Multiplicative constraint (`mult`)
```
y = clamp(hr, min=ε)
out = y × (lr / AvgPool(y))↑4×4
```
Simple scaling of each 4×4 HR block so its mean equals the corresponding LR pixel. Preserves relative texture within blocks. Works because flow output is already approximately positive.

### Softmax constraint (`softmax`)
```
y = exp(hr)
out = y × (lr / AvgPool(y))↑4×4
```
Baseline SmCL (Harder et al.): maps through exp() before scaling. Designed for training end-to-end, not post-hoc application.

## Results

| Metric | Flow+mult | Flow+softmax | Flow (none) | GAN Baseline |
|--------|-----------|--------------|-------------|--------------|
| **CRPS** | **0.2460** | 0.5532 | 0.2516 | 0.3066 |
| MSE | 0.3465 | 1.0773 | 0.3521 | 0.3824 |
| RMSE | 0.5887 | 1.0379 | 0.5934 | 0.6184 |
| MAE | 0.3207 | 0.5887 | 0.3283 | 0.3066 |
| Spread | 0.4267 | 0.0782 | 0.4554 | ~0 |
| Mass Viol | **0.0001** | 0.0000 | 0.0579 | 0.0454 |

## Analysis

1. **Mult constraint improves everything**: CRPS −2.2% vs unconstrained, −19.8% vs baseline. MSE, RMSE, MAE all improved. Mass violation eliminated (0.058 → 0.0001).

2. **SmCL fails post-hoc**: CRPS degrades 2.2× (0.553 vs 0.252). The exp() transform distorts the flow model's calibrated [0,1] output range. The model was not trained to account for exp(), so the relative patterns within blocks get compressed. SmCL only works when the model is trained end-to-end with it.

3. **Spread reduced slightly** (0.455 → 0.427, −6%): Expected — the constraint forces all ensemble members to have the same block means (matching LR), reducing inter-member variability. But intra-block diversity is preserved, so CRPS still benefits from the spread.

4. **Why mult works and softmax doesn't**: The mult constraint makes minimal adjustments — if the model already predicts AvgPool(HR) ≈ LR (mass violation was only 0.058), the scaling factors are close to 1.0. SmCL's exp() introduces a nonlinear transformation that completely changes the distribution.

## Cumulative Results (Best → Worst)

| Model | CRPS | Mass Viol | Notes |
|-------|------|-----------|-------|
| **Flow + mult** | **0.2460** | **0.0001** | Best overall |
| Flow (none) | 0.2516 | 0.0579 | Iteration 2 |
| GAN (none) | 0.3066 | 0.0454 | Baseline (mode-collapsed) |
| Flow + softmax | 0.5532 | 0.0000 | SmCL fails post-hoc |

## Files Modified
- `scripts/flow_downscale.py` — Added `apply_constraint()` function and `--constraint` CLI arg
- `reports/iteration-003.md` — This report

## Concerns Addressed
1. **Training log analysis**: DEC-004 claimed "loss plateaued at epoch 50" — actually continued decreasing to best=0.002009 at unknown epoch. Cosine LR reached 0 by epoch 100, so extending training needs a longer cosine period.
2. **SmCL vs mult for post-hoc**: Confirmed SmCL is wrong approach for post-hoc constraint on a model trained without it. Mult is the correct choice.

## Next Steps
1. **More Euler steps** (50, 100) with mult constraint — may improve sample quality
2. **Train longer** (200 epochs with cosine restart) then apply mult constraint
3. **Train WITH constraint in the loop** — backprop through the multiplicative projection during flow matching training
4. **Explore under-explored directions**: train end-to-end with SmCL, try different noise schedules, or test on additional variables
