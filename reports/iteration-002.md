# Iteration 2: Flow Matching Model

**Date**: 2026-05-03
**Starting commit**: 2df87d1
**GPU**: L40S (node3615, job 13092500, 3h alloc, used 1h48m)
**Wall time**: ~2.5h (00:21–02:17 EDT)

## Objective
Implement a conditional flow matching model to beat the baseline CRPS of 0.307 by generating diverse ensemble predictions.

## Method

**Flow Matching** (Lipman et al., 2022) trains a velocity field v(x_t, t, cond):
- x_t = (1-t)·x_0 + t·x_1 (linear interpolation from noise to data)
- x_0 ~ N(0, I), x_1 = HR target
- Condition: bicubic-upsampled LR concatenated to x_t
- Loss: MSE on predicted velocity vs true velocity (x_1 - x_0)
- Inference: Euler ODE from t=0 to t=1 with 20 steps

**Architecture**: Small UNet with time conditioning (AdaGN-style scale+shift)
- Channels: [32, 64, 128], 3 encoder/decoder levels
- Input: 2 channels (noisy sample + LR condition), 128×128
- Output: 1 channel (predicted velocity), 128×128
- Parameters: 2,341,185 (~10× baseline ResNet)
- Mixed precision (AMP) for training speed

**Training**: 100 epochs, batch_size=256, lr=2e-4 with cosine schedule, AdamW

## Results

### Flow Matching, No Constraints (100 epochs)
- Training: 86.7 min on L40S
- Evaluation: 15.6 min (10 members × 20 Euler steps)

| Metric | Flow (ours) | GAN Baseline | Δ |
|--------|-------------|--------------|---|
| **CRPS** | **0.2516** | 0.3066 | **-18.0%** |
| MSE | 0.3521 | 0.3824 | -7.9% |
| RMSE | 0.5934 | 0.6184 | -4.0% |
| MAE (mean) | 0.3283 | 0.3066 | +7.1% |
| Ensemble Spread | 0.4554 | ~0 | ∞ |
| Mass Violation | 0.0579 | 0.0454 | +27.5% |

### Analysis

1. **CRPS improvement comes from diversity**: CRPS = E|X-y| - 0.5·E|X-X'|. The baseline GAN has zero spread (mode collapse), so CRPS = MAE. The flow model produces meaningful spread (0.455), giving a large spread_term that improves CRPS by 18%.

2. **MAE of ensemble mean is slightly worse**: 0.328 vs 0.307. This is expected — each ensemble member is a stochastic sample, and their mean isn't optimized for pointwise accuracy. The individual members capture uncertainty better.

3. **MSE is actually better**: 0.352 vs 0.382. The flow model's ensemble mean has lower MSE than the mode-collapsed GAN, despite having a UNet architecture vs the GAN's ResNet.

4. **Mass violation increased**: 0.058 vs 0.045. No conservation constraint was applied. Adding SmCL to the flow output should fix this.

## CRPS Verification
- Fast CRPS formula (sorted ensemble identity) verified numerically against naive O(M²) double loop
- Both give identical results to machine precision
- Degenerates to MAE for identical ensemble members (verified)

## Files Created/Modified
- `scripts/flow_downscale.py` — Flow matching model, training, and evaluation
- `models/flow/flow_best.pth` — Best checkpoint (9.4 MB)
- `models/flow/flow_last.pth` — Final checkpoint
- `models/flow/flow_config.pth` — Normalization stats
- `logs/flow_train.log` — Training log
- `logs/flow_eval.log` — Evaluation log
- `reports/iteration-002.md` — This report

## Next Steps
1. **Add SmCL constraint to flow output** — should reduce mass violation and possibly improve CRPS
2. **Try more Euler steps** (50 or 100) — may improve sample quality
3. **Larger model or more epochs** — loss was still slightly decreasing at epoch 100
4. **Compare with constrained GAN baselines** — GAN+softmax still untrained
