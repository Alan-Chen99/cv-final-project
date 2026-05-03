# Iteration 1: Baseline Establishment

**Date**: 2026-05-02 → 2026-05-03
**Starting commit**: 01644d4
**GPU**: L40S (node5002, job 13084027, 2h alloc)
**Wall time**: ~2.5h

## Objective
Establish baseline CRPS numbers for 32×32 → 128×128 ERA5 TCW downscaling using the Harder et al. constrained-downscaling codebase.

## Dataset
- ERA5 Total Column Water (TCW), 4× upsampling
- Train: 40,000 samples, input (1, 1, 32, 32), target (1, 1, 128, 128)
- Val: 10,000 samples
- Test: 10,000 samples
- Values range: ~0.04 to ~135 kg/m²

## CRPS Bug in Baseline Code
The baseline `crps_ensemble()` in `training.py:258-274` has a bug:
- **First loop** uses `fc.shape[-1]**2` (=128²=16384) as denominator
- **Second loop** uses `fc.shape[0]**2` (=M²=100) as denominator
- Both should use ensemble size M squared
- Effect on 128×128 data: **buggy CRPS ≈ 0.5× correct CRPS**
- All CRPS numbers below use the correct formula: **CRPS = E|X-y| - 0.5·E|X-X'|**

## Critical Finding: GAN Mode Collapse
The baseline GAN generates **near-identical predictions** across different noise vectors:
- **Ensemble spread**: 0.000001 (10 members essentially identical)
- **CRPS = MAE = 0.307** (no probabilistic benefit from ensembling)
- Root cause: adversarial loss weight = 0.0001 is too small relative to MSE loss
- The 100-dim noise vector z is effectively ignored

This means the GAN baseline provides **no useful uncertainty quantification**.

## Results

### GAN, No Constraints (200 epochs, batch_size=256)
- Training: 101 min on L40S
- #params: 203,522 (generator) + 286,561 (discriminator)

| Metric | Value |
|--------|-------|
| **CRPS (correct)** | **0.3066** |
| CRPS (buggy baseline) | 0.1677 |
| MSE | 0.3824 |
| RMSE | 0.6184 |
| MAE | 0.3066 |
| PSNR | 45.79 dB |
| SSIM | 0.9875 |
| MS-SSIM | 0.9946 |
| Pearson corr | 0.9977 |
| Mass Violation | 0.0454 |
| Ensemble Spread | ~0 |

### GAN, SmCL (Softmax Constraints)
**Not trained** — GPU allocation expired after GAN none. Needs ~2h allocation.

## Files Created
- `scripts/eval_crps.py` — Correct CRPS evaluation (fixes baseline bug)
- `scripts/eval_all.py` — Batch evaluation script
- `scripts/run_baselines.sh` — Baseline training runner
- `reports/iteration-001.md` — This report

## Key Implications
1. **Baseline CRPS = MAE ≈ 0.307** (due to GAN collapse)
2. Any model with calibrated ensemble spread should beat this
3. Diffusion/flow models are the natural next step (native diversity)
4. Hard constraints still untested — need GAN softmax or apply to diffusion output

## Recommended Next Steps
1. Train GAN+softmax to check if constraints help even with collapsed ensemble
2. Implement conditional diffusion/flow matching model for proper ensembles
3. Target: CRPS < 0.307 via spread_term > 0
