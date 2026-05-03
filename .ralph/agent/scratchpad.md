# Scratchpad — Constrained Downscaling with CRPS

## Iteration 1 — 2026-05-02 21:55 EDT → 2026-05-03 00:18 EDT
**Starting commit**: 01644d4
**Goal**: Set up baselines for 32×32→128×128 ERA5 TCW downscaling, evaluate CRPS

### Understanding
- Task: 32×32 → 128×128 spatial downscaling, ERA5 Total Column Water
- Metric: CRPS (Continuous Ranked Probability Score) — requires ensemble predictions
- Baseline code: `external/constrained-downscaling/` (Harder et al.)
- CRPS only computed for GAN model in baseline (10 ensemble members)
- For deterministic models, CRPS degenerates to MAE
- Training budget: <2hrs per model

### Findings

#### CRPS Bug (Verified)
Baseline `crps_ensemble()` in `training.py:258-274`:
- First loop: `fc.shape[-1]**2 = 128² = 16384` (WRONG)
- Second loop: `fc.shape[0]**2 = 10² = 100` (correct)
- Effect: underestimates CRPS by ~50% on 128×128 data
- Buggy CRPS = 0.168, Correct CRPS = 0.307

#### GAN Collapse (Critical Finding)
The baseline GAN produces **zero ensemble spread**:
- Ensemble spread: 0.000001 (10 members essentially identical)
- CRPS = MAE = 0.307 (no benefit from ensembling)
- The 100-dim noise vector z is ignored — model collapsed to deterministic predictor
- Root cause: tiny adversarial weight (0.0001) + MSE dominance

#### GAN None Results (200 epochs, L40S, 101 min training + 16 min eval)
| Metric | Value |
|--------|-------|
| CRPS (correct) | 0.3066 |
| CRPS (buggy) | 0.1677 |
| MSE | 0.3824 |
| RMSE | 0.6184 |
| MAE | 0.3066 |
| SSIM | 0.9875 |
| Mass Violation | 0.0454 |
| Ensemble Spread | ~0 |

### Concerns (Iteration 1)
1. **CRPS bug**: CONFIRMED. Baseline CRPS numbers are wrong. Using correct formula.
2. **GAN collapse**: CRITICAL. GAN produces no diversity. CRPS = MAE. Need proper probabilistic model.
3. **Time**: GAN softmax could not be trained — GPU allocation expired after GAN none (2hr alloc, 2hr model). Need longer allocation or faster training next time.

### Implications for Next Iterations
1. **GAN baseline is NOT probabilistic** — CRPS = MAE ≈ 0.307 is the true baseline
2. Any model with calibrated spread should beat this (CRPS = MAE - spread_term)
3. Diffusion models are the clear path to improvement — they naturally generate diverse ensembles
4. Hard constraints (SmCL) should still be tested but on a model with actual spread
5. Consider: EDM or flow matching conditioned on LR input, with SmCL on output

### What Remains for Next Iterations
- Train GAN+softmax baseline (to see if constraints affect the collapsed GAN)
- Implement a proper probabilistic model (diffusion or flow matching)
- Target: CRPS < 0.307 with a calibrated ensemble
