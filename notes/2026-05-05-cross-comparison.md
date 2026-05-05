# Cross-Comparison: Flow Matching Experiments for TCW 4x Downscaling

**Date:** 2026-05-05
**Task:** 32x32 -> 128x128 spatial downscaling of Total Column Water (ERA5)
**Dataset:** Harder et al. `era5_sr_data` — 40K train / 10K val / 10K test

## Unified Results (Corrected Energy CRPS)

The corrected CRPS formula is: `CRPS = E|X-y| - 0.5*E|X-X'|` (energy form).
Harder et al.'s original code has a bug (`fc.shape[-1]**2` instead of `fc.shape[0]**2`) that underestimates CRPS by ~50%.

| Model | Branch | Params | CRPS (corrected) | Eval set | RMSE | MAE | Mass Viol |
|-------|--------|--------|------------------|----------|------|-----|-----------|
| **Flow v2 (OT-CFM + attention)** | research2 | 13M | **0.171** | 2K test | 0.456 | 0.242 | 0.000001 |
| Flow v2 (OT-CFM + attention) | research2 | 13M | ~0.174 (est.) | 10K test | 0.458 | 0.247 | 0.000001 |
| Flow attn (LR-anchor) | research | 5.2M | 0.199 | 10K test | 0.481 | 0.258 | 0.000131 |
| GAN baseline (re-evaluated) | research | 204K | 0.307 | 10K test | 0.618 | 0.307 | 0.0454 |
| Bilinear interpolation | research2 | — | — | — | 0.546 | 0.341 | 0.169 |

Notes:
- research2 full 10K corrected CRPS was computed but log lost to preemption. Estimated from 2K→10K ratio (+1.7%).
- Weights for both models exist; re-running eval would give exact numbers.

## Method Comparison

| | research2 (OT-CFM residual) | research (LR-anchor) |
|---|---|---|
| **Flow formulation** | Standard OT-CFM: x_t = (1-t)*noise + t*data | LR-anchor: x_t = (1-t)*(bicubic(LR)+noise) + t*HR |
| **What model learns** | Velocity on residual (HR - bilinear(LR)) | Velocity from LR+noise to HR |
| **Architecture** | UNet, 64/128/256 channels, attention at 16x16 | UNet, 48/96/192 channels, attention at 16x16 |
| **Parameters** | 13.07M | 5.22M |
| **Training** | AdamW, LR=1e-4, cosine (T=40), BS=64, 39 ep | AdamW, LR=2e-4, cosine (T=200), BS=256, 200 ep |
| **Sampling** | 10 Euler steps from noise | 20 Euler steps from LR+noise |
| **Constraint** | AddCL post-hoc | Multiplicative post-hoc + constraint-aware training |
| **Ensemble diversity** | Different noise samples at t=0 | Different noise added to LR at t=0 (sigma=0.3) |
| **Training time** | ~3h on A100 (39 ep) | ~4.5h on L40S (200 ep) |
| **GPU hours total** | ~25h (research2 all iterations) | ~35h (research all iterations) |

## Why research2 Wins on CRPS

The OT-CFM residual approach (research2) achieves ~14% better corrected CRPS (0.171 vs 0.199) despite both using flow matching + attention + constraints. Key differences:

1. **Residual prediction is a stronger inductive bias.** Learning velocity on the residual space (HR - bilinear(LR)) concentrates model capacity on the high-frequency detail. The LR-anchor approach still needs to implicitly separate the LR structure from the detail.

2. **Larger model (13M vs 5.2M).** 2.5x more parameters with wider channels (64/128/256 vs 48/96/192).

3. **Standard OT-CFM vs LR-anchor transport.** Starting from pure noise with optimal transport gives straighter, better-conditioned paths than the LR-anchor interpolation where the model must handle variable noise scales.

4. **Fewer steps needed (10 vs 20).** Straighter OT paths converge in fewer steps, which also benefits CRPS through integration noise acting as beneficial ensemble spread.

## Soft Constraint Experiments (CNN Baseline)

Ran April 27 on L40S. All use Harder et al. CNN architecture (100K params).

| Strategy | Epochs | RMSE | MAE | Mass Violation | Neg pixels/M |
|----------|--------|------|-----|----------------|-------------|
| Baseline soft (alpha=0.99) | 200 | 0.767 | 0.384 | 0.0150 | 339 |
| Low LR (alpha=0.99, lr=1e-4) | 500 | 0.760 | 0.382 | 0.0184 | 463 |
| Curriculum (alpha: 0->0.99) | 400 | 0.640 | 0.314 | 0.0222 | 2 |
| Lagrangian (adaptive alpha) | 400 | 0.627 | 0.310 | 0.0316 | 2 |
| L1 constraint loss | 200 | 2.135 | 1.448 | 0.8814 | 29243 |
| **Hard constraint (AddCL, paper)** | — | **0.580** | **0.290** | **0.000** | 0 |
| **Hard constraint (SmCL, paper)** | — | **0.582** | **0.291** | **0.000** | 0 |

Key findings:
- Better training (curriculum, Lagrangian) improves RMSE but **increases** violation. The tradeoff is fundamental.
- No soft strategy reaches hard-constraint violation levels (<0.001).
- L1 constraint loss catastrophically fails — the L1 penalty on violation dominates MSE gradient.
- Hard constraints (AddCL/SmCL) achieve both best RMSE and zero violation simultaneously.

## Consolidated Key Findings

### What works

| Finding | Evidence |
|---------|----------|
| Flow matching >> GAN/CNN for probabilistic downscaling | 35-44% CRPS improvement across both tracks |
| Hard constraints are free | Zero/negligible CRPS cost, eliminates mass violation |
| AddCL/MultCL are robust post-hoc projections | Work with any generator output; SmCL fails on physical-space values |
| Residual prediction + standard OT-CFM > LR-anchor | 14% better corrected CRPS (0.171 vs 0.199) |
| Self-attention at 16x16 bottleneck | Cheap (2% overhead), consistent ~3% gain |
| 10 Euler steps is sufficient | More steps/higher-order solvers don't help CRPS |

### What fails

| Finding | Evidence |
|---------|----------|
| Soft constraints cannot achieve hard-constraint violation levels | 5 strategies tested, violation floors at 0.015+ |
| SmCL incompatible with flow matching post-hoc | exp() overflow on physical-space values |
| CFG hurts strongly-conditioned tasks | +13% CRPS regression (research2) |
| CRPS-aware loss over-diversifies | +14% CRPS regression (research) |
| EMA redundant with cosine-to-zero schedule | No gain in either track |

### Open questions

1. **Are the two models comparable on identical eval?** Weights exist for both; a single eval script running corrected CRPS on both would give definitive numbers.
2. **Is 13M params necessary or does residual framing explain the gap?** Could test research2 architecture at 5M params.
3. **Does constraint-aware training help the OT-CFM model?** research2 uses AddCL post-hoc only; research uses constraint-aware training loss. Combining both is untested.

## Next Steps (Priority Order)

1. **Unify eval:** Re-run both best models through identical corrected-CRPS eval on full 10K test. Eliminates uncertainty.
2. **Spectral metrics:** CV project outline requires FFT/wavelet evaluation. Neither track computed power spectral density or scale-dependent metrics.
3. **Ablate the gap:** Train research2 architecture at 5.2M params (match research) to isolate residual-vs-LR-anchor effect from capacity effect.
4. **Multi-variable:** Extend to joint TCW + temperature + humidity with cross-variable constraints.
5. **DiT backbone:** Replace UNet with transformer-based score network — untested in climate downscaling.
6. **Spatiotemporal:** TCW T1/T2 tasks from project outline remain untouched.

## Reproduction

Both models' weights are available:
- research2: `models/flow_v2/best_flow.pt` (13M), `src/exp-spatial-4x-crps-v1/flow_matching_v2.py`
- research: `models/flow_attn/flow_best.pth` (5.2M), `scripts/flow_downscale.py`

To get unified corrected CRPS for both:
```bash
# research2 (prints both CRPS formulas)
python src/exp-spatial-4x-crps-v1/flow_matching_v2.py --mode eval \
    --n_ensemble 10 --split test --ode_steps 10 --constraint addcl

# research (already uses corrected CRPS)
python scripts/flow_downscale.py --mode eval \
    --data-dir external/constrained-downscaling/data/era5_sr_data \
    --save-dir models/flow_attn \
    --constraint mult --euler-steps 20 --n-members 10
```
