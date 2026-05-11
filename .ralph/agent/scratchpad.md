# Metrics Task Scratchpad

## Iteration 1 — 2026-05-11 13:05 EDT
Start commit: `8d355d2`

### Objective
Add evaluation metrics beyond pixel-space (CRPS/MAE/RMSE/mass violation) for all trained models. No training. Code in `src/downscaling/`.

### Literature Review
Searched for and downloaded 2 papers on evaluation metrics for climate downscaling:
1. **2410.01776** - "Dynamical-generative downscaling of climate model ensembles" (Lopez-Gomez et al.)
   - Key metrics: CRPS, radially averaged energy spectra, spread-skill ratio, quantile MAE
2. **2604.03275** - "IPSL-AID" (IPSL group)
   - Key metrics: rank histogram, spread-skill ratio, KL divergence, PSD, CRPS, quantile analysis, inter-variable correlation

Existing catalog papers with relevant metrics:
- CorrDiff (2309.15214): power spectra, probability distributions, rank histograms, ensemble spread
- Intercomparison paper (2512.13987): comprehensive comparison of cGANs vs diffusion

### Current State
Existing metrics in `src/downscaling/metrics/`: CRPS (energy + paper variant)
Existing evaluation in `src/downscaling/evaluation/`: MAE, RMSE, mass violation, ensemble evaluation pipeline

Available trained models (NorESM 2x SR):
- Flow matching (wide96-amp): `pool/datasets/noresm-dataset/models/flow-wide96-amp/best_flow.pt`
- Harder CNN (no constraint + softmax): `pool/datasets/noresm-dataset/models/harder/twc_cnn_*.pth`
- Harder GAN (softmax): `pool/datasets/noresm-dataset/models/harder/twc_gan_softmax.pth`
- SwinIR finetuned: `pool/datasets/noresm-dataset/models/swinir_ft/best_swinir.pt`

ERA5 4x SR cached predictions: `pool/datasets/era5_sr_data/prediction/`

### Plan — Metrics to Implement

**Tier 1 (highest diagnostic value):**
1. **Power Spectral Density (PSD)** — Radially averaged power spectrum. THE standard spatial-frequency metric. Shows whether models preserve fine-scale structure vs over-smoothing.
2. **Rank Histogram** — Probabilistic calibration diagnostic. Is the truth equally likely in any rank position among ensemble members?
3. **Spread-Skill Ratio (SSR)** — sigma_ens / RMSE with finite-size correction sqrt((M+1)/M). Optimal at 1.0.

**Tier 2 (complementary):**
4. **SSIM** — Structural similarity index. Captures luminance/contrast/structure beyond pointwise error.
5. **Distribution comparison** — KL divergence between predicted and true distributions.

### Implementation Approach
- `src/downscaling/metrics/spectral.py` — PSD computation
- `src/downscaling/metrics/calibration.py` — Rank histogram, spread-skill ratio
- `src/downscaling/metrics/structural.py` — SSIM
- `src/downscaling/metrics/distributional.py` — KL divergence
- Update `src/downscaling/evaluation/evaluate.py` for extended metrics
- Integration tests in `tests/test_metrics.py`
- Run evaluation on all models, write report

### Allocation Prefix
`qzcs-imez`

### Iteration 1 End
End commit: `74bd78c`
End time: ~13:20 EDT

Work done:
- Literature search: found and downloaded 2 papers (2410.01776, 2604.03275)
- Identified 5 metrics to implement: PSD, rank histogram, SSR, SSIM, KL divergence
- Created 5 tasks for implementation
- Updated CLAUDE.md with new papers
- All committed

## Iteration 2 — 2026-05-11 13:19 EDT
Start commit: `f9e5b00`

### Concerns Review
1. **Workflow**: Iteration 1 was literature-only. No code written. Tasks created but no implementation started. This is fine for a first iteration.
2. **Quality**: The existing CRPS implementation has O(M^2) pairwise loop — acceptable for small M but worth noting.
3. **Literature**: PSD approach is standard and well-documented across CorrDiff, CDSI, R2D2 papers. The CDSI Appendix A has the SSR formula with finite-size correction: sqrt((M+1)/M) * spread/RMSE.

### Work Done
- Implemented PSD metric in `src/downscaling/metrics/spectral.py`:
  - `radial_psd()`: 2D FFT → |F|^2 → azimuthal average over wavenumber bins
  - `ensemble_mean_psd()`: per-member PSD then average (preserves fine-scale variability)
  - `psd_log_ratio()`: scalar summary = mean |log10(P_pred/P_truth)|
- Added 14 integration tests in `tests/test_metrics.py`:
  - White noise flat spectrum, sinusoid peak detection, constant field zero power
  - Output shapes, rectangular fields, ensemble variance reduction
  - Log-ratio analytical values
- All 24 tests pass, ruff clean, basedpyright clean
- Committed: `35db58e`

### Iteration 2 End
End commit: `35db58e`
End time: ~13:24 EDT
Next: Implement rank histogram + spread-skill ratio (task-1778519848-3f8f)

## Iteration 3 — 2026-05-11 13:23 EDT
Start commit: `0c54176`

### Concerns Review
1. **Quality**: PSD implementation is solid — tests cover analytical properties. SSR formula from CDSI paper is the standard reference (Fortin et al., 2014).
2. **Workflow**: Prior iteration implemented PSD correctly, committed, updated scratchpad. No workflow issues.
3. **Literature**: Rank histogram and SSR are standard calibration diagnostics used in CorrDiff, IPSL-AID, CDSI, R2D2. The formulas are well-established. No need for additional literature search.

### Work Done
- Implemented calibration metrics in `src/downscaling/metrics/calibration.py`:
  - `rank_histogram()`: counts rank of truth among M members, returns M+1 bin histogram
  - `spread_skill_ratio()`: bias-corrected SSR = sqrt((M+1)/M) * spread / RMSE
- Added 11 integration tests in `tests/test_metrics.py`:
  - Rank histogram: uniform calibrated ensemble (chi-squared test), U-shape for under-dispersive, deterministic ensemble, output shape, multichannel
  - SSR: perfect calibration ~1.0, underdispersive <0.5, overdispersive >2.0, perfect prediction returns inf, single member raises, finite-size correction verification
- All 35 tests pass (10 CRPS + 14 PSD + 11 calibration), ruff clean, basedpyright clean
- Updated `__init__.py` with re-exports
- Committed: `e62cbd9`

### Iteration 3 End
End commit: `e62cbd9`
End time: ~13:25 EDT
Next: Implement SSIM metric (task-1778519849-067b)

## Iteration 4 — 2026-05-11 13:27 EDT
Start commit: `3846684`

### Concerns Review
1. **Quality**: Prior implementations (PSD, calibration) are solid — correct formulas, good test coverage, clean type checks. No issues.
2. **Workflow**: Prior iterations committed properly, ran all checks. No missing steps.
3. **Design**: SSIM data_range initially computed from truth only, breaking symmetry. Fixed to use joint min/max of both inputs. Also: negated field test was based on wrong assumption about SSIM behavior — negation flips both luminance and structure terms, making their product positive. Replaced with uncorrelated-field test.

### Work Done
- Implemented SSIM metric in `src/downscaling/metrics/structural.py`:
  - `ssim()`: Gaussian-weighted SSIM with scipy.ndimage.gaussian_filter
  - `ensemble_mean_ssim()`: per-member SSIM then average
  - Joint data_range from both inputs for symmetry
  - Default window_sigma=1.5, k1=0.01, k2=0.03 (standard Wang et al. 2004)
- Added 12 integration tests in `tests/test_metrics.py`:
  - Identical fields → 1.0, uncorrelated → near 0, noisy → intermediate (monotone with noise)
  - Constant fields, explicit data_range, shape errors, non-2D errors
  - Symmetry, range bounded [-1, 1], ensemble single-member consistency, better ensemble higher SSIM, non-3D error
- All 47 tests pass (10 CRPS + 14 PSD + 11 calibration + 12 SSIM), ruff clean, basedpyright clean
- Committed: `d6bfbb1`

### Iteration 4 End
End commit: `d6bfbb1`
End time: ~13:30 EDT
Next: Implement KL divergence / distribution comparison (task-1778519850-519e)

## Iteration 5 — 2026-05-11 13:32 EDT
Start commit: `e0635af`

### Concerns Review
1. **Quality**: Prior implementations (PSD, calibration, SSIM) are solid — correct formulas, good test coverage, clean type checks. No issues found.
2. **Workflow**: Prior iterations committed properly, ran all checks (ruff, basedpyright, pytest). No missing steps.
3. **Design**: KL divergence on binned histograms requires shared bin edges from the joint range and Laplace smoothing to avoid log(0). Both implemented correctly.

### Work Done
- Implemented distributional metrics in `src/downscaling/metrics/distributional.py`:
  - `histogram_kl_divergence()`: KL(P||Q) via shared-edge histograms + Laplace smoothing
  - `ensemble_mean_kl_divergence()`: per-member KL then average
  - Handles arbitrary shapes (flattens both inputs), degenerate constant fields
- Added 13 integration tests in `tests/test_metrics.py`:
  - Identical → 0, non-negative (Gibbs), shifted → positive, larger shift → higher KL
  - Scaled distribution, constant fields, shape mismatch, n_bins validation
  - More bins test, arbitrary 3D shape, ensemble single-member consistency, better ensemble lower KL, rejects 1D
- All 60 tests pass (10 CRPS + 14 PSD + 11 calibration + 12 SSIM + 13 KL), ruff clean, basedpyright clean
- Fixed unused import (scipy.ndimage.gaussian_filter) in test file
- Committed: `d9a40bd`

### Iteration 5 End
End commit: `d9a40bd`
End time: ~13:35 EDT
Next: Run comprehensive evaluation on all models + write report (task-1778519851-14b5)

## Iteration 6 — 2026-05-11 13:36 EDT
Start commit: `b9fa495`

### Concerns Review
1. **Workflow**: Prior iterations wrote metrics but never tested on real data. Metrics may have subtle issues only visible with real predictions. Fixed by running full evaluation.
2. **Quality**: Evaluation functions are repetitive — each model has copy-pasted metric code. Wrote a single comprehensive eval that computes ALL metrics for ALL models in one pass.
3. **Quality**: The existing `evaluate_flow_model` doesn't return predictions, only scalar metrics. Wrote separate prediction generators to decouple generation from evaluation, enabling new metrics.

### Work Done
- Wrote `src/downscaling/evaluation/comprehensive.py`:
  - Prediction generators for all model types (flow, CNN, GAN, SwinIR, baselines)
  - Unified `compute_all_metrics()` computing: CRPS, MAE, RMSE, mass violation, SSIM, KL divergence, PSD log-ratio, rank histogram, SSR
  - Diagnostic plot generation: PSD comparison, rank histograms, metrics bar charts
  - Standalone `run_comprehensive_eval()` and `__main__` entry point
- Ran evaluation on GPU (node3302, mit_normal_gpu) on 2000 NorESM test samples, 10 ensemble members
- Generated results JSON + 3 diagnostic plots
- Wrote evaluation report: `src/downscaling/evaluation/results/EVAL_REPORT.md`
- All 60 tests pass, ruff clean, basedpyright clean

### Key Results
| Model | CRPS | MAE | RMSE | MassViol | SSIM | KL | PSD-LR | SSR |
|---|---|---|---|---|---|---|---|---|
| Flow+AddCL | 1.64 | 1.72 | 2.77 | 0.00 | 0.91 | 0.25 | 0.029 | 0.088 |
| CNN(none) | **1.30** | **1.30** | **1.98** | 1.13 | **0.92** | **0.20** | **0.007** | — |
| CNN(softmax) | 1.72 | 1.72 | 2.78 | 0.00 | 0.91 | 0.26 | 0.028 | — |
| GAN(softmax) | 1.76 | 1.78 | 2.85 | 0.00 | 0.88 | 0.27 | 0.036 | 0.044 |
| SwinIR+AddCL | 1.73 | 1.73 | 2.78 | 0.00 | 0.91 | 0.26 | 0.029 | — |
| Bicubic | 1.75 | 1.75 | 2.82 | 0.06 | 0.90 | 0.26 | 0.032 | — |

Key findings:
1. CNN(none) dominates all pointwise metrics but has 1.13K mass violation
2. Constraints hurt on NorESM (LR/HR from different sims — constraint is misspecified)
3. Both ensemble models severely underdispersive (SSR 0.044-0.088)
4. Constrained models barely beat bicubic baseline (~6% CRPS improvement)
5. 2x SR is spectrally easy — all models preserve fine-scale structure

### Iteration 6 End
End commit: (pending)
End time: ~14:10 EDT
Wall clock: ~34 min (including GPU queue wait)
