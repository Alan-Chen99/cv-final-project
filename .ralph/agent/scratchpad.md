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
