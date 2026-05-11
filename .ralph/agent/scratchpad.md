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
