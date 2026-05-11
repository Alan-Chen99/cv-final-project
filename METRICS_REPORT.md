# Evaluation Metrics Report: Climate Downscaling

Comprehensive evaluation across two datasets, 15+ methods, and 7 metrics spanning pixelwise accuracy, spectral fidelity, structural quality, and physical consistency.

## Datasets

| Dataset | Variable | SR Factor | Resolution | Samples | Source |
|---------|----------|-----------|------------|---------|--------|
| ERA5 TCW | Total Column Water | 4x | 32x32 -> 128x128 | 500 (test) | [Google Drive](https://drive.google.com/file/d/1IENhP1-aTYyqOkRcnmCIvxXkvUW2Qbdx) |
| NorESM TAS | Surface Air Temperature | 2x | 32x32 -> 64x64 | 500 (test) | [Google Drive](https://drive.google.com/file/d/1D5tLE7cGcvh3dap-P3VOLEOK_7FqdChF) |

## Metrics

| Metric | Type | Direction | Description | Reference |
|--------|------|-----------|-------------|-----------|
| CRPS (energy) | Probabilistic | Lower better | Continuous Ranked Probability Score via energy form. For deterministic methods, CRPS = MAE. | Gneiting & Raftery 2007 |
| MAE | Pixelwise | Lower better | Mean Absolute Error of ensemble mean | Standard |
| RMSE | Pixelwise | Lower better | Root Mean Square Error of ensemble mean | Standard |
| RALSD (dB) | Spectral | Lower better | Relative Average Log Spectral Distance. Weights errors at all spatial scales equally in log-space. | [Intercomparison (2512.13987)](https://arxiv.org/abs/2512.13987) |
| SSIM | Structural | Higher better | Structural Similarity Index | Wang et al. 2004 |
| PSNR (dB) | Structural | Higher better | Peak Signal-to-Noise Ratio | Standard |
| Mass violation | Physical | Lower better | Coarse-grain constraint mismatch (avg abs diff after downsampling) | [Harder et al. (2208.05424)](https://arxiv.org/abs/2208.05424) |

### RALSD Definition

From the intercomparison paper ([2512.13987](https://arxiv.org/abs/2512.13987)):
1. Compute 2D FFT of each field
2. Radially integrate (bin by wavenumber) -> 1D power spectral density
3. RALSD(dB) = sqrt(1/N * sum_i (10*log10(PSD_true_i / PSD_pred_i))^2)

Captures spectral fidelity: whether predictions reproduce correct spatial frequency content across all scales, not just pixel-level accuracy.

## Methods

### Baselines (4)
- **bilinear** / **bilinear+addcl**: Bilinear interpolation, with/without AddCL
- **bicubic** / **bicubic+addcl**: Bicubic interpolation, with/without AddCL

### Pretrained SR: SwinIR (4)
- **swinir-zeroshot** / **swinir-zeroshot+addcl**: SwinIR x4 pretrained on DF2K, zero-shot transfer
- **swinir-finetuned** / **swinir-finetuned+addcl**: SwinIR finetuned on target dataset

### CNN/GAN: Harder et al. (3)
- **harder-cnn**: CNN with MSE loss, no constraints
- **harder-cnn+smcl**: CNN + SoftMax Correction Layer
- **harder-gan+smcl**: GAN + SoftMax Correction Layer

### Flow Matching — Ours (4)
- **flow-wide96-amp (28M)**: Wide UNet (96 base channels), AMP training
- **flow-uniform-amp (13M)**: Standard UNet (64 channels), uniform timestep sampling
- **flow-logitnorm-ema (13M)**: Standard UNet, logit-normal sampling, EMA weights
- **flow-v2-zscore (13M)**: Standard UNet, z-score normalization

**Evaluation config**: 10 ensemble members, 10 ODE steps (midpoint sampler), AddCL constraint (ERA5) / no constraint (NorESM).

---

## ERA5 TCW 4x Results

**Source**: `eval_results_500.json` (verified, 500 test samples)

Methods sorted by CRPS (best first). All values from verified JSON.

| Method | CRPS | MAE | RMSE | Mass Viol. |
|--------|------|-----|------|------------|
| flow-wide96-amp (28M) | **0.1721** | **0.2513** | **0.4571** | 1.0e-6 |
| flow-uniform-amp (13M) | 0.1753 | 0.2559 | 0.4662 | 1.0e-6 |
| flow-v2-zscore (13M) | 0.1754 | 0.2559 | 0.4673 | 1.0e-6 |
| flow-logitnorm-ema (13M) | 0.1815 | 0.2658 | 0.5003 | 1.0e-6 |
| swinir-finetuned+addcl | 0.2632 | 0.2632 | 0.5094 | 1.4e-6 |
| swinir-finetuned | 0.2649 | 0.2649 | 0.5108 | 0.0247 |
| harder-gan+smcl | 0.2835 | 0.2866 | 0.5540 | 1.1e-6 |
| harder-cnn+smcl | 0.2951 | 0.2951 | 0.5817 | 1.6e-6 |
| swinir-zeroshot+addcl | 0.3106 | 0.3106 | 0.6844 | 1.4e-6 |
| harder-cnn | 0.3129 | 0.3129 | 0.6277 | 0.0362 |
| swinir-zeroshot | 0.3257 | 0.3257 | 0.7025 | 0.0835 |
| bicubic+addcl | 0.3626 | 0.3626 | 0.7408 | 1.4e-6 |
| bicubic | 0.3939 | 0.3939 | 0.7849 | 0.1492 |
| bilinear+addcl | 0.3991 | 0.3991 | 0.8040 | 1.4e-6 |
| bilinear | 0.5191 | 0.5191 | 0.9639 | 0.3203 |

> **Pending**: RALSD, SSIM, PSNR columns require GPU re-evaluation with updated pipeline. Preliminary stdout data from a prior interrupted run (iter5) showed RALSD ranging from 0.19 dB (flow-wide96) to 1.09 dB (bilinear). These values are NOT included here because they were not saved to disk and cannot be verified.

### ERA5 Key Findings

1. **Flow matching dominates**: All 4 flow models occupy the top 4 positions. The best (flow-wide96-amp, 28M params) achieves CRPS 0.172, a 35% improvement over the next-best non-flow method (swinir-finetuned+addcl, 0.263).

2. **Model capacity matters**: flow-wide96-amp (28M, 96 base channels) consistently outperforms the 13M variants (64 base channels) across all metrics. The CRPS gap (0.172 vs 0.175-0.182) is small but consistent.

3. **Flow v2 zscore matches uniform**: flow-v2-zscore (z-score normalization) performs nearly identically to flow-uniform-amp, suggesting z-score normalization is a viable alternative without degradation.

4. **SwinIR finetuning essential**: Zero-shot SwinIR (0.326 CRPS) performs worse than Harder CNN (0.313), but finetuned SwinIR (0.265) beats all Harder variants.

5. **Constraint layers effective on ERA5**: AddCL reduces mass violation to ~1e-6 for all methods. Constrained baselines consistently outperform unconstrained (bicubic+addcl 0.363 vs bicubic 0.394).

6. **CRPS = MAE for deterministic methods**: Baselines, SwinIR, and Harder-CNN variants show identical CRPS and MAE. Only GAN and flow methods differ (CRPS < MAE), confirming correct probabilistic evaluation.

---

## NorESM TAS 2x Results

**Source**: `noresm_eval_results_500.json` (verified, 500 test samples)

Methods sorted by CRPS. Only flow-wide96-amp available (other flow variants not trained on NorESM).

| Method | CRPS | MAE | RMSE | Mass Viol. |
|--------|------|-----|------|------------|
| flow-wide96-amp (28M) | **0.6486** | 0.9669 | 1.5130 | 1.119 |
| swinir-finetuned | 0.9880 | 0.9880 | 1.5337 | 1.065 |
| harder-cnn | 1.1315 | 1.1315 | 1.6945 | 0.943 |
| harder-cnn+smcl | 1.4535 | 1.4535 | 2.2767 | 4.5e-6 |
| swinir-finetuned+addcl | 1.4550 | 1.4550 | 2.2790 | 6.9e-6 |
| bilinear | 1.4725 | 1.4725 | 2.3071 | 0.162 |
| swinir-zeroshot | 1.4753 | 1.4753 | 2.3239 | 0.067 |
| bicubic | 1.4766 | 1.4766 | 2.3187 | 0.061 |
| bilinear+addcl | 1.4779 | 1.4779 | 2.3232 | 7.2e-6 |
| swinir-zeroshot+addcl | 1.4784 | 1.4784 | 2.3244 | 7.1e-6 |
| bicubic+addcl | 1.4794 | 1.4794 | 2.3253 | 7.3e-6 |
| harder-gan+smcl | 1.4809 | 1.5035 | 2.3449 | 1.2e-5 |

### NorESM Key Findings

1. **Flow matching dominates on CRPS even cross-dataset**: flow-wide96-amp achieves CRPS 0.649, a 34% improvement over swinir-finetuned (0.988) and 56% over bilinear (1.473). The flow model trained on NorESM data shows the same dominance pattern as ERA5.

2. **Constraint layers HURT on NorESM TAS**: This is the opposite of ERA5. Adding constraints degrades CRPS severely:
   - swinir-finetuned (0.988) -> swinir-finetuned+addcl (1.455): **47% worse**
   - harder-cnn (1.131) -> harder-cnn+smcl (1.454): **29% worse**
   - bilinear (1.473) -> bilinear+addcl (1.478): marginally worse

   Hypothesis: AddCL enforces that the downsampled HR output matches the LR input. For NorESM TAS 2x (temperature, not water), the LR->HR mapping may have systematic biases that AddCL forces the output to preserve, preventing the model from correcting them.

3. **High mass violation for best methods**: The top 3 methods (flow, swinir-finetuned, harder-cnn) all have mass violation >0.9. Methods achieving low mass violation (<1e-5) cluster at CRPS ~1.45. This suggests a fundamental accuracy-vs-constraint tradeoff on NorESM.

4. **NorESM is harder than ERA5**: Best CRPS on NorESM (0.649) vs ERA5 (0.172). The 2x SR factor should be easier, but NorESM TAS apparently has more complex spatial structure or less predictable variability.

5. **Harder-GAN fails on NorESM**: harder-gan+smcl (CRPS 1.481) is the worst method. The GAN adversarial loss, which helped on ERA5, appears to hurt on NorESM — possibly due to smaller dataset or different data characteristics.

---

## Cross-Dataset Comparison

| Finding | ERA5 TCW 4x | NorESM TAS 2x |
|---------|-------------|---------------|
| Best method | flow-wide96-amp | flow-wide96-amp |
| Best CRPS | 0.172 | 0.649 |
| Flow vs next-best (%) | -35% | -34% |
| Constraint effect | Consistently helps | Consistently hurts |
| GAN benefit | Yes (harder-gan beats harder-cnn) | No (harder-gan worst overall) |
| SwinIR finetuning gap | 0.326 -> 0.265 (-19%) | 1.475 -> 0.988 (-33%) |

## Figures

### ERA5 Figures (`figures/era5/`)
| File | Description | Status |
|------|-------------|--------|
| `metrics_panel.png` | 2x2 bar chart: CRPS, MAE, RMSE, mass_violation (all 15 methods) | Current (4 metrics) |
| `crps_comparison.png` | Sorted CRPS bar chart | Current |
| `flow_vs_baseline.png` | Best flow vs best baseline (3 metrics) | Current |
| `era5_sample_{0-4}_comparison.png` | Per-sample visual comparison (9 methods) | Current |
| `era5_sample_{0-4}_errors.png` | Per-sample error maps | Current |
| `era5_sample_{0-2}_ensemble.png` | Ensemble spread visualization | Current |
| `era5_sample_{3-4}_ensemble.png` | Ensemble spread visualization | **Missing** (code fix applied, pending GPU re-run) |

### NorESM Figures (`figures/noresm/`)
| File | Description | Status |
|------|-------------|--------|
| `metrics_panel.png` | 2x2 bar chart: CRPS, MAE, RMSE, mass_violation (12 methods) | Current (4 metrics) |
| `crps_comparison.png` | Sorted CRPS bar chart | Current |
| `flow_vs_baseline.png` | Best flow vs best baseline | Current |
| `noresm_sample_{0-4}_comparison.png` | Per-sample visual comparison | Current |
| `noresm_sample_{0-4}_errors.png` | Per-sample error maps | Current |
| `noresm_sample_{0-2}_ensemble.png` | Ensemble spread visualization | Current |
| `noresm_sample_{3-4}_ensemble.png` | Ensemble spread visualization | **Missing** (code fix applied, pending GPU re-run) |

### Pending Figures (require GPU eval with updated pipeline)
- Extended metrics panel (all 7 metrics, 3x3 grid)
- Spectral PSD curves (log-log, ground truth vs methods)
- Spectral bias plot (per-frequency bias for each method)
- RALSD bar chart (dedicated spectral metric comparison)

## Implementation

### New Metrics (`src/downscaling/metrics/`)
- `spectral.py`: `radial_psd()`, `radial_psd_batch()`, `ralsd()`, `spectral_bias()`
- `structural.py`: `ssim()`, `psnr()`

### Evaluation Pipeline (`src/downscaling/evaluation/`)
- `batch_metrics.py`: `compute_batch_metrics()`, `compute_spectral_curves()`
- `evaluate.py`: All eval functions support `return_predictions=True` to collect ensemble-mean predictions for batch metric computation
- Incremental saving: JSON written after each method group to prevent data loss from job cancellation

### Plotting (`src/downscaling/plotting/`)
- `spectral.py`: `plot_psd_comparison()`, `plot_spectral_bias()`, `plot_extended_metrics_panel()`, `plot_ralsd_comparison()`

### Test Coverage
- 64 non-GPU tests pass (spectral metrics, structural metrics, batch metrics, plotting)
- Lint (ruff), format (ruff), typecheck (basedpyright): all pass

## Remaining Work

1. **GPU eval with 7 metrics** — Re-run `run_eval.py --max-samples 500` and `run_eval_noresm.py --max-samples 500` on GPU to produce verified JSON with RALSD/SSIM/PSNR + spectral .npz files
2. **Generate spectral figures** — PSD curves, spectral bias, RALSD bar chart from .npz data
3. **Generate extended metrics panel** — 7-metric bar chart (3x3 grid)
4. **Fix ensemble plots** — Re-run `make_figures.py` on GPU to generate samples 3-4 ensemble plots (code fix already applied)
5. **NorESM flow constraint analysis** — Evaluate flow-wide96-amp+addcl on NorESM to quantify constraint impact on the best model
