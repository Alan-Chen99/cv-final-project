# Evaluation Metrics Report: Climate Downscaling

Comprehensive evaluation across two datasets, 15+ methods, and 8 metrics spanning pixelwise accuracy, spectral fidelity, structural quality, distributional fidelity, and physical consistency.

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
| EMD | Distribution | Lower better | Earth Mover Distance (Wasserstein-1) between pixel intensity distributions. | [Intercomparison (2512.13987)](https://arxiv.org/abs/2512.13987), [STVD (2312.06071)](https://arxiv.org/abs/2312.06071) |
| Mass violation | Physical | Lower better | Coarse-grain constraint mismatch (avg abs diff after downsampling) | [Harder et al. (2208.05424)](https://arxiv.org/abs/2208.05424) |

### Metric Selection Grounding

The metric suite covers five complementary evaluation axes, grounded in what recent climate downscaling papers use:

| Axis | Metric(s) | Used by |
|------|-----------|---------|
| Probabilistic accuracy | CRPS | CorrDiff, GenDiff, WassDiff, STVD, CDSI, intercomparison |
| Pixelwise accuracy | MAE, RMSE | All papers |
| Spectral fidelity | RALSD, PSD curves | CorrDiff, GenDiff, conditional diffusion, intercomparison, CDSI |
| Structural quality | SSIM, PSNR | Harder et al., FNO downscaling, conditional diffusion, SwinIR |
| Distribution fidelity | EMD | STVD, intercomparison |
| Physical consistency | Mass violation | Harder et al. |

Additional metrics considered but not included:
- **LHD** (Logarithmic Histogram Distance): Used by the intercomparison paper alongside RALSD. Omitted because EMD is more general (metric-space distance, not histogram-bin-dependent) and is used by more papers.
- **CSI** (Critical Success Index): Precipitation-specific threshold metric. Not applicable to TCW or TAS continuous fields.
- **Spread-Skill Ratio**: Requires per-sample ensemble spread, which our eval pipeline computes for CRPS but not yet exposed as a separate metric. Future work.
- **LPIPS**: Learned perceptual metric (WassDiff uses it). Requires a pretrained VGG network; adds model dependency for marginal benefit over SSIM on climate fields.

### RALSD Definition

From the intercomparison paper ([2512.13987](https://arxiv.org/abs/2512.13987)):
1. Compute 2D FFT of each field
2. Radially integrate (bin by wavenumber) -> 1D power spectral density
3. RALSD(dB) = sqrt(1/N * sum_i (10*log10(PSD_true_i / PSD_pred_i))^2)

Captures spectral fidelity: whether predictions reproduce correct spatial frequency content across all scales, not just pixel-level accuracy.

### EMD Definition

Earth Mover Distance (Wasserstein-1) measures the minimum "work" to transform one distribution into another. Applied to the flattened pixel distributions of ground truth and predictions across the full test set. Unlike RMSE which averages pixel-level errors, EMD captures whether the overall intensity distribution is reproduced — a model with low RMSE could still systematically over/under-represent extreme values, which EMD would detect.

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

**Source**: `eval_results_8metrics.json` (verified, 500 test samples, all 8 metrics)

Methods sorted by CRPS (best first). All values from verified JSON.

| Method | CRPS | MAE | RMSE | RALSD (dB) | SSIM | PSNR (dB) | EMD | Mass Viol. |
|--------|------|-----|------|------------|------|-----------|-----|------------|
| flow-wide96-amp (28M) | **0.1718** | **0.2509** | **0.4561** | **0.19** | **0.9925** | **51.6** | **0.0032** | 1.0e-6 |
| flow-uniform-amp (13M) | 0.1755 | 0.2564 | 0.4672 | 0.22 | 0.9921 | 51.4 | 0.0038 | 1.0e-6 |
| flow-v2-zscore (13M) | 0.1755 | 0.2562 | 0.4678 | 0.21 | 0.9921 | 51.4 | 0.0037 | 1.0e-6 |
| flow-logitnorm-ema (13M) | 0.1812 | 0.2653 | 0.4990 | 0.29 | 0.9914 | 51.1 | 0.0046 | 1.0e-6 |
| swinir-finetuned+addcl | 0.2632 | 0.2632 | 0.5094 | 0.34 | 0.9910 | 51.2 | 0.0055 | 1.4e-6 |
| swinir-finetuned | 0.2649 | 0.2649 | 0.5108 | 0.35 | 0.9910 | 51.2 | 0.0209 | 0.0247 |
| harder-gan+smcl | 0.2835 | 0.2866 | 0.5540 | 0.46 | 0.9896 | 50.3 | 0.0068 | 1.1e-6 |
| harder-cnn+smcl | 0.2951 | 0.2951 | 0.5817 | 0.49 | 0.9887 | 50.0 | 0.0081 | 1.6e-6 |
| swinir-zeroshot+addcl | 0.3106 | 0.3106 | 0.6844 | 0.28 | 0.9873 | 49.4 | 0.0068 | 1.4e-6 |
| harder-cnn | 0.3129 | 0.3129 | 0.6277 | 0.57 | 0.9876 | 49.5 | 0.0129 | 0.0362 |
| swinir-zeroshot | 0.3257 | 0.3257 | 0.7025 | 0.25 | 0.9869 | 49.0 | 0.0129 | 0.0835 |
| bicubic+addcl | 0.3626 | 0.3626 | 0.7408 | 0.70 | 0.9843 | 48.0 | 0.0138 | 1.4e-6 |
| bicubic | 0.3939 | 0.3939 | 0.7849 | 0.90 | 0.9825 | 47.4 | 0.0322 | 0.1492 |
| bilinear+addcl | 0.3991 | 0.3991 | 0.8040 | 0.62 | 0.9815 | 47.0 | 0.0217 | 0.0014 |
| bilinear | 0.5191 | 0.5191 | 0.9639 | 1.09 | 0.9758 | 45.1 | 0.0867 | 0.3203 |

### ERA5 Key Findings

1. **Flow matching dominates**: All 4 flow models occupy the top 4 positions. The best (flow-wide96-amp, 28M params) achieves CRPS 0.172, a 35% improvement over the next-best non-flow method (swinir-finetuned+addcl, 0.263).

2. **Model capacity matters**: flow-wide96-amp (28M, 96 base channels) consistently outperforms the 13M variants (64 base channels) across all metrics. The CRPS gap (0.172 vs 0.175-0.182) is small but consistent.

3. **Flow v2 zscore matches uniform**: flow-v2-zscore (z-score normalization) performs nearly identically to flow-uniform-amp, suggesting z-score normalization is a viable alternative without degradation.

4. **SwinIR finetuning essential**: Zero-shot SwinIR (0.326 CRPS) performs worse than Harder CNN (0.313), but finetuned SwinIR (0.265) beats all Harder variants.

5. **Constraint layers effective on ERA5**: AddCL reduces mass violation to ~1e-6 for all methods. Constrained baselines consistently outperform unconstrained (bicubic+addcl 0.363 vs bicubic 0.394).

6. **CRPS = MAE for deterministic methods**: Baselines, SwinIR, and Harder-CNN variants show identical CRPS and MAE. Only GAN and flow methods differ (CRPS < MAE), confirming correct probabilistic evaluation.

7. **Spectral fidelity confirms flow dominance**: flow-wide96-amp achieves RALSD 0.19 dB (best), while bilinear scores 1.09 dB (worst). SwinIR zero-shot has surprisingly good RALSD (0.25 dB) despite mediocre CRPS — its pretrained weights preserve frequency content even without climate-specific training.

8. **EMD reveals distribution shift in unconstrained methods**: Bilinear without constraint has EMD 0.087 (largest), while flow-wide96 has 0.003 (smallest). AddCL improves EMD for baselines (bilinear: 0.087 -> 0.022) by correcting systematic distributional bias.

---

## NorESM TAS 2x Results

**Source**: `noresm_eval_results_8metrics.json` (verified, 500 test samples, all 8 metrics)

Methods sorted by CRPS. Only flow-wide96-amp available (other flow variants not trained on NorESM).

| Method | CRPS | MAE | RMSE | RALSD (dB) | SSIM | PSNR (dB) | EMD | Mass Viol. |
|--------|------|-----|------|------------|------|-----------|-----|------------|
| flow-wide96-amp (28M) | **0.6492** | **0.9669** | **1.5130** | **0.03** | **0.9891** | **39.0** | **0.1740** | 1.119 |
| swinir-finetuned | 0.9880 | 0.9880 | 1.5337 | 0.04 | 0.9864 | 38.7 | 0.2954 | 1.065 |
| harder-cnn | 1.1315 | 1.1315 | 1.6945 | 0.08 | 0.9782 | 37.4 | 0.2842 | 0.943 |
| harder-cnn+smcl | 1.4535 | 1.4535 | 2.2767 | 0.32 | 0.9676 | 35.3 | 0.4993 | 4.5e-6 |
| swinir-finetuned+addcl | 1.4550 | 1.4550 | 2.2790 | 0.32 | 0.9670 | 35.3 | 0.4992 | 6.9e-6 |
| bilinear | 1.4725 | 1.4725 | 2.3071 | 0.34 | 0.9617 | 35.0 | 0.4897 | 0.162 |
| swinir-zeroshot | 1.4753 | 1.4753 | 2.3239 | 0.30 | 0.9617 | 35.0 | 0.4874 | 0.067 |
| bicubic | 1.4766 | 1.4766 | 2.3187 | 0.32 | 0.9625 | 35.0 | 0.5004 | 0.061 |
| bilinear+addcl | 1.4779 | 1.4779 | 2.3232 | 0.31 | 0.9621 | 35.0 | 0.4995 | 7.2e-6 |
| swinir-zeroshot+addcl | 1.4784 | 1.4784 | 2.3244 | 0.31 | 0.9625 | 35.0 | 0.4993 | 7.1e-6 |
| bicubic+addcl | 1.4794 | 1.4794 | 2.3253 | 0.32 | 0.9621 | 35.0 | 0.5004 | 7.3e-6 |
| harder-gan+smcl | 1.4809 | 1.5035 | 2.3451 | 0.38 | 0.9564 | 34.7 | 0.4979 | 1.2e-5 |

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

6. **Spectral: all methods similar in RALSD range 0.03-0.38 dB**: NorESM RALSD range is much narrower than ERA5 (0.19-1.09 dB). The top 3 methods (flow 0.03, swinir-ft 0.04, harder-cnn 0.08) have near-perfect spectral fidelity. Constrained methods cluster at 0.30-0.38 dB — constraint enforcement corrupts spectral content.

7. **EMD confirms bimodal distribution**: The bottom 9 methods (CRPS >1.45) all have EMD ~0.49-0.50, essentially indistinguishable. Only the top 3 methods separate from this cluster: flow (0.17), harder-cnn (0.28), swinir-ft (0.30).

---

## Cross-Dataset Comparison

| Finding | ERA5 TCW 4x | NorESM TAS 2x |
|---------|-------------|---------------|
| Best method | flow-wide96-amp | flow-wide96-amp |
| Best CRPS | 0.172 | 0.649 |
| Best RALSD | 0.19 dB (flow-wide96) | 0.03 dB (flow-wide96) |
| Best EMD | 0.003 (flow-wide96) | 0.174 (flow-wide96) |
| Flow vs next-best CRPS (%) | -35% | -34% |
| Constraint effect | Consistently helps | Consistently hurts |
| GAN benefit | Yes (harder-gan beats harder-cnn) | No (harder-gan worst overall) |
| SwinIR finetuning gap | 0.326 -> 0.265 (-19%) | 1.475 -> 0.988 (-33%) |
| RALSD range | 0.19 - 1.09 dB | 0.03 - 0.38 dB |
| SSIM range | 0.976 - 0.993 | 0.956 - 0.989 |

## Figures

### Cross-Dataset Summary (`figures/`)
| File | Description | Status |
|------|-------------|--------|
| `dual_metrics_panel.png` | ERA5 vs NorESM side-by-side, 8-row x 2-col grid (all 8 metrics) | Current |
| `dual_crps.png` | Cross-dataset CRPS comparison (sorted bars, ERA5 left, NorESM right) | Current |
| `constraint_impact.png` | Delta CRPS (constrained - unconstrained) for 4 method pairs, both datasets | Current |

### ERA5 Figures (`figures/era5/`)
| File | Description | Status |
|------|-------------|--------|
| `metrics_panel.png` | 2x2 bar chart: CRPS, MAE, RMSE, mass_violation (all 15 methods) | Current |
| `crps_comparison.png` | Sorted CRPS bar chart | Current |
| `flow_vs_baseline.png` | Best flow vs best baseline (3 metrics) | Current |
| `extended_metrics_panel.png` | 3x3 grid: all 8 metrics (CRPS, MAE, RMSE, SSIM, PSNR, RALSD, EMD, mass_violation) | Current |
| `ralsd_comparison.png` | Sorted RALSD bar chart (15 methods) | Current |
| `psd_comparison.png` | Log-log PSD curves: ground truth vs all 15 methods | Current |
| `spectral_bias.png` | Per-frequency spectral bias (dB) for each method | Current |
| `era5_sample_{0-4}_comparison.png` | Per-sample visual comparison (9 methods) | Current |
| `era5_sample_{0-4}_errors.png` | Per-sample error maps | Current |
| `era5_sample_{0-2}_ensemble.png` | Ensemble spread visualization | Current |
| `era5_sample_{3-4}_ensemble.png` | Ensemble spread visualization | **Missing** (code fix applied, pending GPU re-run) |

### NorESM Figures (`figures/noresm/`)
| File | Description | Status |
|------|-------------|--------|
| `metrics_panel.png` | 2x2 bar chart: CRPS, MAE, RMSE, mass_violation (12 methods) | Current |
| `crps_comparison.png` | Sorted CRPS bar chart | Current |
| `flow_vs_baseline.png` | Best flow vs best baseline | Current |
| `extended_metrics_panel.png` | 3x3 grid: all 8 metrics (12 methods) | Current |
| `ralsd_comparison.png` | Sorted RALSD bar chart (12 methods) | Current |
| `psd_comparison.png` | Log-log PSD curves: ground truth vs all 12 methods | Current |
| `spectral_bias.png` | Per-frequency spectral bias (dB) for each method | Current |
| `noresm_sample_{0-4}_comparison.png` | Per-sample visual comparison | Current |
| `noresm_sample_{0-4}_errors.png` | Per-sample error maps | Current |
| `noresm_sample_{0-2}_ensemble.png` | Ensemble spread visualization | Current |
| `noresm_sample_{3-4}_ensemble.png` | Ensemble spread visualization | **Missing** (code fix applied, pending GPU re-run) |

## Implementation

### New Metrics (`src/downscaling/metrics/`)
- `spectral.py`: `radial_psd()`, `radial_psd_batch()`, `ralsd()`, `spectral_bias()`
- `structural.py`: `ssim()`, `psnr()`
- `distribution.py`: `emd()`, `emd_per_sample()`

### Evaluation Pipeline (`src/downscaling/evaluation/`)
- `batch_metrics.py`: `compute_batch_metrics()`, `compute_spectral_curves()`
- `evaluate.py`: All eval functions support `return_predictions=True` to collect ensemble-mean predictions for batch metric computation
- Incremental saving: JSON written after each method group to prevent data loss from job cancellation

### Plotting (`src/downscaling/plotting/`)
- `spectral.py`: `plot_psd_comparison()`, `plot_spectral_bias()`, `plot_extended_metrics_panel()`, `plot_ralsd_comparison()`

### Test Coverage
- 144 non-GPU tests pass (spectral metrics, structural metrics, distribution metrics, batch metrics, plotting)
- Lint (ruff), format (ruff), typecheck (basedpyright): all pass

## Remaining Work

1. ~~**GPU eval with 8 metrics**~~ — **DONE** (iter11).
2. ~~**Generate spectral figures**~~ — **DONE** (iter12). PSD curves, spectral bias, RALSD bars, extended metrics panels for both datasets.
3. ~~**Generate extended metrics panel**~~ — **DONE** (iter12). 8-metric 3x3 grid for ERA5 and NorESM.
4. **Fix ensemble plots** — Re-run `make_figures.py` on GPU to generate ERA5 samples 3-4 and NorESM samples 3-4 ensemble plots (code fix already applied)
5. ~~**Update dual metrics panel**~~ — **DONE** (iter13). Now shows all 8 metrics side-by-side.
