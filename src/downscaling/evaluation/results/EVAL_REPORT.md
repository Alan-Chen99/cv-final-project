# Comprehensive Evaluation Report: Climate Downscaling Models

**Date**: 2026-05-11
**Branch**: metrics
**Metrics**: CRPS, MAE, RMSE, mass violation, SSIM, KL divergence, PSD log-ratio, RALSD, spectral coherence, rank histogram, SSR

## Metric Definitions

| Metric | Direction | Description |
|--------|-----------|-------------|
| CRPS | lower better | Continuous ranked probability score (energy form) |
| MAE | lower better | Mean absolute error (ensemble mean vs truth) |
| RMSE | lower better | Root mean squared error (ensemble mean vs truth) |
| MassViol | lower better | \|avgpool(pred) - lr_orig\| — conservation violation |
| SSIM | higher better | Structural similarity index |
| KL | lower better | Histogram KL divergence (nats) |
| PSD-LR | lower better | Mean \|log10(P_pred/P_truth)\| — spectral power match |
| RALSD | lower better | RMS of dB-scale spectral ratio: sqrt(mean((10*log10(S_ref/S_pred))^2)) |
| Coh | higher better | Mean spectral coherence — phase alignment at each frequency |
| SSR | optimal at 1.0 | Spread-skill ratio with finite-size correction |

---

## Dataset 1: NorESM TAS 2x SR

**Variable**: Near-surface air temperature (TAS)
**Resolution**: 32x32 -> 64x64 (2x upsampling)
**Samples**: 2000 test (of ~12K total)
**Ensemble**: 10 members (flow matching, GAN)
**ODE solver**: midpoint, 10 steps

### Models

| Model | Type | Constraint | Params |
|-------|------|-----------|--------|
| Flow+AddCL | Flow matching (wide96) | AddCL | ~113M |
| CNN(none) | Harder et al. ResNet | None | ~100K |
| CNN(softmax) | Harder et al. ResNet | Softmax | ~100K |
| GAN(softmax) | Harder et al. ResNet+noise | Softmax | ~200K |
| SwinIR+AddCL | SwinIR-M finetuned | AddCL | ~11.7M |

### Results

| Model | CRPS | MAE | RMSE | MassViol | SSIM | KL | PSD-LR | RALSD | Coh | SSR |
|-------|------|-----|------|----------|------|-----|--------|-------|-----|-----|
| Flow+AddCL | 1.6414 | 1.7195 | 2.7741 | 0.0000 | 0.9101 | 0.2510 | 0.0285 | 0.317 | 0.977 | 0.088 |
| CNN(none) | **1.2976** | **1.2976** | **1.9840** | 1.1264 | **0.9231** | **0.2045** | **0.0065** | **0.076** | **0.986** | -- |
| CNN(softmax) | 1.7232 | 1.7232 | 2.7782 | 0.0000 | 0.9113 | 0.2558 | 0.0283 | 0.310 | 0.977 | -- |
| GAN(softmax) | 1.7618 | 1.7818 | 2.8513 | 0.0000 | 0.8751 | 0.2666 | 0.0362 | 0.482 | 0.957 | 0.044 |
| SwinIR+AddCL | 1.7252 | 1.7252 | 2.7806 | 0.0000 | 0.9082 | 0.2550 | 0.0285 | 0.312 | 0.976 | -- |
| Bicubic | 1.7490 | 1.7490 | 2.8156 | 0.0626 | 0.8984 | 0.2611 | 0.0316 | 0.406 | 0.973 | -- |
| Bilinear | 1.7499 | 1.7499 | 2.8142 | 0.1665 | 0.8953 | 0.2668 | 0.0264 | 0.321 | 0.973 | -- |

### NorESM Findings

1. **CNN(none) dominates pointwise metrics but violates conservation.** CRPS 1.30, MAE 1.30, SSIM 0.92 — but mass violation 1.13 K. NorESM LR/HR are from different simulations, so the unconstrained model bypasses the LR bottleneck.

2. **Constraints hurt on NorESM.** Adding softmax constraint to CNN degrades CRPS from 1.30 to 1.72 (+33%). All constrained models cluster at CRPS 1.64-1.76, barely beating bicubic (1.75). The conservation constraint is misspecified because LR/HR are not from the same simulation.

3. **Both ensemble models are severely underdispersive.** Flow+AddCL SSR=0.088, GAN SSR=0.044. Rank histograms show extreme U-shapes — ensemble members are near-copies of each other.

4. **2x SR is spectrally easy.** PSD log-ratio < 0.04 for all models. RALSD confirms: CNN(none) at 0.076 dB, constrained models ~0.31 dB. All preserve fine-scale structure.

5. **Spectral coherence uniformly high (~0.97).** All models achieve strong phase alignment with truth at 2x SR. CNN(none) slightly best (0.986). GAN slightly lower (0.957) due to stochastic texture generation.

---

## Dataset 2: ERA5 TCW 4x SR

**Variable**: Total column water (TCW)
**Resolution**: 32x32 -> 128x128 (4x upsampling)
**Samples**: 2000 test (of 10K total)
**Ensemble**: 10 members (all flow matching variants)
**Predictions**: Pre-cached from research2-6 branches

### Models

| Model | Description | Constraint |
|-------|-------------|-----------|
| Flow(none) | Basic flow matching | None (but predictions appear to have AddCL applied*) |
| ResFlow(none) | Residual flow matching | None |
| FlowV2+AddCL | Flow v2 (z-score norm) | AddCL |
| ResFlow-20s+AddCL | Residual flow, 20 ODE steps | AddCL |
| ResFlow-Heun+AddCL | Residual flow, Heun solver | AddCL |

*Flow(none) mass violation is 1e-6, suggesting AddCL was applied during prediction generation despite the filename.

### Results

| Model | CRPS | MAE | RMSE | MassViol | SSIM | KL | PSD-LR | RALSD | Coh | SSR |
|-------|------|-----|------|----------|------|-----|--------|-------|-----|-----|
| Flow(none) | 0.2187 | 0.2929 | 0.5710 | 0.0000 | 0.9282 | 0.0151 | 0.0297 | 0.391 | 0.916 | 0.936 |
| ResFlow(none) | 0.2190 | 0.2952 | 0.5723 | 0.0289 | 0.9261 | 0.0142 | 0.0296 | 0.391 | 0.916 | 0.953 |
| FlowV2+AddCL | 0.2174 | 0.3052 | 0.5684 | 0.0000 | 0.9014 | 0.0212 | 0.0360 | 0.458 | 0.915 | 1.196 |
| ResFlow-20s+AddCL | **0.2116** | **0.2934** | 0.5729 | 0.0000 | 0.9255 | 0.0157 | **0.0270** | **0.350** | 0.915 | **1.013** |
| ResFlow-Heun+AddCL | 1.6307 | 2.4558 | 3.1016 | 0.0000 | 0.2514 | 0.4108 | 1.1922 | 14.252 | 0.398 | 3.023 |
| Bicubic | 0.3778 | 0.3778 | 0.7646 | 0.1436 | **0.9524** | **0.0089** | 0.0722 | 0.887 | 0.884 | -- |
| Bilinear | 0.4989 | 0.4989 | 0.9404 | 0.3096 | 0.9376 | 0.0198 | 0.0915 | 1.067 | 0.872 | -- |

### ERA5 Findings

1. **ResFlow-20s+AddCL is the best model.** CRPS 0.212 (44% better than bicubic 0.378), near-perfect calibration SSR=1.013, best PSD match (0.027), and lowest RALSD (0.350 dB). The 20-step ODE solver provides better integration accuracy.

2. **ERA5 ensembles are well-calibrated — unlike NorESM.** SSR values: 0.94–1.20 for working models (target 1.0). Rank histograms are nearly flat. This is a stark contrast to NorESM where SSR was 0.04–0.09. ERA5 LR/HR come from the same reanalysis (consistent physics), so flow matching stochasticity maps to genuine uncertainty.

3. **ResFlow-Heun+AddCL is broken.** CRPS 1.63, PSD-LR 1.19, RALSD 14.25 dB, SSIM 0.25, coherence 0.40. The Heun solver diverged, producing flat-spectrum noise at high wavenumbers. RALSD is particularly sensitive here — 40x worse than working models. This model should be excluded from analysis.

4. **4x SR is harder than 2x SR — models differentiate more.** Best-model-vs-bicubic improvement: 44% on ERA5 4x vs 6% on NorESM 2x. The 4x task has more fine-scale structure to recover.

5. **Bicubic has highest SSIM (0.952) despite worst CRPS.** SSIM favors smooth fields; bicubic produces blurry but structurally coherent outputs. This highlights SSIM's bias toward smoothness.

6. **Spectral coherence is uniformly high (~0.915) across working models.** All flow models achieve similar phase alignment with truth. Coherence differentiates broken models (Heun: 0.40) but does not separate working models.

---

## Cross-Dataset Comparison

| Aspect | NorESM 2x SR | ERA5 4x SR |
|--------|-------------|-----------|
| Best CRPS | 1.30 (CNN-none, unconstrained) | 0.21 (ResFlow-20s+AddCL) |
| Best CRPS improvement over bicubic | 6% (constrained) / 26% (unconstrained) | 44% |
| Ensemble calibration (SSR) | 0.04–0.09 (severe underdispersion) | 0.94–1.20 (well-calibrated) |
| PSD preservation (PSD-LR) | All < 0.04 (easy) | Working models < 0.04 (good) |
| RALSD (dB) | 0.08–0.48 (CNN-none best) | 0.35–0.46 working; 14.25 broken Heun |
| Spectral coherence | ~0.97 (all models) | ~0.915 (working models); 0.40 (broken Heun) |
| Constraint effect | Harmful (misspecified LR/HR) | Beneficial (consistent LR/HR) |

### Key Insight: Calibration Gap

The most striking cross-dataset finding is the calibration contrast. NorESM ensembles are near-deterministic (SSR ~ 0.05) while ERA5 ensembles are well-calibrated (SSR ~ 1.0). This is NOT a model architecture difference — the same flow matching framework is used in both. The difference is data consistency: ERA5 LR/HR are derived from the same reanalysis, so the residual distribution is well-behaved and flow matching can explore it. NorESM LR/HR are from different simulations, making the residual distribution harder to learn.

## Diagnostic Plots

Plots are saved per dataset in `pool/datasets/metrics/{noresm,era5}/`:
- `psd_comparison.png` — radially averaged PSD vs truth
- `rank_histograms.png` — calibration diagnostic for ensemble models
- `metrics_summary.png` — bar chart comparison of all scalar metrics

## Reproducibility

- Script: `src/downscaling/evaluation/comprehensive.py`
- NorESM: `python -m downscaling.evaluation.comprehensive --dataset noresm --max-samples 2000`
- ERA5: `python -m downscaling.evaluation.comprehensive --dataset era5 --max-samples 2000`
- Both: `python -m downscaling.evaluation.comprehensive --dataset both --max-samples 2000`
- NorESM checkpoints: `pool/datasets/noresm-dataset/models/`
- ERA5 cached predictions: `pool/datasets/era5_sr_data/prediction/`
- Results JSON: `pool/datasets/metrics/{noresm,era5}/*_comprehensive_results.json`
