# Anvita Checkpoint Comparison

Comparison of Harder et al. CNN/GAN checkpoints trained on this machine vs Anvita's checkpoints (trained externally, 6h each). Evaluated on 2000 test samples using identical data, normalization, and metrics pipeline.

## Checkpoints

Anvita's checkpoints at `/orcd/pool/007/chenxy/datasets/anvita/`. Verified: identical architectures (same state_dict keys, parameter counts, tensor shapes) — only weight values differ.

| File | Dataset | Model | Constraint |
|---|---|---|---|
| `noresm_cnn_none.pth` | NorESM 2x | CNN | none |
| `noresm_cnn_softmax.pth` | NorESM 2x | CNN | SmCL |
| `noresm_gan_none.pth` | NorESM 2x | GAN | none |
| `noresm_gan_softmax.pth` | NorESM 2x | GAN | SmCL |
| `twc_cnn_noconstraints.pth` | ERA5 4x | CNN | none |
| `twc_cnn_softmax.pth` | ERA5 4x | CNN | SmCL |
| `twc_gan_noconstraints.pth` | ERA5 4x | GAN | none |
| `twc_gan_softmax.pth` | ERA5 4x | GAN | SmCL |

## Metrics comparison

### ERA5 TCW 4x

| Config | Source | MAE | RMSE | SSIM | RALSD |
|---|---|---|---|---|---|
| CNN (none) | paper | 0.326 | 0.657 | — | — |
| CNN (none) | ours | 0.300 | 0.611 | 0.988 | 0.546 |
| CNN (none) | anvita | **0.296** | **0.598** | **0.988** | **0.526** |
| CNN (SmCL) | paper | 0.291 | 0.582 | — | — |
| CNN (SmCL) | ours | 0.283 | 0.568 | 0.989 | 0.474 |
| CNN (SmCL) | anvita | 0.283 | **0.565** | **0.989** | **0.436** |
| GAN (none) | paper | 0.313 | 0.628 | — | — |
| GAN (none) | ours | **0.288** | **0.574** | **0.989** | **0.508** |
| GAN (none) | anvita | 0.470 | 0.834 | 0.978 | 0.825 |
| GAN (SmCL) | paper | 0.310 | 0.603 | — | — |
| GAN (SmCL) | ours | **0.274** | **0.541** | **0.990** | **0.441** |
| GAN (SmCL) | anvita | — (killed) | — | — | — |

### NorESM TAS 2x

| Config | Source | MAE | RMSE | SSIM | RALSD |
|---|---|---|---|---|---|
| CNN (none) | paper | 1.559 | 2.348 | — | — |
| CNN (none) | ours | **1.298** | **1.984** | **0.976** | **0.076** |
| CNN (none) | anvita | 1.401 | 2.151 | 0.972 | 0.131 |
| CNN (SmCL) | paper | 1.847 | 2.885 | — | — |
| CNN (SmCL) | ours | 1.723 | 2.778 | **0.967** | 0.310 |
| CNN (SmCL) | anvita | 1.725 | 2.780 | 0.966 | **0.306** |
| GAN (SmCL) | ours | **1.782** | **2.851** | 0.951 | 0.482 |
| GAN (SmCL) | anvita | 1.782 | 2.859 | **0.952** | **0.415** |

## Ensemble collapse

All GAN models exhibit ensemble collapse (noise input is ignored). Ours are more severely collapsed (10-90x less ensemble spread), which paradoxically yields lower MAE by converging to a near-deterministic mean.

| Model | Spread (pixel std) | Max-min range | Pairwise corr |
|---|---|---|---|
| ours ERA5 GAN none | 0.004 | 0.015 | 0.999999 |
| anvita ERA5 GAN none | 0.063 | 0.212 | 0.999666 |
| ours ERA5 GAN SmCL | 0.005 | 0.017 | 0.999999 |
| anvita ERA5 GAN SmCL | 0.019 | 0.064 | 0.999991 |
| ours NorESM GAN SmCL | 0.040 | 0.134 | 0.999991 |
| anvita NorESM GAN SmCL | 0.050 | 0.166 | 0.999993 |
| anvita NorESM GAN none | 0.379 | 1.273 | 0.999661 |

All models have correlation >0.999 and spread <0.4% of value range — complete mode collapse. The GAN architecture from Harder et al. (100-dim noise via ConvTranspose2d concatenated with LR input) does not maintain stochasticity through training.

## Findings

1. **CNN and SmCL configs**: Negligible difference between ours and Anvita (<1% MAE). Both beat the paper. Anvita has marginally better RALSD on SmCL models.

2. **Anvita ERA5 GAN (none) is broken**: RMSE 0.834 vs paper's 0.628 (33% worse). Root cause unknown — loading and normalization verified correct. Likely a bad GAN training run (noisy best-model selection from stochastic validation).

3. **Our NorESM CNN (none) is better**: MAE 1.298 vs Anvita's 1.401. Both beat the paper's 1.559.

4. **All GANs have ensemble collapse**: The noise input has no effect on outputs. Our models are more collapsed, which gives lower pointwise MAE but no meaningful probabilistic predictions. The GAN is effectively a deterministic model.

5. **Longer training (6h) does not help**: Harder et al. reports 3-6h on A100 for 200 epochs. Both sets of checkpoints are in the same regime. The differences are within run-to-run variance, except for Anvita's broken ERA5 GAN (none).
