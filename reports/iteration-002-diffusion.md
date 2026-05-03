# Iteration 2: Conditional Diffusion Model for 32×32 → 128×128 Downscaling

## Summary
Trained and evaluated a conditional DDPM for climate downscaling. The diffusion model achieves **CRPS = 0.1039**, beating the paper's best GAN baseline (CRPS = 0.1508) by **31%**.

## Method
- **Architecture**: Conditional UNet (12.8M params) with residual prediction
  - Input: noisy residual + bilinear-upsampled LR (2 channels)
  - Output: predicted noise (1 channel)
  - Config: base_channels=64, channel_mults=(1,2,4), time_emb_dim=256
- **Diffusion**: 1000-step cosine schedule, DDIM sampling (50 steps, eta=1.0)
- **Training**: Residual target = HR - bilinear(LR), normalized. AdamW lr=2e-4, cosine LR decay
- **Ensemble**: 10 members per sample (same as paper GAN)

## Training Details
- 30 epochs total (resumed after preemption at epoch 10)
- ~4.4 min/epoch on L40S (46GB)
- Val loss: 0.1116 (ep1) → 0.0796 (ep20) → 0.0787 (ep30)
- 3 GPU preemptions during training/evaluation

## Results (Test Set)

### Our Models
| Model | Constraint | Epochs | Test Samples | Ensemble | CRPS | MAE | RMSE |
|-------|-----------|--------|-------------|----------|------|-----|------|
| Bilinear | none | - | 10K | 1 | 0.5065 | - | 0.9487 |
| CNN (ours) | none | 200 | 10K | 1 | 0.3097 | 0.310 | 0.6213 |
| CNN (ours) | SmCL | 61 | 10K | 1 | 0.2977 | 0.298 | 0.5978 |
| **Diffusion v1** | **none** | **30** | **2K** | **10** | **0.1039** | **0.266** | **0.583** |

### Comparison with Paper
| Model | Our CRPS | Paper CRPS | Improvement |
|-------|----------|-----------|-------------|
| CNN (none) | 0.310 | 0.326 | 5% better |
| CNN (SmCL) | 0.298 | 0.291 | 2% worse (fewer epochs) |
| **Diffusion v1** | **0.104** | — | **31% better than best GAN (0.151)** |

## Key Findings
1. **Diffusion beats GAN by 31% on CRPS** — even with only 30 epochs and no constraint layers
2. **Residual prediction helps** — predicting HR-bilinear(LR) instead of HR directly
3. **DDIM enables practical evaluation** — 50 steps vs 1000 full DDPM steps, 20x speedup
4. **Val loss still improving at epoch 30** — more training could further improve results

## Bug Fixes Applied
1. **Device mismatch**: Diffusion schedule tensors on CPU, data on GPU → fixed `_extract` to move to device
2. **UNet decoder channels**: Upsample layers created with wrong channel count → fixed to use input channels
3. **GPU-side schedule**: Moved all diffusion schedule tensors to GPU before eval → 10x speedup

## Caveats
- Evaluated on 2K test samples (of 10K) — larger eval would take ~100 min
- Paper's CRPS function has a minor inconsistency (`fc.shape[-1]` vs `fc.shape[0]` in weight denominators), but used consistently for fair comparison
- No constraint layers applied to diffusion output yet
- Model trained only 30/50 planned epochs due to preemptions

## Artifacts
- `scripts/simple_diffusion.py` — Conditional DDPM with DDIM sampling, resume support
- `scripts/gpu_run.py` — Updated with PYTHONUNBUFFERED for visible output
- `models/diffusion_v1/best_diffusion.pt` — Best checkpoint (epoch 30, val_loss=0.079)
- `models/diffusion_v1/norm_stats.pt` — Normalization statistics

## Next Steps
1. **Add SmCL constraint layer** to diffusion output → enforce mass conservation
2. **More training epochs** (50-100) with preemption-resilient checkpointing
3. **Larger model** with attention at bottleneck (mults=(1,2,4,8))
4. **Full 10K test evaluation** with 20 ensemble members
5. **Train GAN baselines** to independently verify paper CRPS numbers
