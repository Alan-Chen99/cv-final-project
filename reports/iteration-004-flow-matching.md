# Iteration 4: Flow Matching Beats DDPM with Fewer Steps and Less Training

## Summary
Implemented OT conditional flow matching (OT-CFM) with the same UNet backbone as the DDPM model. With only 17 epochs of training and 10 Euler sampling steps, flow matching achieves **CRPS=0.095** — a 5.5% improvement over the DDPM's 0.101 at 60 epochs with 50 DDIM steps. This represents a **37% improvement over the paper's best GAN (0.151)**.

## Method
- **Training**: Learn velocity field v(x_t, t) where x_t = (1-t)*noise + t*data is a linear interpolation
  - Target: v = data - noise (direction from noise to data)
  - Loss: MSE between predicted and true velocity
  - t ~ Uniform(0, 1) during training
- **Sampling**: Euler ODE integration from noise (t=0) to data (t=1)
  - x_{k+1} = x_k + v(x_k, t_k) * dt, with dt = 1/steps
  - 10 steps sufficient (vs 50 DDIM steps for DDPM)
- **Same UNet backbone**: 12.8M params, base_channels=64, channel_mults=(1,2,4)
- **AddCL post-processing**: Same as iter-003, applied after denormalization

## Why Flow Matching Over DDPM?
1. **Simpler**: No noise schedule to tune (linear interpolation is the path)
2. **Fewer steps**: ODE paths are straighter → converge with fewer integration steps
3. **Same architecture**: Only changes training objective and sampling procedure
4. **Under-explored** for climate downscaling (CDSI uses stochastic interpolants, but no paper uses OT-CFM)

## Training Details
- 34 epochs total on L40S across 3 runs (2 preemptions + 1 killed for time)
  - Run 1: epochs 1-13 on node3500, preempted
  - Run 2: epochs 14-17 on node4502, preempted
  - Run 3: epochs 18-34 on node3620, killed at iteration time limit
- Best val loss: 0.255 (epoch 29), down from 0.267 (epoch 17 — when eval was done)
- ~4.4 min/epoch (same as DDPM — identical architecture)
- Cosine LR schedule, AdamW, lr=1e-4
- **Note**: Eval results below are from the 17-epoch checkpoint. The 29-epoch checkpoint should be slightly better but was not re-evaluated due to time.

## Results (2K test samples, 10 ensemble members)

### Flow Matching Sampler Comparison
| Sampler | Steps | Constraint | CRPS (paper) | CRPS (std) | MAE | RMSE | Mass viol |
|---------|-------|-----------|-------------|------------|-----|------|-----------|
| euler | 10 | none | **0.0954** | 0.177 | **0.250** | **0.475** | 0.006 |
| euler | 10 | addcl | 0.0957 | 0.177 | 0.250 | 0.475 | 0.000001 |
| euler | 25 | addcl | 0.0957 | 0.175 | 0.252 | 0.479 | 0.000001 |
| midpoint | 25 | addcl | 0.0961 | 0.175 | 0.254 | 0.483 | 0.000001 |

### Full Progression
| Model | Constraint | Epochs | Steps | CRPS | CRPS (std) | MAE | RMSE | Mass viol |
|-------|-----------|--------|-------|------|-----------|-----|------|-----------|
| Bilinear | none | - | - | 0.507 | - | - | 0.949 | - |
| CNN | none | 200 | - | 0.310 | - | 0.310 | 0.621 | - |
| CNN | SmCL | 61 | - | 0.298 | - | 0.298 | 0.598 | 0.000001 |
| GAN (paper) | ScAddCL | - | - | 0.151 | - | 0.305 | 0.604 | - |
| DDPM | AddCL | 60 | 50 DDIM | 0.101 | 0.186 | 0.262 | 0.574 | 0.000001 |
| **Flow (ours)** | **AddCL** | **17** | **10 Euler** | **0.096** | **0.177** | **0.250** | **0.475** | **0.000001** |

### Key Findings
1. **Flow matching > DDPM**: 5.5% better CRPS with 3.5x less training and 5x fewer sampling steps
2. **10 Euler steps is optimal**: More steps don't improve paper-CRPS (slightly worsen MAE/RMSE)
3. **RMSE improvement is dramatic**: 0.475 vs DDPM's 0.574 = 17% reduction
4. **AddCL still a free lunch**: eliminates mass violation with no quality penalty
5. **Undertrained**: Only 17 epochs — val loss still improving. More training should help further

## Artifacts
- `scripts/flow_matching.py` — Full flow matching implementation
- `models/flow_v1/best_flow.pt` — 29-epoch checkpoint (val_loss=0.255)
- `models/flow_v1/norm_stats.pt` — Normalization statistics

## GPU Preemptions
- node3500 (job 13127325): preempted at epoch 13
- node4502 (job 13131427): preempted at epoch 17
- node3620 (job 13132680): eval completed successfully

## Next Steps
1. Re-evaluate with 29-epoch checkpoint (should be slightly better than 17-epoch)
2. Resume training to 60+ epochs with fresh cosine schedule
3. Self-attention at UNet bottleneck (8x8 spatial, cheap to add)
4. Test with more ensemble members (20 instead of 10)
5. Full 10K test evaluation for final numbers
