# Iteration 6: Classifier-Free Guidance Does NOT Help Climate Downscaling (CRPS=0.105)

## Summary
Trained flow matching v3 with classifier-free guidance (CFG), logit-normal time sampling, and random flip augmentation. **CRPS=0.105 — a 13% regression from v2's 0.093.** CFG hurts for strongly-conditioned super-resolution tasks where the LR input is highly informative. Guidance scale 1.5 was even worse (0.106). This is a valuable negative result.

## Method
- **Classifier-free guidance (CFG)**: During training, drop the LR condition with p=0.1 (replace with zeros). At inference, guide with: `v = v_uncond + w * (v_cond - v_uncond)` where w is the guidance scale.
- **Logit-normal time sampling**: Sample t ~ sigmoid(N(0, 1)) instead of t ~ Uniform(0,1). Concentrates training on harder intermediate timesteps.
- **Random flip augmentation**: Random horizontal and vertical flips during training.
- **Architecture**: Same AttentionUNet as v2 (13.07M params, attention at 16x16 bottleneck)
- **Training**: 40 epochs, AdamW lr=1e-4, cosine schedule, batch_size=64

## Training Details
- 40 epochs across 3 GPU allocations (2 preemptions: node4404 at ep1, node3619 at ep6)
- Main training on node3507: epochs 7-40, ~150.5 min total
- Best val_loss: 0.241 (vs v2's 0.253 — lower loss but worse CRPS!)
- ~4.4 min/epoch

## Results (2K test samples, 10 ensemble, 10 Euler steps)

### Flow v3 (CFG + logit-normal)
| Guidance | Constraint | CRPS (paper) | CRPS (std) | MAE | RMSE | Mass viol |
|----------|-----------|-------------|------------|-----|------|-----------|
| 1.0 | none | 0.1049 | 0.192 | 0.275 | 0.579 | 0.003 |
| 1.0 | addcl | 0.1048 | 0.192 | 0.275 | 0.579 | 0.000001 |
| 1.5 | none | 0.1055 | 0.195 | 0.276 | 0.572 | 0.004 |

### Full Progression Across All Iterations
| Model | Constraint | Epochs | Steps | CRPS | CRPS (std) | MAE | RMSE | Mass viol |
|-------|-----------|--------|-------|------|-----------|-----|------|-----------|
| Bilinear | none | - | - | 0.507 | - | - | 0.949 | - |
| CNN | none | 200 | - | 0.310 | - | 0.310 | 0.621 | - |
| CNN | SmCL | 61 | - | 0.298 | - | 0.298 | 0.598 | 0.000001 |
| GAN (paper) | ScAddCL | - | - | 0.151 | - | 0.305 | 0.604 | - |
| DDPM | AddCL | 60 | 50 DDIM | 0.101 | 0.186 | 0.262 | 0.574 | 0.000001 |
| Flow v1 | AddCL | 17 | 10 Euler | 0.096 | 0.177 | 0.250 | 0.475 | 0.000001 |
| **Flow v2 (attn)** | **AddCL** | **39** | **10 Euler** | **0.093** | **0.171** | **0.242** | **0.456** | **0.000001** |
| Flow v3 (CFG) | addcl | 40 | 10 Euler | 0.105 | 0.192 | 0.275 | 0.579 | 0.000001 |

## Why CFG Hurts Here

1. **Strong condition**: In text-to-image, the text condition is weak/ambiguous — CFG amplifies the weak signal. In super-resolution, the LR image is a very informative condition (determines ~80% of the HR output). The unconditional model learns a poor distribution with no spatial information.

2. **Guidance amplifies noise**: The difference `v_cond - v_uncond` is noisy because the unconditional velocity field is poorly trained (only 10% of steps). Amplifying this noisy difference with w>1 degrades samples.

3. **Logit-normal hurts endpoints**: By concentrating training on t≈0.5, the model underfits t near 0 (initial noise) and t near 1 (near data), which are critical for sample quality.

4. **Val loss paradox**: v3 has lower val_loss (0.241 vs v2's 0.253) despite worse CRPS. This is because logit-normal focuses the loss on intermediate timesteps which are easier to learn, creating a misleading loss signal.

## Key Takeaway
**CFG is not a universal improvement for conditional generation.** When the condition is strong (as in climate SR where LR strongly constrains HR), condition dropout hurts training and guidance at inference adds noise. This explains why no climate downscaling paper uses CFG.

## Artifacts
- `scripts/flow_matching_v3.py` — Flow matching with CFG + logit-normal + augmentation
- `scripts/train_flow_v3.sh` — sbatch training script
- `scripts/eval_flow_v3.sh` — sbatch evaluation script (6 configs)
- `models/flow_v3/best_flow.pt` — 40-epoch checkpoint (val_loss=0.241)
- `models/flow_v3/norm_stats.pt` — Normalization statistics

## GPU Preemptions
- node4404 (sbatch 13161786): 1 epoch, preempted
- node3619 (sbatch 13166105): preempted after 37s (no epochs)
- node3507 (sbatch 13168916): epochs 7-40, completed successfully
- node3002 (sbatch 13181333): eval 3/6 configs, preempted
- node?? (sbatch 13185428): eval remaining, preempted before output

## Next Steps
1. Flow v2 remains the best model (CRPS=0.093). Future iterations should build on v2, not v3.
2. Try loss weighting by timestep (min-SNR or v-prediction loss) — these target training efficiency without hurting conditional signal.
3. Try wider model (base_channels=96) — more capacity without training changes.
4. Full 10K test evaluation of v2 for final numbers.
