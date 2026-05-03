# Iteration 3: More Training + Post-Hoc Constraint Layers on Diffusion

## Summary
Resumed diffusion training from 30 to 60 epochs and applied AddCL (additive constraint layer) as post-processing. CRPS improved from 0.1039 to **0.1008**, a 33% improvement over the paper's best GAN (0.151). AddCL eliminates mass violation with no CRPS penalty.

## Method Changes
- **More training**: Resumed from epoch 30 checkpoint, trained to epoch 60 with cosine LR schedule
- **AddCL post-processing**: Applied additive constraint layer after diffusion sampling
  - `corrected_HR = pred_HR + tile(LR - avgpool4x4(pred_HR))`
  - Enforces: average of each 4x4 HR patch equals corresponding LR pixel (mass conservation)
  - No retraining needed — applied only at inference time
- **Correct CRPS**: Implemented standard CRPS formula alongside paper's version for reference

## Training Details
- 60 epochs total (30 from iter-002, 30 more in iter-003)
- Best val loss: 0.0743 (epoch 41), down from 0.0787 (epoch 30)
- 1 GPU preemption during training (node4302 at epoch 40, resumed on node1632)
- Total new training time: ~135 min on L40S

## Results (2K test samples, 10 ensemble, DDIM-50)

### Progression
| Model | Constraint | Epochs | CRPS | CRPS (std) | MAE | RMSE | Mass viol |
|-------|-----------|--------|------|-----------|-----|------|-----------|
| Bilinear | none | - | 0.507 | - | - | 0.949 | - |
| CNN | none | 200 | 0.310 | - | 0.310 | 0.621 | - |
| CNN | SmCL | 61 | 0.298 | - | 0.298 | 0.598 | 0.000001 |
| GAN (paper) | ScAddCL | - | 0.151 | - | 0.305 | 0.604 | - |
| Diffusion v1 | none | 30 | 0.104 | - | 0.266 | 0.583 | ~0.007 |
| Diffusion v1 | none | 60 | 0.101 | 0.186 | 0.262 | 0.576 | 0.003 |
| **Diffusion v1** | **AddCL** | **60** | **0.101** | **0.186** | **0.262** | **0.574** | **0.000001** |

### Key Findings
1. **More training helps modestly**: 30→60 epochs improved CRPS 0.1039→0.1010 (2.8%)
2. **AddCL is a free lunch**: zero mass violation with no quality penalty (CRPS 0.1010→0.1008)
3. **Diffusion + AddCL beats paper GAN by 33%** on CRPS (0.101 vs 0.151)
4. **CRPS function note**: Paper's `crps_ensemble` has asymmetric weights (`shape[-1]` vs `shape[0]`). All numbers use same formula for fair comparison. Standard CRPS (0.186) also computed.

## CRPS Function Analysis
The paper's CRPS uses `fc.shape[-1]` (=128, image width) for below-obs weights but `fc.shape[0]` (=n_ensemble) for above-obs weights. This makes above-obs weights ~164x larger, creating asymmetry. Since all models use the same function, relative rankings are valid. Standard CRPS = 0.186 is reported alongside for reference.

## Artifacts
- `models/diffusion_v1/best_diffusion.pt` — 60-epoch checkpoint (epoch 41, val_loss=0.0743)
- `scripts/simple_diffusion.py` — Updated with AddCL, correct CRPS, optimizer resume

## Next Steps
1. Try training diffusion with constraint in the loop (SmCL — output in log-space)
2. Larger model with self-attention at bottleneck
3. Flow matching instead of DDPM (potentially faster sampling)
4. Full 10K test evaluation
