# Zero-Shot & Deterministic Baselines for TCW 4x Downscaling

**Date:** 2026-05-06
**Branch:** research6
**Task:** 32x32 → 128x128 spatial downscaling of Total Column Water (ERA5)
**Dataset:** Harder et al. `era5_sr_data` — 40K train / 10K val / 10K test
**Evaluation:** 2K test samples (first 2000), corrected energy CRPS = E|X−y| − 0.5·E|X−X'|

## Question

Can pretrained image super-resolution models (SwinIR) compete with flow matching trained from scratch? How much does ensemble diversity matter vs. individual prediction quality?

## Methods

### Zero-shot SwinIR
- Model: SwinIR-M classical SR x4 (pretrained on DF2K natural images)
- 11.9M parameters, trained for natural image restoration
- Adaptation: single-channel climate data replicated to 3 channels, min-max normalized to [0,1]
- After SR, 3 channels averaged back to 1, denormalized to physical space
- TTA: 8-fold augmentation (flips + 90° rotation combinations) for ensemble diversity

### UNet L1 Regression (trained from scratch)
- SimpleUNetRegression: 3.5M parameters
- Architecture: encoder (32→16→8), decoder (8→16→32), transposed convolution upsample (32→128)
- Channel widths: 64/128/256
- Training: L1 loss, AdamW lr=2e-4, cosine schedule, 100 epochs, batch 128
- Training time: 19.4 minutes on NVIDIA L40S
- Input: normalized LR (32×32), Output: normalized HR (128×128)

### Baselines
- Bicubic interpolation (torch bicubic)
- AddCL constraint (Harder et al.): avgpool(HR) == LR

## Results

| Method | Params | Training | CRPS | RMSE | MAE | Mass Viol | Spread |
|--------|--------|----------|------|------|-----|-----------|--------|
| Bicubic | - | - | 0.378 | 0.765 | 0.378 | 0.144 | 0 |
| Bicubic + AddCL | - | - | 0.348 | 0.721 | 0.348 | 0.000 | 0 |
| SwinIR zero-shot | 11.9M | 0 | 0.311 | 0.664 | 0.311 | 0.083 | 0 |
| SwinIR TTA8 | 11.9M | 0 | 0.293 | 0.662 | 0.310 | 0.083 | 0.029 |
| SwinIR TTA8 + AddCL | 11.9M | 0 | 0.279 | 0.645 | 0.294 | 0.000 | 0.026 |
| UNet L1 deterministic | 3.5M | 19 min | 0.290 | 0.622 | 0.290 | 0.020 | 0 |
| UNet L1 + AddCL | 3.5M | 19 min | 0.289 | 0.620 | 0.289 | 0.000 | 0 |
| UNet L1 TTA8 | 3.5M | 19 min | 0.258 | 0.633 | 0.292 | 0.012 | 0.065 |
| UNet L1 TTA8 + AddCL | 3.5M | 19 min | 0.259 | 0.632 | 0.292 | 0.000 | 0.062 |
| **Prior: OT-CFM** | **13M** | **~3 hr** | **0.171** | **~0.456** | **~0.242** | **0.000** | **~0.07** |

## Key Findings

### 1. Pretrained image SR features provide limited benefit
SwinIR zero-shot (trained on natural images) achieves MAE 0.311 vs bicubic 0.378 — a 17% improvement. However, a simple 3.5M UNet trained from scratch in only 19 minutes achieves MAE 0.290, outperforming the 11.9M pretrained SwinIR. This suggests that climate data patterns (smooth large-scale fields with moderate detail) are sufficiently different from natural images that pretrained features don't transfer well.

### 2. TTA ensemble diversity is limited
8-fold TTA (flips + rotations) provides:
- SwinIR: 6% CRPS improvement (0.311 → 0.293), spread = 0.029
- UNet: 11% CRPS improvement (0.290 → 0.258), spread = 0.065

The UNet benefits more from TTA because it's more sensitive to augmentation (trained without augmentation), but the spread is still small compared to genuine generative ensemble members.

### 3. Generative diversity is essential for CRPS
The flow matching model (CRPS 0.171) achieves:
- MAE 0.242 (worse than UNet's 0.290? No — the flow ensemble mean is better because each member is optimized to cover the distribution)
- Spread that more than compensates for per-member error

The CRPS decomposition shows: generative models benefit from diverse members that bracket the truth, while deterministic models + TTA produce near-identical predictions with minimal spread.

### 4. AddCL has context-dependent value
- For bicubic: AddCL helps CRPS (0.378 → 0.348) because it provides useful structural correction
- For UNet: AddCL is neutral (0.258 → 0.259) because the model already approximately satisfies conservation
- Key: when ensemble diversity exists, AddCL can reduce spread by constraining all members similarly, slightly hurting CRPS

## Implications for Future Work

1. **Don't pursue pretrained natural image SR** — climate data is different enough that training from scratch is preferred
2. **Focus on generative models** — CRPS fundamentally benefits from distributional diversity
3. **Flow matching remains the best approach** — 0.171 CRPS is far ahead of any deterministic method
4. **To beat 0.171, improve the generative model itself** — bigger architecture, better training, or novel approaches (DiT, consistency models, etc.)
5. **Consider CorrDiff-style approach** — deterministic mean (UNet) + generative residuals (flow matching) could potentially improve accuracy while maintaining diversity

## Reproduction

```bash
# Zero-shot SwinIR with TTA and AddCL
python scripts/gpu_run.py JOBID "python src/zero_shot_eval/eval_zeroshot.py --method swinir --tta --constraint addcl --max-samples 2000"

# Train UNet regression (19 min)
python scripts/gpu_run.py JOBID "python src/zero_shot_eval/swinir_refine.py --mode train --model unet --epochs 100 --batch-size 128"

# Evaluate UNet with TTA
python scripts/gpu_run.py JOBID "python src/zero_shot_eval/swinir_refine.py --mode eval --tta --max-samples 2000"
```

## Files

- `src/zero_shot_eval/eval_zeroshot.py` — SwinIR zero-shot evaluation script
- `src/zero_shot_eval/swinir_refine.py` — UNet regression training and evaluation
- `src/zero_shot_eval/network_swinir.py` — SwinIR model definition (from official repo)
- Pool: `research6/weights/swinir_classical_x4.pth` — pretrained SwinIR weights
- Pool: `research6/models/unet_regression/best_model.pt` — trained UNet weights
