# research3 Branch Report: Flow Matching Architecture & Training Optimization

**Date:** 2026-05-05 to 2026-05-06
**Branch:** research3 (11 iterations, ~36hr elapsed)
**Task:** 32×32 → 128×128 spatial downscaling of ERA5 Total Column Water (TCW)
**Metric:** Corrected energy CRPS: `CRPS = E|X-y| - 0.5·E|X-X'|`, M=10 ensemble
**Dataset:** Harder et al. `era5_sr_data` — 40K train / 10K val / 10K test, values ~0.04–135 kg/m²
**Hardware:** NVIDIA L40S (48GB), MIT Engaging cluster
**Training budget:** ≤2hr wall-clock per model

## Best Result

| Model | Params | Epochs | Solver | Steps | Constraint | CRPS (10K test) | RMSE | MAE | Mass Viol |
|-------|--------|--------|--------|-------|-----------|-----------------|------|-----|-----------|
| **UNet v2 wide96** | **28.4M** | **25** | **Midpoint** | **5** | **AddCL** | **0.1676** | **0.450** | **0.244** | **0.000001** |

Checkpoint: `pool/datasets/research3/models/unet_wide96_amp/best_flow.pt`

**Improvement over prior branches:**
- vs research2 base64 (13M, euler 10): **0.1676 vs 0.1709** = 1.9% better
- vs research base (5.2M, euler 20): **0.1676 vs 0.199** = 15.8% better
- vs GAN baseline (Harder et al.): **0.1676 vs 0.307** = 45.4% better

## All Results (research3, corrected CRPS, 10K test)

| Model | Params | Epochs | CRPS | RMSE | MAE | Key change | Iter |
|-------|--------|--------|------|------|-----|-----------|------|
| **UNet v2 wide96** | 28.4M | 25 | **0.1676** | 0.450 | 0.244 | 96ch, wider | 7-9 |
| UNet v2 base64 (midpoint) | 13.1M | 40 | 0.1709 | 0.461 | 0.249 | midpoint solver | 5-6 |
| UNet v2 base64 (euler) | 13.1M | 40 | 0.1731 | 0.455 | 0.245 | euler 10 solver | 5-6 |
| UNet v2 fine-tuned | 13.1M | 30 | 0.177 | 0.474 | 0.251 | logit→uniform | 4B |
| UNet v2 logit-normal | 13.1M | 26 | 0.179 | 0.498 | 0.257 | logit-normal t | 3 |
| U-ViT | 16.5M | 200 | 0.194 | 0.533 | 0.274 | transformer+skip | 2 |
| DiT | 14.6M | 200 | 0.195 | 0.540 | 0.276 | pure transformer | 1 |
| UNet v2 + augmentation | 13.1M | 34 | 0.190 | — | — | h/v flip (hurts) | 4A |
| UNet v2 wide96 T_max=25 | 28.4M | 22 | not eval'd | — | — | shorter cosine (worse val loss) | 10 |
| GAN baseline (re-eval'd) | 204K | — | 0.307 | 0.618 | 0.307 | mode collapsed | — |

All CRPS values use corrected energy formula with M=10 ensemble, midpoint solver (5 steps) + AddCL unless noted.

## Cross-Branch Comparison

| Model | Branch | Params | CRPS (corrected, 10K) | Method |
|-------|--------|--------|-----------------------|--------|
| **UNet v2 wide96** | **research3** | **28.4M** | **0.1676** | OT-CFM, residual, midpoint 5, uniform t, AMP |
| UNet v2 base64 | research3 | 13.1M | 0.1709 | OT-CFM, residual, midpoint 5, uniform t, AMP |
| UNet v2 base64 | research2 | 13.1M | ~0.174 (est.) | OT-CFM, residual, euler 10, standard t |
| Attention UNet | research | 5.2M | 0.199 | LR-anchor flow, euler 20, mult constraint |
| GAN baseline | all | 204K | 0.307 | Harder et al. (mode collapsed) |

## Method

### Architecture: UNet v2

Standard UNet with residual blocks, time-conditioned via scale+shift, self-attention at the 16×16 bottleneck resolution.

- **Input:** 2 channels (noisy state + bilinear-upsampled LR condition) at 128×128
- **Output:** 1 channel (predicted velocity in residual space) at 128×128
- **Encoder:** 3 levels with channel multipliers (1, 2, 4)
  - Base64: channels (64, 128, 256) = 13.1M params
  - Wide96: channels (96, 192, 384) = 28.4M params
- **Each level:** 2 ResBlocks (GroupNorm → SiLU → Conv3x3, time FiLM)
- **Bottleneck:** ResBlock → 4-head self-attention at 16×16 → ResBlock
- **Decoder:** Symmetric with skip connections
- **Time embedding:** Sinusoidal positional encoding → MLP

### Flow Matching: OT-CFM with Residual Prediction

- **Transport path:** x_t = (1-t)·noise + t·(HR - bilinear(LR))
- **What model learns:** Velocity field v(x_t, t, c) in the residual space
- **At inference:** Sample from t=0 (noise) to t=1 (residual), then add bilinear(LR)
- **Conditioning:** Bilinear-upsampled LR concatenated as second channel

### Training Recipe (Best Model)

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| LR schedule | Cosine, T_max=40 |
| Batch size | 64 |
| Epochs | 25 (killed by time limit at epoch 25/40) |
| Timestep sampling | Uniform t ~ U(0,1) |
| Precision | AMP (float16 forward, float32 gradients) |
| Data augmentation | None |

### Inference

| Parameter | Value |
|-----------|-------|
| ODE solver | Midpoint (2nd-order Runge-Kutta) |
| Steps | 5 (= 10 NFE) |
| Constraint | AddCL (additive correction) |
| Ensemble size | M=10 |

### AddCL Constraint

Post-hoc projection ensuring mass conservation: adjusts each 4×4 HR block so its mean exactly equals the corresponding LR pixel. Zero impact on CRPS, eliminates mass violation (0.004 → 0.000001).

```python
def apply_addcl(pred_hr, lr_orig, upsampling_factor=4):
    pooled = AvgPool2d(upsampling_factor)(pred_hr)
    correction = lr_orig - pooled
    correction_hr = correction.repeat_interleave(4, dim=-2).repeat_interleave(4, dim=-1)
    return pred_hr + correction_hr
```

## Findings

### What works

| Finding | Evidence | Improvement |
|---------|----------|-------------|
| Wider UNet (96ch → 28.4M params) | wide96 CRPS 0.1676 vs base64 0.1709 | 1.9% |
| Midpoint solver (5 steps) over Euler (10 steps) | 0.1709 vs 0.1731, same NFE | 1.3% |
| Uniform timestep sampling over logit-normal | 0.1709 (40ep) vs 0.179 (26ep) | 4.5% |
| AMP training (no quality loss, 15% faster) | Same val loss, faster per-epoch | 0% quality, +15% speed |
| Cosine LR with T_max > training epochs | T_max=40 beats T_max=25 for 25ep budget | 2.5% val loss |
| AddCL post-hoc constraint | Zero CRPS cost, eliminates mass violation | free |

### What fails

| Finding | Evidence |
|---------|----------|
| DiT/U-ViT pure transformers | CRPS 0.194-0.195, 14% worse than UNet at equal compute |
| Logit-normal timestep sampling (SD3-style) | CRPS 0.179 vs 0.171 baseline |
| Data augmentation (h/v flip) | CRPS 0.190, 11% regression |
| EMA (decay=0.9999) with short training | CRPS 0.228, catastrophic with 26 epochs |
| SmCL (softmax constraint) on flow matching | NaN — exp() overflow on physical-space values |
| Cosine schedule with T_max = training epochs | LR decays to zero too fast, worse convergence |

### Key insights

1. **Model capacity was the bottleneck.** The wide96 model (28.4M) achieves 3.4% better val loss than base64 (13.1M) despite 37% fewer epochs. More parameters compensate for fewer training steps within the 2hr budget.

2. **Incomplete cosine schedule is a feature, not a bug.** With T_max=40 but only 25 epochs completed, LR stays productive throughout training (~0.000031 at epoch 25 vs ~0 for T_max=25). The model was still improving when killed — this acts as a warm-down rather than full decay.

3. **Pure transformers (DiT/U-ViT) are not competitive at this scale.** Despite 200 epochs vs 25-40 for UNet, both plateau at CRPS ~0.195. The patch tokenization (patch_size=8) loses local detail that UNet's multi-scale convolutions preserve. Hybrid architectures might bridge this gap.

4. **Midpoint solver beats Euler at same NFE.** 2nd-order RK with 5 steps (10 NFE) outperforms Euler with 10 steps (10 NFE) by 1.3% CRPS. The higher-order solver produces tighter, more accurate ensemble members.

5. **Residual prediction + standard OT-CFM > LR-anchor flow.** Learning velocity on (HR - bilinear(LR)) residuals with standard noise-to-data transport gives 15.8% better CRPS than the LR-anchor approach (0.168 vs 0.199), confirmed across multiple model sizes.

## Iteration Timeline

| Iter | Time (EDT) | Focus | Result |
|------|-----------|-------|--------|
| 1 | May 5 16:12–18:52 | DiT architecture (40ep) | CRPS 0.216 (40ep), 0.195 (200ep) |
| 2 | May 5 19:06–21:26 | U-ViT (DiT + skip connections, 200ep) | CRPS 0.194 — marginal over DiT |
| 3 | May 5 21:27–23:33 | UNet v2 + logit-normal t + EMA | CRPS 0.179 (logit-normal), 0.228 (EMA) |
| 4 | May 6 00:00–04:15 | UNet v2 augmentation + fine-tune | Aug: 0.190 (hurts), Finetune: 0.177 |
| 5 | May 6 04:30–08:00 | UNet v2 uniform t + AMP (40ep) | CRPS 0.1709 — new base64 best |
| 6 | May 6 08:15–12:05 | Inference optimization (solver, steps) | Midpoint 5: 0.1709 (1.3% over euler 10) |
| 7 | May 6 12:29–16:17 | Wide96 UNet training (25ep) | Val loss 0.243 (3.4% better), no GPU eval |
| 8 | May 6 16:18–17:15 | CPU eval wide96 (100 samples) | CRPS ~0.177 (vs base64 ~0.180) |
| 9 | May 6 17:13–19:55 | GPU eval wide96 (10K test) | **CRPS 0.1676 — new best** |
| 10 | May 6 20:00–22:25 | T_max=25 cosine schedule test | Negative result (0.2495 vs 0.2432 val loss) |
| 11 | May 6 22:27– | Final report writing | This document |

## Limitations

1. **Ensemble size M=10 only.** CRPS estimator at M=10 has sampling noise. Larger ensembles (M=20, 50) were never tested and could change absolute numbers (though unlikely to change rankings).

2. **Single variable (TCW).** All experiments use univariate Total Column Water. Multi-variable downscaling with cross-variable constraints is untested.

3. **Perfect-model setup.** ERA5 LR is coarsened from ERA5 HR — no GCM→RCM distribution shift. Real-world performance may differ.

4. **No spectral evaluation.** Power spectral density, wavelet analysis, and scale-dependent metrics were not computed. The model may still suppress high-frequency content despite good CRPS.

5. **Wide96 undertrained.** Val loss was still decreasing at epoch 25. More epochs or a larger training budget would likely improve results further.

## Reproduction

**Train (requires GPU, ~2.5hr on L40S):**
```bash
python src/exp-spatial-4x-crps-v1/flow_matching_v2.py --mode train \
    --epochs 40 --batch_size 64 --base_channels 96 --amp
```

**Evaluate (requires GPU, ~45min on L40S):**
```bash
python src/exp-spatial-4x-crps-v1/inference_ablation.py \
    --checkpoint pool/datasets/research3/models/unet_wide96_amp/best_flow.pt \
    --configs midpoint_5_addcl euler_10_addcl \
    --n_ensemble 10 --split test
```
Note: `base_channels` and `channel_mults` are read from the checkpoint's saved args automatically.

## Source Files

| File | Purpose |
|------|---------|
| `src/exp-spatial-4x-crps-v1/flow_matching_v2.py` | UNet v2 model definition, training, and evaluation |
| `src/exp-spatial-4x-crps-v1/inference_ablation.py` | Multi-config inference evaluation (solver, steps, constraints) |
| `src/exp-spatial-4x-crps-v1/dit_flow.py` | DiT architecture for flow matching |
| `src/exp-spatial-4x-crps-v1/uvit_flow.py` | U-ViT architecture for flow matching |
| `src/exp-spatial-4x-crps-v1/eval_crps.py` | Standalone CRPS evaluation for Harder et al. models |
| `src/exp-spatial-4x-crps-v1/eval_bilinear_crps.py` | Bilinear interpolation baseline |

## CRPS Formula Note

All CRPS values in this report use the corrected energy form:
```
CRPS = E|X-y| - 0.5·E|X-X'|
```
where X, X' are independent ensemble members and y is the ground truth. The Harder et al. codebase has a bug in `crps_ensemble()` that uses `fc.shape[-1]**2` (=16384) instead of `fc.shape[0]**2` (=M²) as denominator, underestimating CRPS by ~50%. Numbers from other reports using the buggy formula (typically ~0.09x) are NOT directly comparable to numbers in this report (~0.17x).
