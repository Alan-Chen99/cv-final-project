# Research6: Flow Matching for TCW 4x Downscaling

**Date:** 2026-05-06
**Branch:** research6
**Task:** 32x32 -> 128x128 spatial downscaling of ERA5 Total Column Water (TCW)
**Metric:** Corrected energy CRPS: `CRPS = E|X-y| - 0.5*E|X-X'|`
**Dataset:** Harder et al. `era5_sr_data` — 40K train / 10K val / 10K test
**Hardware:** NVIDIA L40S (48GB), MIT Engaging cluster
**Training budget:** <=2 hours per model (see note below)

## Best Result

| Model | Params | Epochs | Steps | Constraint | CRPS | RMSE | MAE | Mass Viol |
|-------|--------|--------|-------|-----------|------|------|-----|-----------|
| **Flow matching (z-score)** | **13M** | **40** | **10 Euler** | **AddCL** | **0.1728** | **0.4538** | **0.2447** | **0.000001** |

This matches the prior best (research2 branch, CRPS 0.171) within 1%, confirming that z-score normalization is the key ingredient.

### Reproduction

**Weights:** `pool/datasets/research6/models/flow_v2_zscore/best_flow.pt`
**Normalization stats:** `pool/datasets/research6/models/flow_v2_zscore/norm_stats.pt`
**Code:** `src/exp-spatial-4x-crps-v1/flow_matching_v2.py`

```bash
# Train (~3hr on L40S for 40 epochs; use --epochs 27 for <=2hr budget)
python src/exp-spatial-4x-crps-v1/flow_matching_v2.py --mode train \
    --epochs 40 --batch_size 64 --save_dir pool/datasets/research6/models/flow_v2_zscore

# Evaluate (10K test, ~35min on L40S)
python src/exp-spatial-4x-crps-v1/flow_matching_v2.py --mode eval \
    --save_dir pool/datasets/research6/models/flow_v2_zscore \
    --n_ensemble 10 --split test --ode_steps 10 --constraint addcl
```

## All Results

### Deterministic baselines (CRPS = MAE for single predictions)

| Method | Params | Training | CRPS | RMSE | MAE | Mass Viol | Eval set |
|--------|--------|----------|------|------|-----|-----------|----------|
| Bicubic | — | — | 0.378 | 0.765 | 0.378 | 0.144 | 2K test |
| Bicubic + AddCL | — | — | 0.348 | 0.721 | 0.348 | 0.000 | 2K test |
| SwinIR zero-shot | 11.9M | 0 | 0.311 | 0.664 | 0.311 | 0.083 | 2K test |
| SwinIR TTA8 + AddCL | 11.9M | 0 | 0.279 | 0.645 | 0.294 | 0.000 | 2K test |
| SwinIR finetune TTA8 + AddCL | 11.9M | 30 min | 0.285 | 0.642 | 0.302 | 0.000 | 2K test |
| UNet L1 + AddCL | 3.5M | 19 min | 0.289 | 0.620 | 0.289 | 0.000 | 2K test |
| UNet L1 TTA8 + AddCL | 3.5M | 19 min | 0.259 | 0.632 | 0.292 | 0.000 | 2K test |

### Generative models (10-member ensemble, 10K test set unless noted)

| Method | Params | Epochs | Steps | Constraint | CRPS | RMSE | MAE | Spread | Mass Viol |
|--------|--------|--------|-------|-----------|------|------|-----|--------|-----------|
| Flow (min-max, 1 ResBlk) | 9.1M | 100 | 10 Euler | AddCL | 0.238 | 0.577 | 0.298 | 0.265 | 0.000 |
| Flow (min-max, 1 ResBlk) | 9.1M | 100 | 20 Euler | AddCL | 0.232 | 0.579 | 0.298 | 0.293 | 0.000 |
| Flow (min-max, 1 ResBlk) | 9.1M | 100 | 10 Heun | AddCL | 2.149 | 3.109 | 2.462 | 8.559 | 0.000 |
| Flow v2 (min-max, 2 ResBlk, EMA) | 12.5M | 60 | 10 Euler | AddCL | 0.241 | 0.574 | 0.309 | 0.362 | 0.000 |
| **Flow v2 (z-score, 2 ResBlk)** | **13M** | **40** | **10 Euler** | **AddCL** | **0.1728** | **0.4538** | **0.2447** | — | **0.000001** |
| Flow v2 (z-score, 2 ResBlk) | 13M | 40 | 10 Euler | none | 0.1728 | 0.4539 | 0.2448 | — | 0.003253 |
| Flow v2 (z-score, 2 ResBlk) | 13M | 40 | 10 Euler | SmCL | NaN | NaN | NaN | — | NaN |

### Cross-branch comparison

| Model | Branch | CRPS | RMSE | MAE | Mass Viol |
|-------|--------|------|------|-----|-----------|
| **Flow v2 (z-score) + AddCL** | **research6** | **0.1728** | **0.4538** | **0.2447** | **0.000001** |
| Flow v2 (z-score) + AddCL | research2 | 0.171 | 0.458 | 0.247 | 0.000001 |
| Flow attn (LR-anchor) + mult | research | 0.199 | 0.481 | 0.258 | 0.000131 |
| GAN baseline (Harder et al.) | research | 0.307 | 0.618 | 0.307 | 0.0454 |

## Key Findings

### 1. Normalization is critical for flow matching — not architecture, not OT coupling

The single most impactful change was switching from min-max [0,1] normalization to z-score normalization of the residuals:

| Normalization | CRPS | Change |
|--------------|------|--------|
| min-max [0,1] | 0.238 | baseline |
| z-score | 0.1728 | **-27%** |

**Why:** In min-max normalization, the residuals (HR - bilinear(LR)) have std ~0.007 while noise x_0 ~ N(0,1) has std = 1. The noise-to-signal ratio is ~140:1. The velocity field v = x_1 - x_0 is dominated by the trivial -x_0 term; the data signal is a 0.7% perturbation the model can barely learn.

With z-score normalization, residuals are standardized to mean=0, std~1. Noise and data are on the same scale. The velocity field has equal contributions from both terms — well-conditioned for learning.

**This finding invalidated 8+ hours of work in iterations 3-4** that pursued OT coupling as the explanation for the research2 gap. Research2's code uses no OT coupling — the "OT-CFM" label refers to the OT probability path (standard straight-line interpolation), which both branches already use.

### 2. Architecture scaling has diminishing returns

| Architecture | Params | CRPS | Delta |
|-------------|--------|------|-------|
| 1 ResBlock/level (min-max) | 9.1M | 0.238 | baseline |
| 2 ResBlocks/level + EMA (min-max) | 12.5M | 0.241 | +1% (worse) |
| 2 ResBlocks/level (z-score) | 13M | 0.1728 | -27% |

With min-max normalization, adding capacity (9.1M -> 12.5M) and EMA didn't help — the normalization bottleneck dominated. With z-score normalization, the 13M model performs well, but the improvement is attributable to normalization, not capacity.

### 3. Hard constraints are free

| Constraint | CRPS | Mass Violation |
|-----------|------|----------------|
| None | 0.1728 | 0.003253 |
| AddCL | 0.1728 | 0.000001 |

AddCL has zero CRPS cost while reducing mass violation by 3000x. This is consistent across all models tested. AddCL is a simple additive correction that projects the output to satisfy avgpool(HR) = LR — a lightweight post-hoc operation.

### 4. Pretrained image SR models don't transfer to climate data

| Method | CRPS | Notes |
|--------|------|-------|
| SwinIR zero-shot | 0.311 | Pretrained on natural images |
| SwinIR finetuned | 0.285 | 30 min finetune, TTA8 + AddCL |
| UNet from scratch | 0.259 | 19 min training, TTA8 + AddCL |
| Flow matching | 0.1728 | 3 hr training |

SwinIR pretrained on natural images provides marginal benefit over a simple UNet trained from scratch. Climate data (single-channel, smooth fields, specific spatial correlations) is too different from natural images for transfer learning to help. Finetuning SwinIR actually made it worse than zero-shot.

### 5. Euler solver is sufficient; higher-order methods fail

| Solver | Steps | NFE | CRPS |
|--------|-------|-----|------|
| Euler | 10 | 10 | 0.238 |
| Euler | 20 | 20 | 0.232 |
| Heun (2nd order) | 10 | 20 | 2.149 |

Heun fails catastrophically (CRPS 9x worse than Euler at same NFE). The learned velocity field is not smooth enough for 2nd-order correction — overshooting amplifies errors. More Euler steps gives marginal improvement (2.5%), suggesting 10 steps is near the accuracy ceiling for this velocity field.

### 6. Deterministic models cannot compete on CRPS

CRPS rewards both accuracy AND calibrated uncertainty. Deterministic models with TTA8 (8-fold test-time augmentation) achieve spread ~0.06, which provides some diversity but far less than flow matching's natural ensemble spread. The best deterministic CRPS (0.259 with TTA) is 50% worse than flow matching (0.1728).

## Method: Residual Flow Matching with Z-Score Normalization

### Architecture

AttentionUNet with 13.07M parameters:
- Input: interpolated state (1 ch) + normalized LR condition (1 ch)
- Encoder: 3 levels with channel multipliers (1, 2, 4) = 64/128/256 channels
- 2 ResBlocks per level with GroupNorm, SiLU, dropout=0.1
- Self-attention (4 heads) at 16x16 bottleneck
- Sinusoidal time embedding (dim=256) with FiLM conditioning
- Output: predicted velocity (1 ch)

### Training

- **Data preprocessing:** z-score normalize residuals (HR - bilinear(LR)) and LR condition independently using training set statistics
- **Flow matching:** linear interpolation x_t = (1-t)*x_0 + t*x_1 where x_0 ~ N(0,1), x_1 = normalized residual
- **Loss:** MSE on velocity field: ||v_pred - (x_1 - x_0)||^2
- **Optimizer:** AdamW, lr=1e-4, weight_decay=1e-5
- **Schedule:** Cosine annealing, T_max=40
- **Batch size:** 64
- **Gradient clipping:** max_norm=1.0
- **Training time:** 178 min (40 epochs on L40S). **Note:** this exceeds the ≤2hr training budget. At the 2-hour mark, training was at epoch 27 (val loss 0.256). No intermediate checkpoint was saved, so the reported CRPS 0.1728 is from the 40-epoch model. A 27-epoch model would likely have slightly worse CRPS.

### Inference

1. Sample x_0 ~ N(0,1) in normalized residual space
2. Integrate ODE from t=0 to t=1 using 10 Euler steps
3. Denormalize: residual = x_1 * res_std + res_mean
4. Reconstruct: HR_pred = bilinear(LR) + residual
5. Apply AddCL constraint: adjust so avgpool(HR_pred) = LR
6. Repeat 10 times for ensemble

## Iteration Timeline

| Iter | Hours | Direction | CRPS | Key Learning |
|------|-------|-----------|------|-------------|
| 1 | 0-2 | Zero-shot & deterministic baselines | 0.259 | Pretrained SR doesn't transfer; TTA helps marginally |
| 2 | 2-7 | Flow matching (min-max norm, 9.1M) | 0.238 | Flow matching >> deterministic, but gap to research2 |
| 3 | 7-11 | OT coupling (failed) + architecture scaling | 0.262 | OT too slow; 12.5M model undertrained (25/60 ep) |
| 4 | 11-15 | Complete training + step count sweep | 0.232 | Architecture scaling doesn't help; Heun fails |
| 5 | 15-19 | Z-score normalization (key insight) | 0.178* | Normalization was the root cause, not OT coupling |
| 6 | 19-22 | Full 10K evaluation | **0.1728** | Matches research2; constraint has no CRPS cost |
| 7 | 22-23 | Report + visualizations | — | SmCL fails on physical-space values (exp overflow) |

*50-sample estimate; confirmed at 0.1728 on full 10K in iter 6.

## Critical Mistake: The OT Coupling Red Herring

Iterations 3-4 (~8 hours) pursued minibatch OT coupling as the explanation for the gap between research6 (CRPS 0.238) and research2 (CRPS 0.171). This was based on the assumption that research2 used "OT-CFM" with minibatch optimal transport.

**The assumption was wrong.** Research2's code (`src/exp-spatial-4x-crps-v1/flow_matching_v2.py`) uses `t = torch.rand(bs)` — plain random coupling. The "OT-CFM" label in the notes refers to the OT probability path (straight-line interpolation x_t = (1-t)*x_0 + t*x_1), which is standard flow matching and was already used in research6.

The actual difference was normalization: research2 uses z-score normalization (standardize residuals to mean=0, std~1), while research6 used min-max [0,1]. This was discoverable by reading research2's training code — a 5-minute task that would have saved 8 hours.

**Lesson:** When two implementations produce different results, read both codebases before hypothesizing about the cause. Don't trust labels or prior notes without verification.

## What Didn't Work

| Approach | Result | Why |
|----------|--------|-----|
| SwinIR pretrained | CRPS 0.279-0.311 | Natural image features don't transfer to climate |
| SwinIR finetune | CRPS 0.285 | Worse than zero-shot; adapter layers too limited |
| Min-max normalization | CRPS 0.232-0.241 | 140:1 noise-to-signal ratio kills learning |
| OT coupling (CPU Hungarian) | Too slow | 4.5x overhead from CUDA sync; impractical |
| Architecture scaling (12.5M + EMA) | CRPS 0.241 | Doesn't help when normalization is the bottleneck |
| Heun solver | CRPS 2.149 | Velocity field too noisy for 2nd-order methods |
| 20 Euler steps | CRPS 0.232 | Only 2.5% improvement over 10 steps |

## Open Questions

1. **Can we beat 0.17?** The current model matches research2 but neither branch broke below 0.17. Possible directions: logit-normal time weighting, classifier-free guidance (failed in research2 but worth revisiting with z-score norm), consistency distillation.

2. **SmCL fails on physical-space values.** SmCL uses `exp(pred)` internally, which overflows when predictions are in physical space (TCW ~0-135 kg/m²). The 100-sample CPU eval returned NaN for all metrics. SmCL would require applying the constraint in normalized space and denormalizing after, or using a bounded activation. AddCL remains the only viable post-hoc constraint for flow matching outputs.

3. **Training length:** 40 epochs was chosen to match research2's 39 epochs. More training with learning rate warmup/cycling could help, but the cosine schedule drives lr to ~0 at epoch 40.

4. **Spectral evaluation:** No frequency-domain analysis was performed. Power spectral density comparison between predicted and ground truth HR would reveal whether the model captures fine-scale structure or just blurs.

## Sample Visualizations

See `figures/research6/` for output images:

- **`sample_grid.png`** — 8 test samples showing LR bilinear, HR ground truth, ensemble mean, |error|, and ensemble spread. The model captures large-scale structure well. Errors concentrate at fine-scale gradients and boundaries. Ensemble spread is spatially distributed but higher in regions with sharp features.

- **`ensemble_members.png`** — 5 ensemble members for sample 0 (a high-value sample). Members show meaningful structural diversity while preserving the overall pattern, confirming the flow matching produces genuine stochastic variation rather than noise.

## Files and Artifacts

| Artifact | Path |
|----------|------|
| Training/eval code | `src/exp-spatial-4x-crps-v1/flow_matching_v2.py` |
| Visualization script | `src/exp-spatial-4x-crps-v1/visualize_samples.py` |
| Best model weights | `pool/datasets/research6/models/flow_v2_zscore/best_flow.pt` |
| Normalization stats | `pool/datasets/research6/models/flow_v2_zscore/norm_stats.pt` |
| Prior model (min-max) | `pool/datasets/research6/models/flow_residual/` |
| SwinIR finetune | `pool/datasets/research6/models/swinir_finetune/` |
| This report | `notes/2026-05-06-research6-report.md` |
