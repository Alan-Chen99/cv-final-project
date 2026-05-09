# Hard-Constrained Flow Matching for Climate Downscaling

**Task:** 32×32 → 128×128 spatial super-resolution of ERA5 Total Column Water (TCW)
**Metric:** Energy CRPS (corrected): `CRPS = E|X-y| - 0.5·E|X-X'|`, M=10 ensemble
**Dataset:** Harder et al. `era5_sr_data` — 40K train / 10K val / 10K test, values ~0.04–135 kg/m²
**Training budget:** ≤2hr wall-clock per model (NVIDIA L40S, 48GB)

## Summary of Results

Best model achieves **CRPS = 0.168** (10K test) — a 45% improvement over the GAN baseline from Harder et al. Hard constraints (AddCL) eliminate mass conservation violations at zero CRPS cost.

### Main Results Table (200 test samples, M=10 ensemble, midpoint 10 steps)

| Method | Constraint | CRPS ↓ | MAE ↓ | RMSE ↓ | Mass Viol ↓ | Spread |
|--------|-----------|--------|-------|--------|-------------|--------|
| Bilinear | None | 0.538 | 0.538 | 0.972 | 0.329 | 0.000 |
| Bilinear | AddCL | 0.416 | 0.416 | 0.813 | ≈0 | 0.000 |
| Bicubic | None | 0.411 | 0.411 | 0.794 | 0.154 | 0.000 |
| Bicubic | AddCL | 0.379 | 0.379 | 0.750 | ≈0 | 0.000 |
| **Wide96 UNet** | **None** | **0.180** | **0.263** | **0.471** | 0.005 | 0.281 |
| **Wide96 UNet** | **AddCL** | **0.180** | **0.262** | **0.471** | **≈0** | 0.281 |
| Logit-normal UNet | AddCL | 0.181 | 0.265 | 0.478 | ≈0 | 0.287 |
| Base64 z-score | AddCL | 0.184 | 0.268 | 0.482 | ≈0 | 0.288 |
| Base64 uniform | AddCL | 0.184 | 0.268 | 0.482 | ≈0 | 0.288 |
| CFG UNet | AddCL | 0.184 | 0.269 | 0.483 | ≈0 | 0.286 |

Full test set (10K samples) baselines: bilinear = 0.506, bicubic = 0.384, bicubic+AddCL = 0.353.

### Full Test Set (10K) — Best Model

From research3 branch report (midpoint 5 steps + AddCL):

| Model | Params | CRPS (10K) | RMSE | MAE | Mass Viol |
|-------|--------|-----------|------|-----|-----------|
| **Wide96 UNet** | 28.4M | **0.1676** | 0.450 | 0.244 | ≈0 |
| Base64 UNet | 13.1M | 0.1709 | 0.461 | 0.249 | ≈0 |
| GAN (Harder et al.) | 204K | 0.307 | 0.618 | 0.307 | — |

### Solver Comparison (200 samples, wide96 + AddCL)

| Solver | Steps | NFE | CRPS | Time (s) |
|--------|-------|-----|------|----------|
| Euler | 5 | 5 | 0.187 | 34.8 |
| Euler | 10 | 10 | 0.182 | 67.6 |
| Midpoint | 5 | 10 | 0.180 | 67.7 |
| Midpoint | 10 | 20 | 0.180 | 133.7 |

At equal NFE=10: midpoint (5 steps) beats euler (10 steps) by 1.3%. Diminishing returns beyond midpoint-5.

## Method

### Flow Matching (OT-CFM)

We use Optimal Transport Conditional Flow Matching to learn a velocity field in residual space:

- **Transport path:** `x_t = (1-t)·noise + t·residual` where `residual = HR - bilinear(LR)`
- **Model learns:** velocity `v(x_t, t, condition)` that transports noise to residuals
- **Inference:** ODE solve from t=0 (Gaussian noise) to t=1, then add bilinear(LR) to get HR prediction
- **Conditioning:** Bilinear-upsampled LR concatenated as second input channel

This formulation naturally separates:
1. The coarse spatial structure (carried by bilinear LR, always present)
2. The high-frequency detail (learned by flow matching as residual)

### Architecture: Attention UNet

| Component | Details |
|-----------|---------|
| Input | 2 channels (noisy state + LR condition) at 128×128 |
| Output | 1 channel (predicted velocity) at 128×128 |
| Encoder | 3 levels, channel mults (1, 2, 4) |
| Wide96 | Channels (96, 192, 384) = 28.4M params |
| Base64 | Channels (64, 128, 256) = 13.1M params |
| Each level | 2 ResBlocks (GroupNorm → SiLU → Conv3x3, FiLM time conditioning) |
| Bottleneck | ResBlock → 4-head self-attention at 16×16 → ResBlock |
| Decoder | Symmetric with skip connections |
| Time embed | Sinusoidal positional → MLP |

### Hard Constraints (AddCL)

Additive Constraint Layer from Harder et al. (2208.05424) applied as post-processing:

```
correction = LR_orig - AvgPool(pred_HR)
pred_constrained = pred_HR + repeat_interleave(correction, factor=4)
```

This enforces **exact** mass conservation: the mean of each 4×4 HR block equals the corresponding LR pixel. Key findings:

- **Free lunch:** AddCL costs zero CRPS — the flow model already approximately conserves mass (violation ~0.005 without constraint), so the correction is minimal.
- **SmCL overflows:** The softmax constraint (exp then normalize) overflows on physical-space TCW values (0–135 kg/m²). Would require log-space predictions.
- **Architecture-agnostic:** Same constraint layer works on any model output.

### Training Recipe

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| LR schedule | Cosine annealing |
| Batch size | 32 |
| Epochs | 25 (wide96), 40 (base64) |
| AMP | Yes (mixed precision) |
| Gradient clip | 1.0 |
| Timestep sampling | Uniform [0, 1] |
| Data augmentation | None (hurts at <50 epochs) |
| EMA | None (unstable at <50 epochs) |

## Key Findings

1. **Flow matching dominates interpolation baselines.** CRPS 0.180 vs 0.538 (bilinear) and 0.411 (bicubic) — a 56–67% reduction in CRPS.

2. **Hard constraints are free.** AddCL eliminates mass conservation violations (0.005 → ≈0) with no CRPS degradation. This confirms Harder et al.'s finding that constraint layers work well when LR-HR distributions are aligned.

3. **Wider networks help.** Wide96 (28.4M) beats Base64 (13.1M) by ~2% CRPS, and both fit within the 2hr training budget.

4. **Midpoint solver > Euler at equal NFE.** Midpoint with 5 steps (NFE=10) matches or beats Euler with 10 steps (NFE=10), while both take ~68s for 200 samples.

5. **Diminishing returns past midpoint-5.** Going from midpoint-5 (CRPS=0.180) to midpoint-10 (CRPS=0.180) doubles inference time with no gain.

6. **Uniform timestep sampling is optimal for short training.** Logit-normal (research4) slightly hurt CRPS vs uniform. At ≤50 epochs, the model needs uniform coverage of the full trajectory.

7. **CFG provides no benefit.** Classifier-free guidance (research4) matched or slightly worsened base performance.

8. **Z-score normalization matches min-max.** Research6 showed that z-score normalization of residuals performs identically to the default approach.

## Figures

All figures in `figures/`:

| File | Type | Description |
|------|------|-------------|
| `metrics_crps.png` | Data plot | CRPS comparison bar chart across all methods |
| `metrics_mae.png` | Data plot | MAE comparison bar chart |
| `constraint_effect.png` | Data plot | Grouped bars: with/without AddCL |
| `mass_violation.png` | Data plot | Mass conservation violation by method |
| `sample_grid.png` | Output artifact | 8 samples: LR / HR / Mean / |Error| / Spread |
| `ensemble_members_best.png` | Output artifact | Individual ensemble members with per-member MAE |

## Project Structure

```
src/downscaling/
├── __init__.py
├── data.py            # ERA5 loading, NormStats, normalize/denormalize
├── metrics.py         # crps_energy, crps_paper, mae, rmse, mass_violation, spread
├── constraints.py     # apply_addcl, apply_smcl
├── sampling.py        # euler_sample, midpoint_sample, timestep samplers
├── training.py        # OT-CFM training loop (TrainConfig, train_flow_matching)
├── evaluation.py      # evaluate_flow_model, evaluate_deterministic, load_flow_checkpoint
├── visualization.py   # plot_sample_grid, plot_ensemble_members, plot_metrics_comparison
└── models/
    ├── unet.py        # AttentionUNet (full architecture)
    ├── dit.py         # DiT (Vision Transformer variant)
    └── ddpm.py        # DDPMSchedule, ddim_sample

tests/                 # 51 integration tests (35 CPU, 16 GPU-only)
scripts/
├── evaluate_all.py    # Run evaluations on all methods
└── visualize.py       # Generate all figures

experiments/           # Frozen experiment code (4 directories, 6 branches)
results/               # JSON evaluation results (git-tracked)
figures/               # Generated plots (git-tracked)
```

## Reproducibility

All results can be reproduced:

```bash
# Evaluate all methods (requires GPU + pool data)
python scripts/evaluate_all.py --samples 200 --ensemble 10

# Generate figures (requires GPU for sample plots)
python scripts/visualize.py --samples 8

# Run tests (35 pass on CPU, 16 require GPU)
pytest tests/ -v

# Coverage (full coverage requires GPU for model/sampling tests)
pytest tests/ --cov=downscaling --cov-report=term-missing

# Lint/typecheck
ruff check src/ tests/ scripts/
basedpyright src/ tests/ scripts/
```

Pre-trained weights in pool:
- `pool/datasets/research3/models/unet_wide96_amp/best_flow.pt` (best model)
- `pool/datasets/research3/models/unet_wide96_amp/norm_stats.pt`
- Additional models from research3, research4, research6 branches

## Limitations and Future Work

1. **Evaluation on 200 samples only for flow models.** Full 10K evaluation takes ~50 min per model. The 200-sample estimates are consistent with 10K numbers from research3 (0.180 vs 0.168 — within expected variance from subsample size).

2. **SmCL overflow.** The softmax constraint requires log-space predictions. A log-residual formulation could enable SmCL for non-negative variables.

3. **Single variable only.** Only TCW tested. Multi-variable constraints (Tmin/Tmean/Tmax) remain unexplored in this framework.

4. **No temporal modeling.** Each time step is treated independently. Spatiotemporal flow matching (cf. STVD) could improve temporal coherence.

5. **Standard UNet backbone.** DiT and U-ViT were tested (research3) but underperformed UNet at ≤200 epochs. Longer training or pretrained transformer backbones may change this.
