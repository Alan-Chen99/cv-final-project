# Constrained Flow Matching for Climate Downscaling

## 1. Problem

Statistical downscaling maps coarse-resolution climate fields to their fine-scale
counterparts. This project explores whether modern generative models (flow matching,
diffusion) can improve on the hard-constrained CNN/GAN baselines of Harder et al.
(2208.05424), which enforce physical conservation laws via architectural constraint
layers (AddCL, SmCL).

**Task:** 32x32 → 128x128 spatial super-resolution of ERA5 Total Column Water (TCW).
**Dataset:** Harder et al. `era5_sr_data` — 40K train / 10K val / 10K test samples.
**Metric:** Corrected energy CRPS with 10-member ensembles.
**Constraint:** ≤2 hours wall-clock training per model.

## 2. CRPS Correction

The original `constrained-downscaling` codebase has a bug in `crps_ensemble()`:
it uses `fc.shape[-1]**2` (= 128² = 16384) instead of `fc.shape[0]**2` (= M² = 100
for M=10) in the first summation loop. This underestimates the spread term by ~164x,
making reported CRPS approximate MAE.

All results in this report use the corrected energy CRPS:

```
CRPS = E|X - y| - 0.5 * E|X - X'|
```

with the unbiased M(M-1)/2 pairwise estimator for the spread term
(Gneiting & Raftery, 2007). Implementation: `src/downscaling/metrics/crps.py`.

**Consequence:** Published Harder et al. CRPS numbers (CNN=0.115, GAN=0.151) are not
directly comparable — they use the buggy formula that approximates MAE.

## 3. Methods

### 3.1 Baselines (No Training)

| Method | Description |
|--------|-------------|
| Bilinear | `F.interpolate(LR, scale_factor=4, mode='bilinear')` |
| Bilinear + AddCL | Bilinear with additive constraint layer |
| Bicubic | `F.interpolate(LR, scale_factor=4, mode='bicubic')` |
| Bicubic + AddCL | Bicubic with additive constraint layer |

### 3.2 Flow Matching (OT-CFM)

All flow models use optimal transport conditional flow matching on the residual
space (HR − bilinear(LR)), with z-score normalization of the residuals.

**Architecture:** AttentionUNet — encoder-decoder with skip connections, self-attention
at the 16x16 bottleneck (4 heads). Time conditioning via sinusoidal embedding + FiLM.

**Training recipe:**
- Optimizer: AdamW, LR=1e-4, cosine schedule
- Mixed precision (AMP) training
- Transport path: `x_t = (1-t) · noise + t · residual`
- Model predicts velocity `v(x_t, t, c)` in normalized residual space
- Conditioning: bilinear-upsampled LR concatenated as second channel

**Sampling:** ODE integration from t=0 (noise) to t=1 (residual), then
denormalize and add bilinear(LR). Each ensemble member uses a different noise seed.

**Constraint:** AddCL applied post-hoc — adds a spatially-uniform correction so that
`AvgPool(pred_HR) == LR` exactly (mass conservation). Zero cost at inference.

| Model Variant | Params | Key Difference |
|---------------|--------|----------------|
| flow-wide96-amp | 28.4M | 96/192/384 channels, uniform t, midpoint solver |
| flow-v2-zscore | 13.1M | 64/128/256 channels, z-score normalization |
| flow-uniform-amp | 13.1M | 64/128/256 channels, uniform t, AMP |
| flow-logitnorm-ema | 13.1M | Logit-normal timestep sampling + EMA |

### 3.3 Other Methods Explored (Across Research Branches)

| Method | Branch | Result | Verdict |
|--------|--------|--------|---------|
| DDPM (VP-SDE, DDIM 50 steps) | research2,4 | CRPS 0.191 | Works but slower sampling, worse CRPS |
| DiT (Diffusion Transformer) | research3,5 | CRPS 0.195-0.243 | Patch tokenization loses spatial info |
| U-ViT (Transformer + skip) | research3 | CRPS 0.194 | Similar to DiT, no advantage over UNet |
| OT-CFM + CFG (guidance) | research4 | CRPS 0.196 | Hurts — task is strongly conditioned |
| OT-CFM + Spectral Loss | research4 | CRPS 0.204 | FFT loss hurts CRPS |
| LR-Anchor Flow | research | CRPS 0.199 | Starting from LR+noise instead of pure noise |
| SwinIR Multi-Head (K=8) | research5 | CRPS 0.183 | Pretrained backbone, direct CRPS loss |
| SwinIR-Conditioned OT-CFM | research5 | CRPS 0.173 | SwinIR prediction as additional conditioning |
| Min-max normalization | research6 | CRPS 0.238 | 27% worse than z-score; normalization is critical |
| Data augmentation (flips) | research3 | CRPS 0.190 | Hurts — breaks spatial structure |

## 4. Results

### 4.1 Main Comparison (500 test samples, 10-member ensemble, midpoint sampler, AddCL)

Evaluated with `scripts/run_eval.py` using the organized `src/downscaling/` library.
All methods evaluated on the same 500 test samples for fair comparison.

| Method | CRPS ↓ | MAE ↓ | RMSE ↓ | Mass Violation ↓ |
|--------|--------|-------|--------|-----------------|
| **flow-wide96-amp (28M)** | **0.1719** | **0.2511** | **0.4563** | 0.000001 |
| flow-v2-zscore (13M) | 0.1754 | 0.2560 | 0.4668 | 0.000001 |
| flow-uniform-amp (13M) | 0.1756 | 0.2564 | 0.4670 | 0.000001 |
| flow-logitnorm-ema (13M) | 0.1814 | 0.2656 | 0.4987 | 0.000001 |
| bicubic + AddCL | 0.3626 | 0.3626 | 0.7408 | 0.000001 |
| bicubic | 0.3939 | 0.3939 | 0.7849 | 0.1492 |
| bilinear + AddCL | 0.3991 | 0.3991 | 0.8040 | 0.000001 |
| bilinear | 0.5191 | 0.5191 | 0.9639 | 0.3203 |

### 4.2 Baselines on Full 10K Test Set

| Method | CRPS ↓ | MAE ↓ | RMSE ↓ | Mass Violation ↓ |
|--------|--------|-------|--------|-----------------|
| bicubic + AddCL | 0.3533 | 0.3533 | 0.7283 | 0.000001 |
| bicubic | 0.3838 | 0.3838 | 0.7716 | 0.1458 |
| bilinear + AddCL | 0.3888 | 0.3888 | 0.7911 | 0.000001 |
| bilinear | 0.5065 | 0.5065 | 0.9487 | 0.3140 |

### 4.3 Cross-Branch Best Models (10K test, from individual branch evaluations)

These numbers come from the original experiment evaluations on their respective
branches, using the corrected energy CRPS formula.

| Model | Branch | Params | CRPS | RMSE | MAE | Mass Viol |
|-------|--------|--------|------|------|-----|-----------|
| **UNet wide96 (OT-CFM)** | research3 | 28.4M | **0.1676** | 0.450 | 0.244 | 0.000001 |
| UNet base64 (OT-CFM, midpoint) | research3 | 13.1M | 0.1709 | 0.461 | 0.249 | 0.000001 |
| Flow v2 z-score | research6 | 13.0M | 0.1728 | 0.454 | 0.245 | 0.000001 |
| SwinIR-Conditioned OT-CFM | research5 | 13.0M | 0.173 | 0.464 | 0.312 | 0.005 |
| UNet base64 (OT-CFM, euler) | research3 | 13.1M | 0.1731 | 0.455 | 0.245 | 0.000001 |
| OT-CFM + Logit-Normal FT | research4 | 13.0M | 0.1840 | 0.451 | 0.243 | 0.000001 |
| SwinIR Multi-Head K=8 | research5 | 0.8M | 0.183 | — | 0.250 | 0.000 |
| OT-CFM Baseline | research4 | 13.0M | 0.1865 | 0.455 | 0.245 | 0.000001 |
| DDPM VP-SDE | research4 | 13.0M | 0.1907 | 0.478 | 0.250 | 0.000001 |
| Attention UNet (LR-anchor) | research | 5.2M | 0.199 | 0.481 | 0.258 | 0.000131 |
| GAN (Harder et al., re-eval) | research | 204K | 0.307 | 0.618 | 0.307 | 0.0454 |

## 5. Key Findings

### 5.1 What Works

1. **Flow matching >> GAN/CNN for probabilistic downscaling.**
   Best flow model (CRPS 0.168) is 45% better than re-evaluated GAN (0.307) and 2x
   better than the best non-learned baseline (bicubic+AddCL, 0.353).

2. **Hard constraints are free.**
   AddCL reduces mass violation to ~10⁻⁶ with no measurable CRPS cost. All flow
   models achieve identical CRPS with and without AddCL (confirmed in research6:
   0.1728 with vs 0.1728 without). Constraint is a post-hoc projection — no
   retraining needed.

3. **Soft constraints cannot match hard constraints.**
   Five strategies tested (penalty weighting, curriculum, Lagrangian, L1): violation
   floors at 0.015+. Hard constraints achieve 0.000001. The gap is fundamental.

4. **Z-score normalization is critical.**
   Switching from min-max [0,1] to z-score normalization of residuals improved CRPS
   by 27% (0.238 → 0.173). This is more impactful than architecture changes.

5. **Residual prediction + standard OT-CFM > LR-anchor.**
   Learning velocity on residual space (HR − bilinear(LR)) with standard OT transport
   from pure noise outperforms LR-anchor flow (0.171 vs 0.199 = 14% better). Straighter
   optimal transport paths converge in fewer steps (10 vs 20).

6. **Midpoint solver > Euler at same step count.**
   5-step midpoint matches or beats 10-step Euler (research3: 0.1709 vs 0.1731),
   halving inference cost.

7. **Self-attention at bottleneck is cheap and effective.**
   4-head self-attention at 16x16 resolution adds ~2% compute overhead with consistent
   ~3% CRPS improvement.

### 5.2 What Fails

1. **CFG hurts strongly-conditioned tasks.** +13% CRPS regression (research4). The LR
   condition already determines >95% of variance; guidance over-sharpens.

2. **Spectral/FFT loss hurts CRPS.** Frequency-domain supervision conflicts with
   ensemble diversity optimization.

3. **DiT/U-ViT underperform UNet.** Patch tokenization loses fine-scale spatial
   information critical for SR. Both architectures plateau at CRPS ~0.195.

4. **Data augmentation (flips) hurts.** Climate fields have spatially-varying statistics;
   random flips break this structure.

5. **EMA is redundant with cosine-to-zero schedule.** No gain observed in either
   research track when cosine schedule already decays LR to near-zero.

6. **SmCL incompatible with flow matching post-hoc.** The softmax constraint applies
   `exp()` to model output, causing overflow on residual-space predictions. SmCL
   requires integration into training (not tested with flow matching).

### 5.3 Model Scaling

Wider models help, but with diminishing returns:

| Model | Params | CRPS | Relative |
|-------|--------|------|----------|
| Attention UNet (LR-anchor) | 5.2M | 0.199 | baseline |
| UNet base64 (OT-CFM) | 13.1M | 0.171 | −14% |
| UNet wide96 (OT-CFM) | 28.4M | 0.168 | −16% |

The 5.2M → 13M jump also changed the flow formulation (LR-anchor → OT-CFM residual),
so the capacity effect is confounded. The 13M → 28M comparison is clean (same method,
only width differs): +2x params → −2% CRPS. Diminishing returns suggest the 13M
architecture is near-optimal for this dataset/task.

## 6. Figures

All figures generated by `scripts/make_figures.py` from evaluation results in
`eval_results_500.json`. Flow model predictions use the wide96-amp checkpoint
with 10-member ensemble, midpoint sampler (5 steps), and AddCL constraint.

| Figure | Description |
|--------|-------------|
| `figures/crps_comparison.png` | CRPS bar chart across all methods |
| `figures/metrics_panel.png` | 2x2 panel: CRPS, MAE, RMSE, Mass Violation (log scale) |
| `figures/flow_vs_baseline.png` | Grouped bar: best flow model vs best baseline |
| `figures/sample_*_comparison.png` | Side-by-side: LR, HR, bilinear, bicubic, bicubic+AddCL, flow |
| `figures/sample_*_errors.png` | Absolute error heatmaps with shared colorscale |
| `figures/sample_*_ensemble.png` | Ensemble mean, std, and absolute error maps |

## 7. Code Organization

```
src/downscaling/
├── models/unet.py          # AttentionUNet with building blocks
├── metrics/crps.py         # Corrected energy CRPS + paper-compatible version
├── constraints/layers.py   # AddCL and SmCL constraint layers
├── sampling/ode.py         # Euler and midpoint ODE solvers
├── sampling/timesteps.py   # Uniform and logit-normal timestep sampling
├── data/era5.py            # ERA5 TCW 4x data loader
├── training/ema.py         # Exponential moving average
├── training/flow_matching.py  # OT-CFM trainer with TrainConfig
├── evaluation/evaluate.py  # evaluate_ensemble() and evaluate_flow_model()
├── evaluation/baselines.py # Bicubic/bilinear baselines with optional constraints
├── evaluation/checkpoints.py  # Model checkpoint loading
└── plotting/               # Metrics and sample visualization
```

**Integration tests:** 68 tests in `tests/`, 93% coverage (excluding plotting boilerplate).
Tests run on CPU; GPU accelerates but is not required. Data tests need pool disk.

**Experiments:** Frozen snapshots in `experiments/` from 6 research branches.
Each was active during a research iteration and is read-only after merge.

## 8. Reproduction

### Evaluate all methods (requires GPU)

```bash
srun --gres=gpu:1 --mem=32G --time=01:00:00 \
  python scripts/run_eval.py --n-samples 500 --n-ensemble 10
```

### Generate figures (metrics: CPU only; samples: GPU for flow model)

```bash
# Metrics only (from existing JSON)
python scripts/make_figures.py --metrics-only

# Full figures including flow model samples
srun --gres=gpu:1 --mem=32G --time=00:30:00 \
  python scripts/make_figures.py
```

### Run tests

```bash
srun --gres=gpu:1 --mem=16G --time=00:15:00 pytest tests/ -v
```

## 9. Trained Model Checkpoints

All checkpoints stored on pool disk (`/home/chenxy/orcd/pool/datasets/`).

| Model | Path | Params | CRPS (10K) |
|-------|------|--------|------------|
| UNet wide96 (best) | `research3/models/unet_wide96_amp/best_flow.pt` | 28.4M | 0.1676 |
| UNet base64 (midpoint) | `research3/models/unet_base64_midpoint/best_flow.pt` | 13.1M | 0.1709 |
| Flow v2 z-score | `research6/models/flow_v2_zscore/best_flow.pt` | 13.0M | 0.1728 |
| OT-CFM + Logit-Normal | `research4/models/logit_normal_flow/best_flow.pt` | 13.0M | 0.1840 |
| DDPM VP-SDE | `research4/models/ddpm/best_diffusion.pt` | 13.0M | 0.1907 |
| Attention UNet (LR-anchor) | `research/models/flow_attn/flow_best.pth` | 5.2M | 0.199 |

## 10. Limitations

1. **Single variable (TCW).** All experiments use ERA5 total column water. Multi-variable
   downscaling with cross-variable constraints is untested.

2. **Perfect-model setup.** LR is obtained by coarsening HR (average pooling), not from
   an actual GCM. Real GCM→RCM distribution shift may change conclusions.

3. **Spatial only.** Spatiotemporal downscaling (video diffusion) was not explored.

4. **No spectral evaluation.** Power spectral density and wavelet-based metrics were
   not computed, despite being proposed in the project outline. CRPS, MAE, and RMSE
   are all pixel-space metrics.

5. **CRPS formula inconsistency with Harder et al.** Our corrected CRPS numbers are not
   directly comparable to published Harder et al. numbers due to their buggy formula.
   We did not reproduce their baselines with corrected CRPS.

6. **SmCL untested with flow matching.** SmCL causes overflow when applied post-hoc to
   residual-space predictions. Training-time SmCL integration was not attempted.
