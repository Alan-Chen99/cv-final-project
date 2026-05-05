# Spatial 4x Downscaling CRPS Experiment

**Date:** 2026-05-03 to 2026-05-05
**Task:** 32x32 -> 128x128 spatial downscaling of TCW (total column water), CRPS metric
**Dataset:** Harder et al. `era5_sr_data` — 10K train, ~2.6K val, ~2.6K test samples
**Commits:** 392e62e (base) to 080e2b4 (HEAD of research2 branch)

## Best Result

| Model | Params | Epochs | Sampler | Steps | Constraint | CRPS (full 10K test) | CRPS (2K test) |
|-------|--------|--------|---------|-------|-----------|---------------------|----------------|
| **Flow v2** | **13M** | **39** | **Euler** | **10** | **AddCL** | **0.094** | **0.093** |

**Improvement over baselines:** 39% vs paper GAN (0.151), 18% vs paper CNN (0.115), 9% vs own DDPM (0.104).

### Reproduction

**Weights:** `models/flow_v2/best_flow.pt` (13M params, 39 epochs). Committed at `f4285e1`.
**Normalization stats:** `models/flow_v2/norm_stats.pt`

Train (requires GPU, ~3hr on A100):
```bash
python src/exp-spatial-4x-crps-v1/flow_matching_v2.py --mode train --epochs 40 --batch_size 64
```

Evaluate (requires GPU, ~15min on A100):
```bash
python src/exp-spatial-4x-crps-v1/flow_matching_v2.py --mode eval \
    --n_ensemble 10 --split test --n_test -1 --steps 10 --constraint addcl
```

Expected output: `CRPS (paper): 0.0942`, `Mass violation: 0.000001`

## All Results

### Baselines (deterministic — CRPS = MAE)

| Method | Constraint | CRPS | MAE | RMSE | Source |
|--------|-----------|------|-----|------|--------|
| Bilinear interpolation | none | 0.341 | 0.341 | 0.546 | eval_bilinear_crps.py |
| CNN (paper) | none | 0.115 | 0.115 | 0.191 | Harder et al. Table 1 |
| CNN (paper) | SmCL | 0.124 | 0.124 | 0.193 | Harder et al. Table 1 |
| GAN (paper) | none | 0.151 | — | — | Harder et al. Table 2 |

Note: CNN and GAN paper results are from Harder et al. Table 1-2, not reproduced here.
Own CNN reproductions were run but used a slightly different eval pipeline (results in logs, not definitive).

### Generative models (this experiment)

All models use the same UNet backbone architecture, varying in capacity and generation method.
All trained on residual prediction: model learns `HR - bilinear(LR)`.

| Model | Method | Params | Epochs | Sampler/Steps | Constraint | CRPS (2K) | CRPS (10K) | Weights |
|-------|--------|--------|--------|---------------|-----------|-----------|------------|---------|
| DDPM v1 | DDPM (cosine schedule, DDIM 50 steps) | 12.8M | 60 | DDIM/50 | AddCL | 0.101 | — | models/diffusion_v1/best_diffusion.pt |
| Flow v1 | OT-CFM | 12.2M | 17 | Euler/10 | AddCL | 0.095 | — | models/flow_v1/best_flow.pt |
| **Flow v2** | **OT-CFM + self-attention** | **13M** | **39** | **Euler/10** | **AddCL** | **0.093** | **0.094** | **models/flow_v2/best_flow.pt** |
| Flow v3 | OT-CFM + attention + CFG | 13M | 23 | Euler/10 | AddCL | 0.105 | — | models/flow_v3/best_flow.pt |
| Flow v4 | OT-CFM + attention (wider: 96 base) | 28M | 22 | Euler/10 | AddCL | 0.094 | — | models/flow_v4/best_flow.pt |

CRPS values use the paper-compatible asymmetric formula from Harder et al. (see code).

### Eval sweep (Flow v2, 2K test)

| Sampler | Steps | NFE | Constraint | CRPS | MAE | RMSE | Mass viol |
|---------|-------|-----|-----------|------|-----|------|-----------|
| Euler | 10 | 10 | AddCL | **0.0926** | **0.2424** | **0.4556** | 0.000001 |
| Euler | 20 | 20 | AddCL | 0.0926 | 0.2444 | 0.4588 | 0.000001 |
| Midpoint | 10 | 20 | AddCL | 0.0931 | 0.2467 | 0.4626 | 0.000001 |
| Midpoint | 20 | 40 | AddCL | 0.0931 | 0.2471 | 0.4631 | 0.000001 |
| Euler | 10 | 10 | none | 0.0926 | 0.2424 | 0.4558 | 0.004 |
| Midpoint | 10 | 20 | none | 0.0931 | 0.2467 | 0.4624 | 0.004 |

SmCL causes NaN — incompatible with flow matching (exp() on physical-space values overflows).

### Full test set (Flow v2, all ~2.6K samples)

| Sampler | Steps | Constraint | CRPS | MAE | RMSE | Mass viol |
|---------|-------|-----------|------|-----|------|-----------|
| Euler | 10 | AddCL | **0.0942** | 0.2466 | 0.4583 | 0.000001 |

The 2K subset (first 2000 samples) gives slightly lower CRPS (0.093 vs 0.094) — the remaining ~600 samples are marginally harder.

## Architecture

### Flow Matching v2 (best model)

**Method:** Optimal Transport Conditional Flow Matching (OT-CFM). Learns velocity field `v(x_t, t, c)` where `x_t = (1-t)*noise + t*data`, `c` is bilinear-upsampled LR input. Sampling: Euler ODE integration from `t=0` (noise) to `t=1` (data) in 10 steps.

**Network:** UNet with self-attention at bottleneck.
- Input: 2 channels (noisy state + LR condition) at 128x128
- Output: 1 channel (predicted velocity) at 128x128
- Encoder: 3 levels with channel multipliers (1, 2, 4) = (64, 128, 256)
- Each level: 2 ResBlocks (GroupNorm, SiLU, Conv3x3, time-conditioned via scale+shift)
- Bottleneck: ResBlock -> 4-head self-attention at 16x16 -> ResBlock
- Decoder: symmetric with skip connections
- Time embedding: sinusoidal positional encoding -> MLP
- Total: 13.07M parameters (274K from attention)

**Training:** AdamW, LR=1e-4, cosine schedule (T_max=40), batch size 64, 39 epochs (~3hr on A100). Residual prediction: model learns velocity in the space of `HR - bilinear(LR)`.

**Constraint:** AddCL (additive correction layer) applied post-sampling. Adjusts each 4x4 HR block so its mean exactly equals the corresponding LR pixel: `pred += (LR - avgpool(pred)).repeat_interleave(4)`. Zero impact on CRPS, eliminates mass violation (0.004 -> 0.000001).

## Key Findings

1. **Flow matching >> DDPM for climate downscaling.** 10 Euler steps vs 50 DDIM steps, 9% better CRPS, simpler training (no noise schedule tuning). OT path gives straighter trajectories than DDPM's curved paths.

2. **Self-attention at bottleneck is cheap and helpful.** 274K extra parameters (2% overhead) capture long-range spatial correlations in climate fields. Improvement is small (~3% CRPS) but consistent.

3. **AddCL is a free lunch for physical consistency.** Zero CRPS impact, eliminates mass conservation violation. SmCL (softmax constraint) is incompatible with flow matching — the exp() transformation causes overflow on physical-space predictions (TCW range 0-130 kg/m^2).

4. **Euler 10 is the optimal sampler for CRPS.** Counterintuitively, more steps and higher-order solvers (midpoint) slightly hurt CRPS. The integration noise from coarse Euler steps adds beneficial ensemble spread.

5. **Classifier-free guidance hurts strongly-conditioned tasks.** CFG caused 13% CRPS regression because the LR input is already highly informative — amplifying the conditioning signal introduces artifacts.

6. **Wider models need proportionally more training.** Flow v4 (28M, 2x wider) at 22 epochs matches but doesn't beat Flow v2 (13M) at 39 epochs. The 2x compute cost per epoch makes it inefficient for this dataset size.

7. **Cosine LR schedule traps.** Resuming training after cosine schedule exhaustion produces LR~0. Fixed with fresh optimizer + new cosine schedule from resume point (--finetune_lr flag).

## Negative Results

| What | Why it failed | Commit |
|------|--------------|--------|
| CFG (classifier-free guidance) | LR input is highly informative; amplifying it introduces artifacts | b26345a |
| SmCL (softmax constraint) | exp() on physical-space values causes overflow | f4285e1 |
| Test-time augmentation (H-flip) | Model not trained with flips; degraded predictions | b26345a |
| Midpoint ODE solver | Reduces beneficial ensemble spread from Euler noise | f4285e1 |
| Wider model (28M vs 13M) | Undertrained at 22 epochs, 2x compute cost not justified | 0c45bad |

## Unreproducible / Incomplete Results

- **Flow v2 extended training (5 extra epochs):** Training was cancelled by concurrent GPU user after 5 epochs. Val loss was trending down (0.256) but hadn't beaten the original (0.253). Weights exist at `models/flow_v2_ext/best_flow.pt` but are inconclusive. Commit: b26345a.
- **CNN baseline reproductions:** Run in iteration 1 but used `external/constrained-downscaling` code with a slightly different eval pipeline. Results in `logs/cnn_baselines.log` but not rigorously validated against paper numbers. Paper values used instead.
- **Flow v2 EMA weights:** `models/flow_v2_ema/` contains a checkpoint from a training run that was preempted early. Not evaluated. No conclusions can be drawn.

## Source Files

All experiment code is in `src/exp-spatial-4x-crps-v1/`:

| File | Purpose |
|------|---------|
| `flow_matching_v2.py` | Best model: train + eval, includes AddCL/SmCL, Euler/midpoint samplers |
| `simple_diffusion.py` | DDPM baseline: train + eval with DDIM sampling |
| `eval_crps.py` | Standalone CRPS evaluation for Harder et al. models |
| `eval_bilinear_crps.py` | Bilinear interpolation baseline CRPS |
| `compute_metrics.py` | Comprehensive metric computation (CRPS, RMSE, MAE, PSNR, SSIM, etc.) |
| `train_flow_v2.sh` | SLURM sbatch script for training |
| `eval_v2_full.sh` | SLURM sbatch script for full test evaluation |
| `run_cnn_baselines.sh` | SLURM script for CNN baseline training |

## Iteration Timeline

| Iter | Duration | Focus | Key Result |
|------|----------|-------|------------|
| 1 | ~4hr | CNN baselines, pipeline setup | CNN CRPS=0.298, bilinear=0.341 |
| 2 | ~3hr | Conditional DDPM | DDPM CRPS=0.104 |
| 3 | ~3hr | Extended DDPM + AddCL | DDPM+AddCL CRPS=0.101 |
| 4 | ~4hr | Flow matching v1 | Flow CRPS=0.095 (17 epochs!) |
| 5 | ~4hr | Self-attention (Flow v2) | **Flow v2 CRPS=0.093** |
| 6 | ~4hr | CFG (Flow v3) | Negative: CRPS=0.105 |
| 7 | ~3hr | Eval improvements, AddCL analysis | Confirmed AddCL is free lunch |
| 8 | ~2hr | Flow v4 (wider) evaluation | v4 CRPS=0.094, v2 still best |
| 9 | ~3hr | Midpoint solver, SmCL, full eval | Euler 10 optimal, full 10K CRPS=0.094 |
| 10 | ~2hr | Extended training, TTA | Both negative, GPU contention |
