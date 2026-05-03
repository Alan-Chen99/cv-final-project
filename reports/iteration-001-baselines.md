# Iteration 1: CNN Baselines for 32×32 → 128×128 Downscaling

## Task
- Dataset: ERA5 Total Column Water (TCW4), 4× upsampling (32×32 → 128×128)
- Metric: CRPS (Continuous Ranked Probability Score)
- Training budget: <2hr per model
- Hardware: NVIDIA L40S (46GB VRAM) on MIT preemptable partition

## Reference Results (Harder et al. 2208.05424, Table 5 & 6)

### Deterministic Models (CRPS = MAE for single predictions)
| Model | Constraint | RMSE | MAE (=CRPS) |
|-------|-----------|------|-------------|
| Enlarge | none | 1.286 | 0.717 |
| Bicubic | none | 0.800 | 0.401 |
| CNN | none | 0.657 | 0.326 |
| CNN | SmCL | 0.582 | 0.291 |
| CNN | ScAddCL | **0.575** | **0.289** |

### Probabilistic Models (CRPS < MAE due to ensemble spread)
| Model | Constraint | RMSE | MAE | CRPS |
|-------|-----------|------|-----|------|
| GAN | none | 0.628 | 0.313 | 0.1522 |
| GAN | ScAddCL | 0.604 | 0.305 | **0.1508** |
| GAN | SmCL | 0.603 | 0.310 | 0.1520 |

## Our Baseline Results (Test Set, 10K samples)

| Model | Constraint | Epochs | CRPS | RMSE | Mass Viol. | Neg Pixels |
|-------|-----------|--------|------|------|-----------|-----------|
| Bilinear (no train) | none | - | 0.5065 | 0.9487 | - | - |
| CNN (ours) | none | 200 | 0.3097 | 0.6213 | 0.0433 | 219 |
| CNN (ours) | SmCL | 61* | 0.2977 | 0.5978 | 0.000001 | 0 |

*SmCL model trained for 61 epochs before preemption. Loss plateaued by epoch 15.

### Comparison with Paper
| Model | Our CRPS | Paper CRPS | Delta |
|-------|----------|-----------|-------|
| CNN (none) | 0.310 | 0.326 | -0.016 (better) |
| CNN (SmCL) | 0.298 | 0.291 | +0.007 (slightly worse, fewer epochs) |

## Analysis

### Key Findings
1. **CNN results match paper closely** — validating our pipeline and data setup
2. **SmCL reduces mass violation to ~0** and removes all negative predictions
3. **Loss plateaus by epoch 15-20** — 200 epochs is excessive; 50-100 epochs sufficient
4. **Preemptable partition** caused 2 preemptions — need to handle this in future

### Why CRPS Matters
- For deterministic models: CRPS = MAE (single prediction)
- For ensemble models: CRPS = E|X-y| - 0.5·E|X-X'| < MAE
- The GAN's CRPS (0.15) is **~48% lower** than CNN SmCL MAE (0.29)
- This gap comes entirely from ensemble spread — probabilistic models are critical

### Path to Beating GAN Baseline (CRPS = 0.1508)
1. **Diffusion model**: Better distribution matching than GAN → lower CRPS
2. **Constraint layers**: SmCL on diffusion output → marginal improvement
3. **More ensemble members**: 20-50 samples vs GAN's 10 → better CRPS
4. **Residual prediction**: Predict HR - bilinear(LR) instead of HR directly

## Artifacts Created
- `scripts/simple_diffusion.py` — Conditional DDPM ready to train
- `scripts/eval_crps.py`, `scripts/compute_metrics.py` — Evaluation utilities
- `scripts/eval_bilinear_crps.py` — Bilinear baseline
- `external/constrained-downscaling/models/twc_cnn_none.pth` — CNN baseline checkpoint
- `external/constrained-downscaling/models/twc_cnn_softmax.pth` — CNN SmCL checkpoint

## Next Steps (Priority Order)
1. **Run GAN baselines** — CNN→SmCL and CNN→ScAddCL to confirm paper CRPS values
2. **Train conditional diffusion** — Use scripts/simple_diffusion.py, target CRPS < 0.15
3. **Add SmCL to diffusion output** — Apply constraint layer as post-processing
4. **Experiment with larger model** — More channels/blocks, attention in UNet
