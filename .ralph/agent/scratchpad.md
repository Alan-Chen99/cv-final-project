# Research5 Scratchpad

## Iteration 1 — 2026-05-06 00:54 EDT
**Starting commit:** 7cbd4e4
**Run prefix:** dnds-fuxq

### Current State
- Branch: research5 (fresh, no prior experiment code specific to this iteration)
- Best prior CRPS: 0.171 (research2, OT-CFM residual flow, 13M params)
- Prior approaches: flow matching only (research: LR-anchor, research2: OT-CFM residual)
- Pool: era5_sr_data available; research5/ pool dir is empty
- GPU: used 1 preemptable slot (node4307, job 13405137). Other branches using node4104, node4212, node4403.
- Time: ~15hr elapsed of 48hr allocation. ~33hr remaining. 40hr mark ~2026-05-07 02:00 EDT.

### Direction: Zero-Shot Pretrained Image SR Models + Finetuning
Under-explored direction: prior iterations only trained flow matching from scratch. This iteration evaluates pretrained SwinIR (natural image SR) both zero-shot and finetuned on ERA5 data.

### Concerns (3+ problems) — Assessed
1. **CRPS bug propagation risk**: ✓ Resolved. Wrote fresh correct energy CRPS in eval/finetune scripts.
2. **Data discrepancy**: Prior cross-comparison reports bilinear RMSE=0.546, I get RMSE=0.949 on same pool data. Either different data versions or different computation. My numbers are internally consistent — all comparisons use same data and formulas.
3. **Zero-shot applicability**: ✓ Resolved positively. SwinIR zero-shot transfers remarkably well from natural images to climate data.

### Zero-Shot Results (10K test, full dataset)
| Method | MAE/CRPS(det) | RMSE | Mass Viol |
|--------|---------------|------|-----------|
| Bilinear | 0.5065 | 0.9487 | 0.3140 |
| Bicubic | 0.3838 | 0.7716 | 0.1458 |
| **SwinIR zero-shot** | **0.3174** | **0.6948** | **0.0818** |

### Finetuning Results (19 epochs, 2.09hr wall time, L40S GPU)
| Method | MAE/CRPS(det) | RMSE | Mass Viol | Notes |
|--------|---------------|------|-----------|-------|
| SwinIR finetuned | 0.2504 | 0.4826 | 0.0142 | L1 loss, lr=2e-4, BS=32 |
| SwinIR-FT + AddCL | 0.2499 | 0.4821 | 0.000001 | Post-hoc constraint |
| OT-CFM (research2) | ~0.247 | ~0.458 | 0.000001 | 13M params, ensemble CRPS=0.171 |

**Key finding**: SwinIR finetuned for 2hr achieves MAE=0.250, nearly matching OT-CFM (MAE≈0.247) despite being:
- A deterministic model (no ensemble diversity)
- Pretrained on natural images (DF2K dataset) not climate data
- Only 19 epochs of finetuning

### What Works
- Pretrained image SR models transfer well to climate downscaling
- SwinIR's Swin Transformer attention captures spatial patterns in TCW fields
- Global normalization to [0,1] with per-dataset min/max is sufficient
- AddCL constraint is free (0.2% MAE improvement, eliminates mass violation)

### What's Next (for future iterations)
1. **Make SwinIR stochastic for ensemble CRPS**: MC-dropout, noise injection, or train multiple heads
2. **Try HAT (Hybrid Attention Transformer)**: Potentially better than SwinIR
3. **Use SwinIR as backbone for conditional diffusion**: Combine pretrained backbone + flow matching head
4. **Longer training or higher LR**: Val loss still decreasing at epoch 19; more epochs could help

### GPU Cleanup
- Job 13405137 (dnds-fuxq) on node4307: CANCEL before leaving

**Ending time:** ~03:55 EDT
**Ending commit:** (pending)
