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

**Ending time:** ~04:00 EDT
**Ending commit:** e1b97ed

## Iteration 2 — 2026-05-06 04:00 EDT
**Starting commit:** a824136
**Run prefix:** jrut-ohex

### Current State
- Time: ~18hr elapsed. ~22hr to 40hr mark (2026-05-07 02:00 EDT). Plenty of time.
- GPU: prior job (dnds-fuxq/13405137) already cleaned up. Need fresh allocation.
- squeue: 2 normal + 2 preemptable used by others. Can allocate 1 preemptable.
- Best SwinIR-FT: MAE=0.250 deterministic. OT-CFM best: CRPS=0.171 (ensemble).

### Concerns (3+ problems)
1. **Workflow: No ensemble CRPS ever computed on this branch.** The entire objective metric is CRPS, but iteration 1 only measured deterministic MAE (=CRPS for det. model). Comparison to OT-CFM's 0.171 is apples-to-oranges since SwinIR's 0.250 is det. CRPS. Must make model stochastic.
2. **Quality: LR schedule mismatch.** Cosine annealing with T_max=50 but only 19 epochs ran (2hr wall limit). LR at epoch 19/50 is still ~50% of initial LR — schedule never completed its cooldown. This likely caused the val loss oscillation seen around epochs 12-19.
3. **Quality: Val MAE oscillation indicates potential overfitting.** Physical MAE chart shows val bumps at epochs 13-15 while train keeps dropping. The gap widens in later epochs. Full backbone finetuning (11.9M params) on 40K samples may be overdoing it.

### Direction: Multi-Head SwinIR with Direct CRPS Loss
**Why this direction:**
- Directly optimizes the actual metric (CRPS energy score)
- Under-explored: no prior work combines pretrained SR backbone + multi-head CRPS training
- Leverages the finetuned backbone features (iteration 1's work)
- Avoids MC-dropout's typically weak diversity

**Architecture:**
- Freeze SwinIR backbone (through conv_after_body + skip)
- Replace tail (conv_before_upsample + upsample + conv_last) with K=8 parallel branches
- Each branch: Conv+ReLU → PixelShuffle → Conv → output (1, 128, 128)
- ~400K params per branch → 3.2M trainable total

**Loss:** Energy score = (1/K)Σ|y_k - y| - (1/(2K²))ΣΣ|y_k - y_k'|
This is a proper scoring rule: rewards accuracy AND calibrated spread.

### Training Log
- 04:12 — Fix NaN bug (bias std on 1-element tensor)
- 04:24 — Node busy from prior srun; re-allocated → node3206
- 04:43 — Training started: K=8 heads, 3.2M trainable, frozen backbone
- 06:45 — Training complete: 35 epochs in 2.03h, best val loss 0.001434 at epoch 17

### CRPS Ensemble Results (10K test)
| Method | CRPS (energy) | MAE (ens. mean) | RMSE | Spread | Mass Viol |
|--------|---------------|-----------------|------|--------|-----------|
| Multi-Head SwinIR (K=8) | **0.183** | 0.250 | 0.482 | 0.272 | 0.005 |
| Multi-Head + AddCL | 0.222 | 0.250 | — | — | 0.000001 |
| OT-CFM (research2) | **0.171** | 0.247 | 0.458 | — | 0.000001 |
| SwinIR-FT deterministic | 0.250 | 0.250 | 0.483 | 0 | 0.014 |

### Analysis
- **Good:** Ensemble CRPS (0.183) vs deterministic CRPS (0.250) = 27% improvement
- **Bad:** Still behind OT-CFM (0.171) by 7%
- **Surprising:** AddCL *hurts* ensemble CRPS (0.183 → 0.222). Adding the same shift to all members doesn't change spread, but the ensemble members may already be spreading optimally around the unconstrained mean — re-centering onto LR means corrupts the calibration.
- **Concern:** Spread (0.272) is much larger than MAE (0.250), suggesting possible over-dispersion. The heads may not be learning structured uncertainty but rather random noise diversity.

### What's Next
- Re-run with more heads (K=16) or fewer (K=4) to see CRPS sensitivity
- Consider SmCL-style constraint per member (not uniform AddCL)
- The frozen backbone may be limiting — try unfreezing last 2 Swin layers

**Ending time:** ~06:47 EDT
**Ending commit:** 27b5b2c

## Iteration 3 — 2026-05-06 06:47 EDT
**Starting commit:** 27b5b2c
**Run prefix:** zjot-ghfi

### Current State
- Time: ~21hr elapsed. ~19hr to 40hr mark (2026-05-07 02:00 EDT). Plenty of time.
- GPU: no active allocation. Need to acquire one.
- squeue: 1 normal, 3 preemptable (including node1627 CPU). Can allocate 1 preemptable.
- Best CRPS: 0.183 (multi-head SwinIR K=8, frozen backbone)
- Target: OT-CFM CRPS=0.171

### Concerns (3+ problems)

1. **Quality: Over-dispersion in K=8 ensemble.** Spread (0.272) exceeds MAE (0.250) by 9%. The ensemble produces more diversity than accuracy warrants. This suggests heads learn noisy pixel-level diversity rather than structured spatial uncertainty. Reducing spread while maintaining accuracy would lower CRPS.

2. **Quality: Frozen backbone caps ensemble mean accuracy at deterministic SwinIR level.** Ensemble mean MAE=0.250 is identical to single-head finetuned SwinIR MAE=0.250. Multi-head training added ONLY diversity, zero accuracy improvement. The backbone features are shared and frozen — heads can only rearrange, not improve them.

3. **Workflow: No visualization of individual ensemble members.** Prior iteration evaluated aggregate metrics (CRPS, spread) but never examined what individual heads produce. Without this, we don't know if diversity is structured (different patterns in uncertain regions) or random (noise). Critical for understanding why AddCL hurts CRPS.

4. **Quality: AddCL hurting CRPS (0.183→0.222) is suspicious.** The uniform additive shift should NOT change spread (same shift to all members). The 21% CRPS degradation suggests the shift destroys calibration — the members were calibrated around the unconstrained mean, not the LR-consistent mean. This means the heads learned to be diverse around a biased mean.

### Direction: Residual Ensemble Parameterization

**Key insight:** Current multi-head approach replaces the finetuned tail with K independent heads. This discards the deterministic mean accuracy (MAE=0.250) and asks each head to independently learn the full HR mapping + diversity. Result: over-dispersed ensemble.

**Alternative:** Keep the finetuned deterministic mean predictor (frozen) and add K lightweight heads that predict RESIDUALS around it. Output_k = det_mean + residual_k.

**Why this is better:**
- CorrDiff principle: separate deterministic mean from stochastic perturbations
- Residuals center around zero → natural calibration anchor
- Can regularize residual magnitude → control spread
- Ensemble mean starts at finetuned MAE=0.250 (guaranteed floor)

**Architecture:**
- Backbone: frozen (conv_first → Swin layers → conv_after_body + skip)
- Deterministic tail: frozen (conv_before_upsample → upsample → conv_last)
- K=8 residual heads: each Conv(180→64)+ReLU → PixelShuffle(4x) → Conv(64→1) → residual
- Output: det_mean + residual_k
- Loss: energy CRPS + λ*mean(residual²) for regularization
- Init: small Xavier init (gain=0.01) for symmetry breaking

### Training Log
- 06:57 — Allocated node4104, hit CUDA ECC error (hardware fault). Cancelled.
- 06:59 — Reallocated, got node3005 (L40S)
- 07:02 — Training started: K=8 residual heads, 15M total / 3.2M trainable
- 08:52 — Preempted at epoch 30 (108 min). Best val loss 0.001432 at epoch 12.
- 08:56 — Reallocated node3207 (normal GPU) for evaluation

### Residual Ensemble Results (10K test)
| Method | CRPS | MAE | RMSE | Spread | Mass Viol |
|--------|------|-----|------|--------|-----------|
| Residual K=8 | **0.183** | 0.250 | 0.482 | 0.280 | 0.005 |
| Residual + AddCL | 0.206 | 0.250 | — | — | 0.000001 |
| Direct K=8 (iter 2) | **0.183** | 0.250 | 0.482 | 0.272 | 0.005 |
| Direct + AddCL (iter 2) | 0.222 | 0.250 | — | — | 0.000001 |
| OT-CFM (research2) | **0.171** | 0.247 | 0.458 | — | 0.000001 |

### Analysis: Negative Result
**The residual parameterization provides NO improvement.** CRPS = 0.183 for both direct and residual modes. The hypothesis that residual centering would reduce over-dispersion was wrong — spread is actually slightly WORSE (0.280 vs 0.272).

**Why this happened:**
- The CRPS loss is the dominant optimization force, not the parameterization
- Both approaches converge to the same t1/t2 equilibrium (~0.0026/0.0024 in normalized space)
- The frozen backbone is the true bottleneck — all heads share the same features, limiting both accuracy and diversity structure
- Over-dispersion is inherent to the K=8 multi-head + CRPS loss setup

**Key insight:** The bottleneck is NOT how we parameterize the heads (direct vs residual) but the quality/flexibility of the shared backbone features. The frozen backbone caps ensemble mean accuracy at MAE=0.250 regardless of head design.

**Positive observation:** AddCL hurts LESS with residual mode (0.206 vs 0.222). The residual structure makes the ensemble slightly more compatible with post-hoc constraints.

### Implications for Next Iterations
1. **Unfreeze backbone layers** — the only way to improve MAE below 0.250, which directly improves CRPS
2. **Try fundamentally different stochastic mechanism** — multi-head K=8 may cap CRPS regardless. Flow matching or diffusion on residuals could produce better-calibrated diversity.
3. **K sensitivity** still untested but likely won't overcome the backbone bottleneck

**Ending time:** 08:58 EDT
**Ending commit:** (pending)
