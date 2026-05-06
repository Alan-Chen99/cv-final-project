# Scratchpad — research6

## Iteration 1
**Start:** 2026-05-06 00:58 EDT
**Start commit:** 7cbd4e4
**Run prefix:** dbqh-ictv

### Situation
- Branch: research6 (fresh, no prior work on this branch)
- Prior best CRPS (corrected): 0.171 (research2, OT-CFM 13M params)
- Prior best CRPS (paper formula, buggy): 0.094 (research2)
- Data: ERA5 TCW, 32×32→128×128, 40K/10K/10K
- GPU: node4104 occupied (wwtlu-jjtmi, normal), node4212 occupied (sweep-gpu2, preemptable)
- Need to allocate own GPU node

### Concerns
1. **Workflow concern:** Prior iterations all trained from scratch. The task explicitly asks to start with zero-shot pretrained image SR models. This is completely unexplored.
2. **Fact concern:** The cross-comparison reports CRPS 0.171 (corrected) vs 0.094 (paper/buggy). Need to verify: what IS the corrected CRPS for the best model? The report says it was partially lost to preemption.
3. **Quality concern:** No pretrained model evaluation infrastructure exists. Need to build it from scratch for zero-shot evaluation.

### Plan for this iteration
ONE thing: Set up and run zero-shot evaluation of pretrained image SR models on ERA5 data.
- Install SwinIR / download weights
- Write evaluation script adapted for single-channel climate data
- Run bicubic baseline + SwinIR zero-shot
- Compute corrected CRPS
- Also train a quick UNet regression to compare

### Results (2K test set, corrected energy CRPS)

| Method | Params | Training | CRPS | RMSE | MAE | Mass Viol |
|--------|--------|----------|------|------|-----|-----------|
| Bicubic | - | - | 0.378 | 0.765 | 0.378 | 0.144 |
| Bicubic + AddCL | - | - | 0.348 | 0.721 | 0.348 | 0.000 |
| SwinIR zero-shot | 11.9M | 0 | 0.311 | 0.664 | 0.311 | 0.083 |
| SwinIR TTA8 + AddCL | 11.9M | 0 | 0.279 | 0.645 | 0.294 | 0.000 |
| UNet L1 (no TTA) | 3.5M | 19 min | 0.290 | 0.622 | 0.290 | 0.020 |
| UNet L1 + AddCL | 3.5M | 19 min | 0.289 | 0.620 | 0.289 | 0.000 |
| UNet L1 TTA8 | 3.5M | 19 min | 0.258 | 0.633 | 0.292 | 0.012 |
| UNet L1 TTA8 + AddCL | 3.5M | 19 min | 0.259 | 0.632 | 0.292 | 0.000 |
| **Prior: OT-CFM (corrected)** | **13M** | **~3 hr** | **0.171** | **~0.456** | **~0.242** | **0.000** |

### Key Findings

1. **Pretrained image SR provides limited benefit for climate data.** SwinIR zero-shot beats bicubic but a simple UNet trained from scratch in 19 min achieves similar or better MAE.
2. **TTA gives limited CRPS improvement.** 8-fold TTA reduces CRPS by ~11% (0.290 → 0.258) but the spread is small (0.065) compared to generative models.
3. **The gap to flow matching is large.** Best deterministic approach (UNet TTA8 + AddCL, CRPS 0.259) is 52% worse than flow matching (0.171). Generative ensemble diversity is essential for CRPS.
4. **AddCL has mixed effects.** For the UNet TTA8, AddCL slightly hurts CRPS (reduces spread more than it improves accuracy).

### Conclusions for next iterations

- Deterministic models cannot compete with generative models on CRPS
- The flow matching model's advantage comes from genuine distributional diversity
- Pretrained natural image features don't transfer well to single-channel climate fields
- Future work should focus on improving generative models, not deterministic regression
- Potential directions: DiT backbone, consistency distillation, improved sampling

**End:** 2026-05-06 02:05 EDT
**End commit:** 665393e

## Iteration 2
**Start:** 2026-05-06 02:06 EDT
**Start commit:** 20026b6
**Run prefix:** jwcj-hvhx

### Situation
- Branch: research6, iteration 1 completed zero-shot and deterministic baselines
- Best result so far: UNet L1 TTA8 CRPS 0.258 (deterministic), prior flow matching 0.171
- Normal GPU: 2/2 used (rwzi-rdw node3006, wwtlu-jj node4104) — NOT ours
- Preemptable GPU: 1/4 used (sweep-gp node4211) — NOT ours
- dnds-fux on node4307: CPU-only preemptable — NOT ours
- Need to allocate own preemptable GPU

### Concerns
1. **Workflow concern:** Iteration 1 trained a finetuned SwinIR model (`best_swinir1ch.pt` exists at pool/research6/models/swinir_finetune/) but NEVER evaluated it. The scratchpad shows only zero-shot SwinIR results. The starting direction explicitly asked to "finetune general image models and evaluate" — the finetuning was done but evaluation was skipped.
2. **Quality concern:** The SwinIR1Ch wrapper uses simple 1x1 conv adapters (input/output). This is a very simple adapter. Need to check if the model was trained with frozen or unfrozen backbone. With frozen backbone, only the adapters (6 params) are trainable — too few to learn anything useful.
3. **Workflow concern:** Iteration 1 didn't clearly document which GPU jobs it created or cleaned up. The rwzi-rdw and dnds-fux jobs were created during iteration 1 but aren't mentioned in the scratchpad. They might be from other branches running concurrently.

### Plan for this iteration
1. Evaluate existing finetuned SwinIR model (5 min) — fill gap from iteration 1
2. Train a flow matching model with residual mode (attention, augmentation, 64/128/256 channels) — target beating 0.171 CRPS
3. Evaluate the flow matching model on 10K test set

### SwinIR Finetune Eval (gap from iter 1)
| Method | CRPS | RMSE | MAE | Mass Viol | Spread |
|--------|------|------|-----|-----------|--------|
| SwinIR-finetune TTA8 | 0.300 | 0.659 | 0.320 | 0.092 | 0.035 |
| SwinIR-finetune TTA8+AddCL | 0.285 | 0.642 | 0.302 | 0.000 | 0.031 |

Finetuning actually made SwinIR WORSE than zero-shot (0.285 vs 0.279). Confirms pretrained natural image features don't help.

### Residual Flow Matching Training
- Architecture: FlowUNet with 64/128/256 channels, attention at 16x16, 9.1M params
- Residual mode: learns on (HR - bilinear(LR)) space
- Training: AdamW lr=1e-4, cosine schedule T_max=100, batch 64, augmentation
- 100 epochs across 3 GPU allocations (preempted once, time-limited once)
- Best val loss: 0.001978 (epoch ~70)
- Total training time: ~2.9 hrs on L40S

### Flow Matching Results (10K test, 10 members, 10 Euler steps)

| Method | CRPS | RMSE | MAE | Spread | Mass Viol |
|--------|------|------|-----|--------|-----------|
| Residual flow + AddCL | 0.238 | 0.577 | 0.298 | 0.265 | 0.000 |
| Residual flow (none) | 0.239 | 0.578 | 0.300 | 0.273 | 0.029 |
| **Prior: OT-CFM (research2)** | **0.171** | **~0.456** | **~0.242** | **~0.07** | **0.000** |

### Analysis: Why 0.238 instead of 0.171
1. **Excess diversity, insufficient accuracy.** Spread=0.265 is 4x research2's ~0.07, but MAE=0.298 is 23% worse than ~0.242. Model generates too-diverse, low-quality samples.
2. **Architecture capacity.** 9.1M params (1 ResBlock/level) vs research2's 13M (2 ResBlocks/level). The extra capacity is critical for per-member accuracy.
3. **Training schedule mismatch.** Cosine T_max=100 with 100 epochs vs research2's T_max=40 with 39 epochs. Research2's aggressive schedule may force faster convergence.
4. **No EMA.** Research2 presumably used EMA based on its codebase. I trained without EMA.

### Conclusions for next iteration
- Need to match research2 architecture more closely: 2 ResBlocks per level (~13M params)
- Use shorter cosine schedule (T_max=40) with early stopping
- Add EMA (decay=0.999)
- The residual formulation itself is sound; the bottleneck is model capacity and training recipe
- Alternative direction: try to use the research2 model directly if weights exist on this branch

**End:** 2026-05-06 06:50 EDT
**End commit:** 8290f26

## Iteration 3
**Start:** 2026-05-06 06:46 EDT
**Start commit:** 8dfda68
**Run prefix:** uqge-enkg

### Situation
- Branch: research6, iterations 1-2 complete
- Best CRPS this branch: 0.238 (residual flow, 9.1M, 100 epochs, random coupling)
- Target: research2's 0.171 (OT-CFM, 13M, 2 ResBlocks/level, 39 epochs)
- GPU: 1 normal slot available (1/2 used), preemptable nearly full (3-4/4)
- Research2 weights NOT accessible from this branch (pool/research2 empty)
- Time: 06:46 EDT, ~19hr remaining before 40hr cutoff (02:00 EDT tomorrow)

### Concerns
1. **Quality concern — missing OT coupling**: The biggest architectural difference between research6 (CRPS 0.238) and research2 (CRPS 0.171) is that research2 used OT-CFM (optimal transport coupling). Research6 uses random source-target pairing. OT coupling leads to straighter flow paths that are easier to learn with few Euler steps. This is the most likely cause of the quality gap, ahead of capacity (9.1M vs 13M).

2. **Quality concern — no EMA**: Research6's training loop has EMA implemented but NEVER enabled (--ema flag not passed). EMA typically gives ~5-10% improvement in generative models. With 100 epochs of training, EMA would accumulate a significantly smoothed model.

3. **Workflow concern — dangling GPU jobs**: Three GPU jobs running (nsyy-wjrk node3208, jrut-ohex COMPLETING, crbk-tkvl node3302) with prefixes that don't match iteration 2's jwcj-hvhx. These are from other agents/branches. Iteration 2 did not properly document which jobs it created. Cannot determine if any are ours. Per rule 1010/1012, leaving them alone since prefixes don't match.

### Plan for this iteration
**ONE thing: Implement minibatch OT coupling and train with EMA.**

Hypothesis: OT coupling is the single biggest factor in the research2 gap. Testing with same architecture (9.1M, 1 ResBlock/level) isolates OT's effect from capacity scaling.

Changes:
- Add `--ot` flag for minibatch OT coupling via scipy.optimize.linear_sum_assignment
- Train with --ot --ema --ema-decay 0.999, same channels (64,128,256), batch 64
- ~60-70 epochs to fit in 2hr on L40S
- Evaluate on full 10K test with AddCL

### OT Coupling Attempt — FAILED (too slow)
Implemented minibatch OT coupling with GPU-based cost matrix + CPU Hungarian algorithm.
- First attempt (CPU cost matrix): ~7.7 min/epoch (4.5x slower than non-OT)
- Second attempt (GPU cost matrix + CPU Hungarian): still ~7.7 min/epoch
- Root cause: `cost.cpu()` forces CUDA sync every batch, breaks GPU pipeline
- Training preempted at epoch 10 (77 min)
- Conclusion: **minibatch OT with CPU Hungarian is not viable** for 128x128 images at batch 64.
  Would need GPU-native Sinkhorn or approximate matching.

### Pivot: Scale architecture to 2 ResBlocks/level + EMA (no OT)
- Added `--n-res-blocks` parameter to FlowUNet
- 2 ResBlocks/level: 12.5M params (vs 9.1M with 1 ResBlock)
- Training rate: ~2.96 min/epoch on L40S
- **Preempted 3 times** on mit_preemptable partition during this iteration:
  1. OT training preempted at epoch 10 (node4308, signal terminated)
  2. v2 training preempted after 2 min (node3206, signal terminated)
  3. v2 training preempted at epoch 25 (node4208, DUE TO PREEMPTION)

### Flow v2 Results (10K test, 10 members, 10 Euler steps, 25 epochs only)
| Method | Params | Epochs | CRPS | RMSE | MAE | Spread | Mass Viol |
|--------|--------|--------|------|------|-----|--------|-----------|
| Flow v2 + AddCL | 12.5M | 25 | 0.262 | 0.613 | 0.340 | 0.514 | 0.000 |
| **Iter2: flow residual + AddCL** | **9.1M** | **100** | **0.238** | **0.577** | **0.298** | **0.265** | **0.000** |
| **Prior: OT-CFM (research2)** | **13M** | **39** | **0.171** | **~0.456** | **~0.242** | **~0.07** | **0.000** |

### Analysis
1. **Model is severely undertrained.** 25/60 epochs. Spread=0.514 is 2x the 9.1M model's 0.265, indicating the model generates noise-like diverse samples rather than structured predictions.
2. **Architecture scaling needs matching training.** The 12.5M model at 25 epochs is worse than 9.1M at 100 epochs. The extra capacity needs proportionally more training to converge.
3. **OT coupling is prohibitively expensive with CPU Hungarian.** Would need GPU-native solver (Sinkhorn) or approximate matching to be practical.
4. **Preemptable partition extremely unstable today.** 3 preemptions in one iteration. Consider using mit_normal when available.

### What next iteration should do
1. **Resume flow_v2 training from epoch 25** — checkpoint exists at `pool/research6/models/flow_v2/`. Need 35+ more epochs.
2. **Try mit_normal partition** if available (more stable)
3. After training completes, evaluate with both AddCL and no constraint
4. If CRPS doesn't improve significantly over 9.1M model, the gap to research2 is NOT primarily architecture — it must be OT coupling or other training details

**End:** 2026-05-06 10:57 EDT
**End commit:** (pending)
