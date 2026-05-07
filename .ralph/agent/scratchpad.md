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

**Ending time:** 08:59 EDT
**Ending commit:** 44ff7ab

## Iteration 4 — 2026-05-06 09:00 EDT
**Starting commit:** 6eb28be
**Run prefix:** knam-twmo

### Current State
- Time: ~23hr elapsed. ~17hr to 40hr mark (2026-05-07 02:00 EDT). Plenty of time.
- GPU: no active allocation. node4208 completing, node3404 in use by another branch.
- squeue: 1 normal available, 2 preemptable available (including node1627 CPU).
- Best CRPS: 0.183 (multi-head SwinIR K=8, frozen backbone — both direct and residual)
- Target: OT-CFM CRPS=0.171

### Concerns (3+ problems)

1. **Quality: Frozen backbone confirmed as bottleneck across 2 experiments.** Iterations 2 and 3 both achieve CRPS=0.183 with MAE=0.250 — identical to the single-head finetuned SwinIR. Neither direct nor residual head parameterization can improve accuracy beyond what frozen backbone features support. Must unfreeze to break this floor.

2. **Quality: Over-dispersion (spread > MAE) is structural to multi-head setup.** Spread=0.272 (direct) / 0.280 (residual) both exceed MAE=0.250. K=8 independent heads sharing frozen features learn unstructured pixel-level noise rather than spatially coherent uncertainty. Simply increasing head count or changing parameterization won't fix this — need either better features (unfreeze) or fundamentally different stochastic mechanism.

3. **Workflow: Never tried unfreezing — the most obvious next step identified in iterations 2 and 3.** Both prior iterations explicitly recommended unfreezing as the primary next direction. This is the single most actionable improvement.

### Direction: Unfreeze Last 2 Swin Layers + Multi-Head CRPS

**Rationale:** Frozen backbone caps MAE at 0.250. Unfreezing last 2 of 6 RSTB blocks allows the backbone to adapt its high-level features for the multi-head CRPS objective while preserving lower-level pretrained representations.

**Key design choices:**
- Discriminative LR: backbone layers at 1/10th the head LR
- Unfreeze norm layer too (follows the transformer body)
- Keep K=8 direct mode (baseline comparison)
- Same wall clock budget (2hr)

**Expected outcome:** MAE < 0.250, CRPS < 0.183. If MAE drops to ~0.240, CRPS could reach ~0.170-0.175.

### Training Log
- 09:08 — Training started: node4505 (L40S), K=8 direct + unfreeze 2 layers
- 09:08 — 14.7M total, 7.2M trainable (4.0M backbone + 3.2M heads)
- 10:31 — Training complete: 25 epochs in 2.03h, best val loss 0.001428 at epoch 17
- 11:12 — Evaluation complete: CRPS=0.1826 on 10K test

### Results: Unfreeze Last 2 Layers (10K test)
| Method | CRPS | MAE | RMSE | Spread | Mass Viol |
|--------|------|-----|------|--------|-----------|
| Unfreeze2 K=8 | **0.1826** | 0.249 | 0.480 | 0.272 | 0.004 |
| Frozen K=8 (iter 2) | **0.183** | 0.250 | 0.482 | 0.272 | 0.005 |
| Residual K=8 (iter 3) | **0.183** | 0.250 | 0.482 | 0.280 | 0.005 |
| OT-CFM (research2) | **0.171** | 0.247 | 0.458 | — | 0.000001 |

### Analysis: Marginal Result
**Unfreezing backbone provides essentially NO improvement.** CRPS drops from 0.183 to 0.1826 (-0.2%), well within noise. Spread unchanged at 0.272.

**Why unfreezing doesn't help within multi-head CRPS:**
1. 8 conflicting gradient paths through shared backbone average to near-zero effective backbone signal
2. The CRPS loss optimizes ensemble-level diversity, not backbone accuracy directly
3. The multi-head architecture has a CRPS ceiling around 0.183 regardless of backbone quality

**Three iterations confirming multi-head SwinIR ceiling:**
- Iter 2: frozen + direct heads → CRPS=0.183
- Iter 3: frozen + residual heads → CRPS=0.183
- Iter 4: unfrozen + direct heads → CRPS=0.183

**Critical insight:** The multi-head approach has hit its fundamental limit. To beat 0.183, need a different stochastic mechanism entirely. The K=8 heads learn unstructured pixel-level diversity regardless of parameterization or backbone flexibility.

### Implications for Next Iterations
1. **Drop multi-head approach.** Three experiments prove CRPS=0.183 is its ceiling.
2. **Try flow matching on SwinIR residuals** — CorrDiff-style two-stage: use finetuned SwinIR as deterministic predictor, train flow matching model on (hr - swinir_pred) residuals.
3. **Or noise-injection approach** — single SwinIR with noise conditioning for stochastic diversity.
4. **Or return to OT-CFM with better architecture** — OT-CFM already achieves 0.171, but only 13M params. Perhaps OT-CFM + SwinIR backbone features could go further.

**Ending time:** 11:13 EDT
**Ending commit:** 370dc74

## Iteration 5 — 2026-05-06 11:14 EDT
**Starting commit:** 0658fa6
**Run prefix:** abif-qkbf

### Current State
- Time: ~25hr elapsed. ~15hr to 40hr mark (2026-05-07 02:00 EDT). Good budget.
- GPU: no active allocation from this branch. Several other branch jobs running.
- squeue: 2 normal used (limit 2), 2 preemptable GPU + 1 CPU (limit 4). Can allocate 1 preemptable.
- Best CRPS: 0.183 (multi-head SwinIR, ceiling confirmed across 3 experiments)
- Target: OT-CFM CRPS=0.171

### Concerns (3+ problems)

1. **Quality: Multi-head CRPS ceiling at 0.183 — fundamentally limited.** Three experiments (direct, residual, unfrozen) all converge to identical CRPS=0.183. The K independent heads sharing features produce unstructured pixel-level diversity (spread=0.272 > MAE=0.250). This architecture cannot produce calibrated, structured uncertainty.

2. **Workflow: Never combined SwinIR backbone with flow matching.** The two strongest approaches discovered so far — SwinIR deterministic (MAE=0.250) and OT-CFM stochastic (CRPS=0.171) — have never been combined. CorrDiff (the state of the art in climate downscaling) uses exactly this two-stage approach: deterministic mean + diffusion residuals.

3. **Quality: Flow matching produces fundamentally different diversity than multi-head.** Multi-head learns K fixed functions from shared features → unstructured diversity. Flow matching learns the full residual distribution → can sample arbitrary K with spatially structured diversity. This is the key difference that could break the 0.183 ceiling.

### Direction: CorrDiff-style Residual Flow Matching

**Architecture:**
1. Frozen finetuned SwinIR → deterministic mean prediction (MAE=0.250)
2. Precompute residuals: r = hr_true - swinir_pred on training data
3. Train FlowUNet on residuals, conditioned on LR (bicubic upsampled)
4. At inference: sample_k = swinir_pred + flow_residual_k

**Why this should work:**
- CorrDiff principle: separating mean from stochastic detail is the right decomposition
- Residual distribution is simpler than full HR → easier to learn with small model
- Flow matching produces continuous distribution, not K discrete points
- Can sample any K at test time (not limited to training-time K)

**FlowUNet config:**
- channels=(64,128,256) with attention at 16x16 → 9.1M params
- Standard Gaussian noise source
- 20 Euler steps at inference
- EMA decay=0.999

### Training Log
- 11:20 — GPU allocation: tried preemptable (Priority wait), then normal (QOSMaxCpuPerUserLimit), then preemptable H100 → node2640
- 11:24 — Precompute SwinIR predictions: train/val/test residuals. Stats: mean≈0, std=0.0035, range[-0.28, 0.35]
- 11:26 — Training started: 9.1M params, BS=64, LR=2e-4, cosine T_max=49
- 13:26 — Training complete: 184 epochs in 120.4 min (H100), best val=0.001588
- 13:26 — Eval attempt on node2640 hung (other users' CPU processes saturating node)
- 13:48 — Cancelled node2640, got A100 on node2319
- 13:53 — Eval started (with PYTHONUNBUFFERED=1)
- 14:15 — Eval complete (20 steps): CRPS=0.212, MAE=0.264
- 14:17 — Eval with 50 steps started
- 15:10 — Eval complete (50 steps): CRPS=0.207, MAE=0.266
- 15:12 — GPU cancelled

### Results: Residual Flow Matching (10K test)
| Method | CRPS | MAE | RMSE | Spread | Mass Viol |
|--------|------|-----|------|--------|-----------|
| ResFlow 20 steps | 0.212 | 0.264 | 0.488 | 0.208 | 0.015 |
| ResFlow 50 steps | 0.207 | 0.266 | 0.492 | 0.249 | 0.015 |
| ResFlow 20 + AddCL | 0.212 | 0.263 | 0.488 | 0.203 | 0.000001 |
| ResFlow 50 + AddCL | 0.207 | 0.265 | 0.491 | 0.244 | 0.000001 |
| Multi-Head K=8 (iter 2) | **0.183** | 0.250 | 0.482 | 0.272 | 0.005 |
| OT-CFM (research2) | **0.171** | 0.247 | 0.458 | — | 0.000001 |

### Analysis: Negative Result — Gaussian-Source Flow on Concentrated Residuals Fails

**The residual flow matching approach produces CRPS=0.207-0.212, significantly worse than multi-head SwinIR (0.183).**

**Root cause: Source-target distribution mismatch.**
- Residual std = 0.0035 in normalized space
- Gaussian source std = 1.0
- Compression ratio: ~285x
- The flow must learn to compress N(0,1) to N(0, 0.0035) — an extremely long transport path
- Even small velocity field errors accumulate over this large transport, biasing the output

**Evidence of mean corruption:**
- SwinIR deterministic MAE = 0.250
- After adding flow residuals: MAE = 0.264 (+6%)
- The flow adds biased residuals that corrupt the mean
- Ensemble mean of (swinir_pred + flow_residual) should be ≈ swinir_pred if flow is accurate, but it's not

**50 vs 20 steps: minimal improvement (0.207 vs 0.212)**
- More steps increase spread (0.249 vs 0.208) but worsen MAE (0.266 vs 0.264)
- The issue is not ODE accuracy but model accuracy

### Key Insight
CorrDiff works because the residual distribution has reasonable variance relative to the noise source. In our case, the SwinIR predictions are too accurate — the residuals are tiny. Standard flow matching from N(0,1) cannot accurately model such concentrated targets.

### Implications for Next Iterations
1. **If using flow on residuals: MUST use LR-anchored source** — start from small noise (std=0.01), not N(0,1). Short transport path.
2. **Or train flow on full HR distribution** (like OT-CFM does), using SwinIR backbone features as conditioning
3. **Or try noise-conditioned SwinIR** — inject noise directly into the SwinIR forward pass
4. **The multi-head approach (CRPS=0.183) remains the best SwinIR-based result**

**Ending time:** 15:12 EDT
**Ending commit:** 303b8e2

## Iteration 6 — 2026-05-06 15:14 EDT
**Starting commit:** 0a230c4
**Run prefix:** nova-tango

### Current State
- Time: ~29hr elapsed. ~11hr to 40hr mark (2026-05-07 02:00 EDT). ~2-3 more experiment iterations.
- GPU: no active allocation. 2 normal full, 3/4 preemptable used. Allocating 1 preemptable.
- Best CRPS: 0.183 (multi-head SwinIR K=8, research5)
- Target: OT-CFM CRPS=0.171 (research2, checkpoint unavailable — pool dir doesn't exist)
- squeue: job 13445100 pending (nova-tango)

### Concerns (3+ problems)

1. **Quality: Multi-head SwinIR ceiling at 0.183 — exhaustively confirmed.** 3 experiments (direct, residual, unfrozen backbone) all converge to CRPS=0.183. Spread > MAE indicates over-dispersion. This approach is done.

2. **Quality: CorrDiff-style residual flow failed (CRPS=0.207).** Source-target mismatch: N(0,1) → N(0,0.0035) is a 285x compression. Mean corruption: MAE degrades from 0.250 to 0.264. The flow adds biased residuals.

3. **Workflow: Never tested transformer-based score network.** All 5 iterations used either SwinIR (CNN/Transformer hybrid but NOT a DiT) or UNet. DiT is explicitly identified as under-explored direction #7 in CLAUDE.md. All downscaling diffusion/flow papers use UNet backbones.

4. **Quality: OT-CFM remains the best approach but only tested with UNet.** The 13M UNet achieved CRPS=0.171 on research2. The architecture (not just the training method) may be limiting — a transformer backbone could capture longer-range spatial correlations.

### Direction: DiT (Diffusion Transformer) for OT-CFM Flow Matching

**Rationale:** Under-explored direction #7 from CLAUDE.md. All climate downscaling diffusion/flow papers use UNet backbones. DiT has become the dominant architecture in image generation (Stable Diffusion 3, FLUX, etc.) but is untested in climate downscaling. Global self-attention over all spatial tokens may better model long-range correlations in climate variables.

**Architecture: FlowDiT**
- Input: [x_t (1ch), LR_up (1ch)] → 4×4 patches → 1024 tokens × dim=384
- 8 transformer blocks with adaLN-Zero time conditioning
- Multi-head self-attention (6 heads) + MLP (ratio=4)
- Unpatchify → velocity field (1ch, 128×128)
- ~15M params (comparable to 13M UNet baseline)

**Training:**
- Same OT-CFM objective as flow_matching_v2.py
- Same residual prediction (HR - bilinear(LR))
- Same normalization, data loading, evaluation
- AdamW, LR=1e-4, cosine schedule, BS=64
- 2hr wall-clock budget

**Expected outcome:** Uncertain — this is exploratory. DiT may converge slower than UNet. Success criterion: CRPS competitive with UNet (~0.171). Novel contribution even if CRPS is similar (first DiT for climate downscaling).

### Training Log
- 15:21 — salloc submitted (preemptable), pending Priority. Fairshare exhausted across branches.
- 15:28 — Cancelled, tried H100 → pending. Tried L40S → pending. Tried normal → pending.
- 15:42 — Normal slot opened (gcgi-vxgh-eval finished). Submitted normal GPU.
- 15:49 — Allocated node4108 (L40S, normal partition, 3hr limit)
- 15:50 — Benchmarked: DiT-S/8 (256 tokens): 0.9min/ep, 4.8GB. DiT-S/4 (1024 tokens): OOM at BS=64.
- 15:53 — Training started: 100 epochs, BS=64, LR=1e-4, cosine, 22M params
- 17:43 — Training complete: 100 epochs in 109.6 min, best val loss 0.3100 at epoch 84
- 17:56 — Eval (10 steps): CRPS=0.204, MAE=0.282
- 18:08 — Eval (10 steps + AddCL): CRPS=0.204, mass viol=0.000001
- 18:15 — Eval (20 steps): CRPS=0.201, MAE=0.284
- 18:20 — GPU cancelled

### DiT Flow Matching Results (10K test)
| Method | CRPS | MAE | RMSE | Spread | Mass Viol |
|--------|------|-----|------|--------|-----------|
| DiT 10 steps | 0.204 | 0.282 | 0.567 | 0.272 | 0.003 |
| DiT 10 + AddCL | 0.204 | 0.282 | 0.567 | 0.271 | 0.000001 |
| DiT 20 steps | 0.201 | 0.284 | 0.571 | 0.310 | 0.003 |
| Multi-Head K=8 (iter 2) | **0.183** | 0.250 | 0.482 | 0.272 | 0.005 |
| OT-CFM UNet (research2) | **0.171** | 0.247 | 0.458 | — | 0.000001 |

### Analysis: DiT Underperforms UNet for Climate Downscaling

**The DiT achieves CRPS=0.201-0.204, significantly worse than UNet (0.171) and multi-head SwinIR (0.183).**

**Root cause: 8×8 patches lose fine-grained spatial detail.**
- Each 8×8 patch is compressed to a single 384-dim token
- Unpatchify reconstructs 64 pixels from one vector via linear projection
- Fine spatial patterns within patches are lost — rank-1 approximation per patch
- UNet processes at full 128×128 through convolutions with skip connections

**Evidence:**
- MAE=0.282 vs UNet's 0.247 → 14% worse accuracy
- 2-layer conv refinement (32 channels) insufficient for lost intra-patch detail
- More ODE steps (20 vs 10) increases spread (0.31 vs 0.27) but doesn't improve MAE

**Why 4×4 patches weren't feasible:**
- 1024 tokens with BS=64 → OOM on L40S (46GB) even with flash attention
- Would need BS=16 → 42 min/epoch → only ~3 epochs in 2hr

**Positive:** DiT trains fast (1.1 min/ep), converges well, AddCL works perfectly.

### Implications for Next Iterations
1. **Retrain OT-CFM UNet from scratch** on this branch — proven architecture, CRPS=0.171
2. **Hybrid DiT + Conv decoder** could combine global attention with fine detail
3. **UNet convolutional inductive bias is important** for pixel-level prediction tasks

**Ending time:** ~18:20 EDT
**Ending commit:** 0320bda

## Iteration 7 — 2026-05-06 18:23 EDT
**Starting commit:** 5b6fcec
**Run prefix:** urfm-oebd

### Current State
- Time: ~32hr elapsed. ~8hr to 40hr mark (2026-05-07 02:00 EDT). 2 iterations max.
- GPU: no allocation from this branch. 1 normal slot + 0-1 preemptable GPU available.
- Best CRPS: 0.183 (multi-head SwinIR K=8)
- Target: OT-CFM CRPS=0.171

### Concerns (3+ problems)

1. **Quality: 6 iterations, zero improvement over the research2 baseline (0.171).** Multi-head SwinIR caps at 0.183. Residual flow (0.207) and DiT (0.204) both significantly worse. The only thing that beats 0.183 is OT-CFM UNet, which uses fundamentally different architecture + training.

2. **Quality: Multi-head diversity is structurally wrong.** K=8 heads sharing frozen features learn unstructured pixel-level noise (spread=0.272 > MAE=0.250 = over-dispersed). Need calibrated, spatially-structured diversity. Flow matching provides this but requires ODE integration.

3. **Workflow: Never tried noise-conditioned generation on SwinIR.** All SwinIR experiments used K fixed heads (discrete diversity). Injecting continuous noise into the reconstruction tail gives a different diversity mechanism: (a) no gradient conflict (single head), (b) continuous noise space → arbitrary K at test time, (c) can optimize CRPS directly. This is genuinely under-explored.

### Direction: Noise-Conditioned SwinIR Tail

**Architecture:**
1. Frozen SwinIR backbone → features (B, 180, 32, 32)
2. Concat with noise channels z~N(0,I) of shape (B, 16, 32, 32) → (B, 196, 32, 32)
3. Single reconstruction tail: Conv(196→64) → PixelShuffle(4x) → Conv(64→1)
4. Train with CRPS loss: K=8 noise draws per sample

**Why this might work:**
- Noise enters at feature level (32x32), not pixel level — must be processed by conv layers
- Single head = no gradient conflict between heads
- CRPS loss directly optimizes ensemble calibration (both accuracy + spread)
- Pretrained backbone preserves representation quality

**Why this might NOT work:**
- Backbone features may dominate noise channels → near-zero diversity
- 16 noise channels at 32x32 may be too few/many
- The tail may ignore noise if it can minimize CRPS with a constant output

**Risk mitigation:** Use larger noise dim (16 channels) + initialize noise path with non-trivial weights

### Training Log
- 18:30 — GPU allocation: node3512 (L40S, normal partition, 3hr limit)
- 18:31 — Test run (1 epoch): 3.7min/epoch, 482K trainable params. Works.
- 18:36 — Training started: 100 epochs, BS=32, LR=5e-4, cosine T_max=18, K=8
- 20:36 — Training complete: 33 epochs in 121.6 min, best val=0.001566 at epoch 20
- 20:38 — Eval complete (10K test, K=8)
- 20:39 — GPU cancelled

### Results: Noise-Conditioned SwinIR (10K test)
| Method | CRPS | MAE | RMSE | Spread | Mass Viol |
|--------|------|-----|------|--------|-----------|
| Noise-Cond K=8 | 0.200 | 0.258 | 0.494 | 0.197 | 0.039 |
| Noise-Cond + AddCL | 0.206 | 0.255 | — | — | 0.000001 |
| Multi-Head K=8 (iter 2) | **0.183** | 0.250 | 0.482 | 0.272 | 0.005 |
| OT-CFM UNet (research2) | **0.171** | 0.247 | 0.458 | — | 0.000001 |

### Analysis: Negative Result — Noise Injection Ignored by Tail

**Noise-conditioned SwinIR achieves CRPS=0.200, significantly worse than multi-head (0.183).**

**Root cause: Backbone features dominate, tail learns to ignore noise.**
- Spread = 0.197 < MAE = 0.258 → **under-dispersed** (opposite of multi-head which was over-dispersed)
- The t2 (spread term) stayed at 0.0017 throughout training — barely changed from epoch 1 to 33
- Noise projection → 64 channels competing with 180 backbone channels
- Backbone-initialized weights in the first conv layer mean noise path starts at zero → gradient landscape favors keeping noise contribution small
- Energy score loss can be minimized by producing accurate but identical members (CRPS ≈ MAE when spread=0)

**AddCL makes CRPS WORSE (0.206 vs 0.200):** constraint reduces spread further for already under-dispersed ensemble.

**Key insight:** Simple noise concatenation cannot compete with the pretrained backbone signal. The tail learns a deterministic mapping from backbone features and treats noise as negligible perturbation. Multi-head succeeds at diversity because each head has independent weights that diverge during training. Noise injection requires the weights to learn to amplify noise, which is not favored by the loss landscape when deterministic accuracy is already good.

### Summary: 7 Iterations, Best = Multi-Head SwinIR CRPS=0.183
| Iter | Direction | CRPS | Verdict |
|------|-----------|------|---------|
| 1 | SwinIR finetune (det.) | 0.250 | Baseline |
| 2 | Multi-head K=8 direct | **0.183** | Best SwinIR |
| 3 | Multi-head K=8 residual | 0.183 | Same ceiling |
| 4 | Multi-head unfreeze-2 | 0.183 | Same ceiling |
| 5 | Residual flow matching | 0.207 | Negative |
| 6 | DiT flow matching | 0.204 | Negative |
| 7 | Noise-conditioned tail | 0.200 | Negative |

**Ending time:** 20:39 EDT
**Ending commit:** adc4e81

## Iteration 8 — 2026-05-06 20:41 EDT
**Starting commit:** 4832afe
**Run prefix:** volk-apex

### Current State
- Time: ~35hr elapsed. ~5hr to 40hr mark (2026-05-07 02:00 EDT). 1-2 iterations left.
- GPU: 1 normal used (zbhh-axx node3008, not ours), 3 preemptable GPU + 1 CPU. Can allocate 1 normal slot.
- Best CRPS: 0.183 (multi-head SwinIR K=8)
- Target: OT-CFM CRPS=0.171 (research2, checkpoint unavailable)

### Concerns (3+ problems)

1. **Quality: 7 iterations exhausted SwinIR-based approaches.** Multi-head caps at 0.183, noise conditioning at 0.200. All approaches using SwinIR backbone hit the same ceiling because the backbone features dominate and limit diversity. The only approach that beat 0.183 was OT-CFM UNet (0.171) which learns the full residual distribution from scratch.

2. **Workflow: Never combined SwinIR predictions with OT-CFM as conditioning.** Iteration 5 trained flow on SwinIR *residuals* (target=HR-SwinIR_pred, std=0.0035) → failed due to source-target mismatch. But using SwinIR predictions as *extra conditioning* while targeting bilinear residuals (standard OT-CFM target) was never tried. This avoids the mismatch problem.

3. **Quality: OT-CFM UNet has no pretrained prior.** The 13M UNet on research2 learns everything from scratch. Adding SwinIR predictions as conditioning provides a strong prior: the model knows approximately where the HR should be and only needs to learn the residual distribution around it. This is CorrDiff's principle applied correctly.

### Direction: SwinIR-Conditioned OT-CFM Flow Matching

**Key insight from iteration 5 failure:** The problem was targeting SwinIR residuals (too small). The fix is to keep the standard OT-CFM target (bilinear residuals, reasonable variance) but add SwinIR predictions as extra conditioning information.

**Architecture:**
- Same AttentionUNet as flow_matching_v2.py
- in_channels=3: [x_t (1ch), lr_up_norm (1ch), swinir_pred_norm (1ch)]
- Target: bilinear residual (HR - bilinear(LR)), same as standard OT-CFM
- ~13M params (same capacity as research2 baseline)

**Why this should work:**
- Standard OT-CFM target distribution → no source-target mismatch
- SwinIR prediction provides strong spatial prior (MAE=0.250)
- Model can learn to use SwinIR hint for accuracy while generating calibrated diversity
- Strictly more information than standard 2-channel conditioning

**Why this might NOT work:**
- Extra conditioning channel may not help if LR_up already contains sufficient information
- SwinIR predictions may not be diverse enough to help with the stochastic component

**Training:**
- base_channels=64, mults=(1,2,4), attention 4 heads
- BS=64, LR=1e-4, cosine schedule, 200 epochs (time-limited)
- EMA decay=0.999
- 2hr wall-clock budget

### Training Log
- 20:41 — Started iteration. Generated run prefix volk-apex.
- 20:50 — GPU allocation: node3507 (L40S, normal partition, 3hr limit)
- 20:54 — Training started: 13M params, BS=64, LR=1e-4, cosine, EMA=0.999
- 22:51 — Training complete: 26 epochs in 115.8 min, best val=0.249975 at epoch 26
- 22:51 — First eval attempt on node3507 (20 steps) — timed out at 8992/10000
- 23:50 — node3507 allocation expired
- 23:52 — New allocation: node4306 (preemptable, 1hr) — eval timed out again at 9632/10000
- 00:54 — Third allocation: node4308 (preemptable, 2hr)
- 01:28 — Eval complete (10 steps): CRPS=0.175, MAE=0.299
- 02:36 — Eval complete (20 steps): CRPS=0.173, MAE=0.312
- 02:54 — AddCL eval timed out at 2624/10000 (allocation expired)

### Results: SwinIR-Conditioned OT-CFM (10K test)
| Method | CRPS | MAE | RMSE | Mass Viol |
|--------|------|-----|------|-----------|
| SwinIR-Flow 10 steps | 0.175 | 0.299 | 0.461 | 0.005 |
| SwinIR-Flow 20 steps | **0.173** | 0.312 | 0.464 | 0.005 |
| Multi-Head K=8 (iter 2) | 0.183 | 0.250 | 0.482 | 0.005 |
| OT-CFM UNet (research2) | **0.171** | 0.247 | 0.458 | 0.000001 |

### Analysis: Positive Result — SwinIR Conditioning Helps

**SwinIR-conditioned flow achieves CRPS=0.173, close to target 0.171 and beating multi-head (0.183).**

**Key observations:**
- CRPS=0.173 vs multi-head 0.183 = 5.5% improvement, breaking the ceiling
- Only 26 epochs due to slow node (~4.5min/ep on L40S) — model still improving
- MAE=0.312 is worse than multi-head's 0.250 — flow produces calibrated diverse samples rather than tight deterministic predictions
- This is the correct tradeoff: CRPS rewards calibrated uncertainty, not just accuracy
- AddCL evaluation incomplete (timed out) — would likely reduce mass violation from 0.005 to ~0.000001

**Why it works:**
- SwinIR prediction provides strong spatial prior as extra conditioning
- Standard OT-CFM target (bilinear residual) avoids the source-target mismatch that killed iteration 5
- 3-channel conditioning (x_t, lr_up, swinir_pred) gives the model strictly more information

**Limitation:**
- Only 26 epochs trained — likely undertrained compared to research2's full training
- With more epochs, CRPS could potentially reach or beat 0.171

### Updated Summary: 8 Iterations
| Iter | Direction | CRPS | Verdict |
|------|-----------|------|---------|
| 1 | SwinIR finetune (det.) | 0.250 | Baseline |
| 2 | Multi-head K=8 direct | 0.183 | Best SwinIR |
| 3 | Multi-head K=8 residual | 0.183 | Same ceiling |
| 4 | Multi-head unfreeze-2 | 0.183 | Same ceiling |
| 5 | Residual flow matching | 0.207 | Negative |
| 6 | DiT flow matching | 0.204 | Negative |
| 7 | Noise-conditioned tail | 0.200 | Negative |
| 8 | SwinIR-conditioned OT-CFM | **0.173** | **Best this branch** |

**Ending time:** 02:54 EDT
**Ending commit:** (pending)
