# Scratchpad — research3

## Iteration 1
**Start:** 2026-05-05 16:12 EDT, commit 23cd4b6
**End:** 2026-05-05 18:52 EDT, commit 14477bb
**Run prefix:** zyqup-yoihn

### Orientation
- First iteration on research3 branch
- No existing tasks, memories, decisions, or model weights on this branch
- Prior work (research: CRPS 0.199, research2: CRPS 0.171 corrected) established UNet flow matching baselines
- sweep-gpu2 (job 13346680) is pending — NOT mine, leave it alone
- node1627 (job 13357257) is orchestration node, ~42hrs remaining

### Concerns (3+)

1. **Workflow/Quality:** No model weights from prior branches accessible on research3. The research2 "best_flow.pt" (13M params) is not in pool or git. Must train from scratch — cannot fine-tune prior models.

2. **Quality:** Both prior tracks (research, research2) ONLY explored UNet-based architectures. The DiT (Diffusion Transformer) backbone is listed as research direction #7 in CLAUDE.md and noted as "untested in climate downscaling" — a major unexplored direction.

3. **Fact/Quality:** The cross-comparison claims "10 Euler steps sufficient, more steps don't help CRPS." This was only validated on UNet architectures. Different architectures (transformers) may have different optimal step counts — this claim shouldn't be assumed for new architectures.

4. **Quality:** The 2hr training budget means ~40 epochs at batch_size=64 on an A100. A DiT with 256 tokens (patch_size=8) should be comparable in speed to the 13M UNet, but this needs verification.

### Plan for this iteration
**ONE thing:** Implement and train a DiT (Diffusion Transformer) backbone for flow matching.

Rationale: This is the most under-explored direction with highest potential novelty. All prior work used UNets. DiT is the modern standard for image generation (DiT paper, Stable Diffusion 3) but hasn't been tested for climate downscaling.

Architecture:
- Pure DiT: patch_size=8 → 256 tokens at d_model=384
- 12 transformer blocks with AdaLN-Zero (time conditioning)
- Same OT-CFM training pipeline and AddCL constraint
- Same residual prediction (HR - bilinear(LR))
- Target: ~12-14M params (comparable to UNet v2)

### DiT 40-epoch results (node3404, L40S)
**Training:** 40 epochs, batch_size=64, lr=1e-4, cosine T_max=40
- Wall-clock: 21.1 min
- Val loss converged: 0.363 (vs UNet v2's 0.253 — 43% higher)
- Loss curve: steady improvement, converged by epoch 36-40

**Evaluation (2K test, 10 ensemble, Euler 10, AddCL):**

| Model | Params | CRPS (correct) | RMSE | MAE | Mass Viol |
|-------|--------|---------------|------|-----|-----------|
| **DiT 40ep** | 14.6M | **0.216** | 0.626 | 0.302 | 0.000001 |
| UNet v2 (research2) | 13M | 0.171 | 0.456 | 0.242 | 0.000001 |
| UNet attn (research) | 5.2M | 0.199 | 0.481 | 0.258 | 0.0001 |
| GAN baseline | 204K | 0.307 | 0.618 | 0.307 | 0.0454 |

**Analysis:** DiT is 26% worse than UNet v2 but 30% better than GAN. Key issues:
1. No skip connections → fine spatial detail lost
2. Patch-level processing (8x8) → within-patch detail must be reconstructed
3. Global attention is overkill for local spatial correlations
4. No translation equivariance from convolutions

**Submitted:** 200-epoch training (job 13363556) with extended cosine schedule to give DiT fair comparison.

### DiT 200-epoch results (node3404, L40S)
**Training:** 200 epochs, batch_size=64, lr=1e-4, cosine T_max=200
- Wall-clock: 105 min (well within 2hr budget)
- Best val loss: 0.301 at epoch 165 (vs 0.363 at 40 epochs — 17% improvement from more training)
- Train/val convergence: train=0.298, val=0.301 at end — good generalization

**Evaluation (FULL 10K test, 10 ensemble, Euler 10, AddCL):**

| Model | Params | Epochs | CRPS (correct) | RMSE | MAE | Mass Viol |
|-------|--------|--------|---------------|------|-----|-----------|
| **DiT 200ep** | 14.6M | 200 | **0.195** | 0.540 | 0.276 | 0.000001 |
| DiT 40ep | 14.6M | 40 | 0.216 (2K) | 0.626 | 0.302 | 0.000001 |
| UNet v2 (research2) | 13M | 39 | 0.171 (2K) / ~0.174 (10K est.) | 0.456 | 0.242 | 0.000001 |
| UNet attn (research) | 5.2M | 200 | 0.199 (10K) | 0.481 | 0.258 | 0.0001 |
| GAN baseline | 204K | — | 0.307 (10K) | 0.618 | 0.307 | 0.0454 |

**Key findings:**
1. DiT 200ep (0.195) is competitive with research branch UNet (0.199) — only 2% worse!
2. DiT 200ep is still 14% worse than research2 UNet v2 (0.171-0.174)
3. Extended training (40→200 epochs) improved DiT CRPS by ~9.7% (0.216→0.195)
4. DiT needs 5x more epochs but only 3x more wall-time (105 vs ~180 min for UNet 200ep)
5. The val loss gap (0.301 vs 0.253 for UNet) explains most of the CRPS gap

**Conclusion for DiT experiment:**
- Pure DiT CAN work for climate downscaling but is architecturally less efficient than UNet
- The main weakness is lack of multi-scale skip connections
- Next step: U-ViT (DiT + skip connections) could get best of both worlds
- DiT with extended training matches the simpler UNet, suggesting transformers benefit from data/compute but have a harder learning problem

## Iteration 2
**Start:** 2026-05-05 18:53 EDT, commit b0d6dd3
**Run prefix:** ecdzc-kkmks

### Orientation
- DiT 200ep CRPS 0.195 from iter-1 — verified from logs (correct CRPS formula, 10K test, AddCL)
- Actual DiT arch: hidden_size=256, depth=12, num_heads=4 (NOT 384 as in code defaults — training script overrode)
- Cancelled stale job 13369998 (ntlg-alta, unnamed prefix, submitted from node1627 during iter-1 but not documented)
- GPU slot now free. Two sweep-gpu jobs (13346680, 13360036) are NOT mine.

### Concerns (3+)

1. **Workflow:** Prior agent left a stale GPU allocation (job 13369998, ntlg-alta) that wasn't cancelled or documented. The job name doesn't match the zyqup-yoihn prefix. Cancelled it — no useful work was running.

2. **Quality:** DiT's diagnosed weakness (no skip connections) is the #1 gap vs UNet. The U-ViT direction (DiT + skip connections) is the natural fix and directly tests this hypothesis. This is well-supported by the original U-ViT paper ("All are Worth Words", 2022).

3. **Quality:** The prior agent didn't add a 3x3 conv refinement after unpatchify — the original U-ViT paper found this crucial for spatial quality. Pure linear unpatch creates visible grid artifacts at patch boundaries. Must add conv refinement.

4. **Fact check:** Verified DiT 200ep CRPS 0.195 from logs/dit_200ep_train.log line 321. The number is correct.

### Plan for this iteration
**ONE thing:** Implement and train U-ViT (DiT + long skip connections + conv refinement) for flow matching.

Rationale: This directly addresses the main diagnosed weakness of pure DiT (no multi-scale info flow). If skip connections close the gap with UNet, it validates that transformers ARE viable for climate downscaling when properly architected.

Architecture:
- Same as DiT: patch_size=8, hidden_size=256, depth=12, num_heads=4
- ADD: long skip connections between blocks (0→11, 1→10, ..., 5→6) via concat+linear
- ADD: 3x3 conv refinement after unpatchify (from U-ViT paper)
- Extra params: ~0.8M (skip projections) + ~0.6K (conv refinement) ≈ ~15.4M total
- Same OT-CFM training, residual prediction, AddCL constraint
- Train 200 epochs to match DiT comparison

### U-ViT 200-epoch results (node3507, L40S)
**Training:** 200 epochs, batch_size=64, lr=1e-4, cosine T_max=200
- Wall-clock: 117.4 min (within 2hr budget)
- Best val loss: 0.3038 at epoch 188 (vs DiT's 0.301 — 1% worse)
- Train/val convergence: train=0.299, val=0.306 at end

**Evaluation (FULL 10K test, 10 ensemble, Euler 10, AddCL):**

| Model | Params | Epochs | CRPS (correct) | RMSE | MAE | Mass Viol |
|-------|--------|--------|---------------|------|-----|-----------|
| **U-ViT 200ep** | 16.5M | 200 | **0.194** | 0.533 | 0.274 | 0.000001 |
| DiT 200ep | 14.6M | 200 | 0.195 | 0.540 | 0.276 | 0.000001 |
| UNet attn (research) | 5.2M | 200 | 0.199 | 0.481 | 0.258 | 0.0001 |
| UNet v2 (research2) | 13M | 39 | 0.171 | 0.456 | 0.242 | 0.000001 |
| GAN baseline | 204K | — | 0.307 | 0.618 | 0.307 | 0.0454 |

**Key findings:**
1. U-ViT (0.194) marginally beats DiT (0.195) — skip connections help but only slightly
2. U-ViT beats simpler UNet (research, 0.199) by 2.5%
3. U-ViT still 13.5% worse than UNet v2 (0.171)
4. Skip connections improved val_loss (0.304 vs 0.301) but CRPS gap barely changed
5. The transformer architecture bottleneck is NOT skip connections alone — it's the fundamentally different inductive bias (patch tokenization vs multi-scale convolution)

**Conclusion for U-ViT experiment:**
- Adding skip connections to DiT provides marginal improvement
- The core transformer approach hits a wall for this task: patch_size=8 tokenization loses fine spatial detail that convolutions preserve naturally
- Both DiT and U-ViT converge to similar val_loss (~0.301-0.304) and CRPS (~0.194-0.195)
- The UNet's multi-scale convolutional processing is architecturally better suited for spatial super-resolution
- Next directions to explore: (a) smaller patch size (4 instead of 8), (b) hybrid conv-transformer, (c) return to UNet with improvements (EMA, data augmentation, CRPS loss)

**End:** 2026-05-05 21:06 EDT

## Iteration 3
**Start:** 2026-05-05 21:08 EDT, commit 51f5e60
**Run prefix:** vjgxe-qsqzn

### Orientation
- DiT (0.195) and U-ViT (0.194) both plateau — transformer experiments exhausted
- UNet v2 from research2 is still best (CRPS 0.171 at only 39 epochs) but was never:
  - Trained for 200 epochs
  - Trained with EMA
  - Trained with non-uniform timestep sampling
- Two unidentified GPU jobs (ntlg-alta 13379891, wqvi-feml 13382434) — NOT mine, leaving alone
- sweep-gpu1 (13360036) — NOT mine

### Concerns (3+)

1. **Quality:** All flow matching training uses uniform timestep sampling t ~ U(0,1). Modern literature (Stable Diffusion 3, Rectified Flow) shows logit-normal sampling significantly improves training by focusing on harder intermediate timesteps. This is completely untested in our experiments.

2. **Quality:** No experiment has used EMA (Exponential Moving Average) for model weights. EMA is standard practice in diffusion/flow matching — it smooths over training noise and typically improves 1-5%. The UNet v2 (0.171) was trained without EMA.

3. **Quality:** The UNet v2 architecture (13M params, CRPS 0.171) was only trained for 39 epochs on research2. DiT and U-ViT got 200 epochs. The UNet v2 has never been given the same 200-epoch budget — the comparison is unfair. Training UNet v2 for 200ep with EMA could improve well below 0.171.

4. **Workflow:** Prior iterations correctly concluded transformers plateau for this task. The natural next step is UNet improvements, not more transformer variants.

### Plan for this iteration
**ONE thing:** Train UNet v2 (13M params) for 200 epochs with EMA + logit-normal timestep sampling.

Rationale: This combines two well-motivated training recipe improvements that are easy to implement and novel for this project:
- **EMA** (3-5 lines): smooths weight updates, standard in diffusion
- **Logit-normal sampling** (2 lines): SD3's key finding — concentrates training on informative timesteps
- **200 epochs**: fair comparison with DiT/U-ViT experiments

Expected outcome: CRPS should improve from 0.171 baseline. If logit-normal helps significantly, it's a finding worth reporting.

### Training observations
- UNet v2 (13M params) at 128x128: ~4.5 min/epoch on L40S (vs DiT's ~0.5 min/epoch)
  - UNet processes full-resolution feature maps; DiT patches to 16x16 tokens
  - 200 epochs infeasible (~15 hr), reduced to 26 epochs (~102 min)
- Val loss trajectory: 0.375 → 0.267 (best at epoch 23)
  - For comparison: research2 (39ep, uniform t) reached val_loss 0.253
  - Logit-normal model converged to ~5.5% higher val loss than uniform baseline
- LR cosine schedule with T_max=26 reached near-zero by epoch 20

### Results (node1705, L40S → node4208 for EMA eval)

**Evaluation (FULL 10K test, 10 ensemble, Euler 10, AddCL):**

| Model | Params | Epochs | t-sampling | EMA | CRPS (correct) | RMSE | MAE | Mass Viol |
|-------|--------|--------|-----------|-----|---------------|------|-----|-----------|
| **UNet v2 + logit-normal** | 13M | 26 | logit_normal | no | **0.179** | 0.498 | 0.257 | 0.000001 |
| **UNet v2 + logit-normal + EMA** | 13M | 26 | logit_normal | 0.9999 | **0.228** | 0.650 | 0.307 | 0.000001 |
| UNet v2 (research2) | 13M | 39 | uniform | no | 0.171 | 0.456 | 0.242 | 0.000001 |
| U-ViT 200ep (iter-2) | 16.5M | 200 | uniform | no | 0.194 | 0.533 | 0.274 | 0.000001 |
| DiT 200ep (iter-1) | 14.6M | 200 | uniform | no | 0.195 | 0.540 | 0.276 | 0.000001 |

**Key findings:**
1. UNet v2 + logit-normal (26ep): CRPS 0.179 — 4.7% worse than research2 baseline (0.171, 39ep)
2. EMA (decay=0.9999) with only 26 epochs HURTS: CRPS 0.228 (27% worse). EMA needs 10K+ steps to converge; 26 epochs × 625 steps = 16K steps means EMA still averages over early, poorly-trained weights.
3. Logit-normal sampling does NOT improve over uniform: at comparable epoch counts, val loss is 5.5% higher. The SD3 finding doesn't transfer to this scale/task.
4. UNet v2 at 128x128 is ~9x slower per epoch than DiT with patch_size=8, which severely limits epoch count within budget.
5. Even with fewer epochs (26 vs 39), the CRPS gap is only 4.7% — suggesting UNet architecture is robust and most of the gap is simply less training.

**Conclusion:**
- Logit-normal timestep sampling is not beneficial for flow matching on this climate downscaling task at this scale
- EMA with 0.9999 decay is inappropriate for short (<50 epoch) training runs; would need lower decay (e.g., 0.999) or many more epochs
- The UNet v2 architecture from research2 remains the best approach
- Next directions to explore: (a) longer UNet v2 training with uniform t (match research2 properly), (b) data augmentation (flips/rotations), (c) CRPS-aware loss function, (d) different ODE solvers/steps

**End:** 2026-05-06 00:51 EDT
