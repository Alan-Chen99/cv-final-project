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
