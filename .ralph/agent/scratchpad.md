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
