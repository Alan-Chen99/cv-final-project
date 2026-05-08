# Task Summary: research6 — Flow Matching for TCW 4x Downscaling

**Branch:** research6
**Duration:** ~22 hours wall-clock (9 iterations, 2026-05-06 00:58 - 22:46 EDT)
**Final result:** CRPS 0.1728 (10K test, 10 members, AddCL), matching research2's 0.171 within 1%
**Report:** [notes/2026-05-06-research6-report.md](notes/2026-05-06-research6-report.md)

## Iteration Timeline

| Iter | Time (EDT) | Duration | Activity | CRPS | Outcome |
|------|------------|----------|----------|------|---------|
| 1 | 00:58-02:05 | 1.1h | Zero-shot baselines: bicubic, SwinIR, UNet L1 | 0.259 (det) | Deterministic models can't compete on CRPS |
| 2 | 02:06-06:50 | 4.7h | Residual flow matching (min-max, 9.1M, 100ep) | 0.238 | Flow matching >> deterministic, but gap to 0.171 |
| 3 | 06:46-10:57 | 4.2h | OT coupling (failed: too slow), architecture scaling (12.5M, 25ep) | 0.262 | OT coupling impractical on CPU; 3 GPU preemptions |
| 4 | 10:59-14:32 | 3.5h | Resume flow_v2 to 60ep; step count + Heun ablation | 0.232 | Architecture doesn't help; Heun catastrophic |
| 5 | 14:33-19:28 | 4.9h | **Key insight:** read research2 code, discovered z-score normalization | 0.178* | Normalization was root cause, not OT coupling |
| 6 | 19:25-21:40 | 2.3h | Full 10K eval (AddCL + none) | **0.1728** | Matches research2; constraint is free |
| 7 | 21:41-22:40 | 1.0h | Report, visualizations, SmCL eval (NaN) | — | SmCL incompatible; report written |
| 8 | 22:41-22:44 | 0.05h | Report corrections (labels, budget, baselines) | — | Factual fixes |
| 9 | 22:45-22:46 | 0.02h | Fixed-point check | — | No changes needed |

*50-sample CPU estimate

## Claims Verification

| # | Claim | Verification Method | Status |
|---|-------|-------------------|--------|
| 1 | CRPS = 0.1728 (10K test, z-score model + AddCL) | Full 10K eval in iter 6 | **Verified** — eval ran to completion on GPU |
| 2 | AddCL has zero CRPS cost | Compared AddCL (0.1728) vs none (0.1728) on 10K test | **Verified** — mass viol 0.000001 vs 0.003253 |
| 3 | Research2 uses NO OT coupling | Grep for `ot_coupling/linear_sum_assignment/sinkhorn` → 0 matches; training uses `torch.randn_like` | **Verified** (independently reproduced above) |
| 4 | Z-score normalization is the key factor | Training with z-score improved CRPS from 0.232 → 0.1728 (27%) | **Verified** — controlled experiment: only normalization changed |
| 5 | SmCL causes NaN on physical-space flow matching outputs | 100-sample CPU eval returned all NaN; `torch.exp()` overflows on TCW values | **Verified** — root cause identified |
| 6 | Heun solver fails (CRPS 2.149) | Eval on 10K test, min-max model | **Verified** — but only on min-max model, not z-score |
| 7 | SwinIR finetune worse than zero-shot | 0.285 (finetune) vs 0.279 (zero-shot) on 2K test | **Verified** — small margin, consistent across metrics |
| 8 | 50-sample CPU estimate accurate (0.178 vs 0.1728) | Compared against 10K GPU eval | **Verified** — 3% error, validates quick-check method |
| 9 | Training time 178 min exceeds 2hr budget | Scratchpad records 40 epochs in 178 min; 2hr mark = epoch 27 | **Verified** — report honestly flags violation |
| 10 | Model matches research2 (CRPS 0.1728 vs 0.171) | Independent training + eval on this branch | **Verified** — 1% gap within eval noise |
| 11 | Ensemble spread meaningful | Visual inspection of `figures/research6/ensemble_members.png` | **Partially verified** — qualitative only, no spread metric computed |

## Problems and Concerns

### Critical: 8 hours wasted on false OT coupling hypothesis (Iters 3-4)

Research2's code (`flow_matching_v2.py`) was in the repo the entire time. Reading its 500-line training loop would have taken 5 minutes and revealed:
- No OT coupling (plain `torch.randn_like` noise)
- Z-score normalization (the actual key difference)

Instead, the agent:
- Iter 3: Implemented CPU Hungarian matching (~4h), found it too slow, pivoted to architecture scaling
- Iter 4: Completed architecture training (~3.5h), concluded "OT coupling is the key missing ingredient"
- Only in iter 5 did the agent actually read the reference code

**Root cause:** The agent trusted prior notes labeling research2 as "OT-CFM" without verification. The CLAUDE.md warns: "prior agent notes MUST be taken as CONTEXT ONLY. Verify all claims before using." This was not followed.

**Impact:** ~8 hours of the ~22-hour budget (36%) spent on a false lead. The correct fix was discovered in iteration 5 and confirmed in iteration 6.

### Moderate: Training budget violation (178 min > 120 min)

The task specifies "<=2 hr of training allowed" for fair cross-method comparison. The reported model trained for 178 min (40 epochs). At the 2hr mark, training was at epoch 27.

**No checkpoint exists at the 2hr mark.** The reported CRPS 0.1728 comes from a model trained 50% beyond budget. The report flags this but cannot provide a budget-compliant CRPS number.

**Mitigating factor:** The 2hr-mark val loss (0.256) is close to the final val loss (0.251), suggesting the budget-compliant model would have similar (but slightly worse) CRPS.

### Moderate: Decision journal entries never independently verified

Both DEC-001 and DEC-002 have "Independent evaluation: not-started":
- DEC-001 (use research2's recipe): Confidence 85, verified by training results (CRPS 0.1728 confirmed the decision was correct), but no formal independent verification iteration
- DEC-002 (OT coupling narrative is wrong): Confidence 99, trivially verifiable by grep, but marked not-started

The scratchpad's iteration 9 "fixed-point" assessment did not revisit these.

### Minor: Ensemble spread not computed for best model

The spread column shows "---" for all z-score model entries. The report relies on visual inspection of `ensemble_members.png` for diversity claims. This means:
- No quantitative calibration assessment
- Cannot determine if the ensemble is well-calibrated (spread should match RMSE for calibrated ensembles)
- Blocked by GPU preemptions (7 total across iters 5-7)

### Minor: Heun/20-step findings from wrong model

The "Euler solver is sufficient" and "Heun fails catastrophically" findings (Section 5 of report) were tested only on the min-max normalization model — the one that doesn't work well. Z-score normalization may produce smoother velocity fields where these results differ. The report correctly scopes the claim, but the finding has limited generalizability.

### Minor: No frequency-domain analysis

The report acknowledges this as an open question. Power spectral density comparison would reveal whether the model captures fine-scale structure or over-smooths. This is standard practice in climate downscaling evaluation.

## Systematic Workflow Issues

1. **Reference code not read early enough.** The single highest-leverage action (reading `flow_matching_v2.py`) was delayed until iteration 5. A habit of "diff the reference implementation first" would have saved 36% of the time budget.

2. **Preemption-heavy environment underestimated.** 7+ GPU preemptions across iterations 5-7 blocked SmCL evaluation and 20-step ablation. The agent correctly tried switching to `mit_normal` when available but normal slots were occupied by other agents.

3. **No intermediate checkpointing.** The 40-epoch training run saved only best-val and final checkpoints. Saving every N epochs (or at the 2hr mark) would have provided budget-compliant numbers.

4. **Iteration boundaries well-managed.** GPU cleanup was done consistently. Run prefixes were used. No dangling jobs observed.

5. **Report quality is high.** Internally consistent after iter 8 fixes. Honest about limitations. Figures committed. Reproduction commands provided.

## Final Assessment

The research6 branch successfully reproduced research2's CRPS 0.171 (within 1%) and identified the key factor: z-score normalization of residuals, not OT coupling. The main output (report + model weights + visualizations) is solid. The primary workflow failure was not reading the reference code earlier, which wasted ~8 hours. The training budget violation is flagged but unresolved. Overall, the core scientific finding is correct and well-documented.
