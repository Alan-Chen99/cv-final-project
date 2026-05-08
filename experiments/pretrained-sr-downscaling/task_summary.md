# Task Summary: research5 -- Pretrained Image SR for Climate Downscaling

**Branch:** research5
**Duration:** ~42 hours (2026-05-06 00:45 to 2026-05-07 03:10 EDT)
**Iterations:** 12 total (8 experiments + 3 report/review + 1 summarizer)
**Best CRPS:** 0.173 (SwinIR-conditioned OT-CFM, 20 Euler steps, 10K test)
**Report:** `notes/2026-05-07-research5-pretrained-sr-report.md`

## Iteration Log

| Iter | Time (EDT) | Direction | CRPS | Key Finding |
|------|-----------|-----------|------|-------------|
| 1 | 00:54-04:00 | SwinIR zero-shot + finetune | 0.250 | Pretrained SR transfers to climate; MAE=0.250 deterministic |
| 2 | 04:00-06:47 | Multi-head K=8 direct CRPS | **0.183** | 27% improvement over deterministic; over-dispersed (spread=0.272 > MAE=0.250) |
| 3 | 06:47-08:59 | Multi-head K=8 residual | 0.183 | Negative: residual param identical result; not the bottleneck |
| 4 | 09:00-11:13 | Multi-head unfreeze last 2 layers | 0.183 | Negative: unfreezing backbone doesn't help; multi-head ceiling confirmed |
| 5 | 11:14-15:12 | CorrDiff-style residual flow | 0.207 | Negative: N(0,1)->N(0,0.0035) source-target mismatch; corrupts SwinIR mean |
| 6 | 15:14-18:20 | DiT flow matching | 0.204 | Negative: 8x8 patches lose fine spatial detail; UNet better for pixel tasks |
| 7 | 18:23-20:39 | Noise-conditioned SwinIR tail | 0.200 | Negative: backbone features dominate; tail ignores noise |
| 8 | 20:41-02:54 | SwinIR-conditioned OT-CFM | **0.173** | Best result: SwinIR pred as 3rd conditioning channel for standard OT-CFM |
| 9 | 02:56-03:05 | Write report | -- | Comprehensive report written |
| 10 | 03:02-03:06 | Report review | -- | Fixed 3 factual errors (epochs, GPU type, stale inventory entry) |
| 11 | 03:07-03:08 | Report review | -- | Fixed final commit placeholder |
| 12 | 03:09-03:10 | Fixed-point check | -- | No changes; declared fixed-point |

## Claims Verification

| # | Claim | Verification Method | Status |
|---|-------|-------------------|--------|
| 1 | Best CRPS=0.173 (10K test, 20 steps) | Scratchpad training log at iter 8 | Logged, not re-run (needs GPU) |
| 2 | Multi-head ceiling at CRPS=0.183 | 3 independent experiments converge | Consistent pattern across iters 2-4 |
| 3 | research5 0.173 matches research2 ~0.174 (10K) | Cross-comparison note: research2 0.171 on 2K, estimated +1.7% for 10K | **Estimate only** -- research2 10K CRPS was never directly measured |
| 4 | Training scripts exist (10 listed) | `ls src/exp-pretrained-sr/` + `scripts/eval_crps.py` | **Verified** -- all 10 files confirmed |
| 5 | Model checkpoints exist (8 dirs) | `ls pool/datasets/research5/models/` | **Verified** -- all 8 directories confirmed |
| 6 | Figure committed (dit_flow_training.png) | `ls figures/` | **Verified** |
| 7 | AddCL works on best model | Partial eval timed out at 2624/10000 | **Incomplete** -- extrapolated from other experiments |
| 8 | Zero-shot CRPS=1.28 (report) | Scratchpad iter 1 shows MAE=0.3174 for zero-shot | **Discrepancy** (see below) |
| 9 | Iter 2 trained 18 epochs (report compute table) | Scratchpad says 35 total, best at epoch 17 | **Discrepancy** (see below) |
| 10 | Iter 3 trained 27 epochs (report compute table) | Scratchpad says 30 total (preempted), best at epoch 12 | **Discrepancy** (see below) |
| 11 | Iter 4 trained 20 epochs (report compute table) | Scratchpad says 25 total, best at epoch 17 | **Discrepancy** (see below) |
| 12 | Iter 5: 184 epochs on H100 | Scratchpad confirmed; report corrected in iter 10 | **Verified** (after correction) |

## Problems and Concerns

### 1. Zero-shot CRPS discrepancy (report factual error)

The report states: "Zero-shot SwinIR produced CRPS=1.28 -- completely failed on climate data."
The scratchpad (iter 1) records: SwinIR zero-shot MAE/CRPS(det) = **0.3174** on 10K test.

These are clearly different numbers. The scratchpad is the primary log, and all other scratchpad metrics are internally consistent and used throughout the workflow. The 1.28 figure appears only in the report (written at iter 9) with no provenance in the scratchpad. This is likely a report-writing error that was **not caught by 3 review iterations** (10, 11, 12).

### 2. Epoch counts in compute table don't match scratchpad

The report's compute summary uses numbers that don't match the scratchpad for iters 2-4:
- Iter 2: report=18, scratchpad=35 total / best at 17
- Iter 3: report=27, scratchpad=30 total / best at 12
- Iter 4: report=20, scratchpad=25 total / best at 17

The report's "Epochs" column may be using checkpoint metadata (filename or best-epoch marker) rather than total training epochs. This is ambiguous and misleading -- the column header implies total epochs trained. Three review iterations did not cross-reference these with the scratchpad.

### 3. Review process failed to catch factual errors

Iterations 10-12 performed three review passes. They caught: stale inventory entry, epoch/GPU type for iter 5, and a placeholder commit hash. But they missed the zero-shot CRPS discrepancy and epoch count inconsistencies. **The review was narrow** -- it checked whether files and checkpoints existed, but did not systematically verify numbers in the report against the scratchpad.

### 4. Only 1 figure committed (rule violation)

The task requires: "Check key graphs and outputs into git." Only `figures/dit_flow_training.png` (from iter 6) was committed. Seven of eight experiments have no training curve or output visualization in git. Iteration 11 noted this as a limitation but didn't attempt retrospective figure generation.

### 5. Best model undertrained (26/200 epochs)

The SwinIR-conditioned OT-CFM model was configured for 200 epochs but only completed 26 due to a slow L40S node (~4.5 min/epoch). The model was still improving at termination. The reported CRPS=0.173 is a lower bound on what this approach could achieve with full training. The report correctly notes this limitation.

### 6. research2 comparison rests on estimated baseline

The central claim -- "research5 (0.173) matches research2 (~0.174)" -- depends on an estimated 10K CRPS for research2. The actual measured research2 number is 0.171 on 2K test; the 10K number was extrapolated via a "+1.7% ratio" from cross-comparison notes, and the original 10K log was lost to preemption. A fair comparison would require re-running research2 eval on 10K test.

### 7. No independent decision verification

All 5 entries in `decisions.md` have "Independent evaluation: not-started" (or "not needed" for DEC-003). The guardrails require independent verification of decisions. No iteration circled back to verify earlier decisions.

### 8. AddCL evaluation incomplete for best model

The AddCL constraint evaluation for the best model (iter 8) timed out at 2624/10000 samples. The report says AddCL is "confirmed on partial eval; consistent with all prior experiments." This is reasonable extrapolation but not a direct measurement. All other experiments showed AddCL reduces mass violation to ~0.000001, so the claim is plausible but unverified for this specific model.

## Systematic Failure Patterns

1. **Review scope too narrow.** Reviews checked file existence and specific known issues but didn't systematically cross-reference report numbers against scratchpad logs. A checklist-based review ("for each number in the report, find its source in the scratchpad") would have caught the discrepancies.

2. **Figures treated as optional.** Despite a MUST requirement, most iterations didn't generate or commit visualizations. Only the DiT experiment (iter 6) produced a figure, possibly because it was a novel architecture. The iterative loop's pressure to finish within 4hr per iteration crowded out visualization work.

3. **Eval infrastructure fragile.** Iteration 8 required 3 separate GPU allocations for evaluation (two timed out). Eval taking as long as training suggests the eval code is not optimized for large test sets, or the allocation time limits are too tight for eval workloads.

## Workflow Assessment

**Strengths:**
- Systematic exploration: 8 experiments covered a wide design space (pretrained SR, multi-head ensembles, residual parameterization, flow matching, DiT, noise conditioning, SwinIR conditioning)
- Each experiment had clear hypothesis, architecture, and analysis of failure modes
- Negative results were well-documented with root cause analysis
- The iter 5 failure directly informed the iter 8 success (fixing source-target mismatch by switching from "target residuals" to "condition on predictions")
- Report is comprehensive and structurally sound
- Decisions journal captured key choices with alternatives and biases

**Weaknesses:**
- Report contains at least 2 categories of factual errors not caught by review
- Visualization rule not followed
- Best model potentially left performance on the table (undertrained)
- Comparison to research2 baseline is approximate, not definitive
