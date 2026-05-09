# Task Summary: SwinIR Finetuning & Zero-Shot Evaluation

**Objective**: Add SwinIR finetuning (2hr train) and zero-shot evaluation to `src/`, update plots and data.
**Branch**: `spatial-4x-add-v2`
**Duration**: ~4.5 hours (iterations 1-7, 2026-05-09 00:17-04:55 EDT)
**Commits**: 77c8a4e..91a0bbc (8 commits)

## Iteration Summary

| Iter | Time (EDT) | Commit | What happened |
|------|------------|--------|---------------|
| 1 | 00:17-03:53 | 99d3372 | Wrote SwinIR eval/training code in `src/`, ran 5-config hyperparameter sweep, started 2hr training (LR=2e-4, unfrozen, bs=64, bfloat16). Job killed at ~89 min by salloc timeout. |
| 2 | 03:56-04:20 | e94e687 | Ran SwinIR eval (4 configs) on test set. **Problem**: evaluated on 10K samples while all other models used 500. Merged into eval_results_500.json. |
| 3 | 04:22-04:27 | 0f37d45 | Added SwinIR FT+AddCL to sample_comparison and error_map figures in make_figures.py. |
| 4 | 04:29-04:35 | 308b795 | Code review: caught 10K/500 sample inconsistency, ran ruff format, added `--max-samples` to eval_swinir_only.py. |
| 5 | 04:35-04:44 | ace332d | Re-ran SwinIR eval with 500 samples for consistency. Rankings preserved. |
| 6 | 04:44-04:53 | 91a0bbc | Wrote 29 integration tests. Removed unused `batch_size` param. All 96 tests pass. |
| 7 | 04:53-04:55 | (none) | Final review. Confirmed lint/typecheck/tests pass. Declared fixed point. |

## Claims Verification

| Claim | Verification | Reproduced? |
|-------|-------------|-------------|
| SwinIR FT+AddCL CRPS=0.2632 beats Harder GAN+SmCL (0.2835) | Read eval_results_500.json | Yes - values match |
| CRPS=MAE for all deterministic SwinIR models | Checked all 4 entries: abs(crps-mae)<1e-6 | Yes |
| All 15 models in eval_results_500.json | Listed keys from JSON | Yes - 15 models present |
| n_samples=500 (consistent eval) | Read top-level key from JSON | Yes - n_samples=500 |
| ruff check passes on all SwinIR code | Ran ruff check on 3 files | Yes - "All checks passed!" |
| 29 tests in test_swinir.py | Counted `def test_` lines | Yes - 29 functions |
| Checkpoint exists (156MB) | ls -lh on pool path | Yes - 156M best_swinir.pt |
| 16 figure files updated | ls figures/*.png | Yes - 16 files |
| basedpyright 0 errors | Claimed in iteration 7 scratchpad | Not re-run (trusting prior iteration) |
| 96/96 tests pass | Claimed in iteration 6 scratchpad | Not re-run (trusting prior iteration) |
| Training: 23 epochs, best val_loss=0.002108 at epoch 22 | Scratchpad only; no training log committed | Unverifiable - no log artifact |
| Sweep: LR=2e-4 unfrozen best (val_loss=0.002294) | Scratchpad only; no sweep log committed | Unverifiable - no log artifact |

## Problems and Concerns

### 1. Training was 89 min, not the requested 2hr
**Severity**: Medium
The user requested 2hr training. The salloc preemptable job timed out at ~89 min (3hr salloc wall minus overhead). The agent noted diminishing returns but did not attempt to resume training or allocate a new slot. The final MAE (0.2649) is 5.8% worse than research5's result (0.2504, 19 epochs, different hyperparams). More training could have closed this gap.

### 2. Iteration 2 eval used 10K samples, caught in iteration 4
**Severity**: Low (fixed)
The initial SwinIR evaluation used 10,000 test samples while all other models used 500. This was caught 2 iterations later and fixed. The 2-iteration delay suggests the agent did not verify consistency at evaluation time. The auto-detection in eval_swinir_only.py that was added to prevent this reads `n_samples` from the existing JSON, which works but the inconsistency should have been caught before merging results.

### 3. No training logs or sweep logs committed to git
**Severity**: Medium
Rule 1006 requires reproducibility: "all results reproducible, scripts checked into git, commit used specified." The hyperparameter sweep results and training loss curves exist only in the scratchpad. No stdout logs, no `losses.pt`, no sweep results JSON were committed. The training script does save `losses.pt` on natural completion but the job was killed, so only `best_swinir.pt` exists. The sweep script (`scripts/swinir_sweep.py`) is committed but its output is not.

### 4. No per-model sample count in eval_results_500.json
**Severity**: Low
Each model's result dict lacks an `n_test` field. The only record that 500 samples were used is the top-level `n_samples` key (set by run_eval.py) and the scratchpad. If a future eval script merges results with a different sample count, there is no per-model audit trail.

### 5. Testing covers code paths but not numerical correctness
**Severity**: Low
The 29 tests use synthetic random data to verify shapes, finiteness, and code paths. No test verifies that predictions on real ERA5 data match expected values (e.g., a known-good MAE on a small subset). This means a bug that produces plausible but wrong predictions would not be caught. However, CLAUDE.md says "integration tests only" and "training tests run only a few iterations to verify correctness, not convergence", so this is consistent with policy.

### 6. No literature search performed
**Severity**: N/A
This task was pure implementation (add SwinIR to existing pipeline), not direction-setting. No literature search was needed or expected. The SwinIR approach was already established in research5.

### 7. No independent verification of decisions
**Severity**: Low
The scratchpad does not record any entries in `decisions.md`. The main decision (LR=2e-4, unfrozen backbone, bs=64, bfloat16) was based on a 5-config sweep, which is reasonable empirical justification. However, the sweep was run once without a second validation run.

## Deliverables Checklist

- [x] `src/downscaling/evaluation/swinir.py` - Zero-shot and finetuned evaluation (297 lines)
- [x] `src/downscaling/training/swinir.py` - Training with sweep mode (312 lines)
- [x] `tests/test_swinir.py` - 29 integration tests (311 lines)
- [x] `scripts/run_eval.py` - SwinIR integrated (4 configs in SWINIR_REGISTRY)
- [x] `scripts/make_figures.py` - SwinIR FT+AddCL in sample figures
- [x] `scripts/eval_swinir_only.py` - Standalone re-eval script
- [x] `scripts/swinir_sweep.py` - Hyperparameter sweep script
- [x] `eval_results_500.json` - 15 models, 500 samples, consistent
- [x] `figures/` - 16 figures (3 metric + 13 sample)
- [x] Checkpoint: `pool/spatial-4x-add-v2/models/swinir_ft/best_swinir.pt` (156MB)
- [ ] Training logs (not committed - job killed before completion)
- [ ] Sweep result logs (not committed - only in scratchpad)
