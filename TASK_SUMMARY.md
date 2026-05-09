# Task Summary — organize1 Branch

**Objective:** Organize experiment code from 6 branches into a proper Python project, run evaluations, generate visualizations, write report.

**Duration:** 12 iterations, ~2.5 hours (2026-05-08 19:09–21:35 EDT)

**Verdict:** All 7 task items completed. Code quality is solid. Report numbers are verified. A few systematic workflow issues described below.

## Iteration Table

| Iter | Focus | Key Output | Duration |
|------|-------|-----------|----------|
| 1 | Project setup + core extraction | pyproject.toml, src/downscaling/ (7 modules), 22 tests | 16 min |
| 2 | CRPS formula bug fix | Fixed M*(M-1) denominator, added regression tests | 6 min |
| 3 | Training code extraction | training.py (OT-CFM loop), 5 GPU tests | 8 min |
| 4 | Evaluation baselines + GPU run | evaluate_all.py, 4 results JSONs, verified on GPU | 63 min |
| 5 | Visualization module | visualization.py (5 plot funcs), 6 figures, 7 tests | 13 min |
| 6 | Report | REPORT.md (comprehensive) | 3 min |
| 7 | Review: test count + coverage config | Fixed test count 38→50→51, removed unreachable fail_under | 8 min |
| 8 | Review: train→eval compat | Fixed checkpoint args, removed duplicate deps | 5 min |
| 9 | Review: scripts/ lint | Fixed 9 ruff errors, added scripts/ to basedpyright | 4 min |
| 10 | Review: stale report commands | Updated lint commands and test count in repro section | 3 min |
| 11 | Review: rounding error | Fixed logit-normal RMSE 0.478→0.477 | 7 min |
| 12 | Verification | No changes — fixed point declared | 1 min |

## Claims Verification

| # | Claim | Source | Verification | Reproducible? |
|---|-------|--------|-------------|---------------|
| 1 | Wide96 CRPS=0.180 (200 samples) | Report L20 | JSON: 0.17978→rounds to 0.180 | Yes (check JSON) |
| 2 | Wide96 CRPS=0.1676 (10K) | Report L35 | research3 branch report (not canonical pipeline) | **No** — from experiment code, not independently verified |
| 3 | Bilinear CRPS=0.506 (10K) | Report L27 | eval_baselines_full.json: 0.50646→0.506 | Yes (check JSON) |
| 4 | Bicubic+AddCL CRPS=0.353 (10K) | Report L27 | eval_baselines_full.json: 0.35334→0.353 | Yes (check JSON) |
| 5 | Logit-normal RMSE=0.477 | Report L22 | JSON: 0.47749→0.477 | Yes (check JSON, fixed in iter 11) |
| 6 | AddCL is "free" (no CRPS cost) | Report L115 | JSON: wide96 0.17997→0.17978 (Δ=0.001) | Yes — within noise |
| 7 | Midpoint > Euler at equal NFE | Report L48 | solver_comparison.json: 0.1800 vs 0.1824 | Yes (check JSON) |
| 8 | 35 CPU tests, 16 GPU-skip | Report L158 | `pytest tests/ -q`: 35 passed, 16 skipped | Yes (ran and confirmed) |
| 9 | ruff clean | Report L186 | `ruff check src/ tests/ scripts/`: All checks passed | Yes (ran and confirmed) |
| 10 | basedpyright clean | Report L187 | `basedpyright src/ tests/ scripts/`: 0 errors | Yes (ran and confirmed) |
| 11 | 100% coverage on core logic | iter1 scratchpad | Never measured with pytest-cov until iter7; iter7 confirms 100% on metrics/constraints/data | Partially — actual coverage run only happened in iter7 |
| 12 | GAN baseline CRPS=0.307 | Report L37 | From Harder et al. paper — not rerun | **No** — literature value, accepted |
| 13 | "6 figures in figures/" | Report L131-139 | `ls figures/`: 6 .png files present | Yes (files exist) |
| 14 | SmCL overflows on TCW | iter4 scratchpad | Documented as NaN in eval_results.json | Yes — NaN entries confirm |
| 15 | 51 total tests collected | Report L158 | `pytest --co -q`: 51 tests collected | Yes (ran and confirmed) |

## Problems and Concerns

### 1. Systematic: "ruff clean" was false for 8 iterations

Iterations 1–8 all claimed "ruff clean" or "0 ruff errors," but only checked `src/ tests/`. The `scripts/` directory had 9 lint errors the entire time. This was not caught until iteration 9. **Root cause:** the ruff config's `src` key only includes `["src", "tests"]`, and the check command matched that scope. No iteration questioned whether scripts/ should be included.

**Impact:** Low (scripts are standalone, not library code). But it demonstrates a blind spot in verification — the agent trusted a partial scope check without questioning whether the scope was complete.

### 2. Systematic: Test counts were wrong across iterations

| Iteration | Claimed count | Actual at that time |
|-----------|--------------|---------------------|
| 5 | "31 non-GPU tests" | Unclear — may have been correct at the time |
| 6 | "38 integration tests" (in REPORT.md) | Should have been ~50 |
| 7 | Fixed to "51 (35 CPU, 16 GPU-only)" | Correct |
| 8 | "50→51" after adding 1 test | Correct |

The report initially said "38 tests" (stale from iter 3 count) and wasn't updated when visualization tests were added in iter 5. Each iteration had a different count. **Root cause:** test counts were copied from earlier scratchpad entries rather than re-measured.

### 3. 10K flow model results were never independently verified

The report's "Full Test Set (10K)" table (CRPS=0.1676 for Wide96) comes from the research3 branch report, produced by experiment-specific code — not by the canonical `scripts/evaluate_all.py`. The report acknowledges this ("From research3 branch report") but the claim remains unverified through the canonical pipeline.

**Impact:** Medium. The 200-sample results (0.180) are consistent with the 10K claim (0.168), and the formula is the same, but the stated purpose of this branch was to create a *single canonical eval pipeline to verify all claims* (iter 1 concern #3).

### 4. Visualization tests verify "file exists," not correctness

All 7 visualization tests (test_visualization.py) only assert:
- A PNG file was created
- The file has non-zero size

None verify plot content, axis labels, data correctness, or that the right number of subplots exist. A function that saves a blank image would pass all tests.

**Impact:** Low for a research project, but violates the spirit of "100% coverage on core logic" — the visualization module is core logic too.

### 5. Coverage claim was unmeasurable until iteration 7

Iteration 1 claimed "Coverage: 100% on core logic" but `pytest-cov` was not installed until iter 7. The `fail_under = 90` in pyproject.toml was unreachable on CPU. Iter 7 fixed this, confirmed actual coverage, and removed the misleading config.

**Root cause:** the claim was based on manual inspection ("I wrote tests for every function") rather than measurement.

### 6. No decisions.md created

The workflow instructs using `.ralph/agent/decisions.md` for consequential decisions with confidence scores. The file was never created despite multiple architectural decisions (e.g., SmCL dropped due to overflow, training features excluded, checkpoint format chosen). These decisions are documented in the scratchpad but without the structured format that enables independent verification.

### 7. eval_results.json is redundant

Contains identical baseline data to eval_200samples.json (same numbers, 20-minute timestamp difference). Two files with overlapping data. Acknowledged in iters 5 and 11 but never resolved (reasonable — removing could break references).

### 8. Failure classes not caught by existing tests

The current test suite would NOT catch:
- **Incorrect plot content** (see concern #4)
- **Normalization bugs in evaluation** — tests don't run the full evaluate→denormalize→compare pipeline on real data
- **Checkpoint loading with wrong architecture** — test_load_without_args_uses_defaults verifies weight loading but doesn't check that the loaded model produces correct outputs
- **Data ordering/shuffling issues** — no test verifies deterministic evaluation ordering
- **Off-by-one in CRPS ensemble size** — the M*(M-1) fix was caught by manual review, not by a test that would detect M*M vs M*(M-1) difference (the regression test was added after the fix)

### 9. Scratchpad iteration 4 baseline numbers don't match JSON

Iter 4 scratchpad reports "bilinear=0.507" but eval_baselines_full.json shows 0.5064→0.506. The scratchpad used 200-sample numbers (0.538) in the table but 10K numbers (0.507) in the text. Minor inconsistency in working notes — the report has the correct numbers.

## What Went Well

1. **CRPS formula bug caught early** (iter 2) — this was the most impactful finding. The O(M²) reference implementation comparison was a strong verification.
2. **Train→eval incompatibility caught** (iter 8) — would have caused a KeyError for anyone training with the canonical pipeline.
3. **Iterative refinement worked** — each iteration caught genuine issues from prior ones.
4. **Report numbers verified against JSON** — all 3-decimal-place roundings were cross-checked.
5. **Clean final state** — all linters pass, tests pass, git clean.
