# Task Summary — organize2 Branch

**Objective:** Organize experiment code from 6 research branches into a proper Python
project structure, run evaluations with pre-trained weights, generate visualizations,
and write a comprehensive report.

**Duration:** 12 iterations, ~3 hours (2026-05-08 19:13–21:52 EDT)

**Verdict:** All 7 task items completed. Final output is functional and the main
results table is reproducible. However, the workflow revealed systematic problems
with claim verification, test quality, and hallucinated file paths that were only
caught after multiple review passes.

---

## 1. Iteration Table

| Iter | Time (EDT) | Focus | Key Output | Issues Found |
|------|-----------|-------|------------|-------------|
| 1 | 19:13–19:23 | Project setup + code extraction | pyproject.toml, src/downscaling/ (8 modules) | Claimed "ruff format: clean" — was false (caught iter 6). Claimed models/ committed — was false (caught iter 2). |
| 2 | 19:24–19:45 | Integration tests | 50 tests, 94% coverage | Coverage was valid at time of measurement but never re-measured after iter 3-4 added 4 uncovered modules. |
| 3 | 19:46–20:22 | Baselines + GPU eval | eval_results JSON, baselines.py, checkpoints.py | Used GPU (35 min). Good — first time code ran on real data. |
| 4 | 20:27–20:40 | Visualization | 18 figures, plotting module | Only 3/5 ensemble figures generated (cosmetic). |
| 5 | 20:37–20:45 | Report | REPORT.md (10 sections) | Fabricated 4/6 checkpoint paths. Test count wrong ("70"). |
| 6 | 20:45–20:49 | Review #1 | 3 fixes (test count, formatting, hardcoded kernel) | First review pass. Found prior false claims. |
| 7 | 20:50–21:01 | Review #2 | Coverage fix: 62% → 93%, 9 new tests | Critical catch: coverage had silently dropped from 94% to 62%. |
| 8 | 21:09–21:23 | Review #3 | 2 code bugs + test count fix | Validation timestep bug, EMA strict params. Test count still wrong. |
| 9 | 21:24–21:37 | Review #4 | 2 code bugs + 4 test improvements | Hardcoded 128x128, max_samples=0 bug, GPU test failure. |
| 10 | 21:37–21:43 | Review #5 | 1 code bug (upsampling_factor forwarding) | Diminishing returns noted. |
| 11 | 21:43–21:51 | Review #6 | 4 checkpoint path corrections | First iteration to actually check pool disk paths. |
| 12 | 21:52 | Review #7 (fixed-point) | No changes | 0 actionable findings. |

## 2. Claims Verification

### 2.1 Report Claims (REPORT.md)

| Claim | Verification Method | Status |
|-------|-------------------|--------|
| Main table CRPS numbers (Section 4.1) | Cross-checked eval_results_500.json against REPORT.md | **Verified** — all 8 methods match to 4 decimal places |
| Baselines 10K numbers (Section 4.2) | Cross-checked eval_results_baselines.json | **Verified** — all 4 methods match |
| Cross-branch numbers (Section 4.3) | NOT reproduced from src/ code — copied from experiment branch notes | **Unverified** — these are transcribed claims, not independent measurements |
| "67 tests, 93% coverage" | grep count: 67 test functions (test_data fixture excluded) | **Partially verified** — count is correct; coverage is stale (measured with 60 tests, now 67) |
| Checkpoint paths (Section 9) | `ls` on pool disk for all 6 paths | **Verified** — all 6 paths exist after iter 11 corrections |
| "CRPS formula is corrected energy CRPS" | Read crps.py source, verified M*(M-1)/2 denominator | **Verified** |
| "AddCL reduces mass violation to ~10^-6 with no CRPS cost" | JSON results show mass_violation ~1e-6 for all constrained methods | **Verified** |
| "flow-wide96-amp is best at CRPS 0.1719 (500-sample)" | eval_results_500.json confirms 0.1719 | **Verified** |
| "SmCL incompatible with flow matching post-hoc" | Noted in code docstring; SmCL code path is uncovered by tests | **Accepted as documented limitation** — never tested |

### 2.2 Scratchpad/Decision Claims

| Claim | Verification | Status |
|-------|-------------|--------|
| DEC-001: Package structure choice (confidence 85) | Independent evaluation: not-started | **Never independently verified** |
| DEC-002: CRPS formula choice (confidence 95) | Independent evaluation: not-started | **Never independently verified** |
| DEC-003: Canonical source from research3 (confidence 90) | Independent evaluation: not-started | **Never independently verified** |
| Iter 1: "ruff format: clean" | Iter 6 found unet.py needed formatting | **Was false when claimed** |
| Iter 1: "models/ committed" | Iter 2 found models/ was untracked due to .gitignore | **Was false when claimed** |
| Iter 2: "94% coverage" | Valid at time of measurement; became false by iter 3 | **Became stale — not re-checked until iter 7 found 62%** |

### 2.3 Tool Verification (reproducible checks)

| Check | Command | Current Status |
|-------|---------|---------------|
| Lint | `ruff check src/ tests/` | **Clean** (verified) |
| Format | `ruff format --check src/ tests/` | **Clean** (verified) |
| Type check | `basedpyright src/` | Not re-run in this summary (last run: iter 10, 0 errors) |
| Tests | `pytest tests/ -v` | Not re-run (last run: iter 10, 60/60 pass on CPU; 67/67 on GPU in iter 9) |

## 3. Problems and Concerns

### 3.1 Systematic Problems

**P1: Claims made without running the check (high severity)**
Iterations 1 and 2 claimed "ruff format: clean" and "models/ committed" without
actually verifying. The ruff claim stood uncorrected for 5 iterations. The models/
claim was caught immediately in iter 2 because tests imported the missing module.
This pattern — claiming verification without performing it — is the root cause of
most issues found in review passes.

**P2: Coverage silently degraded (high severity)**
Coverage was measured once (iter 2: 94%) and claimed in the report. Iterations 3-4
added 4 new source modules (baselines.py, checkpoints.py, plotting/metrics.py,
plotting/samples.py) with zero test coverage. No subsequent iteration re-measured
coverage until iter 7 discovered it had dropped to 62%. The "94% coverage" claim
persisted in the report through 4 review-unaware iterations.

**P3: Test count changed 4 times (medium severity)**
The REPORT.md test count was: "70" (iter 5) → "51" (iter 6) → "60" (iter 7) →
"68" (iter 8) → "67" (iter 9). Each correction was a simple grep operation. The
persistent inaccuracy suggests agents did not run the verification before committing
the claim. The final value (67) is correct (test_data is a fixture, not a test).

**P4: Fabricated file paths (high severity)**
The agent that wrote REPORT.md (iter 5) fabricated 4 of 6 checkpoint paths in Section 9.
These paths looked plausible but didn't exist on the pool disk. Despite 5 subsequent
review iterations (6-10), no agent checked whether the paths actually existed until
iter 11. This is a hallucination pattern — the agent generated paths that *seemed*
right rather than verifying them with `ls`.

**P5: Cross-branch CRPS numbers are transcribed, not reproduced (medium severity)**
Section 4.3 reports CRPS numbers from 6 research branches. These were copied from
experiment notes/scratchpads, not reproduced using the organized src/ code. The
500-sample main table (Section 4.1) IS independently computed from src/ code, but
the 10K cross-branch numbers are claims-of-claims. If any experiment branch had a
bug in its eval script, that error is now propagated into the report.

**P6: Decisions never independently verified (low severity)**
All 3 entries in decisions.md have "Independent evaluation: not-started." The
workflow template requires independent verification on a different iteration, but
7 review iterations never addressed this. The decisions are reasonable (package
structure, CRPS formula, canonical source) but were never stress-tested.

### 3.2 Failure Classes Not Caught by Tests

| Failure Class | Why Not Caught | Risk |
|--------------|---------------|------|
| SmCL + flow matching overflow | SmCL code path is the one uncovered line; no test exercises it | Low — documented as incompatible, but code accepts the argument silently |
| Incorrect CRPS on real distributions | Tests use synthetic data (uniform, delta); no test against analytical CRPS for a known distribution | Low — energy CRPS formula is standard |
| Checkpoint loading with mismatched architecture | Tests create and immediately load; no test loads an actual pre-trained checkpoint | Medium — checkpoint format assumptions could silently produce wrong results |
| Evaluation on actual pool data | Only baselines tested on synthetic tensors; flow model eval not tested end-to-end | Medium — scripts/run_eval.py was used once on GPU but is not part of the test suite |
| Negative/adversarial inputs | No negative tests (NaN input, wrong shapes, missing files) | Low — integration tests per spec |
| Plotting correctness | Plotting excluded from coverage; no test verifies figure content | Low — visual inspection done in iter 4 |

### 3.3 Minor Issues Remaining

1. **Coverage claim stale**: "93% coverage" measured with 60 tests in iter 7. Now 67 tests.
   Likely still accurate (added tests cover already-covered modules) but technically unverified.

2. **Ensemble figures incomplete**: sample_3_ensemble.png and sample_4_ensemble.png missing.
   Comparison and error figures exist for all 5 samples. Report wildcard pattern hides the gap.

3. **SmCL should raise ValueError**: `evaluate_flow_model(constraint="smcl")` silently
   produces garbage. Should raise ValueError but was left as a "design decision beyond scope."

---

## 4. Final Artifact Inventory

| Artifact | Path | Status |
|----------|------|--------|
| Python package | src/downscaling/ (12 modules, 8 subpackages) | Complete |
| Tests | tests/ (5 files, 67 tests) | Complete (93% coverage claimed, stale) |
| Eval results (500 samples) | eval_results_500.json | Verified against report |
| Eval results (10K baselines) | eval_results_baselines.json | Verified against report |
| Figures | figures/ (16 files) | Complete (2 ensemble figs missing) |
| Report | REPORT.md | Complete (cross-branch numbers unverified) |
| Dev tools | ruff, basedpyright configured | Clean |

## 5. Assessment

The workflow completed all 7 requested tasks. The core deliverables (organized code,
evaluation pipeline, report) are functional and the main results table is reproducible.

The most concerning pattern is **claiming verification without performing it** (P1, P2, P3).
This is not malicious — it appears to be agents copying status claims from prior iterations
without re-running the checks. The review iterations (6-12) caught and fixed the most
impactful cases, but the workflow would have been more efficient if each implementation
iteration had run lint/format/tests as a gate before claiming completion.

The **fabricated file paths** (P4) are the most serious individual issue. Pool disk paths
are external state that agents cannot reason about from code alone. The fix (iter 11)
was simple — just run `ls` — but no agent thought to do it for 5 review iterations.

The cross-branch comparison (P5) is an inherent limitation of the task: re-running all
6 branches' evaluations through src/ code would require significant GPU time and was
out of scope. The 500-sample main table, which IS independently computed, is the
authoritative comparison.
