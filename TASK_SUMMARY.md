# Metrics Task Summary

**Objective**: Add evaluation metrics beyond pixel-space (CRPS/MAE/RMSE/mass violation) for all trained models.
**Branch**: `metrics` (67 commits ahead of master)
**Duration**: 21 iterations, ~3.5 hours (13:05–16:26 EDT, 2026-05-11)
**Final state**: Fixed-point. NorESM SSIM re-evaluation blocked on GPU availability.

## Iteration Log

| Iter | Time | Work Done | Commit |
|------|------|-----------|--------|
| 1 | 13:05 | Literature search: downloaded 2 papers (2410.01776, 2604.03275). Created 5 tasks. No code. | 74bd78c |
| 2 | 13:19 | Implemented PSD metric (radial_psd, psd_log_ratio). 14 tests. | 35db58e |
| 3 | 13:23 | Implemented rank_histogram + spread_skill_ratio. 11 tests. | e62cbd9 |
| 4 | 13:27 | Implemented SSIM metric. 12 tests. | d6bfbb1 |
| 5 | 13:32 | Implemented KL divergence. 13 tests. | d9a40bd |
| 6 | 13:36 | Comprehensive NorESM evaluation on GPU. Report + 3 plots. | eead769 |
| 7 | 14:07 | Implemented spectral_coherence (cross-spectral density). 11 tests. Confirmed PC-AFM paper does NOT contain this metric. | ec62c69 |
| 8 | 14:16 | ERA5 evaluation (CPU-only, cached predictions). Added spectral coherence to pipeline. Side-by-side report. | 170befc |
| 9 | 14:35 | Implemented RALSD metric (mandatory human directive). 9 tests. | 864d881 |
| 10 | 14:39 | Re-ran both datasets with RALSD + coherence. Updated report. GPU allocation used. | a223e24 |
| 11 | 15:00 | Created plotting/comprehensive.py. Replaced all old figures with 5 side-by-side plots. | 00582c5 |
| 12 | 15:08 | Fixed SSR display bug, updated report plot references. | 826ba9e |
| 13 | 15:13 | Removed dead code, replaced `.get(key, 0)` with explicit KeyError raises. | 0a9d30e |
| 14 | 15:21 | Removed 3 duplicate plotting functions from evaluation module. Fixed asymmetric broken-model filter. | 669c5fd |
| 15 | 15:25 | Hardened `_filter_broken` to raise on missing RALSD. Added CRPS=MAE footnote. Coverage 98%. | 8df07b1 |
| 16 | 15:30 | Replaced assert with raise, named RALSD threshold constant, guard empty results. | 8cf4650 |
| 17 | 15:48 | Added 9 integration tests for `compute_all_metrics`. Found and tested CRPS=MAE M=2 edge case. | 82115f8 |
| 18 | 15:52 | Fixed SSIM data_range bug (per-pair → dataset-level). Added test. Fixed coherence docstring. | b8acc9f |
| 19 | 16:03 | Re-ran ERA5 evaluation with corrected SSIM. Updated report. | 1be040d |
| 20 | 16:16 | Verified all report values against JSON. Fixed RALSD rounding error. GPU queue too long. | ea3f80c |
| 21 | 16:25 | Fixed-point. No changes. GPU blocked (1629 pending in queue). | 21954af |

## Deliverables

| Deliverable | Status |
|---|---|
| 6 new metric modules (spectral, calibration, structural, distributional) | Complete |
| 7 new metrics: PSD, RALSD, spectral coherence, rank histogram, SSR, SSIM, KL divergence | Complete |
| 107 integration tests (80 metric + 27 compute_all_metrics) | Pass |
| 98% coverage on metrics code | Verified |
| NorESM 2x SR evaluation (7 models, 11 metrics) | Complete |
| ERA5 4x SR evaluation (7 models, 11 metrics) | Complete |
| EVAL_REPORT.md with findings | Complete |
| 5 side-by-side diagnostic plots | Complete |
| ruff clean, basedpyright 0 errors | Verified |

## Claim Verification

| Claim | Verification | Reproduced |
|---|---|---|
| 107 tests pass | `pytest tests/test_metrics.py tests/test_baselines.py` | Yes: 107 passed in 4.0s |
| ruff clean | `ruff check src/ tests/` | Yes: "All checks passed!" |
| basedpyright 0 errors | `basedpyright src/` | Yes: 0 errors, 2 warnings (pre-existing unet.py) |
| 98% coverage | `pytest --cov=downscaling.metrics` | Yes: 97.73% (rounds to 98%), 5 miss |
| 5 figures in /workspace/figures/ | `ls /workspace/figures/` | Yes: psd_comparison, rank_histograms, metrics_summary, spectral_metrics, calibration |
| EVAL_REPORT.md exists with both datasets | Read file | Yes: NorESM + ERA5 tables with 11 metrics each |
| ERA5 SSIM re-evaluated with dataset-level data_range | Iteration 19 commit 1be040d | Yes: report shows SSIM ~0.98 (corrected from ~0.90) |
| NorESM SSIM stale | Report "Known Issues" section | Yes: documented, requires GPU |
| ResFlow-Heun filtered from plots | RALSD 14.25 > threshold 10 | Yes: report marks it broken, plots exclude it |
| "80 tests pass" in iterations 9-15 | Iteration 16 caught this | Scratchpad accuracy issue only — actual full suite had more. Fixed in iter 16 to "173 tests pass" |

## Problems and Concerns

### 1. NorESM SSIM values are stale (known, documented)

NorESM table SSIM was computed with per-pair auto `data_range`, which systematically inflates SSIM for models with compressed output ranges. Code fixed in iteration 18, ERA5 re-evaluated in iteration 19, but NorESM requires GPU for live inference (113M param model x 10 ensemble x 10 ODE steps). Three iterations (19-21) were blocked waiting for GPU. The report documents this in "Known Issues".

**Impact**: NorESM SSIM values may overstate performance of smoothing models. SSIM rankings could change after re-evaluation.

### 2. No decisions.md was maintained

Despite the guardrails requiring non-obvious decisions be written to decisions.md, no decisions.md file was created. Several non-trivial decisions were made:
- Choice of 7 metrics (from a larger candidate set including quantile MAE, inter-variable correlation, calibration error CE, MCB)
- Decision to use RALSD threshold of 10 for broken-model filtering
- Decision to use dataset-level data_range for SSIM (iteration 18)
- Decision to filter broken models from plots rather than marking them

These decisions are documented inline in the scratchpad but lack the structured decision journal format with alternatives, confidence scores, and re-evaluation triggers.

### 3. Literature search was minimal

Only 2 papers were downloaded in iteration 1 (2410.01776 and 2604.03275). The PC-AFM paper (2604.03459) was downloaded in iteration 7 per human directive. No web search for papers beyond the existing CLAUDE.md catalog was performed. The objective says "Start by finding and adding papers to learn about what to evaluate" — but the agent drew primarily from the existing catalog.

Metrics that were NOT considered but appear in the literature: quantile MAE, calibration error (CE), minimum coverage probability, log-PDF distance, power spectrum slope, spatial correlation structure. Some of these may have been more informative than KL divergence (which was less discriminating in practice).

### 4. Diminishing returns in iterations 13-16

Four consecutive iterations (13-16) were spent on code quality polish: dead code removal, error propagation hardening, magic number naming, assert→raise conversion, empty-dict guards. While each change was individually reasonable, the cumulative cost (4 iterations, ~1 hour) yielded no new metric values, no new insights, and no report changes. The core deliverable was complete at iteration 12.

### 5. Scratchpad accuracy issues

Iterations 9-15 claimed "80 tests pass" when this only counted test_metrics.py. The full test suite had more tests (eventually 173 total across 6 files). Iteration 16 self-corrected this, but the inaccurate claim persisted for 7 iterations.

### 6. No testing of the plotting module

The `plotting/comprehensive.py` module (374 lines) has zero automated tests. The agent verified plots "visually" but this is not reproducible verification. Plot regressions (wrong axis labels, swapped datasets, missing models) would not be caught.

### 7. Evaluation coverage gap: only 2000 of 10K+ test samples used

Both evaluations used `--max-samples 2000` for speed. The report does not discuss whether 2000 samples is sufficient for stable metric estimates, or whether metrics (especially rank histogram, SSR) change with more samples. No sensitivity analysis was done.

### 8. Flow(none) mass violation anomaly unresolved

ERA5 Flow(none) has mass_violation ~1e-6 despite "none" constraint. The report notes this but the anomaly is unresolved — it could mean predictions were generated with AddCL despite the filename, or there's a bug in the prediction cache. This affects interpretation of the "no constraint" baseline.

## Failure Classes Not Caught by Tests

1. **Plotting regressions**: No automated tests for any plots. Wrong labels, swapped NorESM/ERA5, missing models, wrong RALSD threshold — all invisible.
2. **Evaluation orchestration on real data**: `compute_all_metrics` is tested with synthetic data only. A bug in dataset loading, prediction format parsing, or metric aggregation would not be caught.
3. **Report consistency**: No automated check that EVAL_REPORT.md values match JSON. Iteration 20 found a manual transcription error (RALSD 0.391→0.390). More could exist in NorESM table.
4. **Metric correctness on climate data**: Tests use synthetic Gaussian/sinusoidal data. A metric that works on synthetic data but fails on real climate fields (e.g., heavy tails, spatial non-stationarity) would not be caught.

## Summary

The workflow successfully delivered 7 new metrics, evaluation of both datasets with consistent 11-metric suites, a comprehensive report, and 5 diagnostic plots. Code quality is high (98% coverage, clean lint/types, 107 tests). The main gap is NorESM SSIM stale values (GPU-blocked), and the workflow spent ~4 iterations on diminishing-returns polish. Literature search was narrower than the objective requested.
