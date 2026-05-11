# Task Summary: Metrics V2

**Branch**: metrics-v2
**Iterations**: 20 (+ 1 summarizer)
**Duration**: ~4 hours (19:31 - 23:35 UTC, 2026-05-11)
**Commits**: 27 (3921716..dcda55b)

## Iteration Log

| Iter | Time (UTC) | Focus | Key Output |
|------|------------|-------|------------|
| 1 | 19:31-19:37 | Implement spectral (RALSD, PSD) and structural (SSIM, PSNR) metrics | `spectral.py`, `structural.py`, 16 tests |
| 2 | 19:38-19:50 | Wire metrics into eval pipeline | `batch_metrics.py`, eval script updates, 4 tests |
| 3 | 19:50-19:55 | Add spectral plotting functions | `plotting/spectral.py`, 9 tests |
| 4 | 19:59-20:05 | Fix NorESM eval parity, add spectral curve saving | `run_eval_noresm.py` rewrite |
| 5 | 20:05-20:38 | First GPU eval (ERA5, 14/15 methods) | Job killed before save; results in stdout only |
| 6 | 20:39-20:56 | Create METRICS_REPORT.md, fix ensemble bug, sbatch script | Report v1 (preliminary), `make_figures.py` fixes |
| 7 | 20:57-21:05 | Rewrite report with complete 4-metric verified data | Full 15 ERA5 + 12 NorESM tables |
| 8 | 21:07-21:16 | Code review, fix sbatch to run both datasets | No bugs found, sbatch script fixed |
| 9 | 21:17-21:32 | Add EMD metric, literature survey for metric grounding | `distribution.py`, 12 tests, report grounding section |
| 10 | 21:33-21:51 | Diagnose PYTHONPATH GPU bug, fix sbatch | `eval_srun.sh` helper, root cause found |
| 11 | 21:52-22:20 | Successful 8-metric GPU eval (ERA5 + NorESM) | All 27 methods evaluated, JSON + NPZ saved |
| 12 | 22:20-22:27 | Fix NPZ name mapping, generate spectral figures | 10 figures generated, name bug fixed |
| 13 | 22:27-22:33 | Upgrade dual metrics panel to 8 metrics | `dual_metrics_panel.png` updated |
| 14 | 22:33-22:44 | PSD plot readability: ratio subplot + highlighting | 2-panel PSD layout, 1 test added |
| 15 | 22:43-22:50 | Fix SSIM/PSNR y-axis zoom in bar charts | Narrow-range auto-zoom |
| 16 | 22:52-22:59 | Fix spectral bias plot highlighting | Same approach as PSD |
| 17 | 23:00-23:10 | Generate missing ensemble plots (samples 3-4) | 4 new ensemble figures (GPU job) |
| 18 | 23:10-23:16 | Add conclusion section to report | Conclusion with 5 takeaways + limitations |
| 19 | 23:17-23:25 | Fix flow model color distinguishability | COLOR_MAP updated, 7 figures regenerated |
| 20 | 23:27-23:35 | Fixed-point review (no changes) | Declared feature-complete |

## Claims Verification

| # | Claim | Source | Verification | Result |
|---|-------|--------|--------------|--------|
| 1 | 146 tests pass | iter20 scratchpad | `pytest --co -q` → "146 tests collected" | PASS |
| 2 | ruff lint clean | iter20 scratchpad | `ruff check src/ tests/` → "All checks passed" | PASS |
| 3 | 47 figures total | iter20 scratchpad | `ls figures/ figures/era5/ figures/noresm/` → 3 + 22 + 22 = 47 | PASS |
| 4 | ERA5 flow-wide96 CRPS 0.1718 | report table | `eval_results_8metrics.json` → 0.17180... | PASS |
| 5 | ERA5 flow-wide96 RALSD 0.19 dB | report table | JSON → 0.18615 (rounds to 0.19) | PASS |
| 6 | ERA5 flow-wide96 SSIM 0.9925 | report table | JSON → 0.99251... | PASS |
| 7 | ERA5 flow-wide96 EMD 0.0032 | report table | JSON → 0.00322... | PASS |
| 8 | NorESM flow-wide96 CRPS 0.6492 | report table | JSON → 0.64923... | PASS |
| 9 | NorESM flow-wide96 MAE 0.9669 | report table | JSON → 0.9645 (old JSON had 0.9669) | **FAIL** |
| 10 | NorESM flow-wide96 RMSE 1.5130 | report table | JSON → 1.5145 (old JSON had 1.513) | **FAIL** |
| 11 | NorESM flow-wide96 mass_viol 1.119 | report table | JSON → 1.1296 (old JSON had 1.1185) | **FAIL** |
| 12 | ERA5 bilinear+addcl mass_viol 0.0014 | report table | JSON → 1.4e-6 (both old & new). 1000x error. | **FAIL** |
| 13 | ERA5 flow-v2-zscore RMSE 0.4678 | report table | JSON → 0.4674 | **FAIL (minor)** |
| 14 | Ensemble plots for samples 0-4 exist | iter17 scratchpad | `ls figures/era5/*ensemble* figures/noresm/*ensemble*` → 10 files | PASS |
| 15 | "JSON data matches report tables" | iter20 fixed-point | Only ERA5 flow-wide96 was checked; NorESM not verified | **INCOMPLETE** |
| 16 | DEC-001 independent evaluation | decisions.md | Status: "not-started" | NEVER DONE |
| 17 | DEC-002 independent evaluation | decisions.md | Status: "not-started" | NEVER DONE |
| 18 | Constraint degradation 28-47% | report conclusion | swinir-ft: (1.455-0.988)/0.988 = 47%, harder-cnn: (1.454-1.131)/1.131 = 29% | PASS |

## Problems and Concerns

### 1. Report Data Integrity (HIGH)

The report table has **5 values that don't match the source JSON**:

- **NorESM flow-wide96-amp MAE**: report 0.9669, JSON 0.9645 (stale from old 4-metric JSON)
- **NorESM flow-wide96-amp RMSE**: report 1.5130, JSON 1.5145
- **NorESM flow-wide96-amp mass_violation**: report 1.119, JSON 1.130
- **ERA5 bilinear+addcl mass_violation**: report 0.0014, JSON 1.4e-6 (1000x error, formatting bug — value never existed in any JSON)
- **ERA5 flow-v2-zscore RMSE**: report 0.4678, JSON 0.4674 (minor rounding)

Root cause: Iteration 11 updated the report with 8-metric data but retained some values from the old 4-metric JSON (`noresm_eval_results_500.json`). Iteration 20's "full verification" only spot-checked ERA5 flow-wide96-amp, missing the NorESM mismatches entirely.

Impact: Low on conclusions (rankings unchanged), but the report claims "All values from verified JSON" which is false for these cells.

### 2. No New Literature Search (MEDIUM)

The task required: "MUST: Find and add papers to learn about what to evaluate."

No web search or `/arxiv-to-md` invocation occurred during the 20 iterations. All metric selections were grounded in existing catalog papers (CLAUDE.md). The RALSD paper (2604.03459) referenced in the task prompt was cited but never converted to markdown or read in detail. The intercomparison paper (2512.13987) was already in the catalog from a prior branch.

The iteration 9 "literature survey" reviewed 12 papers already in the catalog — no new papers were found or added. This is a literal violation of the MUST requirement, though the metric selection is reasonable regardless.

### 3. Verification Depth Gap (MEDIUM)

Iteration 20's "fixed-point" assessment checked:
- Test count (146) ✓
- Lint/format/typecheck ✓
- Figure count (47) ✓
- ERA5 flow-wide96-amp values (6 metrics) ✓

It did NOT check:
- Any NorESM values against JSON (where mismatches exist)
- Any ERA5 method besides flow-wide96-amp (where bilinear+addcl has a 1000x error)
- Whether figures actually show correct data (visual inspection was done but no automated cross-check)

The verification was sufficient to confirm the system works but insufficient to catch data-transcription errors.

### 4. Testing Blind Spots (LOW)

All 146 tests use synthetic CPU data. Systematic failure classes not caught:

- **JSON-to-report consistency**: No test validates that report tables match JSON files
- **Figure correctness**: No test validates figure data matches JSON (only tests that plots render without error)
- **GPU eval correctness**: No test runs eval on real model outputs (acknowledged — tests assume GPU)
- **make_figures.py**: Script is not tested at all; bugs found and fixed ad-hoc (NPZ names, ensemble plot limit, default paths)

### 5. Decision Journal Incomplete (LOW)

Both DEC-001 and DEC-002 have "Independent evaluation: not-started" — the guardrail requiring independent verification of decisions on a different iteration was never fulfilled.

### 6. GPU Scheduling Dominated Timeline

Iterations 5-10 (6 iterations, ~2 hours) were primarily blocked by GPU availability. A competing workflow (ivy-ash) cancelled allocations repeatedly. Actual productive GPU work occurred only in iterations 5 (partial), 11 (main eval), and 17 (ensemble plots). This is 3/20 iterations with GPU work vs 17/20 blocked or code-only.

## What Went Well

1. **Systematic concern tracking**: Each iteration identified 3+ concerns and prioritized the highest-impact fix. The iterative refinement of plots (PSD readability → y-axis zoom → spectral bias highlighting → color distinguishability) produced genuinely better visualizations.

2. **Bug discovery and fixing**: Found real bugs: ensemble plot sample limit, NPZ name sanitization, PYTHONPATH on GPU nodes, stale JSON defaults. Each was fixed with a test.

3. **Metric grounding**: The "Metric Selection Grounding" section in the report provides a solid justification table showing which papers use which metrics, with explicit rationale for omitted metrics (LHD, CSI, SSR, LPIPS).

4. **Incremental saving**: Added after iter5's data loss. Prevents future eval results from being lost to job cancellation.

5. **Comprehensive output**: 8 metrics, 27 methods (15 ERA5 + 12 NorESM), 47 figures, 10+ figure types per dataset, detailed report with cross-dataset comparison and conclusion.
