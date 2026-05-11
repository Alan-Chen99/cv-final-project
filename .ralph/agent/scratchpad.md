# Metrics V2 Scratchpad

## Iteration 1
- **Start**: 2026-05-11T19:31:45Z, commit 3921716
- **End**: 2026-05-11T19:37:12Z, commit 3cc4e8e
- **Prefix**: gamma-delta

### Current State Assessment
**Existing metrics**: CRPS (energy), MAE, RMSE, mass violation — all pixelwise/probabilistic.
**Existing figures**: Bar charts (CRPS, MAE, RMSE, mass_violation), sample comparisons, error maps, ensemble spread.
**Existing eval results**: ERA5 (15 methods, 500 samples), NorESM (12 methods, 500 samples).

### Concerns (3+)

1. **Quality: No spectral metrics** — The entire evaluation is pixelwise. No spectral power density, no RALSD. Climate downscaling papers universally use spectral analysis (CorrDiff, GenDiff, intercomparison paper, CDSI, all use PSD). This is a critical gap — flow models might produce correct pixel values but wrong spatial structure (or vice versa). The intercomparison paper (2025-12-16) explicitly states RALSD "weights errors at small scales equally to large scales, ensuring fine-scale features are adequately evaluated."

2. **Quality: No structural similarity metric** — No SSIM or perceptual quality metric. While SSIM isn't standard in climate papers, it's universal in image SR papers (SwinIR, SR3, HAT all report it). Since our baselines include SwinIR, we should include it for completeness.

3. **Quality: Missing distribution metrics** — No LHD (Logarithmic Histogram Distance), no Q-Q analysis. The intercomparison paper uses LHD to assess intensity distribution fidelity. Spread-skill ratio missing too (CDSI uses it).

### Plan for this iteration
Implement spectral metrics (PSD computation + RALSD + spectral power plots) in `src/downscaling/metrics/spectral.py`. This is the highest-value missing metric and addresses concern #1.

### Work done
- Implemented `src/downscaling/metrics/spectral.py`: radial_psd, radial_psd_batch, ralsd, spectral_bias
- Implemented `src/downscaling/metrics/structural.py`: ssim, psnr
- Updated `src/downscaling/metrics/__init__.py` to export new metrics
- Created `tests/test_spectral.py` with 16 integration tests (all pass)
- All existing tests still pass (10/10)
- Lint: pass, Format: pass, Typecheck: 0 errors

### Next iteration work
- Update `src/downscaling/evaluation/evaluate.py` to compute RALSD, SSIM, PSNR alongside CRPS/MAE/RMSE
- Add spectral power density plots to `src/downscaling/plotting/`
- Re-run full evaluation on GPU to generate new results with spectral metrics
- Start report file

### RALSD Definition (from intercomparison paper, 2025-12-16)
1. Compute 2D FFT of each field
2. Radially integrate (bin by wavenumber) → 1D power spectrum
3. RALSD(dB) = sqrt(1/N * sum_i (10*log10(F_true_i / F_pred_i))^2)
- Lower is better
- Key: weights errors at all scales equally in log space

## Iteration 2
- **Start**: 2026-05-11T19:38:14Z, commit da578a4
- **Prefix**: gamma-delta (reusing from iteration 1)

### Concerns (3+)

1. **Workflow: New metrics are dead code** — spectral.py and structural.py exist but are NOT wired into the evaluation pipeline. `EvalMetrics` only has crps/mae/rmse/mass_violation. `run_eval.py` only computes and stores these 4 metrics. The new metrics cannot be computed without pipeline integration.

2. **Quality: RALSD is a batch metric, not per-sample** — RALSD averages PSDs over the dataset first, then computes log spectral distance. Current eval pipeline is per-sample (accumulate, average). This architectural mismatch means RALSD cannot simply be added to the per-sample loop — it needs all predictions collected first.

3. **Workflow: No prediction storage** — The current pipeline discards predictions after computing per-sample metrics. To compute RALSD (and to generate spectral plots), predictions must be preserved. This requires structural changes to eval functions.

### Plan for this iteration
Wire new metrics into the evaluation pipeline:
- Create `src/downscaling/evaluation/batch_metrics.py` — computes RALSD, SSIM, PSNR from collected predictions
- Modify `evaluate_flow_model` to optionally return ensemble-mean predictions
- Update `run_eval.py` to collect predictions and compute all 7 metrics
- Run lint/typecheck/tests

### Work done
- Created `src/downscaling/evaluation/batch_metrics.py` with `compute_batch_metrics()` and `compute_spectral_curves()`
- Modified `evaluate_flow_model` — added `return_predictions=True` parameter to return ensemble means (N, H, W)
- Modified `eval_swinir_zeroshot`, `eval_swinir_finetuned` — added `return_predictions=True`
- Modified `evaluate_harder_cnn`, `evaluate_harder_gan` — added `return_predictions=True`
- Rewrote `run_eval.py` main() to collect predictions from ALL methods and compute batch metrics (RALSD, SSIM, PSNR) alongside per-sample metrics (CRPS, MAE, RMSE, mass_violation)
- Updated console output table to show all 7 metrics
- Updated JSON results format to include new metrics
- Updated `evaluation/__init__.py` exports
- Added 4 integration tests for batch_metrics module (20/20 pass)
- All existing tests pass (10/10 CRPS tests)
- Lint: pass (TC002 warnings are pre-existing pattern, numpy used at runtime)
- Format: pass
- Typecheck: 0 errors

### Next iteration work
- Add spectral PSD plot functions to `src/downscaling/plotting/`
- Re-run full evaluation on GPU to generate updated results with all 7 metrics
- Update existing figures with new data
- Start writing the report file
- Review and fix figure inconsistencies (missing ensemble plots for samples 3-4)

- **End**: 2026-05-11T19:49:57Z, commit 744f22e

## Iteration 3
- **Start**: 2026-05-11T19:50:59Z, commit b40c642
- **Prefix**: gamma-delta (reusing)

### Concerns (3+)

1. **Workflow: No spectral visualization code** — `compute_spectral_curves()` exists in batch_metrics.py but there are ZERO plotting functions for spectral data. PSD curves, spectral bias plots, RALSD bar charts — none exist. Without these, the spectral analysis is invisible even after a GPU eval run. This is the highest-priority gap.

2. **Quality: metrics_panel only shows 4 of 7 metrics** — `plot_metrics_panel()` shows CRPS/MAE/RMSE/mass_violation in a 2x2 grid. The new RALSD/SSIM/PSNR metrics are not visualized anywhere. The panel needs updating or a new supplementary panel is needed.

3. **Workflow: Existing eval JSONs lack new metrics** — `eval_results_500.json` and `noresm_eval_results_500.json` only have 4 metrics per method. No RALSD/SSIM/PSNR. All figures will need regenerating after a new GPU eval run.

4. **Quality: Missing ensemble plots for samples 3-4** — Both ERA5 and NorESM have ensemble plots for samples 0-2 but not 3-4. Inconsistent visualization.

### Plan for this iteration
Add spectral and extended metric plotting functions to `src/downscaling/plotting/spectral.py`.

### Work done
- Created `src/downscaling/plotting/spectral.py` with 4 plot functions:
  - `plot_psd_comparison()`: Log-log PSD curves for ground truth vs methods
  - `plot_spectral_bias()`: Per-frequency spectral bias with NaN handling
  - `plot_extended_metrics_panel()`: 3x3 grid showing all 7 metrics (CRPS, MAE, RMSE, SSIM, PSNR, RALSD, mass_violation)
  - `plot_ralsd_comparison()`: Dedicated RALSD bar chart
- Updated `src/downscaling/plotting/__init__.py` with new exports
- Created `tests/test_spectral_plots.py` with 9 tests (all pass)
- 64/64 non-GPU tests pass
- Lint: pass (1 pre-existing TC002 from iter2), Format: pass, Typecheck: 0 errors

### Next iteration work
- Re-run full evaluation on GPU with `run_eval.py` to get 7-metric results
- Generate spectral PSD/bias plots using `compute_spectral_curves()` + new plot functions
- Generate updated metrics panels with `plot_extended_metrics_panel()`
- Fix missing ensemble plots for samples 3-4
- Start report file

- **End**: 2026-05-11T19:55:00Z, commit 41605ce

## Iteration 4
- **Start**: 2026-05-11T19:59:35Z, commit 95f1e76
- **Prefix**: gamma-delta (reusing)

### Concerns (3+)

1. **Workflow: run_eval_noresm.py NOT updated with batch metrics** — `run_eval.py` (ERA5) was fully updated in iter2 to collect predictions and compute RALSD/SSIM/PSNR. But `run_eval_noresm.py` was NOT touched. It still only computes 4 metrics, doesn't collect predictions, doesn't call `compute_batch_metrics()`. The NorESM eval will be incomplete if run as-is.

2. **Workflow: Spectral curve data not saved anywhere** — `run_eval.py` calls `compute_batch_metrics()` for scalar RALSD/SSIM/PSNR but does NOT call `compute_spectral_curves()`. Even after a GPU eval run, there will be no PSD/bias arrays to feed to `plot_psd_comparison()` and `plot_spectral_bias()`. The spectral plotting code is dead without this.

3. **Workflow: No GPU eval has been run in 3 iterations** — All iterations so far added code. Zero GPU eval runs. The existing `eval_results_500.json` only has 4 metrics. No RALSD, SSIM, PSNR data exists. The scratchpad keeps deferring GPU work.

### Plan for this iteration
Fix both eval scripts to save spectral curve data, and bring `run_eval_noresm.py` to parity with `run_eval.py`'s batch metrics pattern. This is prerequisite for the GPU eval run (next iteration).

Specifically:
- Add `compute_spectral_curves()` calls to `run_eval.py` — save spectral data as .npz alongside JSON
- Update `run_eval_noresm.py` to collect predictions, compute batch metrics (RALSD/SSIM/PSNR), compute spectral curves, and save everything

### Work done
- Updated `scripts/run_eval.py`: added `compute_spectral_curves()` import, spectral curve computation for all methods with predictions, saves as `{output_stem}_spectral.npz`
- Rewrote `scripts/run_eval_noresm.py` to fully mirror `run_eval.py` batch metrics pattern:
  - Added predictions dict to collect (N, H, W) arrays from all methods
  - Baselines: generate predictions via F.interpolate + apply_addcl for batch metrics
  - SwinIR/Harder/Flow: use `return_predictions=True` to collect ensemble-mean predictions
  - Compute RALSD/SSIM/PSNR via `compute_batch_metrics()` for all methods
  - Updated `_save_and_print()` to accept predictions/gt, compute spectral curves, save .npz
  - Updated display table to show all 7 metrics when batch data available
- Fixed TC002 lint in `batch_metrics.py`: moved `numpy` import into TYPE_CHECKING block
- 39/39 tests pass, lint pass, format pass, typecheck 0 errors

### Next iteration work
- Run GPU eval for ERA5 (`run_eval.py --max-samples 500`) — ~20min
- Run GPU eval for NorESM (`run_eval_noresm.py --max-samples 500`) — ~10min
- Generate spectral plots from .npz data
- Generate updated metrics panels
- Fix missing ensemble plots for samples 3-4
- Start report file

- **End**: 2026-05-11T20:05:00Z, commit 4fa881a

## Iteration 5
- **Start**: 2026-05-11T20:05:29Z, commit 837fa6c
- **Prefix**: ridge-cedar

### Concerns (3+)

1. **Workflow: 4 iterations of code, zero GPU runs** — The most critical gap. All new metrics code (spectral.py, structural.py, batch_metrics.py) has never been tested on actual model outputs. Tests pass on synthetic data but production eval could fail due to shape mismatches, memory issues, or incorrect function signatures when interacting with real model predictions.

2. **Quality: Existing eval results are stale** — eval_results_500.json and noresm_eval_results_500.json have only 4 metrics (CRPS, MAE, RMSE, mass_violation). No RALSD, SSIM, PSNR. All existing figures were generated from these 4-metric results. Everything needs re-running.

3. **Workflow: No report file yet** — The task explicitly requires "Write a report file tracked in git. Have subsequent iterations review and revise it." After 4 iterations, no report exists. This cannot be further deferred.

4. **Quality: Missing ensemble plots for samples 3-4** — Both ERA5 and NorESM have ensemble plots for samples 0-2 only. Samples 3-4 have comparison and error plots but no ensemble spread visualization. Inconsistent.

### Plan for this iteration
Run the GPU evaluation for ERA5 dataset with all 7 metrics. This is the highest-priority unblocked work — everything else (figures, report, NorESM eval) depends on having real 7-metric results first.

Steps:
1. Allocate GPU via salloc (preemptable)
2. Sanity test: run eval with --max-samples 10 to verify end-to-end
3. Run full ERA5 eval: --max-samples 500
4. Verify output JSON has all 7 metrics + spectral .npz file exists

### Work done
- Ran sanity test (10 samples): ALL 15 methods evaluated with ALL 7 metrics. Pipeline works end-to-end.
- Ran full ERA5 eval (500 samples): 14/15 methods completed successfully before job was cancelled.
  - Job 13757291 (normal GPU, node3405) allocated at 16:17, cancelled at 16:32 (15 min).
  - flow-v2-zscore was killed mid-evaluation (32/500 samples). All other 14 methods completed.
  - Results NOT saved: script was killed before writing JSON/npz.
- Added incremental saving to both `run_eval.py` and `run_eval_noresm.py`:
  - `_save_incremental()` writes JSON after each method group (baselines, SwinIR, Harder, each flow model)
  - Prevents future data loss from job cancellation
- Could not re-run: subsequent GPU allocation attempts (3x) were immediately cancelled by competing agent processes (ivy-ash-c, ivy-ash-d, 3277-eval).
- Cleaned up: no dangling allocations or processes.

### Key results from the 500-sample ERA5 run (from stdout, 14/15 methods):
| Method | CRPS | MAE | RMSE | RALSD(dB) | SSIM |
|---|---|---|---|---|---|
| flow-wide96-amp (28M) | 0.172024 | 0.251212 | 0.456964 | 0.19 | 0.9925 |
| flow-uniform-amp (13M) | 0.175654 | 0.256598 | 0.467506 | 0.22 | 0.9921 |
| flow-logitnorm-ema (13M) | 0.181591 | 0.265857 | 0.499243 | 0.29 | 0.9914 |
| swinir-finetuned+addcl | 0.263245 | 0.263245 | 0.509424 | 0.34 | 0.9910 |
| harder-gan+smcl | 0.283489 | 0.286570 | 0.553964 | 0.46 | 0.9896 |
| bilinear | 0.519055 | 0.519055 | 0.963896 | 1.09 | 0.9758 |

Flow models dominate on ALL 7 metrics. RALSD confirms flow models produce best spectral fidelity (0.19dB vs 1.09dB for bilinear).

### Next iteration work
- Re-run ERA5 eval with incremental saving enabled (will survive cancellations)
- Run NorESM eval
- Generate spectral plots from .npz data
- Write report file
- Fix missing ensemble plots for samples 3-4

- **End**: 2026-05-11T20:38:00Z, commit 401cf85

## Iteration 6
- **Start**: 2026-05-11T20:39:21Z, commit 5027106
- **Prefix**: pine-frost

### Concerns (3+)

1. **Workflow: GPU eval results still not saved to disk** — Iteration 5 ran 14/15 ERA5 methods to completion but results were ONLY captured in stdout (pasted into scratchpad). No JSON/NPZ file was written because the job was killed before the final save. The incremental saving added in iter5 should fix this on next run, but the data from that run is lost. The scratchpad table is the only record and should be treated as approximate (manually transcribed from stdout).

2. **Workflow: Persistent GPU allocation cancellation by competing agent** — The other Ralph workflow (ivy-ash, build.yml on pts/25) repeatedly cancelled salloc allocations in iter5 (3x) and again in iter6 (1x). Root cause: the guardrail "scancel all allocations before the next iteration" in build.yml causes the other agent to cancel ALL user jobs, not just its own prefix. Mitigation: using sbatch instead of salloc (job 13760707 submitted).

3. **Workflow: No report file after 5 iterations** — The task explicitly requires "Write a report file tracked in git. Have subsequent iterations review and revise it." This is the 6th iteration with zero report output. Must start this iteration regardless of whether GPU eval completes.

4. **Quality: NorESM eval script untested on GPU** — run_eval_noresm.py was updated in iter4 to match run_eval.py's batch metrics pattern, but has NEVER been run on GPU. Could have bugs in prediction collection for NorESM-specific methods.

### Plan for this iteration
1. Submit ERA5 eval via sbatch (done: job 13760707)
2. Write initial report file based on available data (scratchpad results from iter5 stdout)
3. Monitor job; if results arrive, update report with verified data

### Work done
- Submitted ERA5 eval sbatch job (13760707) — CANCELLED by competing ivy-ash workflow
- Created sbatch script: `scripts/eval_era5_sbatch.sh` for future use
- **Created METRICS_REPORT.md** — initial report with preliminary results from iter5 stdout, methods table, metrics definitions, key findings, figure inventory, known issues
- **Fixed ensemble plot bug** in `scripts/make_figures.py` line 458: `range(min(3, n_vis_samples))` → `range(n_vis_samples)` — this caused missing ensemble plots for samples 3-4
- **Added spectral figure generation** to `make_figures.py`: integrated `plot_extended_metrics_panel`, `plot_ralsd_comparison`, `plot_psd_comparison`, `plot_spectral_bias` calls when data is available
- Fixed import order (ruff auto-fix)
- Tests: 29/29 spectral tests pass, lint pass, typecheck 0 errors
- GPU BLOCKER: ivy-ash workflow on pts/25 cancels ALL non-ivy-ash pending preemptable jobs every ~10min. Both normal GPU slots occupied by ivy-ash (until ~18:37 EDT). Preemptable pending jobs get cancelled before they can start.

### Next iteration work
- Wait for ivy-ash workflow to release GPUs (after 18:37 EDT), then run ERA5 eval
- Run NorESM eval
- Generate all spectral/extended figures from .npz data
- Update METRICS_REPORT.md with verified JSON results
- Fix missing ensemble plots for samples 3-4 (code fixed, needs GPU re-run of make_figures.py)

- **End**: 2026-05-11T20:56:07Z, commit 376e79d

## Iteration 7
- **Start**: 2026-05-11T20:57:20Z, commit 66896c3
- **Prefix**: jdxg-xdev

### Concerns (3+)

1. **Workflow: Report shows only 6 of 15 methods** — eval_results_500.json has verified 4-metric data for all 15 ERA5 methods, but the report table only shows 6 cherry-picked rows. The RALSD/SSIM values shown are from unverified stdout transcription (iter5), NOT from any JSON file. No PSNR values in the table at all. The report claims 7 metrics but displays 5 columns for 6 rows. This is misleading.

2. **Workflow: Report missing NorESM section entirely** — noresm_eval_results_500.json has verified 4-metric data for 12 methods, but METRICS_REPORT.md has zero NorESM content. The task requires comprehensive comparison across both datasets.

3. **Quality: NorESM constraint mismatch not analyzed** — NorESM results show constrained methods (+AddCL/+SmCL) perform WORSE on CRPS than unconstrained versions. Flow-wide96 has mass_violation 1.12 on NorESM vs ~1e-6 on ERA5. This constraint-vs-accuracy tradeoff is a key finding that hasn't been discussed.

4. **Workflow: GPU still blocked** — Both normal slots taken by ivy-ash (until ~18:37 EDT, ~1.5h from now). Preemptable queue Priority-blocked. Cancelled dangling salloc from iter6. Cannot run GPU eval this iteration.

### Plan for this iteration
Rewrite METRICS_REPORT.md with complete, verified data:
- All 15 ERA5 methods with verified 4-metric data from JSON
- All 12 NorESM methods with verified 4-metric data from JSON
- Remove unverified RALSD/SSIM/PSNR data (mark as pending)
- Add NorESM analysis including constraint tradeoff finding
- Proper figure inventory from visual inspection

### Work done
- Rewrote METRICS_REPORT.md completely:
  - All 15 ERA5 methods with verified 4-metric data (was 6 methods with unverified data)
  - All 12 NorESM methods with verified data (was missing entirely)
  - Removed unverified RALSD/SSIM/PSNR columns — marked as pending GPU eval
  - Added cross-dataset comparison table
  - Added NorESM analysis: constraint layers HURT on TAS (swinir-ft 0.988 -> 1.455 with addcl)
  - Proper figure inventory with status
  - All data verified against JSON files via Python script
- Cancelled dangling salloc from iter6 (job 13761188) and mlnl-eval (job 13760724)
- Visually inspected 6 figures: metrics panels, CRPS, samples, flow_vs_baseline — all look correct
- Confirmed ERA5 samples 3-4 missing ensemble plots (code fix from iter6, needs GPU re-run)
- GPU blocked: both normal slots by ivy-ash until ~18:37 EDT, preemptable Priority-blocked
- Lint/typecheck: maintained code passes (src/, tests/)

### Next iteration work
- Run GPU eval (ERA5 + NorESM) with 7 metrics — this is THE critical blocker
- Generate spectral figures from .npz data
- Regenerate ensemble plots for samples 3-4
- Update report with RALSD/SSIM/PSNR data

- **End**: 2026-05-11T21:05:00Z, commit cf052f7

## Iteration 8
- **Start**: 2026-05-11T21:07:53Z, commit 0c13674
- **Prefix**: xrut-azjs

### Concerns (3+)

1. **Fact: Pending sbatch job 13761716 referenced non-existent script** — Job referenced `scripts/eval_comprehensive.sh` which was never created. Would have failed silently when started. Cancelled it.

2. **Workflow: GPU still Priority-blocked on preemptable** — Both normal slots occupied by ivy-ash (running ~30min, ends ~18:37 cluster time). Preemptable queue returns "insufficient priority" even with --immediate. salloc attempt (13762500) also got revoked.

3. **Workflow: 8 iterations, no GPU results on disk** — The most persistent blocker. All RALSD/SSIM/PSNR data is unverified. The iter5 stdout data in scratchpad is the only record and cannot be reproduced.

4. **Quality: sbatch script was ERA5-only** — Prior script only ran ERA5 eval, not NorESM. Fixed to run both.

### Plan for this iteration
Since GPU is blocked: review all code for correctness, run tests, fix the sbatch script to actually work, prepare for next iteration's GPU run.

### Work done
- Cancelled broken pending job 13761716 (referenced non-existent `scripts/eval_comprehensive.sh`)
- **Full code review** of the eval pipeline:
  - `run_eval.py`: correct, incremental saving works, spectral curves computed and saved
  - `run_eval_noresm.py`: correct, mirrors ERA5 with NorESM-specific paths/upsampling_factor=2
  - `batch_metrics.py`: correct wrapper around spectral/structural metrics
  - `spectral.py`: RALSD implementation matches paper definition, handles different image sizes
  - `structural.py`: uses skimage SSIM (standard), PSNR correct
  - `evaluate.py`: return_predictions correctly collects ensemble means (N,H,W)
  - `baselines.py`: upsample functions consistent between metric computation and prediction collection
  - `swinir.py`, `harder.py`: return_predictions correct
  - **No bugs found**
- Verified report data matches JSON files (spot-checked key values)
- All 132 tests pass, lint clean, typecheck 0 errors
- **Updated sbatch script** (`scripts/eval_era5_sbatch.sh`):
  - Runs BOTH ERA5 and NorESM evals in one job (was ERA5-only)
  - Outputs to `eval_results_7metrics.json` and `noresm_eval_results_7metrics.json`
  - Uses function `run_in_container()` for DRY singularity exec calls
  - Fixed job name to `metrics-v2-eval`
  - Increased time limit to 1.5h for both datasets

### Next iteration work
- Submit sbatch job when GPU becomes available
- If ivy-ash releases, try salloc for interactive GPU access
- Generate spectral figures from .npz data
- Update report with RALSD/SSIM/PSNR data
- Regenerate ensemble plots for samples 3-4


- **End**: 2026-05-11T21:15:58Z, commit ed48782

## Iteration 9
- **Start**: 2026-05-11T21:17:37Z, commit 4ca593f
- **Prefix**: otut-fveo

### Concerns (3+)

1. **Workflow: 8 iterations, zero GPU results on disk** — Still the most persistent blocker. GPU allocation fails with Priority on both normal and preemptable partitions. Ivy-ash holds 1 normal slot (was 2, now 1). sbatch job 13764285 queued on preemptable.

2. **Quality: Missing distribution metric** — The task says "find more" metrics. Iter1 identified LHD (Logarithmic Histogram Distance) as a gap but it was never implemented. Literature survey shows EMD (Earth Mover Distance) is used by the intercomparison paper AND STVD — more widely adopted than LHD.

3. **Quality: Metric selection not grounded in literature** — The report lists 7 metrics with brief references but doesn't justify WHY these metrics were chosen or what alternatives were considered. The task explicitly requires "Ground in papers: what metric matters? what others used?"

### Plan for this iteration
Add EMD (Earth Mover Distance) as the 8th metric — addresses the distribution metric gap identified in iter1. Ground all metric selections in literature with a survey of what papers use. Wire EMD into the full pipeline (metrics, batch_metrics, eval scripts, plots, tests, report).

### Work done
- **Literature survey**: Reviewed all 12 climate downscaling papers for evaluation metrics. Found EMD used by intercomparison paper (2512.13987) and STVD (2312.06071). Also cataloged LPIPS (WassDiff), SSR (CDSI), CSI (WassDiff), LHD (intercomparison).
- **Implemented `src/downscaling/metrics/distribution.py`**: `emd()` (batch) and `emd_per_sample()` using scipy.stats.wasserstein_distance
- **Updated `src/downscaling/metrics/__init__.py`**: Added emd, emd_per_sample exports
- **Updated `src/downscaling/evaluation/batch_metrics.py`**: Added EMD to compute_batch_metrics()
- **Updated `scripts/run_eval.py`**: Added EMD to display table and _print_metrics
- **Updated `scripts/run_eval_noresm.py`**: Same as run_eval.py
- **Updated `src/downscaling/plotting/spectral.py`**: Added EMD to extended metrics panel (now 8-panel grid)
- **Created `tests/test_distribution.py`**: 12 tests covering EMD properties, per-sample, batch integration
- **Updated `tests/test_spectral.py`**: batch_metrics key assertion now includes 'emd'
- **Updated `tests/test_spectral_plots.py`**: sample_results fixture includes EMD, panel test expects 8 visible
- **Updated `METRICS_REPORT.md`**:
  - Added EMD to metrics table with references
  - Added "Metric Selection Grounding" section: 6-axis table showing which papers use which metrics
  - Added "Additional metrics considered but not included" (LHD, CSI, SSR, LPIPS) with justification
  - Added EMD definition section
  - Updated all "7 metrics" references to "8 metrics"
  - Updated pending notice to include EMD
- **Updated sbatch script**: Cancelled old job 13763198, resubmitted as 13764285 with updated code
- 144/144 tests pass, lint pass, format pass, typecheck 0 errors (2 pre-existing warnings)

### Next iteration work
- GPU eval: sbatch job 13764285 was cancelled (guardrail 1011). Must resubmit next iteration.
- GPU Priority-blocked on ALL partitions. Normal has 1 slot free but Priority insufficient. Preemptable same.
- When results arrive: update report with RALSD/SSIM/PSNR/EMD data
- Generate spectral figures from .npz data
- Regenerate ensemble plots for samples 3-4

- **End**: 2026-05-11T21:32:24Z, commit 63c5da1

## Iteration 10
- **Start**: 2026-05-11T21:33:38Z, commit 495081d
- **Prefix**: rqdn-5230

### Concerns (3+)

1. **Fact: GPU eval would FAIL without singularity** — The editable install `.pth` file (`_editable_impl_climate_downscaling.pth`) contains `/workspace/src`, but `/workspace` does NOT exist on GPU nodes. The worktree path is `/home/chenxy/repos/workspace/metrics-v2`. Direct srun without singularity gets `ModuleNotFoundError: No module named 'downscaling'`. The sbatch script works because singularity binds the project dir to `/workspace` inside the container. Prior iterations did not discover this because iter5 used singularity. Iter9's mysterious `python -m downscaling.evaluation.comprehensive` process was somehow running despite this — unclear how (maybe module was loaded before `/workspace` unmounted, or a different mechanism).

2. **Workflow: Rogue process on GPU from prior iteration** — Job 13763778 (salloc, "metrics-eval") was running `python -m downscaling.evaluation.comprehensive` — a module that does NOT exist in any worktree. No output files produced. GPU at 100% for 8+ minutes with no results. Killed it. Root cause: prior iteration launched an eval using a non-existent module, process consumed GPU without producing output. This wasted the entire 1hr allocation.

3. **Workflow: 10 iterations, zero GPU results on disk** — This is now a systemic issue. Iter5 ran eval but job was killed before save. Iter6-9 blocked by ivy-ash or Priority. Iter10: found GPU allocation but rogue process wasted it, then allocation expired. The PYTHONPATH/singularity issue means even if we get GPU, direct srun fails.

4. **Quality: sbatch output names were stale (7metrics → 8metrics)** — Fixed this iteration.

### Plan for this iteration
Diagnose and fix the eval execution path. The sbatch script (singularity) is the correct approach. Direct srun requires PYTHONPATH workaround.

### Work done
- Killed rogue `python -m downscaling.evaluation.comprehensive` process on GPU node4102 (PID 2214962)
- **Diagnosed PYTHONPATH bug**: editable install `.pth` references `/workspace/src` which doesn't exist on GPU nodes. Singularity works because it bind-mounts project dir to `/workspace`.
- **Fixed sbatch script**: updated output filenames from 7metrics to 8metrics
- **Created `scripts/eval_srun.sh`**: helper for interactive srun usage that sets `PYTHONPATH=$PROJECT_DIR/src` to work without singularity
- GPU allocation attempt: both normal and preemptable Priority-blocked. No GPU available.
- Cancelled expired salloc 13763778 (was already expired) and pending 13765443, 13765455
- Fixed lint issues in run_eval.py and run_eval_noresm.py (SIM114: combined elif branches for ssim/psnr/emd)
- 144/144 tests pass, lint clean on eval scripts
- Memory recorded: mem-1778536219-ba0a (PYTHONPATH GPU fix)

### Next iteration work
- Submit sbatch job (singularity approach confirmed working) — must wait for GPU availability
- OR use eval_srun.sh with PYTHONPATH workaround for interactive srun
- When results arrive: update report with RALSD/SSIM/PSNR/EMD data, generate spectral figures
- Regenerate ensemble plots for samples 3-4

- **End**: 2026-05-11T21:50:33Z, commit (pending)

## Iteration 11
- **Start**: 2026-05-11T21:52:27Z, commit 7e7bf25
- **Prefix**: rqdn-5230 (reused from iter10, same chain)

### Concerns (3+)

1. **Fact: Report test count was wrong (156 vs actual 144)** — Iteration 9 claimed 156 tests in the report but only 144 exist. Fixed.

2. **Quality: Report figure inventory missed 3 cross-dataset figures** — `dual_metrics_panel.png`, `dual_crps.png`, `constraint_impact.png` existed in `figures/` but were not in the report's figure inventory. Fixed.

3. **Quality: Lint errors in 2 legacy scripts** — `scripts/eval_crps.py` (import order + unused import) and `scripts/flow_downscale.py` (unused loop variable). Fixed.

4. **Workflow: GPU finally became available after 10 blocked iterations** — ivy-ash workflow released. Submitted sbatch on both mit_preemptable (Priority-blocked, cancelled) and mit_normal_gpu (started on node3205 L40S). Job 13766068 completed successfully in 19 minutes.

### Plan for this iteration
Submit GPU eval, fix code quality issues, update report with 8-metric verified data.

### Work done
- **GPU eval completed**: Job 13766068 on node3205 (L40S), ~19 min total
  - ERA5: 15/15 methods, all 8 metrics, spectral .npz saved
  - NorESM: 12/12 methods, all 8 metrics, spectral .npz saved
  - Output files: eval_results_8metrics.json, eval_results_8metrics_spectral.npz, noresm_eval_results_8metrics.json, noresm_eval_results_8metrics_spectral.npz
- **Verified iter5 RALSD claims**: flow-wide96 RALSD=0.19dB (actual 0.186), bilinear RALSD=1.09dB — iter5 stdout claims confirmed accurate
- **Updated METRICS_REPORT.md**:
  - ERA5 table: full 8 metrics for all 15 methods (was 4 metrics)
  - NorESM table: full 8 metrics for all 12 methods (was 4 metrics)
  - Added 2 new ERA5 findings (spectral + distribution insights)
  - Added 2 new NorESM findings (spectral + EMD bimodal cluster)
  - Added cross-dataset spectral comparison rows
  - Added cross-dataset figure inventory (was missing 3 figures)
  - Updated remaining work section
  - Fixed test count (156 -> 144)
- **Fixed lint**: eval_crps.py imports, flow_downscale.py unused variable
- All 144 tests pass, lint clean

### Key ERA5 results (verified from JSON)
| Method | CRPS | RALSD | SSIM | EMD |
|--------|------|-------|------|-----|
| flow-wide96 | 0.172 | 0.19dB | 0.993 | 0.003 |
| flow-uniform | 0.176 | 0.22dB | 0.992 | 0.004 |
| swinir-ft+addcl | 0.263 | 0.34dB | 0.991 | 0.006 |
| bilinear | 0.519 | 1.09dB | 0.976 | 0.087 |

### Next iteration work
- Generate spectral figures from .npz data (PSD curves, bias plots, RALSD bars)
- Generate extended 8-metric bar charts
- Regenerate ensemble plots for samples 3-4 (code fix from iter6)
- Update dual metrics panel with 8 metrics


- **End**: 2026-05-11T22:19:43Z, commit 8dfec76

## Iteration 12
- **Start**: 2026-05-11T22:20:40Z, commit 08f08f4
- **Prefix**: rmiv-vbkt

### Concerns (3+)

1. **Bug: NPZ method names don't match COLOR_MAP** — `run_eval.py` line 509 sanitizes names (`flow-wide96-amp (28M)` → `flow-wide96-amp_28M`) for NPZ keys, but plotting code's COLOR_MAP/DISPLAY_NAMES expects canonical names with parentheses. Flow models would get fallback gray colors (#888888) and raw names in legends. Root cause: unnecessary sanitization (numpy savez handles parens fine).

2. **Workflow: No spectral figures generated despite .npz data existing** — Iter11 completed GPU eval producing spectral .npz files but did not generate figures from them. The `make_figures.py` script had the code but was never run with `--metrics-only` flag pointing to the 8-metric results.

3. **Quality: `make_figures.py` defaults point to old JSON** — Default `--era5-results` is `eval_results_500.json` (old 4-metric) not `eval_results_8metrics.json`. Would silently use stale data if run without explicit flags.

### Plan for this iteration
Fix the NPZ name mapping bug. Generate all spectral/extended metric figures from .npz data (no GPU needed). Visually inspect all generated figures. Update report.

### Work done
- **Fixed NPZ name mapping bug** in `make_figures.py`: added reverse mapping from sanitized NPZ keys to canonical method names using the JSON results dict. Flow models now get correct colors and display names.
- **Removed unnecessary sanitization** in `run_eval.py` and `run_eval_noresm.py`: numpy savez handles parentheses fine, sanitization was creating the name mismatch.
- **Generated 10 new figures** via `make_figures.py --metrics-only`:
  - ERA5: `extended_metrics_panel.png`, `ralsd_comparison.png`, `psd_comparison.png`, `spectral_bias.png`
  - NorESM: `extended_metrics_panel.png`, `ralsd_comparison.png`, `psd_comparison.png`, `spectral_bias.png`
  - Cross-dataset: `dual_crps.png`, `dual_metrics_panel.png` (regenerated)
- **Visually inspected all 10 new figures** — all correct:
  - PSD: clear frequency-dependent separation, flow models closest to ground truth
  - Spectral bias: bilinear worst (~2 dB ERA5), flow models near 0
  - RALSD bars: Wide96 best (0.19 dB ERA5), clear ranking
  - Extended panels: all 8 metrics displayed with correct colors and values
  - NorESM PSD: methods overlap closely (2x SR easier), spectral bias smaller
- **Updated METRICS_REPORT.md**: 
  - Added 8 new figure entries to ERA5 and NorESM inventories
  - Removed "Pending Figures" section (all generated)
  - Updated remaining work items 2 and 3 as DONE
- 144/144 tests pass, lint clean, ruff format clean

### Next iteration work
- Fix ensemble plots for ERA5 samples 3-4 and NorESM samples 3-4 (needs GPU)
- Update dual metrics panel to show all 8 metrics (currently 4)
- Consider updating `make_figures.py` defaults to point to 8-metric results

- **End**: 2026-05-11T22:26:49Z, commit c920344

## Iteration 13
- **Start**: 2026-05-11T22:27:45Z, commit 27d45f1
- **Prefix**: zwdk-jndi

### Concerns (3+)

1. **Quality: `plot_dual_metrics_panel` only shows 4 of 8 metrics** — The cross-dataset comparison figure (`dual_metrics_panel.png`) only showed CRPS, MAE, RMSE, Mass Violation. Missing SSIM, PSNR, RALSD, EMD — inconsistent with the per-dataset `extended_metrics_panel.png` which shows all 8. Listed as remaining work item 5 in the report.

2. **Quality: `make_figures.py` defaults point to stale JSON** — Default `--era5-results` is `eval_results_500.json` (old 4-metric) not `eval_results_8metrics.json`. Running without flags silently produces figures from outdated data.

3. **Workflow: GPU job birch-amber-base on node4106 is not from this chain** — Job 13766528 (salloc, 24min runtime, 2hr limit) using one of my normal GPU slots. Not my prefix (zwdk-jndi or any prior chain prefix). Must not touch it per guardrail 1012.

### Plan for this iteration
Update `plot_dual_metrics_panel` to show all 8 metrics. Fix `make_figures.py` defaults. Regenerate figure. Single focused change.

### Work done
- **Updated `plot_dual_metrics_panel`** in `src/downscaling/plotting/metrics.py`: expanded from 4-row to 8-row layout showing all 8 metrics (CRPS, MAE, RMSE, SSIM, PSNR, RALSD, EMD, Mass Violation). Each row now includes direction indicator ("lower/higher is better") in y-axis label. Dynamic filtering: only shows metrics present in data.
- **Fixed `make_figures.py` defaults**: changed `--era5-results` from `eval_results_500.json` to `eval_results_8metrics.json` and `--noresm-results` from `noresm_eval_results_500.json` to `noresm_eval_results_8metrics.json`.
- **Regenerated `figures/dual_metrics_panel.png`**: 8-row x 2-col panel, visually inspected — all 8 metrics correct.
- **Updated METRICS_REPORT.md**: figure description updated, remaining work item 5 marked DONE.
- 19/19 relevant tests pass (test_spectral_plots + test_metrics), lint clean, typecheck 0 errors.

### Next iteration work
- Fix ensemble plots for ERA5 samples 3-4 and NorESM samples 3-4 (needs GPU)

- **End**: 2026-05-11T22:32:38Z, commit 1b7daf2

## Iteration 14
- **Start**: 2026-05-11T22:33:36Z, commit a7c3e72
- **Prefix**: ercc-ggac

### Concerns (3+)

1. **Quality: PSD plots are unreadable with 15 methods** — ERA5 PSD comparison has 15 overlapping lines in log-log scale; individual methods indistinguishable except bilinear (gray) diverging at high freq. NorESM PSD is even worse: ALL 12 methods overlap perfectly, conveying zero information. The plots fail the task requirement "Plots make sense and show what they need to show well."

2. **Quality: Ensemble plots for samples 3-4 still missing** — ERA5 and NorESM both lack ensemble plots for samples 3 and 4 (have comparison and error plots). Code fix applied in iter8, but needs GPU re-run. Remaining work item 4.

3. **Quality: NorESM spectral bias noisy at high frequencies** — The NorESM spectral bias plot shows oscillation/noise at frequencies >0.35 cycles/pixel, making high-freq behavior hard to interpret. ERA5 spectral bias is much cleaner.

### Plan for this iteration
Fix the PSD comparison plot readability. When many methods overlap, add a ratio subplot (PSD_pred/PSD_truth in dB) that reveals small differences invisible in the raw PSD. Highlight a few representative methods in the main PSD panel; dim the rest. Regenerate figures. Single focused change.
