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
