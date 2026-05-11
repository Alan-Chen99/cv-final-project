# Metrics Task Scratchpad

## Iteration 1 — 2026-05-11 13:05 EDT
Start commit: `8d355d2`

### Objective
Add evaluation metrics beyond pixel-space (CRPS/MAE/RMSE/mass violation) for all trained models. No training. Code in `src/downscaling/`.

### Literature Review
Searched for and downloaded 2 papers on evaluation metrics for climate downscaling:
1. **2410.01776** - "Dynamical-generative downscaling of climate model ensembles" (Lopez-Gomez et al.)
   - Key metrics: CRPS, radially averaged energy spectra, spread-skill ratio, quantile MAE
2. **2604.03275** - "IPSL-AID" (IPSL group)
   - Key metrics: rank histogram, spread-skill ratio, KL divergence, PSD, CRPS, quantile analysis, inter-variable correlation

Existing catalog papers with relevant metrics:
- CorrDiff (2309.15214): power spectra, probability distributions, rank histograms, ensemble spread
- Intercomparison paper (2512.13987): comprehensive comparison of cGANs vs diffusion

### Current State
Existing metrics in `src/downscaling/metrics/`: CRPS (energy + paper variant)
Existing evaluation in `src/downscaling/evaluation/`: MAE, RMSE, mass violation, ensemble evaluation pipeline

Available trained models (NorESM 2x SR):
- Flow matching (wide96-amp): `pool/datasets/noresm-dataset/models/flow-wide96-amp/best_flow.pt`
- Harder CNN (no constraint + softmax): `pool/datasets/noresm-dataset/models/harder/twc_cnn_*.pth`
- Harder GAN (softmax): `pool/datasets/noresm-dataset/models/harder/twc_gan_softmax.pth`
- SwinIR finetuned: `pool/datasets/noresm-dataset/models/swinir_ft/best_swinir.pt`

ERA5 4x SR cached predictions: `pool/datasets/era5_sr_data/prediction/`

### Plan — Metrics to Implement

**Tier 1 (highest diagnostic value):**
1. **Power Spectral Density (PSD)** — Radially averaged power spectrum. THE standard spatial-frequency metric. Shows whether models preserve fine-scale structure vs over-smoothing.
2. **Rank Histogram** — Probabilistic calibration diagnostic. Is the truth equally likely in any rank position among ensemble members?
3. **Spread-Skill Ratio (SSR)** — sigma_ens / RMSE with finite-size correction sqrt((M+1)/M). Optimal at 1.0.

**Tier 2 (complementary):**
4. **SSIM** — Structural similarity index. Captures luminance/contrast/structure beyond pointwise error.
5. **Distribution comparison** — KL divergence between predicted and true distributions.

### Implementation Approach
- `src/downscaling/metrics/spectral.py` — PSD computation
- `src/downscaling/metrics/calibration.py` — Rank histogram, spread-skill ratio
- `src/downscaling/metrics/structural.py` — SSIM
- `src/downscaling/metrics/distributional.py` — KL divergence
- Update `src/downscaling/evaluation/evaluate.py` for extended metrics
- Integration tests in `tests/test_metrics.py`
- Run evaluation on all models, write report

### Allocation Prefix
`qzcs-imez`

### Iteration 1 End
End commit: `74bd78c`
End time: ~13:20 EDT

Work done:
- Literature search: found and downloaded 2 papers (2410.01776, 2604.03275)
- Identified 5 metrics to implement: PSD, rank histogram, SSR, SSIM, KL divergence
- Created 5 tasks for implementation
- Updated CLAUDE.md with new papers
- All committed

## Iteration 2 — 2026-05-11 13:19 EDT
Start commit: `f9e5b00`

### Concerns Review
1. **Workflow**: Iteration 1 was literature-only. No code written. Tasks created but no implementation started. This is fine for a first iteration.
2. **Quality**: The existing CRPS implementation has O(M^2) pairwise loop — acceptable for small M but worth noting.
3. **Literature**: PSD approach is standard and well-documented across CorrDiff, CDSI, R2D2 papers. The CDSI Appendix A has the SSR formula with finite-size correction: sqrt((M+1)/M) * spread/RMSE.

### Work Done
- Implemented PSD metric in `src/downscaling/metrics/spectral.py`:
  - `radial_psd()`: 2D FFT → |F|^2 → azimuthal average over wavenumber bins
  - `ensemble_mean_psd()`: per-member PSD then average (preserves fine-scale variability)
  - `psd_log_ratio()`: scalar summary = mean |log10(P_pred/P_truth)|
- Added 14 integration tests in `tests/test_metrics.py`:
  - White noise flat spectrum, sinusoid peak detection, constant field zero power
  - Output shapes, rectangular fields, ensemble variance reduction
  - Log-ratio analytical values
- All 24 tests pass, ruff clean, basedpyright clean
- Committed: `35db58e`

### Iteration 2 End
End commit: `35db58e`
End time: ~13:24 EDT
Next: Implement rank histogram + spread-skill ratio (task-1778519848-3f8f)

## Iteration 3 — 2026-05-11 13:23 EDT
Start commit: `0c54176`

### Concerns Review
1. **Quality**: PSD implementation is solid — tests cover analytical properties. SSR formula from CDSI paper is the standard reference (Fortin et al., 2014).
2. **Workflow**: Prior iteration implemented PSD correctly, committed, updated scratchpad. No workflow issues.
3. **Literature**: Rank histogram and SSR are standard calibration diagnostics used in CorrDiff, IPSL-AID, CDSI, R2D2. The formulas are well-established. No need for additional literature search.

### Work Done
- Implemented calibration metrics in `src/downscaling/metrics/calibration.py`:
  - `rank_histogram()`: counts rank of truth among M members, returns M+1 bin histogram
  - `spread_skill_ratio()`: bias-corrected SSR = sqrt((M+1)/M) * spread / RMSE
- Added 11 integration tests in `tests/test_metrics.py`:
  - Rank histogram: uniform calibrated ensemble (chi-squared test), U-shape for under-dispersive, deterministic ensemble, output shape, multichannel
  - SSR: perfect calibration ~1.0, underdispersive <0.5, overdispersive >2.0, perfect prediction returns inf, single member raises, finite-size correction verification
- All 35 tests pass (10 CRPS + 14 PSD + 11 calibration), ruff clean, basedpyright clean
- Updated `__init__.py` with re-exports
- Committed: `e62cbd9`

### Iteration 3 End
End commit: `e62cbd9`
End time: ~13:25 EDT
Next: Implement SSIM metric (task-1778519849-067b)

## Iteration 4 — 2026-05-11 13:27 EDT
Start commit: `3846684`

### Concerns Review
1. **Quality**: Prior implementations (PSD, calibration) are solid — correct formulas, good test coverage, clean type checks. No issues.
2. **Workflow**: Prior iterations committed properly, ran all checks. No missing steps.
3. **Design**: SSIM data_range initially computed from truth only, breaking symmetry. Fixed to use joint min/max of both inputs. Also: negated field test was based on wrong assumption about SSIM behavior — negation flips both luminance and structure terms, making their product positive. Replaced with uncorrelated-field test.

### Work Done
- Implemented SSIM metric in `src/downscaling/metrics/structural.py`:
  - `ssim()`: Gaussian-weighted SSIM with scipy.ndimage.gaussian_filter
  - `ensemble_mean_ssim()`: per-member SSIM then average
  - Joint data_range from both inputs for symmetry
  - Default window_sigma=1.5, k1=0.01, k2=0.03 (standard Wang et al. 2004)
- Added 12 integration tests in `tests/test_metrics.py`:
  - Identical fields → 1.0, uncorrelated → near 0, noisy → intermediate (monotone with noise)
  - Constant fields, explicit data_range, shape errors, non-2D errors
  - Symmetry, range bounded [-1, 1], ensemble single-member consistency, better ensemble higher SSIM, non-3D error
- All 47 tests pass (10 CRPS + 14 PSD + 11 calibration + 12 SSIM), ruff clean, basedpyright clean
- Committed: `d6bfbb1`

### Iteration 4 End
End commit: `d6bfbb1`
End time: ~13:30 EDT
Next: Implement KL divergence / distribution comparison (task-1778519850-519e)

## Iteration 5 — 2026-05-11 13:32 EDT
Start commit: `e0635af`

### Concerns Review
1. **Quality**: Prior implementations (PSD, calibration, SSIM) are solid — correct formulas, good test coverage, clean type checks. No issues found.
2. **Workflow**: Prior iterations committed properly, ran all checks (ruff, basedpyright, pytest). No missing steps.
3. **Design**: KL divergence on binned histograms requires shared bin edges from the joint range and Laplace smoothing to avoid log(0). Both implemented correctly.

### Work Done
- Implemented distributional metrics in `src/downscaling/metrics/distributional.py`:
  - `histogram_kl_divergence()`: KL(P||Q) via shared-edge histograms + Laplace smoothing
  - `ensemble_mean_kl_divergence()`: per-member KL then average
  - Handles arbitrary shapes (flattens both inputs), degenerate constant fields
- Added 13 integration tests in `tests/test_metrics.py`:
  - Identical → 0, non-negative (Gibbs), shifted → positive, larger shift → higher KL
  - Scaled distribution, constant fields, shape mismatch, n_bins validation
  - More bins test, arbitrary 3D shape, ensemble single-member consistency, better ensemble lower KL, rejects 1D
- All 60 tests pass (10 CRPS + 14 PSD + 11 calibration + 12 SSIM + 13 KL), ruff clean, basedpyright clean
- Fixed unused import (scipy.ndimage.gaussian_filter) in test file
- Committed: `d9a40bd`

### Iteration 5 End
End commit: `d9a40bd`
End time: ~13:35 EDT
Next: Run comprehensive evaluation on all models + write report (task-1778519851-14b5)

## Iteration 6 — 2026-05-11 13:36 EDT
Start commit: `b9fa495`

### Concerns Review
1. **Workflow**: Prior iterations wrote metrics but never tested on real data. Metrics may have subtle issues only visible with real predictions. Fixed by running full evaluation.
2. **Quality**: Evaluation functions are repetitive — each model has copy-pasted metric code. Wrote a single comprehensive eval that computes ALL metrics for ALL models in one pass.
3. **Quality**: The existing `evaluate_flow_model` doesn't return predictions, only scalar metrics. Wrote separate prediction generators to decouple generation from evaluation, enabling new metrics.

### Work Done
- Wrote `src/downscaling/evaluation/comprehensive.py`:
  - Prediction generators for all model types (flow, CNN, GAN, SwinIR, baselines)
  - Unified `compute_all_metrics()` computing: CRPS, MAE, RMSE, mass violation, SSIM, KL divergence, PSD log-ratio, rank histogram, SSR
  - Diagnostic plot generation: PSD comparison, rank histograms, metrics bar charts
  - Standalone `run_comprehensive_eval()` and `__main__` entry point
- Ran evaluation on GPU (node3302, mit_normal_gpu) on 2000 NorESM test samples, 10 ensemble members
- Generated results JSON + 3 diagnostic plots
- Wrote evaluation report: `src/downscaling/evaluation/results/EVAL_REPORT.md`
- All 60 tests pass, ruff clean, basedpyright clean

### Key Results
| Model | CRPS | MAE | RMSE | MassViol | SSIM | KL | PSD-LR | SSR |
|---|---|---|---|---|---|---|---|---|
| Flow+AddCL | 1.64 | 1.72 | 2.77 | 0.00 | 0.91 | 0.25 | 0.029 | 0.088 |
| CNN(none) | **1.30** | **1.30** | **1.98** | 1.13 | **0.92** | **0.20** | **0.007** | — |
| CNN(softmax) | 1.72 | 1.72 | 2.78 | 0.00 | 0.91 | 0.26 | 0.028 | — |
| GAN(softmax) | 1.76 | 1.78 | 2.85 | 0.00 | 0.88 | 0.27 | 0.036 | 0.044 |
| SwinIR+AddCL | 1.73 | 1.73 | 2.78 | 0.00 | 0.91 | 0.26 | 0.029 | — |
| Bicubic | 1.75 | 1.75 | 2.82 | 0.06 | 0.90 | 0.26 | 0.032 | — |

Key findings:
1. CNN(none) dominates all pointwise metrics but has 1.13K mass violation
2. Constraints hurt on NorESM (LR/HR from different sims — constraint is misspecified)
3. Both ensemble models severely underdispersive (SSR 0.044-0.088)
4. Constrained models barely beat bicubic baseline (~6% CRPS improvement)
5. 2x SR is spectrally easy — all models preserve fine-scale structure

### Iteration 6 End
End commit: `eead769`
End time: ~14:07 EDT
Wall clock: ~31 min (including GPU queue wait)

## Iteration 7 — 2026-05-11 14:07 EDT
Start commit: `36cfc8e`
Allocation prefix: `wafer-plug`

### Concerns Review
1. **Fact**: Task references arxiv 2604.03459 for "spectral coherence" but the paper (PC-AFM) does NOT contain this metric. It uses RALSD, CE, log-PDF distance, MCB, CRPS. The spectral coherence concept is a standard signal processing metric (cross-spectral density normalized by auto-spectra) — no paper attribution needed.
2. **Quality**: Spectral coherence for a SINGLE field pair is trivially 1.0 at every frequency (|X*conj(Y)|^2 / (|X|^2 * |Y|^2) = 1). Meaningful estimation requires averaging cross-spectra across multiple samples before computing coherence. The API must take batches of (N, H, W) pairs.
3. **Quality**: The paper 2604.03459 introduces RALSD = sqrt(mean((10*log10(S_ref/S_pred))^2)) which is a slightly different formulation from our psd_log_ratio (mean |log10|). Worth noting but not blocking — our formulation is comparable.

### Work Done
- Downloaded paper 2604.03459 (PC-AFM) — confirmed it does NOT contain "spectral coherence" metric. Uses RALSD, CE, log-PDF, MCB instead. Spectral coherence is a standard signal processing concept.
- Implemented spectral coherence in `src/downscaling/metrics/spectral.py`:
  - `spectral_coherence(preds, truths)`: batch (N,H,W) → (wavenumbers, γ²(k)) via azimuthally-averaged cross-spectral density
  - `mean_spectral_coherence(preds, truths)`: scalar summary (mean coherence across wavenumbers)
  - `_wavenumber_grid()`: shared helper for radial wavenumber computation
- Added 11 integration tests in `tests/test_metrics.py`:
  - Identical → 1.0, independent → near 0, single sample independent → <0.5
  - Bounded [0,1], noisy copy ordering, output shape, shape mismatch, rejects 2D
  - Scalar summary, ordering, frequency-dependent coherence (low-k > high-k with high-freq noise)
- All 71 tests pass, ruff clean, basedpyright clean
- Updated `__init__.py` with re-exports
- Updated CLAUDE.md with new paper entry
- Key insight: single-sample coherence is NOT trivially 1.0 for independent fields — azimuthal averaging within wavenumber bins causes phase cancellation. This is correct behavior.

### Iteration 7 End
End commit: `ec62c69`
End time: ~14:15 EDT

### HUMAN GUIDANCE (2026-05-11 18:16:11 UTC)

Two new directives before you finalize the eval report:

1. NEW METRIC: Implement a spectral coherence metric based on arxiv 2604.03459. Download the paper first with /arxiv-to-md. Spectral coherence measures phase alignment between predicted and truth fields at each spatial frequency — complementary to PSD which only captures power magnitude. Add to src/downscaling/metrics/ with integration tests. A task has been added for this.

2. CONSISTENCY: Ensure the comprehensive evaluation compares NorESM 2x SR models AND ERA5 4x SR models consistently — same metrics computed for both datasets, presented side by side. Do not evaluate one dataset with metrics the other lacks.

### HUMAN GUIDANCE (2026-05-11 18:16:11 UTC)

CORRECTION to prior guidance: I was wrong about 'spectral coherence' being in arxiv 2604.03459. Your concern #1 in iteration 7 is correct — the paper (PC-AFM) uses RALSD, CE, log-PDF distance, MCB. Use your judgment on which of those metrics are worth adding given what we already have. The spectral coherence idea (cross-spectral density) is still worth implementing as a standalone metric if you think it adds value — it just isn't from this paper.

The NorESM/ERA5 consistency directive still stands: make sure both datasets are evaluated with the same metric set.

## Iteration 8 — 2026-05-11 14:16 EDT
Start commit: `d2f804e`
Allocation prefix: N/A (CPU-only iteration — ERA5 predictions are cached)

### Concerns Review
1. **Quality**: `compute_all_metrics()` does NOT include spectral coherence despite it being implemented in iteration 7. The metric exists in spectral.py but was never wired into the evaluation pipeline.
2. **Workflow**: Comprehensive evaluation is NorESM-only. Human directive explicitly requires ERA5 evaluation with the same metric set. ERA5 predictions are cached (5 flow models × 10K samples × 10 members), so no GPU inference is needed.
3. **Quality**: `compute_all_metrics()` hardcodes `UPSAMPLING_FACTOR = 2` (NorESM). ERA5 uses 4x. Must parameterize.

### Plan
Single deliverable: Add ERA5 evaluation to comprehensive.py with same metrics.
- Parameterize `compute_all_metrics` (upsampling_factor, add spectral coherence)
- Add `run_era5_eval()` that loads cached predictions + truth
- Run evaluation (CPU-only)
- Update report with both datasets side-by-side

### Work Done
- Added `spectral_coherence` (mean_spectral_coherence) to `compute_all_metrics()` — was missing since iteration 7
- Parameterized `upsampling_factor` in `compute_all_metrics()` (was hardcoded to 2)
- Added `run_era5_eval()` loading cached ERA5 predictions (5 flow models)
- Added CLI `--dataset noresm|era5|both` flag
- Ran ERA5 evaluation on 2000 test samples (CPU-only, ~2 min)
- Visually verified all 3 diagnostic plots (PSD, rank histograms, metrics summary)
- Updated EVAL_REPORT.md with both datasets side-by-side + cross-dataset analysis
- All 71 tests pass, ruff clean, basedpyright clean

### Key ERA5 Results
| Model | CRPS | SSR | Coh |
|---|---|---|---|
| ResFlow-20s+AddCL | **0.212** | **1.013** | 0.915 |
| FlowV2+AddCL | 0.217 | 1.196 | 0.915 |
| Flow(none) | 0.219 | 0.936 | 0.916 |
| ResFlow-Heun+AddCL | 1.631 | 3.023 | 0.398 | ← broken

### Anomalies
1. Flow(none) mass_violation=1e-6 despite "no constraint" — predictions likely generated with AddCL (filename mismatch)
2. ResFlow-Heun+AddCL clearly diverged — excluded from analysis

### Cross-Dataset Insight
ERA5 ensembles well-calibrated (SSR ~1.0) vs NorESM severely underdispersive (SSR ~0.05). Same flow matching framework. Difference: ERA5 LR/HR from same reanalysis (consistent physics), NorESM LR/HR from different sims.

### Iteration 8 End
End commit: `170befc`
End time: ~14:25 EDT

### HUMAN GUIDANCE (2026-05-11 18:35:41 UTC)

MANDATORY: Implement RALSD (Relative Average Log Spectral Distance) from arxiv 2604.03459 (PC-AFM). This metric is required — not optional. Formula: RALSD = sqrt(mean((10*log10(S_ref/S_pred))^2)). This is distinct from the existing psd_log_ratio (which uses mean|log10|) — RALSD uses RMS of dB-scale ratio. Add it to the metrics suite and include it in both NorESM and ERA5 evaluations.

## Iteration 9 — 2026-05-11 14:35 EDT
Start commit: `3ae9240`

### Concerns Review
1. **Quality**: RALSD not yet implemented despite mandatory human directive. Prior iteration 8 did not address it (focused on ERA5 eval). Must implement now.
2. **Quality**: Summary tables and plots in comprehensive.py need to include RALSD column/bar. Without this, even after implementing the metric function, it won't appear in evaluation output.
3. **Workflow**: After implementing RALSD, evaluation results JSON files are stale (computed without RALSD). Need to re-run evaluation on both datasets to include RALSD values.

### Work Done
- Implemented `ralsd()` in `src/downscaling/metrics/spectral.py`:
  - Formula: `sqrt(mean((10*log10(S_ref/S_pred))^2))` — RMS of dB-scale spectral ratio
  - Handles zero-power bins (returns inf), validates wavenumber grid match
- Added 9 integration tests in `tests/test_metrics.py`:
  - Identical → 0, non-negative, analytical value (ratio=10 → 10 dB), larger mismatch → higher
  - Distinct from psd_log_ratio, zero-power → inf, mismatched grids → ValueError
  - Symmetric in dB (squared removes sign), works on real radial_psd output
- Wired RALSD into `compute_all_metrics()` in comprehensive.py
- Updated summary table headers and plot metric list to include RALSD
- Updated `__init__.py` exports
- All 80 tests pass (71 prior + 9 RALSD), ruff clean, basedpyright clean
- Did NOT re-run evaluation (needs next iteration — GPU for NorESM, CPU for ERA5)

### Iteration 9 End
End commit: `864d881`
End time: ~14:38 EDT
Next: Re-run evaluation on both datasets (NorESM needs GPU, ERA5 is CPU-only) to get RALSD values. Update EVAL_REPORT.md with RALSD column.

## Iteration 10 — 2026-05-11 14:39 EDT
Start commit: `07785b3`
Allocation prefix: `ipjpj-czlki`

### Concerns Review
1. **Workflow**: Both JSON result files are stale — computed before RALSD was added. NorESM additionally lacks spectral coherence (added in iteration 7, but NorESM eval last ran in iteration 6). Prior agent correctly identified this but deferred to this iteration.
2. **Quality**: EVAL_REPORT.md NorESM table has no Coh or RALSD columns. ERA5 table has Coh but no RALSD. Cross-dataset table says "Spectral coherence | Not computed (rerun needed)" for NorESM. Must fix all.
3. **Quality**: The NorESM results JSON is at `pool/datasets/metrics/comprehensive_results.json` (top-level, from iteration 6) but the code now writes to `pool/datasets/metrics/noresm/`. After re-running, the old file becomes dead. Need to be aware of path discrepancy.

### Plan
1. Run ERA5 eval (CPU-only, ~2 min)
2. Allocate GPU, run NorESM eval (~15 min)
3. Update EVAL_REPORT.md with both complete datasets (RALSD + Coh for both)

### Work Done
- Re-ran ERA5 evaluation (CPU-only, ~6 min on head node) — now includes RALSD and spectral coherence
- Allocated GPU (node3202, mit_normal_gpu), ran NorESM evaluation (~7 min) — now includes RALSD and spectral coherence
- Released GPU allocation
- Visually verified all 6 diagnostic plots (3 per dataset): PSD, rank histograms, metrics summary
- Updated EVAL_REPORT.md: added RALSD column to both tables, added Coh column to NorESM table, updated findings and cross-dataset comparison
- Both JSON result files confirmed to have all 11 metrics for all models
- ruff clean, basedpyright clean (0 errors)
- Committed: `a223e24`

### Key New RALSD Values
**NorESM**: CNN(none) 0.076 (best), constrained models ~0.31, GAN 0.48 (worst working)
**ERA5**: ResFlow-20s+AddCL 0.350 (best), Flow/ResFlow ~0.39, Heun 14.25 (broken)
RALSD is more discriminating than PSD-LR for NorESM (CNN-none 0.076 vs 0.007 — same ranking but 4x dynamic range improvement with RALSD)

### Iteration 10 End
End commit: `a223e24`
End time: ~15:00 EDT
All human directives addressed: RALSD implemented (iter 9), both datasets evaluated with consistent 11-metric set, report updated.

### HUMAN GUIDANCE (2026-05-11 19:00:45 UTC)

This evaluation supersedes everything in /workspace/figures/. Remove all existing files there and regenerate all diagnostic plots in a consistent manner — same metric set, same format, both NorESM and ERA5 datasets, side by side.

## Iteration 11 — 2026-05-11 15:00 EDT
Start commit: `f5c1f0c`

### Concerns Review
1. **Quality**: Old figures in `/workspace/figures/` are from a different evaluation pipeline (organize2 branch, May 11 13:03). They use different model names, different metrics (only 4), and different data format. Must be completely replaced, not incrementally updated.
2. **Quality**: The comprehensive.py evaluation module saves plots to pool/datasets/metrics/ (per-dataset subdirs), not to /workspace/figures/. These are per-dataset separate plots (3 each), not side-by-side. Human directive requires side-by-side format.
3. **Workflow**: ResFlow-Heun+AddCL is clearly broken (RALSD 14.25, SSR 3.023). Prior iterations noted this but never filtered it from plots. It compresses the scale and makes other models indistinguishable. Must auto-filter broken models.

### Work Done
- Created `src/downscaling/plotting/comprehensive.py`: standalone plotting module that reads cached JSON results and generates 5 consistent side-by-side figures
- Cleared all old files from `/workspace/figures/` (17 ERA5 files, 17 NorESM files, 3 top-level files)
- Generated 5 new plots, all NorESM + ERA5 side-by-side:
  1. `psd_comparison.png` — PSD curves with RALSD values in legend
  2. `rank_histograms.png` — 2 NorESM + 4 ERA5 ensemble models
  3. `metrics_summary.png` — all 10 scalar metrics, best model highlighted green
  4. `spectral_metrics.png` — PSD-LR, RALSD, Coherence focused panel
  5. `calibration.png` — SSR with red/green color coding (0.8-1.2 = well-calibrated)
- Auto-filtered broken ResFlow-Heun+AddCL (RALSD > 10 threshold)
- Updated `src/downscaling/plotting/__init__.py` with new exports
- Visually verified all 5 plots — physically reasonable
- 80 tests pass, ruff clean, basedpyright clean

### Iteration 11 End
End commit: `00582c5`
End time: ~15:07 EDT

## Iteration 12 — 2026-05-11 15:08 EDT
Start commit: `8abaa15`

### Concerns Review
1. **Quality**: SSR in SCALAR_METRICS was marked "lower is better" but SSR optimal is 1.0. The metrics_summary plot showed non-ensemble models at SSR=0 highlighted green as "best". This is semantically wrong — SSR=0 means N/A, not optimal. The calibration panel already handles SSR correctly with |SSR-1| distance and ensemble-only filtering.
2. **Quality**: EVAL_REPORT.md Diagnostic Plots section referenced stale pool paths and listed only 3 plots. The canonical plots are 5 side-by-side figures in /workspace/figures/ (regenerated in iteration 11).
3. **Workflow**: All prior iteration claims verified: 80 tests pass, ruff clean, basedpyright 0 errors, all 5 figures visually correct and physically reasonable.

### Work Done
- Removed SSR from SCALAR_METRICS in plotting/comprehensive.py (SSR already handled by calibration panel)
- Updated EVAL_REPORT.md Diagnostic Plots section: correct paths (/workspace/figures/), all 5 plots listed, regeneration command
- Regenerated all 5 figures
- Verified: ruff clean, basedpyright clean

### Iteration 12 End
End commit: `826ba9e`
End time: ~15:12 EDT

## Iteration 13 — 2026-05-11 15:13 EDT
Start commit: `9483cef`

### Concerns Review
1. **Quality**: Dead `print_summary_table()` function (lines 708-730 in evaluation/comprehensive.py) — exact copy of inline table printing above it. Maintenance hazard: two copies of the same logic can diverge silently.
2. **Quality**: Plotting code uses `.get(key, 0)` for missing metrics, silently rendering 0 in plots instead of failing. If a metric fails to compute, the plot would show it as 0 — misleading. Violates error propagation rule.
3. **Quality**: `_filter_broken()` used `.get("ralsd", 0)` — if RALSD is missing from a model, it would NOT be filtered (0 < 10 passes). Should explicitly check key existence.

### Work Done
- Removed dead `print_summary_table()` function from evaluation/comprehensive.py
- Consolidated duplicate ssim import into single import block (ruff auto-split back to two blocks for isort compliance — that's fine)
- Replaced all `.get(key, 0)` in plotting/comprehensive.py with explicit KeyError raises for missing metrics
- Fixed `_filter_broken()` to check `"ralsd" in r` before filtering
- Replaced `r.get("ralsd", 0)` in PSD legend with `r["ralsd"]`
- Verified: ruff clean, basedpyright 0 errors, 80 tests pass, all 5 plots regenerate correctly
- All changes are code cleanup — no metric values or report content changed

### Iteration 13 End
End commit: `0a9d30e`
End time: ~15:17 EDT

## Iteration 14 — 2026-05-11 15:21 EDT
Start commit: `171e69c`

### Concerns Review
1. **Quality**: `evaluation/comprehensive.py` has 3 duplicate plotting functions (lines 300-397) that diverge from the canonical `plotting/comprehensive.py`. They have bugs:
   - `plot_metrics_summary` uses `.get(metric, 0)` — silently shows 0 for missing metrics (iteration 13 fixed this in plotting/ but not evaluation/)
   - `plot_rank_histograms` uses `r.get('ssr', '?'):.3f` — `.3f` format on string fallback would crash (masked by ensemble-only filtering)
   - Per-dataset plots don't filter broken models, don't highlight best models, don't show RALSD in PSD legend
2. **Quality**: `_filter_broken` in `plotting/comprehensive.py` only applied to ERA5 (line 333), not NorESM. Asymmetric — if NorESM had a broken model, it would pass through unfiltered.
3. **Workflow**: Per-dataset plots (saved to pool alongside JSON) are generated by the stale evaluation plotting code, not the canonical `plotting/comprehensive.py`. Anyone looking at pool directory sees inferior, buggy plots.

### Work Done
- Removed 3 duplicate plotting functions (plot_psd_comparison, plot_rank_histograms, plot_metrics_summary) from evaluation/comprehensive.py — these had bugs (.get(metric, 0), .3f format on string fallback) and diverged from the canonical plotting/comprehensive.py
- Removed matplotlib imports from evaluation/comprehensive.py (now unused)
- Removed plot generation calls from both run_comprehensive_eval() and run_era5_eval()
- Fixed asymmetric _filter_broken in plotting/comprehensive.py: now filters both NorESM and ERA5 (was ERA5-only)
- Regenerated all 5 canonical figures — visually verified metrics_summary.png
- ruff clean, basedpyright 0 errors (2 warnings in unet.py — pre-existing)
- No dangling processes found

### Iteration 14 End
End commit: `669c5fd`
End time: ~15:25 EDT

## Iteration 15 — 2026-05-11 15:25 EDT
Start commit: `82d8fb2`

### Concerns Review
1. **Workflow**: Coverage was never verified in any prior iteration. Ran it: 98% (225 stmts, 5 miss). Uncovered lines are all defensive guards for degenerate edge cases (empty wavenumber bins, constant+non-allclose fields). Acceptable per testing policy.
2. **Quality**: `_filter_broken()` in plotting/comprehensive.py silently removes models with missing RALSD key (`"ralsd" in r` fails → model excluded from ALL plots). This violates error propagation rules — if RALSD computation fails for a model, the model disappears from plots with no warning. Should raise on missing key, filter only on value > threshold.
3. **Quality**: EVAL_REPORT.md NorESM table shows CRPS = MAE for all deterministic models (CNN-none 1.2976, CNN-softmax 1.7232, SwinIR 1.7252). This is mathematically correct (CRPS with M=1 = MAE) but readers will think it's a copy-paste error. Should add a footnote.

### Plan
Fix concern #2: harden `_filter_broken` to raise on missing RALSD. Add footnote for concern #3.

### Work Done
- Hardened `_filter_broken()` in plotting/comprehensive.py: now raises KeyError if any model is missing RALSD, instead of silently filtering it out. Value-based filtering (RALSD > 10) unchanged.
- Added footnote to EVAL_REPORT.md explaining CRPS = MAE equivalence for M=1 deterministic models
- Verified: 80 tests pass, ruff clean, basedpyright clean, all 5 plots regenerate correctly
- Coverage verified: 98% (225 stmts, 5 miss — all defensive guards)
- No dangling processes

### Iteration 15 End
End commit: `8df07b1`
End time: ~15:30 EDT
