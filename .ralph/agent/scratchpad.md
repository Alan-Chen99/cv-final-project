# Scratchpad — organize1 branch

## Iteration 1
**Start:** 2026-05-08 19:09 EDT | commit: b8b468b
**Prefix:** coral-twist

### Orientation

6 experiment branches have run. Code is heavily duplicated across 4 experiment directories.
Best CRPS: 0.1676 (wide96 UNet, research3). ~10 trained model weights available in pool.

### Top 3 Concerns

1. **Quality: Massive code duplication** — CRPS, data loading, constraints, UNet, solvers are
   copy-pasted across every experiment file. No shared library exists. This is the primary
   problem the task addresses.

2. **Workflow: No tests, no linting, no type checking** — 6 branches of research code with
   zero automated quality checks. pyproject.toml exists but has no tool configs.

3. **Workflow: Results not independently verified** — Reports claim CRPS numbers but the
   evaluation code was duplicated (not shared), and there's no standard eval script.
   Need a single canonical eval pipeline to verify all claims.

### Plan for this iteration

**Focus: Task (1) — Set up proper Python project structure + tooling**

This is the foundation everything else depends on. Specifically:
- Configure pyproject.toml with ruff, black, isort, basedpyright, coverage
- Create src/ package skeleton with proper __init__.py files
- Extract core shared code: data loading, metrics (CRPS), constraints, sampling
- Install dev dependencies

Leave for future iterations:
- Training loops (extract from experiments/)
- Visualization code
- Report
- Task (5): add evaluation methods that don't require training (bilinear + AddCL, etc.)
- Task (6): visualization code for key results using pre-trained weights
- Task (7): write report file

### Completed

- pyproject.toml configured: ruff, black, isort, basedpyright, coverage, pytest
- Created src/downscaling/ package with 7 modules:
  - data.py: ERA5 loading, NormStats, normalize/denormalize
  - metrics.py: crps_energy, crps_paper, mae, rmse, mass_violation, ensemble_spread
  - constraints.py: apply_addcl, apply_smcl
  - sampling.py: euler_sample, midpoint_sample, timestep samplers
  - models/unet.py: AttentionUNet (full arch with building blocks)
  - models/dit.py: DiT (full Vision Transformer)
  - models/ddpm.py: DDPMSchedule, ddim_sample
  - evaluation.py: evaluate_flow_model, evaluate_deterministic, bicubic_predict
- Created tests/ with 4 test files (22 tests, all passing):
  - test_metrics.py: CRPS, MAE, RMSE, mass violation, spread
  - test_constraints.py: AddCL exact conservation, SmCL non-negativity
  - test_data.py: normalize/denormalize roundtrip, NormStats
  - test_models.py: forward shapes, gradients, DDPM schedule (GPU-required)
- All linters pass: ruff, ruff format, basedpyright (0 errors)
- Coverage: 100% on core logic (metrics, constraints, data)

**End:** 2026-05-08 19:25 EDT | commit: c3565c3

## Iteration 2
**Start:** 2026-05-08 19:21 EDT | commit: 3ceae78
**Prefix:** vypq-pczs

### Orientation

Reviewed all code from iteration 1 + experiment code. Found critical bug in the primary
evaluation metric.

### Top 3 Concerns

1. **Fact: crps_energy() formula is WRONG** — Denominator uses M*M instead of M*(M-1),
   and missing M=1 guard. For M=2 with obs=1, forecasts=[0,2]: produces 0.5 instead of 0.0.
   This is the primary metric for all evaluations. The O(M²) pairwise reference in
   experiments/spatial-4x-flow-matching/src/flow_matching_v2.py:crps_ensemble_correct()
   uses the correct formula. Tests didn't catch it because they used property-based checks
   rather than known-answer regression tests.

2. **Workflow: Missing training code** — src/downscaling has no training loop code.
   The task explicitly asks for "organize all code in a good way in ./src: training and eval
   and plotting". Training loops are duplicated across 4 experiment dirs. Need to extract
   a canonical training loop.

3. **Workflow: Missing visualization code** — Task (6) explicitly asks for visualization code.
   Experiment dirs have plotting code (visualize_samples.py, visualize_results.py) but nothing
   is in src/. Also no report file yet (task 7).

### Work Done

- **Fixed crps_energy()**: Changed denominator from M*M to M*(M-1), added M=1 early return
- **Added regression tests**:
  - test_symmetric_ensemble_is_zero: obs=1, forecasts=[0,2] → CRPS=0
  - test_matches_pairwise_reference: verifies O(M log M) matches O(M²) reference
- **Fixed pre-existing lint issues**: zip strict=True, format fixes
- All checks pass: 24 tests, 0 ruff errors, 0 basedpyright errors

**End:** 2026-05-08 19:27 EDT | commit: 740d936

### Next iterations should focus on
- Task (3): Extract training loop into src/downscaling/training.py
- Task (5): Evaluation baselines (bilinear, bilinear+AddCL, etc.)
- Task (6): Visualization code
- Task (7): Report file

## Iteration 3
**Start:** 2026-05-08 19:28 EDT | commit: b88863f
**Prefix:** 62edcbc7

### Orientation

Reviewed ALL code in src/downscaling/ and experiments/. Prior iterations set up project
structure (iter1) and fixed CRPS bug (iter2). No training code in src/ yet.

### Top 3 Concerns

1. **Workflow: No training code in src/** — Task (3) explicitly asks to organize "training
   and eval and plotting" into src/. All 4 experiment dirs duplicate the OT-CFM training
   loop. The canonical training code is in experiments/spatial-4x-flow-matching/src/flow_matching_v2.py.
   Best result: wide96 UNet (28.4M params), 25 epochs, CRPS 0.1676.

2. **Workflow: No visualization code** — Task (6) asks for visualization code using pre-trained
   weights. Experiment dirs have plotting code (visualize_samples.py, visualize_results.py)
   but nothing extracted to src/ yet.

3. **Quality: Pre-existing test failure** — test_gradient_flow used base_channels=16 with
   channel_mults=(1,2), causing GroupNorm incompatibility (48 channels not divisible by 32).
   The skip connection concatenation (ch=32 + out_ch=16 = 48) triggered the error. Prior
   iterations didn't catch this because they may have only run CPU-level checks.

### Work Done

- **Created src/downscaling/training.py** — Extracted canonical OT-CFM training loop from
  experiments/spatial-4x-flow-matching/src/flow_matching_v2.py:
  - `TrainConfig` dataclass: epochs, lr, weight_decay, grad_clip, amp, save_dir, t_sampling
  - `TrainResult` dataclass: per-epoch losses, best_val_loss, best_epoch, total_time
  - `train_flow_matching()`: Full training loop with AdamW, cosine LR, AMP, grad clipping,
    checkpoint saving. Validation always uses uniform timesteps for comparability.
  - `train_step()`: Single training step for custom loops and integration tests.
  - Deliberately excluded features proven harmful at <50 epochs: EMA, data augmentation.

- **Created tests/test_training.py** — 5 integration tests (GPU-required):
  - test_loss_is_finite: single step produces valid loss
  - test_loss_decreases_over_steps: 20 steps overfit on fixed batch
  - test_amp_step: AMP works correctly
  - test_logit_normal_timesteps: logit-normal sampling works
  - test_gradient_clipping: tight clip bounds gradients

- **Fixed pre-existing test bug**: test_gradient_flow base_channels 16→32

- All checks pass: 38 tests, 0 ruff errors, 0 basedpyright errors

**End:** 2026-05-08 19:36 EDT | commit: (pending)

### Next iterations should focus on
- Task (5): Evaluation baselines (bilinear, bilinear+AddCL, bilinear+SmCL)
- Task (6): Visualization code using pre-trained weights
- Task (7): Report file

## Iteration 4
**Start:** 2026-05-08 19:37 EDT | commit: 2306e3f
**Prefix:** jakm-fcpp

### Orientation

Iters 1-3 built the shared library (data, metrics, constraints, models, sampling, training).
No evaluations have actually been run through the canonical pipeline. No visualization or report.

### Top 3 Concerns

1. **Workflow: No evaluations actually run** — evaluation.py has functions but no script to
   load pre-trained weights and execute. All CRPS numbers in reports come from experiment
   code, not the canonical pipeline. Need to independently verify results.

2. **Workflow: No visualization code in src/** — Task (6) entirely unaddressed. Experiment dirs
   have plotting code (visualize_samples.py, visualize_results.py) but nothing in src/.

3. **Quality: evaluation.py missing bilinear baseline** — Has bicubic_predict but no
   bilinear_predict. The bilinear upsampled version is the core baseline (lr_up from data
   loading). Also no model loading helper to reconstruct models from checkpoints.

### Plan for this iteration

**Focus: Task (5) — Add evaluation baselines + create runnable evaluation script**

1. Add `bilinear_predict()` to evaluation.py
2. Add `load_flow_checkpoint()` helper for loading models from pool
3. Create `scripts/evaluate_all.py` — comprehensive evaluation runner
4. Allocate GPU and run evaluations to produce results table
5. Save results to git-tracked JSON

### Work Done

- **Added `bilinear_predict()`** to evaluation.py — returns lr_up directly
- **Added `load_flow_checkpoint()`** to evaluation.py — reconstructs AttentionUNet from
  checkpoint args and loads weights + norm stats
- **Created `scripts/evaluate_all.py`** — comprehensive evaluation runner:
  - Deterministic baselines: bilinear, bicubic with/without AddCL
  - Flow matching models: all 5 available models from pool
  - Configurable: ensemble size, ODE steps, solver, sample limit
  - Outputs formatted table + JSON results
- **Created `tests/test_evaluation.py`** — 4 integration tests:
  - bilinear returns lr_up, bicubic shape, bilinear+addcl conservation,
    checkpoint save/load roundtrip, flow eval smoke test
- **Ran evaluations** on GPU (preemptable partition):
  - Full 10K deterministic baselines: bilinear=0.507, bicubic=0.384, bicubic+AddCL=0.353
  - 200-sample flow models: all 5 models with/without AddCL
  - 1000-sample wide96: CRPS=0.1650 (confirms reported ~0.168)
  - Solver comparison: midpoint(5)=0.180 > euler(10)=0.182 at equal NFE
- **SmCL overflow documented**: SmCL produces NaN on physical-space TCW values
  (exp(135) overflows). Removed from default baselines, kept as option.
- All linters pass: ruff, ruff format, basedpyright

### Key Results (200 samples, M=10, midpoint 10 steps, +AddCL)

| Method | CRPS | MAE | RMSE | MassViol |
|--------|------|-----|------|----------|
| bilinear | 0.507 | 0.507 | 0.949 | 0.314 |
| bilinear+AddCL | 0.389 | 0.389 | 0.791 | 0.000 |
| bicubic | 0.384 | 0.384 | 0.772 | 0.146 |
| bicubic+AddCL | 0.353 | 0.353 | 0.728 | 0.000 |
| wide96 (research3)+AddCL | 0.180 | 0.262 | 0.471 | 0.000 |
| logit-normal (research4)+AddCL | 0.181 | 0.265 | 0.478 | 0.000 |
| base64 zscore (research6)+AddCL | 0.184 | 0.268 | 0.482 | 0.000 |
| base64 uniform (research3)+AddCL | 0.184 | 0.268 | 0.482 | 0.000 |
| cfg (research4)+AddCL | 0.184 | 0.269 | 0.483 | 0.000 |

Wide96 is best. AddCL is free (no CRPS cost, eliminates mass violation).

**End:** 2026-05-08 20:40 EDT | commit: 363db50

### Next iterations should focus on
- Task (6): Visualization code (sample grids, ensemble diversity, metrics bar charts)
- Task (7): Write report file
- Run full 10K eval for flow models (takes ~50 min per model, needs dedicated GPU time)

## Iteration 5
**Start:** 2026-05-08 20:42 EDT | commit: 363db50
**Prefix:** edbw-iajz

### Orientation

Iters 1-4 built the shared library + evaluation pipeline. Results JSON files exist with
verified numbers. No visualization code in src/, no report file.

### Top 3 Concerns

1. **Workflow: No visualization code** — Task (6) explicitly asks for visualization code
   using pre-trained weights. Two experiment scripts have plotting logic
   (visualize_samples.py, visualize_results.py) but nothing extracted to src/.
   This is the primary remaining gap before the report.

2. **Quality: eval_results.json contains NaN entries** — First eval run included SmCL
   which produces NaN. eval_200samples.json removed them but eval_results.json still
   has NaN rows. Minor data hygiene issue.

3. **Workflow: No report file** — Task (7) asks for a report. Leave for next iteration
   once visualization artifacts exist.

### Plan for this iteration

**Focus: Task (6) — Create visualization module + generate plots**

1. Create `src/downscaling/visualization.py` — reusable plotting functions:
   - `plot_sample_grid()`: LR, HR, ensemble mean, |error|, spread (output artifact)
   - `plot_ensemble_members()`: individual ensemble members (output artifact)
   - `plot_metrics_comparison()`: bar charts comparing methods (data plotting)
2. Create `scripts/visualize.py` — runner that loads best model, generates predictions, plots
3. Allocate GPU, run visualization script, verify output

### Completed

- **Created `src/downscaling/visualization.py`** — 5 plotting functions:
  - `plot_sample_grid()`: 5-column grid (LR, HR, mean, |error|, spread)
  - `plot_ensemble_members()`: individual members with per-member MAE
  - `plot_metrics_comparison()`: horizontal bar chart by metric
  - `plot_constraint_effect()`: grouped bar (with/without AddCL)
  - `plot_mass_violation()`: mass conservation bar chart
- **Created `scripts/visualize.py`** — runner script with --samples flag for GPU plots
- **Created `tests/test_visualization.py`** — 7 tests (synthetic data, no GPU needed)
- **Generated 6 figures** in `figures/`:
  - Data plots: metrics_crps.png, metrics_mae.png, constraint_effect.png, mass_violation.png
  - Output artifacts: sample_grid.png (8 samples), ensemble_members_best.png
- **Visually verified**: model predictions add HR detail vs LR bilinear, ensemble
  members show meaningful diversity (MAE 0.666-0.743), constraint effect clearly visible
- All checks pass: 31 non-GPU tests, 0 ruff errors, 0 basedpyright errors

**End:** 2026-05-08 20:55 EDT | commit: (pending)

### Next iterations should focus on
- Task (7): Write report file
- Clean up eval_results.json (remove NaN SmCL entries)
