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
