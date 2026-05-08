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

**End:** 2026-05-08 19:XX EDT | commit: TBD (about to commit)
