# Scratchpad — organize2

## Iteration 1
**Start**: 2026-05-08 19:13 EDT | **Commit**: b8b468b
**Prefix**: organize2-2047

### Orientation
First iteration. No prior work on this branch — fresh start.
Branch has experiments/ (4 dirs from research3-6) and external/ (3 baseline codebases).
No src/, no tests, no dev tools configured.

SLURM: 2 preemptable jobs running (not ours). 0 normal. Can use either partition.

### Top Concerns

1. **Workflow (highest priority)**: No project structure exists. Must create src/ with proper
   packaging before any code organization or evaluation can happen. Tool configuration
   (ruff, basedpyright, etc.) needs to be set up first.

2. **Quality**: Experiment code has massive duplication across 4 directories. Each experiment
   re-implements CRPS, constraint layers, data loading, models from scratch. Need to extract
   canonical versions carefully — the "best" version of each component is scattered.

3. **Fact verification**: Pool weights verified to exist:
   - research3/models/unet_wide96_amp/ (best: CRPS 0.1676)
   - research4/models/ (DDPM, DiT, UNet-CFG, logit-normal variants)
   - research5/models/ (SwinIR finetune, CRPS ensemble, noise-SwinIR, residual flow, swinir flow)
   - research5/pretrained_weights/ (SwinIR classical SR x4)
   - research6/models/ (flow_v2_zscore, flow_v2, swinir_finetune, etc.)

### Plan for this iteration
**DO ONE THING**: Set up project structure (tasks 1+2 from prompt)
- Configure pyproject.toml with dev deps and tool settings
- Create src/ package skeleton
- Add initial __init__.py files
- Configure ruff, basedpyright, black, isort, coverage
- Document src/ vs experiments/ pattern

### Iteration 1 Progress
**Completed tasks 1+2+3 (partial)**:
- pyproject.toml: renamed to climate-downscaling, added dev deps (basedpyright, ruff, black, isort, coverage, pytest, pytest-cov), configured all tools
- Created src/downscaling/ with 8 subpackages: models, data, metrics, constraints, sampling, training, evaluation, plotting
- Extracted canonical code from experiments into organized modules:
  - models/unet.py: AttentionUNet with all building blocks (from flow_matching_v2.py)
  - metrics/crps.py: Both energy CRPS (correct) and paper CRPS (bug-fixed)
  - constraints/layers.py: AddCL and SmCL
  - sampling/ode.py: Euler and Midpoint ODE solvers
  - sampling/timesteps.py: Uniform and logit-normal sampling
  - data/era5.py: ERA5 TCW 4x data loader
  - training/ema.py: EMA helper
  - training/flow_matching.py: Full FlowMatchingTrainer with TrainConfig
  - evaluation/evaluate.py: evaluate_ensemble() and evaluate_flow_model()
- All imports verified working
- Ruff lint: clean, ruff format: clean, basedpyright: 0 errors, 2 warnings (PyTorch stubs)
- Updated CLAUDE.md with directory structure and src/ vs experiments/ documentation
- Updated .gitignore for coverage artifacts

### Still needed (future iterations)
- Task 4: Write integration tests (100% coverage on core logic)
- Task 5: Additional evaluation methods (bicubic baseline, SwinIR zero-shot, etc.)
- Task 6: Visualization code for key results
- Task 7: Report file

**End**: 2026-05-08 19:23 EDT | **Ending commit**: a230ceb

## Iteration 2
**Start**: 2026-05-08 19:24 EDT | **Commit**: b591dfb
**Prefix**: dram-boggle

### Orientation
Iteration 1 completed project structure (tasks 1-3). All source code reviewed. No ready tasks.

### Top Concerns

1. **Workflow (highest priority)**: No tests exist at all. This is the most critical gap — the code
   was "verified" by the prior agent only by reading, not by actually running it. Must write and
   run integration tests to verify correctness.

2. **Quality**: `evaluate_flow_model()` hardcodes `pool = nn.AvgPool2d(kernel_size=4)` on line 106,
   ignoring any upsampling_factor parameter. Minor since we always use 4x, but inconsistent with
   the parameterized versions in constraint layers and evaluate_ensemble.

3. **Quality**: data loader `era5.py` uses `inp[:, 0, :, :, :]` which assumes a specific temporal
   dimension in the data. This was likely correct for the experiment code but needs verification
   against actual pool data.

### Plan for this iteration
**DO ONE THING**: Write integration tests (task 4 from prompt)
- Test metrics (CRPS correctness with known analytical results)
- Test constraints (conservation enforcement)
- Test full pipeline (model → sampling → evaluation)
- Test training (2 epochs with synthetic data)
- Run tests, verify coverage ≥90%
