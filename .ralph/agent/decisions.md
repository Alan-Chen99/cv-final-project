# Decision Journal

## DEC-001
- **Decision**: Package naming and structure
- **Chosen Option**: `src/downscaling/` with subpackages (models, data, metrics, constraints, sampling, training, evaluation, plotting)
- **Confidence**: 85
- **Alternatives Considered**:
  - Flat `src/` as package: Too generic, collides with common Python convention
  - `src/climate_downscaling/`: Verbose; `downscaling` is sufficient in project context
  - Monolithic single module: Doesn't scale for different experiment methods
- **Reasoning**: Standard Python src layout. Short import paths (`from downscaling.models import AttentionUNet`). Subpackages mirror the conceptual decomposition (models, metrics, constraints are independent concerns).
- **When to re-evaluate**: If the package is published externally (may need more specific name)
- **Independent evaluation**: not-started
- **Framing Biases**: None identified — follows standard Python packaging conventions
- **Timestamp**: 2026-05-08T23:30:00Z

## DEC-002
- **Decision**: Which CRPS implementation to use as primary
- **Chosen Option**: Energy CRPS (E|X-y| - 0.5*E|X-X'|) as `crps_energy()`, paper version as `crps_paper()` with bug fix
- **Confidence**: 95
- **Alternatives Considered**:
  - Only energy CRPS: Would lose comparability with prior published results
  - Only paper CRPS with bug: Would propagate known error
- **Reasoning**: Energy CRPS is the mathematically correct formula (Gneiting & Raftery 2007). Paper version is included for backward compatibility but with the fc.shape[-1]**2 bug fixed. Both are clearly named to avoid confusion.
- **When to re-evaluate**: Never — this is the standard correct formula
- **Independent evaluation**: not-started
- **Framing Biases**: None
- **Timestamp**: 2026-05-08T23:30:00Z

## DEC-003
- **Decision**: Which model code to use as canonical source
- **Chosen Option**: `flow_matching_v2.py` from spatial-4x-flow-matching (research3 branch) as canonical for UNet, constraints, ODE solvers, and CRPS
- **Confidence**: 90
- **Alternatives Considered**:
  - Merge from multiple experiment files: Risk of introducing inconsistencies
  - Use pretrained-sr-downscaling code: Has more features but also more coupling
- **Reasoning**: This file produced the best result (CRPS 0.1676 with wide96 config). All other experiment files derive from it. Clean, self-contained implementations of all core components.
- **When to re-evaluate**: If a new architecture outperforms UNet
- **Independent evaluation**: not-started
- **Framing Biases**: Selecting by best result may miss code quality issues in the source
- **Timestamp**: 2026-05-08T23:30:00Z
