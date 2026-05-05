# Memories

## Patterns

### mem-1778016803-d5f3
> GPU workflow: use gpu_run.py wrapper for running commands on GPU nodes. Allocate with salloc, run with gpu_run.py JOBID 'command'. Can't use srun --overlap while training srun step is active (step slot busy). Model checkpoint writes to NFS-mounted workspace.
<!-- tags: gpu, workflow, slurm | created: 2026-05-05 -->

## Decisions

### mem-1778016803-875c
> DiT (Diffusion Transformer) flow matching with 14.6M params, 40 epochs: corrected CRPS=0.243 on full 10K test. Significantly worse than UNet flow matching (0.199 for 5.2M LR-anchor, ~0.183 for 13M OT-CFM). Val loss 0.381 vs UNet 0.253. DiT lacks local inductive bias valuable for climate fields. May need more training, smaller patch size, or positional encoding improvements.
<!-- tags: dit, flow-matching, crps, architecture | created: 2026-05-05 -->

## Fixes

### mem-1778016803-ba70
> CRPS formula discrepancy: flow_downscale.py uses Gneiting M^2 formula (standard energy CRPS), flow_matching_v2.py crps_ensemble_correct uses M*(M-1) (unbiased estimator). For M=10, difference is ~7%. Cross-comparison report mixed formulas. Use Gneiting M^2 consistently.
<!-- tags: crps, evaluation, metrics | created: 2026-05-05 -->

## Context
