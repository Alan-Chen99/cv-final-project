# Memories

## Patterns

### mem-1778016803-d5f3
> GPU workflow: use gpu_run.py wrapper for running commands on GPU nodes. Allocate with salloc, run with gpu_run.py JOBID 'command'. Can't use srun --overlap while training srun step is active (step slot busy). Model checkpoint writes to NFS-mounted workspace.
<!-- tags: gpu, workflow, slurm | created: 2026-05-05 -->

## Decisions

### mem-1778096424-d7fd
> Spectral loss (FFT L1 on reconstructed x1, weight=0.1) + data augmentation (random H/V flips) on OT-CFM flow matching: CRPS=0.2036 on full 10K test (Gneiting M^2, AddCL). 9.2% WORSE than MSE-only baseline (0.1865). Spectral loss fails because x1_pred = x_t + (1-t)*v_pred is noisy at small t, making FFT targets meaningless. Data augmentation was not isolated.
<!-- tags: spectral-loss, crps, flow-matching, augmentation | created: 2026-05-06 -->

### mem-1778080603-3a5c
> DDPM VP-SDE full 10K eval: CRPS=0.1907 (Gneiting M^2, AddCL, stochastic DDIM 20 steps eta=1.0). Confirmed 2.3% worse than OT-CFM flow matching (0.1865). 1K→10K gap: 0.1877→0.1907 (+1.6%). SmCL cannot be applied post-hoc (exp overflow on residual predictions). salloc+srun is unreliable from container — use sbatch for all GPU work.
<!-- tags: ddpm, crps, evaluation, flow-matching | created: 2026-05-06 -->

### mem-1778068979-793a
> DDPM (VP-SDE) score-based diffusion with same 13M AttentionUNet: CRPS=0.1877 on 1K test (stochastic DDIM 20 steps, eta=1.0, AddCL). 2% worse than OT-CFM flow matching (0.184 on 1K). EMA decay=0.9999 is harmful for short training runs (CRPS=0.263). Full 10K eval not completed (hung srun). DDPM underperforms because OT-CFM learns straight paths needing fewer sampling steps.
<!-- tags: ddpm, crps, flow-matching, architecture | created: 2026-05-06 -->

### mem-1778053249-59c1
> Full 10K eval results for UNet 55ep flow matching: Euler 10 steps CRPS=0.1865, Heun 10 steps CRPS=0.1885. Higher-order/more steps don't help: OT-CFM trains with straight-line interpolation so Euler is the matched solver. AddCL constraint, Gneiting M^2 formula. This is the definitive number for UNet flow matching on research4.
<!-- tags: crps, evaluation, flow-matching, solver | created: 2026-05-06 -->

### mem-1778041528-e693
> UNet 55-epoch continued training: resumed from iter2's 25-epoch model with fresh cosine LR schedule (finetune_lr=5e-5, T_max=34). cfg_prob=0 (CFG disabled). Val loss 0.272→0.251 (7.7% improvement). CRPS=0.184 on 1K test (Gneiting M^2), vs iter2's 0.193 (5% improvement). Now competitive with research2's estimated ~0.183. Full 10K eval pending.
<!-- tags: crps, training, flow-matching | created: 2026-05-06 -->

### mem-1778030992-f8c3
> UNet CFG flow matching: 13M AttentionUNet with 10% condition dropout (CFG), 25 epochs, CRPS=0.196 (Gneiting M^2) on full 10K test. CFG guidance does NOT help for climate downscaling — guidance_scale=1.0 is optimal, higher guidance hurts CRPS. LR conditioning is already strong/unambiguous unlike text-to-image. Model weights in pool at research4/models/unet_cfg_best.pt.
<!-- tags: cfg, crps, architecture, flow-matching | created: 2026-05-06 -->

### mem-1778016803-875c
> DiT (Diffusion Transformer) flow matching with 14.6M params, 40 epochs: corrected CRPS=0.243 on full 10K test. Significantly worse than UNet flow matching (0.199 for 5.2M LR-anchor, ~0.183 for 13M OT-CFM). Val loss 0.381 vs UNet 0.253. DiT lacks local inductive bias valuable for climate fields. May need more training, smaller patch size, or positional encoding improvements.
<!-- tags: dit, flow-matching, crps, architecture | created: 2026-05-05 -->

## Fixes

### mem-1778080609-8eec
> salloc+srun (via gpu_run.py) is unreliable for long-running commands from within Apptainer container on node1627. Quick commands work but eval/training srun steps hang silently. sbatch works reliably because singularity exec runs directly on the GPU node. Always use sbatch for eval and training jobs.
<!-- tags: gpu, workflow, slurm | created: 2026-05-06 -->

### mem-1778016803-ba70
> CRPS formula discrepancy: flow_downscale.py uses Gneiting M^2 formula (standard energy CRPS), flow_matching_v2.py crps_ensemble_correct uses M*(M-1) (unbiased estimator). For M=10, difference is ~7%. Cross-comparison report mixed formulas. Use Gneiting M^2 consistently.
<!-- tags: crps, evaluation, metrics | created: 2026-05-05 -->

## Context
