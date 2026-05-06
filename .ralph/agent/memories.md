# Memories

## Patterns

### mem-1778016803-d5f3
> GPU workflow: use gpu_run.py wrapper for running commands on GPU nodes. Allocate with salloc, run with gpu_run.py JOBID 'command'. Can't use srun --overlap while training srun step is active (step slot busy). Model checkpoint writes to NFS-mounted workspace.
<!-- tags: gpu, workflow, slurm | created: 2026-05-05 -->

## Decisions

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

### mem-1778016803-ba70
> CRPS formula discrepancy: flow_downscale.py uses Gneiting M^2 formula (standard energy CRPS), flow_matching_v2.py crps_ensemble_correct uses M*(M-1) (unbiased estimator). For M=10, difference is ~7%. Cross-comparison report mixed formulas. Use Gneiting M^2 consistently.
<!-- tags: crps, evaluation, metrics | created: 2026-05-05 -->

## Context
