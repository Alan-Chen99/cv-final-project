# Memories

## Patterns

## Decisions

### mem-1778084896-80d9
> Midpoint ODE solver (2nd-order RK) with 5 steps improves CRPS by 1.3% over Euler 10 steps for flow matching inference at same NFE cost (10 evaluations). Tested on UNet v2 with 10K test, 10 ensemble. midpoint_5_addcl CRPS=0.1709 vs euler_10_addcl CRPS=0.1731. More midpoint steps (10) don't help further. SmCL constraint produces NaN on residual models (exp overflow). TTA hurts when model not trained with augmentation.
<!-- tags: flow-matching, inference, ode-solver | created: 2026-05-06 -->

### mem-1778043144-d810
> Logit-normal timestep sampling (SD3-style) does NOT help flow matching for 32->128 TCW climate downscaling. CRPS 0.179 (26ep logit-normal) vs baseline 0.171 (39ep uniform). EMA decay=0.9999 HURTS with short training (26ep): CRPS 0.228. UNet v2 at 128x128 takes ~4.5 min/epoch on L40S (~9x slower than DiT with patch_size=8), limiting to ~26 epochs in 2hr budget.
<!-- tags: flow-matching, training-recipe, timestep-sampling | created: 2026-05-06 -->

### mem-1778029644-4342
> U-ViT (DiT + skip connections + conv refinement, 16.5M params, 200ep) tested for OT-CFM flow matching. CRPS 0.194 vs DiT 0.195 (marginal improvement). Skip connections improved val_loss (0.304 vs 0.301) but CRPS barely changed. Both transformer architectures (DiT, U-ViT) plateau at CRPS ~0.194-0.195, significantly worse than UNet v2 (0.171). Bottleneck is patch tokenization (patch_size=8), not information flow. Next: try smaller patch, hybrid conv-transformer, or return to UNet with improvements.
<!-- tags: uvit, architecture, flow-matching | created: 2026-05-06 -->

### mem-1778021458-3477
> DiT (Diffusion Transformer, patch_size=8, hidden=256, depth=12, 14.6M params) tested for OT-CFM flow matching on 32x32->128x128 TCW downscaling. Result: CRPS 0.195 (200ep) vs UNet 0.171 (research2, 39ep). Pure DiT is 14% worse than UNet due to lack of skip connections and multi-scale processing. However, DiT is competitive with simpler UNet (research branch: 0.199). Extended training helps DiT (40ep: 0.216 → 200ep: 0.195). Next: try U-ViT (DiT + skip connections).
<!-- tags: dit, architecture, flow-matching | created: 2026-05-05 -->

## Fixes

### mem-1778021451-be12
> GPU nodes cannot access /workspace directly. Must use singularity container via sbatch. The workflow: (1) use gpu_run.py wrapper or (2) write sbatch with 'module load apptainer' + 'singularity exec --nv --mount source=PROJECT_DIR,dest=/workspace SIF bash -c ...' where PROJECT_DIR=/home/chenxy/repos/workspace/research3 and SIF=/home/chenxy/orcd/pool/cuda:13.0.2-cudnn-devel-ubuntu24.04.sif. Outputs go to /home/chenxy/orcd/scratch/logs/. The mit_normal_gpu partition (6hr limit) often gets allocated faster than mit_preemptable (2-day limit, 363 pending jobs).
<!-- tags: gpu, slurm, singularity | created: 2026-05-05 -->

## Context
