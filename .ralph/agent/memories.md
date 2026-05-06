# Memories

## Patterns
- SwinIR zero-shot (pretrained on DF2K natural images) gets MAE=0.317 on ERA5 TCW 4x downscaling — 17% better than bicubic. Pretrained image SR models transfer to climate data.
- SwinIR requires channel adaptation for 1ch climate data: modify conv_first/conv_last AND set model.mean to (1,1,1,1) to avoid channel broadcasting.
- SLURM node4104 has CUDA ECC error — always exclude from GPU allocations.

## Context
- ERA5 SR data in pool: input [40K,1,1,32,32], target [40K,1,1,128,128], TCW range [0.04, 131] kg/m²
- Both eval scripts in repo (eval_crps.py, compute_metrics.py) have buggy CRPS formula. Always use energy CRPS: E|X-y| - 0.5*E|X-X'|
- Prior cross-comparison report MAE/RMSE numbers may use different data normalization — internal consistency is key
