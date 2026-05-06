#!/bin/bash
# Inference ablation: test solver, constraint, steps, TTA on best model
# Run on GPU node via: srun --jobid=JOBID singularity exec ...

set -e

MODEL_DIR="/home/chenxy/orcd/pool/datasets/research3/models/unet_uniform_amp"
BASEDIR="external/constrained-downscaling"

# First: quick ablation on 2K samples to identify best settings
echo "=== Phase 1: Quick ablation (2K samples) ==="
python src/exp-spatial-4x-crps-v1/inference_ablation.py \
    --save_dir "$MODEL_DIR" \
    --basedir "$BASEDIR" \
    --split test \
    --max_samples 2000 \
    --n_ensemble 10 \
    --eval_batch_size 32 \
    --configs "euler_10_addcl,euler_10_smcl,euler_10_none,midpoint_5_addcl,midpoint_5_smcl,euler_20_addcl,euler_20_smcl,euler_10_addcl_tta,euler_10_smcl_tta,midpoint_5_smcl_tta"

echo ""
echo "=== Phase 2: Full 10K eval on top 3 configs ==="
# Will run the best configs on full 10K after seeing Phase 1 results
# (Manually select top configs based on Phase 1)
