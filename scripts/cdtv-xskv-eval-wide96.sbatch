#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=cdtv-xskv-eval
#SBATCH --output=/home/chenxy/orcd/scratch/logs/cdtv-xskv-eval-wide96-%j.out
#SBATCH --chdir=/tmp

echo "=== Job started on $(hostname) at $(date) ==="
module load apptainer 2>&1

PROJECT_DIR=/home/chenxy/repos/workspace/research3
SIF=/home/chenxy/orcd/pool/cuda:13.0.2-cudnn-devel-ubuntu24.04.sif

# Evaluate wide96 model (28.4M params) on full 10K test
# Compare midpoint_5_addcl (best solver from iter-6) and euler_10_addcl (baseline)
singularity exec --nv \
    --cleanenv \
    --mount type=bind,source=/orcd,destination=/orcd \
    --mount type=bind,source=$PROJECT_DIR,destination=/workspace \
    --env PYTHONUNBUFFERED=1 \
    $SIF \
    bash -c 'cd /workspace && source .venv/bin/activate && \
    python -u src/exp-spatial-4x-crps-v1/inference_ablation.py \
        --save_dir /home/chenxy/orcd/pool/datasets/research3/models/unet_wide96_amp \
        --basedir external/constrained-downscaling \
        --split test --n_ensemble 10 --eval_batch_size 32 \
        --configs "midpoint_5_addcl,euler_10_addcl"'

echo "=== Job finished at $(date), rc=$? ==="
