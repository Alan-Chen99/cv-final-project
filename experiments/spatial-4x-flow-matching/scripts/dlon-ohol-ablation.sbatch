#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=dlon-ohol
#SBATCH --output=/home/chenxy/orcd/scratch/logs/dlon-ohol-ablation-%j.out
#SBATCH --chdir=/tmp

echo "=== Job started on $(hostname) at $(date) ==="

module load apptainer 2>&1
echo "Module loaded, rc=$?"

PROJECT_DIR=/home/chenxy/repos/workspace/research3
SIF=/home/chenxy/orcd/pool/cuda:13.0.2-cudnn-devel-ubuntu24.04.sif

echo "Launching singularity..."
singularity exec --nv \
    --cleanenv \
    --mount type=bind,source=/orcd,destination=/orcd \
    --mount type=bind,source=$PROJECT_DIR,destination=/workspace \
    --env PYTHONUNBUFFERED=1 \
    $SIF \
    bash -c 'cd /workspace && source .venv/bin/activate && \
    echo "Python: $(python --version)" && \
    echo "GPU: $(python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NO GPU\")")" && \
    python -u src/exp-spatial-4x-crps-v1/inference_ablation.py \
        --save_dir /home/chenxy/orcd/pool/datasets/research3/models/unet_uniform_amp \
        --basedir external/constrained-downscaling \
        --split test --max_samples 2000 --n_ensemble 10 --eval_batch_size 32 \
        --configs "euler_10_addcl,euler_10_smcl,euler_10_none,midpoint_5_addcl,midpoint_5_smcl,euler_20_addcl,euler_20_smcl,euler_10_addcl_tta,euler_10_smcl_tta,midpoint_5_smcl_tta"'

echo "=== Job finished at $(date), rc=$? ==="
