#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -J gcgi-vxgh-train
#SBATCH -G 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o /home/chenxy/orcd/scratch/logs/train-spectral-%j.log
#SBATCH -e /home/chenxy/orcd/scratch/logs/train-spectral-%j.log

set -euo pipefail

PROJECT_DIR='/home/chenxy/repos/workspace/research4'
SIF='/home/chenxy/orcd/pool/cuda:13.0.2-cudnn-devel-ubuntu24.04.sif'

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

module load apptainer

# Train with spectral loss + augmentation
echo ""
echo "=== Training: spectral loss (0.1) + augmentation ==="
echo "Start: $(date)"

singularity exec --nv \
  --cleanenv \
  --mount 'type=bind,source=/orcd,destination=/orcd' \
  --mount 'type=bind,source=/home/chenxy/nix_store,destination=/nix,ro' \
  --mount "type=bind,source=$PROJECT_DIR,destination=/workspace" \
  --env PREPEND_PATH=/nix/state/profile/bin:/nix/nix_path/bin \
  --env NIX_REMOTE=daemon \
  --env BASH_ENV="$HOME/.bashrc" \
  "$SIF" \
  bash -c 'cd /workspace && source .venv/bin/activate && python src/exp-spatial-4x-crps-v1/unet_cfg_flow.py \
    --mode train \
    --epochs 40 \
    --batch_size 64 \
    --lr 1e-4 \
    --cfg_prob 0 \
    --base_channels 64 \
    --channel_mults 1,2,4 \
    --attn_heads 4 \
    --augment \
    --spectral_weight 0.1 \
    --save_dir models/unet_spectral'

echo "Training finished at $(date)"

# Evaluate on 1K test first (quick sanity check)
echo ""
echo "=== Eval: 1K test, Euler 10, AddCL ==="
echo "Start: $(date)"

singularity exec --nv \
  --cleanenv \
  --mount 'type=bind,source=/orcd,destination=/orcd' \
  --mount 'type=bind,source=/home/chenxy/nix_store,destination=/nix,ro' \
  --mount "type=bind,source=$PROJECT_DIR,destination=/workspace" \
  --env PREPEND_PATH=/nix/state/profile/bin:/nix/nix_path/bin \
  --env NIX_REMOTE=daemon \
  --env BASH_ENV="$HOME/.bashrc" \
  "$SIF" \
  bash -c 'cd /workspace && source .venv/bin/activate && python src/exp-spatial-4x-crps-v1/unet_cfg_flow.py \
    --mode eval \
    --save_dir models/unet_spectral \
    --n_ensemble 10 \
    --ode_steps 10 \
    --constraint addcl \
    --guidance_scale 1.0 \
    --max_samples 1000 \
    --split test'

echo "1K eval finished at $(date)"

# Full 10K eval
echo ""
echo "=== Eval: 10K test, Euler 10, AddCL ==="
echo "Start: $(date)"

singularity exec --nv \
  --cleanenv \
  --mount 'type=bind,source=/orcd,destination=/orcd' \
  --mount 'type=bind,source=/home/chenxy/nix_store,destination=/nix,ro' \
  --mount "type=bind,source=$PROJECT_DIR,destination=/workspace" \
  --env PREPEND_PATH=/nix/state/profile/bin:/nix/nix_path/bin \
  --env NIX_REMOTE=daemon \
  --env BASH_ENV="$HOME/.bashrc" \
  "$SIF" \
  bash -c 'cd /workspace && source .venv/bin/activate && python src/exp-spatial-4x-crps-v1/unet_cfg_flow.py \
    --mode eval \
    --save_dir models/unet_spectral \
    --n_ensemble 10 \
    --ode_steps 10 \
    --constraint addcl \
    --guidance_scale 1.0 \
    --split test'

echo "10K eval finished at $(date)"

echo ""
echo "Job $SLURM_JOB_ID finished at $(date)"
