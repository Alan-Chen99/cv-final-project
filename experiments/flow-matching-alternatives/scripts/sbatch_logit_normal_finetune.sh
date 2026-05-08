#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -J zbhh-axxo
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o /home/chenxy/orcd/scratch/logs/logit-normal-ft-%j.log
#SBATCH -e /home/chenxy/orcd/scratch/logs/logit-normal-ft-%j.log

set -euo pipefail

PROJECT_DIR='/home/chenxy/repos/workspace/research4'
SIF='/home/chenxy/orcd/pool/cuda:13.0.2-cudnn-devel-ubuntu24.04.sif'

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

module load apptainer

run_cmd() {
    local NAME="$1"
    shift
    echo ""
    echo "=== $NAME ==="
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
      bash -c "cd /workspace && source .venv/bin/activate && $*"
    echo "End: $(date)"
}

# --- SETUP: copy best model for resume ---
echo "=== SETUP: copying best 55ep model ==="
mkdir -p "$PROJECT_DIR/models/unet_logit_normal"
cp "$PROJECT_DIR/models/unet_cfg/best_flow.pt" "$PROJECT_DIR/models/unet_logit_normal/best_flow.pt"
cp "$PROJECT_DIR/models/unet_cfg/norm_stats.pt" "$PROJECT_DIR/models/unet_logit_normal/norm_stats.pt"
echo "Copied best model (epoch 51, val_loss=0.251)"

# --- TRAINING: fine-tune 15 epochs with logit-normal t ---
run_cmd "Fine-tune 15ep logit-normal" \
    python src/exp-spatial-4x-crps-v1/unet_cfg_flow.py \
    --mode train \
    --epochs 67 \
    --batch_size 64 \
    --cfg_prob 0 \
    --resume \
    --finetune_lr 5e-5 \
    --t_schedule logit_normal \
    --logit_normal_mean 0.0 \
    --logit_normal_std 1.0 \
    --save_dir models/unet_logit_normal

echo "Training complete"

# Copy model to pool
POOL_DIR="/home/chenxy/orcd/pool/datasets/research4/models"
mkdir -p "$POOL_DIR"
cp "$PROJECT_DIR/models/unet_logit_normal/best_flow.pt" "$POOL_DIR/unet_logit_normal_best.pt"
cp "$PROJECT_DIR/models/unet_logit_normal/norm_stats.pt" "$POOL_DIR/unet_logit_normal_norm_stats.pt"
echo "Model copied to pool"

# --- EVAL 1: 1K test, AddCL ---
run_cmd "Eval 1K AddCL" \
    python src/exp-spatial-4x-crps-v1/unet_cfg_flow.py \
    --mode eval --save_dir models/unet_logit_normal \
    --n_ensemble 10 --ode_steps 10 --constraint addcl \
    --guidance_scale 1.0 --max_samples 1000 --split test

# --- EVAL 2: 10K test, AddCL ---
run_cmd "Eval 10K AddCL" \
    python src/exp-spatial-4x-crps-v1/unet_cfg_flow.py \
    --mode eval --save_dir models/unet_logit_normal \
    --n_ensemble 10 --ode_steps 10 --constraint addcl \
    --guidance_scale 1.0 --split test

echo ""
echo "Job $SLURM_JOB_ID finished at $(date)"
