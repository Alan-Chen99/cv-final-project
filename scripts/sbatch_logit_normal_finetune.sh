#!/bin/bash
#SBATCH --job-name=zbhh-axxo
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/workspace/logs/logit_normal_finetune_%j.out

set -e

echo "=== LOGIT-NORMAL FINE-TUNE + EVAL ==="
echo "Job: $SLURM_JOB_ID, Node: $(hostname), GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
date '+%Y-%m-%d %H:%M:%S %Z'

CONTAINER="/home/chenxy/.apptainer/images/pytorch_24.07-py3.sif"
WORKSPACE="/workspace"
SCRIPT="src/exp-spatial-4x-crps-v1/unet_cfg_flow.py"
SAVE_DIR="models/unet_logit_normal"
POOL_DIR="/home/chenxy/orcd/pool/datasets/research4/models"

# --- SETUP: copy best model for resume ---
echo ""
echo "=== SETUP: copying best 55ep model ==="
mkdir -p "$WORKSPACE/$SAVE_DIR"
cp "$WORKSPACE/models/unet_cfg/best_flow.pt" "$WORKSPACE/$SAVE_DIR/best_flow.pt"
cp "$WORKSPACE/models/unet_cfg/norm_stats.pt" "$WORKSPACE/$SAVE_DIR/norm_stats.pt"
echo "Copied best model (epoch 51, val_loss=0.251)"

# --- TRAINING: fine-tune with logit-normal t ---
echo ""
echo "=== TRAINING: fine-tune 15 epochs with logit-normal t ==="
date '+%Y-%m-%d %H:%M:%S'

singularity exec --nv \
  --bind /home/chenxy/orcd/pool:/home/chenxy/orcd/pool \
  --bind "$WORKSPACE:$WORKSPACE" \
  "$CONTAINER" \
  python "$WORKSPACE/$SCRIPT" \
    --mode train \
    --basedir "$WORKSPACE/external/constrained-downscaling" \
    --save_dir "$WORKSPACE/$SAVE_DIR" \
    --resume \
    --finetune_lr 5e-5 \
    --epochs 67 \
    --batch_size 64 \
    --cfg_prob 0 \
    --t_schedule logit_normal \
    --logit_normal_mean 0.0 \
    --logit_normal_std 1.0

echo ""
echo "=== TRAINING COMPLETE ==="
date '+%Y-%m-%d %H:%M:%S'

# Copy model to pool
mkdir -p "$POOL_DIR"
cp "$WORKSPACE/$SAVE_DIR/best_flow.pt" "$POOL_DIR/unet_logit_normal_best.pt"
cp "$WORKSPACE/$SAVE_DIR/norm_stats.pt" "$POOL_DIR/unet_logit_normal_norm_stats.pt"
echo "Model copied to pool"

# --- EVAL 1: 1K test, AddCL ---
echo ""
echo "=== EVAL: logit-normal finetune, 1K test, AddCL ==="
date '+%Y-%m-%d %H:%M:%S'

singularity exec --nv \
  --bind /home/chenxy/orcd/pool:/home/chenxy/orcd/pool \
  --bind "$WORKSPACE:$WORKSPACE" \
  "$CONTAINER" \
  python "$WORKSPACE/$SCRIPT" \
    --mode eval \
    --basedir "$WORKSPACE/external/constrained-downscaling" \
    --save_dir "$WORKSPACE/$SAVE_DIR" \
    --split test \
    --max_samples 1000 \
    --n_ensemble 10 \
    --ode_steps 10 \
    --constraint addcl \
    --solver euler

# --- EVAL 2: 10K test, AddCL (if time permits) ---
echo ""
echo "=== EVAL: logit-normal finetune, 10K test, AddCL ==="
date '+%Y-%m-%d %H:%M:%S'

singularity exec --nv \
  --bind /home/chenxy/orcd/pool:/home/chenxy/orcd/pool \
  --bind "$WORKSPACE:$WORKSPACE" \
  "$CONTAINER" \
  python "$WORKSPACE/$SCRIPT" \
    --mode eval \
    --basedir "$WORKSPACE/external/constrained-downscaling" \
    --save_dir "$WORKSPACE/$SAVE_DIR" \
    --split test \
    --n_ensemble 10 \
    --ode_steps 10 \
    --constraint addcl \
    --solver euler

echo ""
echo "=== ALL DONE ==="
date '+%Y-%m-%d %H:%M:%S'
