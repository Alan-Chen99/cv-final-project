#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -J gcgi-vxgh-eval
#SBATCH -G 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /home/chenxy/orcd/scratch/logs/eval-spectral-%j.log
#SBATCH -e /home/chenxy/orcd/scratch/logs/eval-spectral-%j.log

set -euo pipefail

PROJECT_DIR='/home/chenxy/repos/workspace/research4'
SIF='/home/chenxy/orcd/pool/cuda:13.0.2-cudnn-devel-ubuntu24.04.sif'

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

module load apptainer

run_eval() {
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
      bash -c "cd /workspace && source .venv/bin/activate && python src/exp-spatial-4x-crps-v1/unet_cfg_flow.py $*"
    echo "End: $(date)"
}

# 1K quick check
run_eval "Spectral 1K AddCL" \
    --mode eval --save_dir models/unet_spectral \
    --n_ensemble 10 --ode_steps 10 --constraint addcl \
    --guidance_scale 1.0 --max_samples 1000 --split test

# Full 10K eval
run_eval "Spectral 10K AddCL" \
    --mode eval --save_dir models/unet_spectral \
    --n_ensemble 10 --ode_steps 10 --constraint addcl \
    --guidance_scale 1.0 --split test

# Also eval the baseline UNet (55ep, no spectral) for comparison with exact same code
run_eval "Baseline 10K AddCL" \
    --mode eval --save_dir models/unet_cfg \
    --n_ensemble 10 --ode_steps 10 --constraint addcl \
    --guidance_scale 1.0 --split test

echo ""
echo "Job $SLURM_JOB_ID finished at $(date)"
