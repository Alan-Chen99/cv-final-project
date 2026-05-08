#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -J hrgq-sauw-eval
#SBATCH -G 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /home/chenxy/orcd/scratch/logs/eval-ddpm-%j.log
#SBATCH -e /home/chenxy/orcd/scratch/logs/eval-ddpm-%j.log

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
    echo "=========================================="
    echo "Running: $NAME at $(date)"
    echo "=========================================="

    singularity exec --nv \
        --cleanenv \
        --mount "type=bind,source=/orcd,destination=/orcd" \
        --mount "type=bind,source=/home/chenxy/nix_store,destination=/nix,ro" \
        --mount "type=bind,source=$PROJECT_DIR,destination=/workspace" \
        --env PREPEND_PATH=/nix/state/profile/bin:/nix/nix_path/bin \
        --env NIX_REMOTE=daemon \
        --env "BASH_ENV=$HOME/.bashrc" \
        "$SIF" \
        bash -c "cd /workspace && source .venv/bin/activate && python $*"

    echo "Finished: $NAME at $(date)"
}

# 1. DDPM 10K eval with AddCL, stochastic DDIM (main result)
run_eval "ddpm_10k_addcl_eta1" \
    src/exp-spatial-4x-crps-v1/ddpm_unet.py --mode eval \
    --ddim_steps 20 --ddim_eta 1.0 --n_ensemble 10 \
    --constraint addcl --eval_batch_size 32 --split test

# 2. DDPM 10K eval without constraint (for comparison)
run_eval "ddpm_10k_none_eta1" \
    src/exp-spatial-4x-crps-v1/ddpm_unet.py --mode eval \
    --ddim_steps 20 --ddim_eta 1.0 --n_ensemble 10 \
    --constraint none --eval_batch_size 32 --split test

# 3. UNet flow matching 10K eval without constraint (for AddCL impact comparison)
run_eval "unet_flow_10k_none" \
    src/exp-spatial-4x-crps-v1/unet_cfg_flow.py --mode eval \
    --ode_steps 10 --guidance_scale 1.0 --n_ensemble 10 \
    --constraint none --eval_batch_size 32 --split test

echo ""
echo "All evaluations complete. Job $SLURM_JOB_ID finished at $(date)"
