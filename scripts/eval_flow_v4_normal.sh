#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -J flow-v4-eval-n
#SBATCH -G l40s:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /home/chenxy/orcd/scratch/logs/flow_v4_eval_n-%j.log
#SBATCH -e /home/chenxy/orcd/scratch/logs/flow_v4_eval_n-%j.log

set -euo pipefail

PROJECT_DIR='/home/chenxy/repos/workspace/research2'
SIF='/home/chenxy/orcd/pool/cuda:13.0.2-cudnn-devel-ubuntu24.04.sif'

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
nvidia-smi

module load apptainer

singularity exec --nv \
  --cleanenv \
  --mount 'type=bind,source=/orcd,destination=/orcd' \
  --mount 'type=bind,source=/home/chenxy/nix_store,destination=/nix,ro' \
  --mount "type=bind,source=$PROJECT_DIR,destination=/workspace" \
  --env PREPEND_PATH=/nix/state/profile/bin:/nix/nix_path/bin \
  --env NIX_REMOTE=daemon \
  --env BASH_ENV="$HOME/.bashrc" \
  --env PYTHONUNBUFFERED=1 \
  "$SIF" \
  bash -c 'cd /workspace && source .venv/bin/activate && \
    echo "=== Eval 1: no constraint ===" && \
    python -u scripts/flow_matching_v4.py --mode eval --split test --max_samples 2000 \
      --n_ensemble 10 --ode_steps 10 --constraint none --save_dir models/flow_v4 && \
    echo "=== Eval 2: with AddCL ===" && \
    python -u scripts/flow_matching_v4.py --mode eval --split test --max_samples 2000 \
      --n_ensemble 10 --ode_steps 10 --constraint addcl --save_dir models/flow_v4'

echo "Job $SLURM_JOB_ID finished at $(date)"
