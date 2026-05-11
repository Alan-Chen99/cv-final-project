#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -J eval-noresm
#SBATCH -G l40s:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH -o /home/chenxy/orcd/scratch/logs/eval-noresm-%j.log
#SBATCH -e /home/chenxy/orcd/scratch/logs/eval-noresm-%j.log

set -euo pipefail

PROJECT_DIR='/home/chenxy/repos/workspace/main'
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
  --env PYTHONUNBUFFERED=1 \
  "$SIF" \
  bash -c 'cd /workspace && source .venv/bin/activate && python -m downscaling.evaluation.comprehensive --dataset noresm'

echo "Job $SLURM_JOB_ID finished at $(date)"
