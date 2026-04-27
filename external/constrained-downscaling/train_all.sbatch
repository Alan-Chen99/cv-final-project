#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -J harder-repro
#SBATCH -G l40s:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --requeue
#SBATCH -o /home/chenxy/orcd/scratch/logs/harder-repro-%j.log
#SBATCH -e /home/chenxy/orcd/scratch/logs/harder-repro-%j.log

set -euo pipefail

PROJECT_DIR='/home/chenxy/repos/workspace/main'
SIF='/home/chenxy/orcd/pool/cuda:13.0.2-cudnn-devel-ubuntu24.04.sif'

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
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
  "$SIF" \
  bash -c 'cd /workspace/external/constrained-downscaling && bash run_all.sh'

echo "Job $SLURM_JOB_ID finished at $(date)"
