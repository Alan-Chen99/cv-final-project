#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -J pine-frost-era5
#SBATCH -G 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH -o /home/chenxy/orcd/scratch/logs/eval-era5-%j.log
#SBATCH -e /home/chenxy/orcd/scratch/logs/eval-era5-%j.log

set -euo pipefail

PROJECT_DIR='/home/chenxy/repos/workspace/metrics-v2'
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
  --env PYTHONUNBUFFERED=1 \
  "$SIF" \
  bash -c 'cd /workspace && source .venv/bin/activate && python scripts/run_eval.py --max-samples 500 --output eval_results_7metrics.json'

echo "Job $SLURM_JOB_ID finished at $(date)"
