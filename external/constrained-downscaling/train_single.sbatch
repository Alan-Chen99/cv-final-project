#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -G l40s:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH --requeue
#SBATCH -o /home/chenxy/orcd/scratch/logs/harder-%x-%j.log
#SBATCH -e /home/chenxy/orcd/scratch/logs/harder-%x-%j.log

# Usage: sbatch -J <model_id> train_single.sbatch <model> <model_id> <constraints> <loss> [extra_args...]
# Example: sbatch -J twc_cnn_mult train_single.sbatch cnn twc_cnn_mult mult mse

set -euo pipefail

MODEL="$1"
MODEL_ID="$2"
CONSTRAINTS="$3"
LOSS="$4"
shift 4
EXTRA_ARGS="$@"

PROJECT_DIR='/home/chenxy/repos/workspace/main'
SIF='/home/chenxy/orcd/pool/cuda:13.0.2-cudnn-devel-ubuntu24.04.sif'

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "Model: $MODEL_ID ($MODEL, $CONSTRAINTS, $LOSS)"
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
  bash -c "cd /workspace/external/constrained-downscaling && source /workspace/.venv/bin/activate && python main.py --dataset era5_sr_data --model $MODEL --model_id $MODEL_ID --constraints $CONSTRAINTS --loss $LOSS --batch_size 256 --epochs 200 $EXTRA_ARGS"

echo "Job $SLURM_JOB_ID finished at $(date)"
