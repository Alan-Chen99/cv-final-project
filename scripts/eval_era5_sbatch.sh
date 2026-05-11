#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -J metrics-v2-eval
#SBATCH -G 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 1:30:00
#SBATCH -o /home/chenxy/orcd/scratch/logs/eval-8metrics-%j.log
#SBATCH -e /home/chenxy/orcd/scratch/logs/eval-8metrics-%j.log

# Runs ERA5 (500 samples) + NorESM (500 samples) evaluation with all 8 metrics.
# Incremental saving: JSON written after each method group.
# Outputs: eval_results_8metrics.json, eval_results_8metrics_spectral.npz,
#          noresm_eval_results_8metrics.json, noresm_eval_results_8metrics_spectral.npz

set -euo pipefail

PROJECT_DIR='/home/chenxy/repos/workspace/metrics-v2'
SIF='/home/chenxy/orcd/pool/cuda:13.0.2-cudnn-devel-ubuntu24.04.sif'

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

module load apptainer

run_in_container() {
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
    bash -c "cd /workspace && source .venv/bin/activate && $1"
}

echo "=== ERA5 TCW 4x evaluation ==="
run_in_container 'python scripts/run_eval.py --max-samples 500 --output eval_results_8metrics.json'
echo "ERA5 eval completed at $(date)"

echo "=== NorESM TAS 2x evaluation ==="
run_in_container 'python scripts/run_eval_noresm.py --max-samples 500 --output noresm_eval_results_8metrics.json'
echo "NorESM eval completed at $(date)"

echo "Job $SLURM_JOB_ID finished at $(date)"
