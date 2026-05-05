#!/bin/bash
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:30:00
#SBATCH --job-name=zyqup-yoihn-dit
#SBATCH --output=/home/chenxy/orcd/scratch/logs/dit_train_%j.log
#SBATCH --error=/home/chenxy/orcd/scratch/logs/dit_train_%j.log

PROJECT_DIR="/home/chenxy/repos/workspace/research3"
SIF="/home/chenxy/orcd/pool/cuda:13.0.2-cudnn-devel-ubuntu24.04.sif"

echo "=== DiT Flow Training ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "JobID: $SLURM_JOB_ID"

module load apptainer

singularity exec --nv \
    --cleanenv \
    --mount type=bind,source=/orcd,destination=/orcd \
    --mount type=bind,source=/home/chenxy/nix_store,destination=/nix,ro \
    --mount type=bind,source=${PROJECT_DIR},destination=/workspace \
    --env PREPEND_PATH=/nix/state/profile/bin:/nix/nix_path/bin \
    --env NIX_REMOTE=daemon \
    ${SIF} \
    bash -c 'cd /workspace && source .venv/bin/activate && python src/exp-spatial-4x-crps-v1/dit_flow.py --mode train --epochs 40 --batch_size 64 --lr 1e-4 --hidden_size 256 --depth 12 --num_heads 4 --save_dir models/dit_flow && echo "TRAINING COMPLETE" && python src/exp-spatial-4x-crps-v1/dit_flow.py --mode eval --n_ensemble 10 --split test --ode_steps 10 --constraint addcl --save_dir models/dit_flow --max_samples 2000 && echo "EVAL COMPLETE"'

echo ""
echo "=== Done ==="
echo "Date: $(date)"
