#!/bin/bash
#SBATCH --job-name=flow-v3
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/flow_v3_train.log
#SBATCH --chdir=/home/chenxy/repos/workspace/research2

source /home/chenxy/orcd/scratch/venvs/research2/bin/activate

echo "[$(date)] Starting flow v3 training (CFG + logit-normal) on $(hostname)"

python -u scripts/flow_matching_v3.py \
    --mode train \
    --epochs 40 \
    --batch_size 64 \
    --lr 1e-4 \
    --p_uncond 0.1 \
    --time_sampling logit_normal \
    --time_std 1.0 \
    --random_flip \
    --save_dir models/flow_v3 \
    --resume

echo "[$(date)] Training finished"
