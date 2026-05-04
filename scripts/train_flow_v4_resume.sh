#!/bin/bash
#SBATCH --job-name=flow-v4-res
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/chenxy/repos/workspace/research2
#SBATCH --output=/home/chenxy/repos/workspace/research2/logs/train_flow_v4_resume_%j.log

source /home/chenxy/orcd/scratch/venvs/research2/bin/activate

python scripts/flow_matching_v4.py \
    --mode train \
    --epochs 40 \
    --batch_size 64 \
    --resume \
    --base_channels 96 \
    --channel_mults "1,2,4" \
    --attn_heads 4

echo "TRAINING_DONE"
