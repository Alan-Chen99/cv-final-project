#!/bin/bash
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=flow-v2-train
#SBATCH --output=/home/chenxy/repos/workspace/research2/logs/flow_v2_train_sbatch.log

source /home/chenxy/orcd/scratch/venvs/research2/bin/activate
cd /home/chenxy/repos/workspace/research2

echo "[$(date)] Starting flow v2 training on $(hostname)"
python -u scripts/flow_matching_v2.py --mode train --epochs 40 --batch_size 64 --resume
echo "[$(date)] Training finished with exit code $?"
