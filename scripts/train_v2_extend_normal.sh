#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=flow-v2-ext
#SBATCH --chdir=/home/chenxy/repos/workspace/research2
#SBATCH --output=/home/chenxy/repos/workspace/research2/logs/flow_v2_ext_%j.log

source /home/chenxy/orcd/scratch/venvs/research2/bin/activate

echo "[$(date)] Starting extended v2 training on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Resume from epoch 39, train 26 more epochs (to epoch 65)
# Fresh cosine schedule: starts at 5e-5, T_max=26, decays to 0
python -u scripts/flow_matching_v2.py \
    --mode train \
    --epochs 65 \
    --batch_size 64 \
    --resume \
    --finetune_lr 5e-5 \
    --save_dir models/flow_v2_ext

echo "[$(date)] Extended training finished with exit code $?"
