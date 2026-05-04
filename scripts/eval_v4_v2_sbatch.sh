#!/bin/bash
#SBATCH --job-name=eval-crps
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/chenxy/repos/workspace/research2
#SBATCH --output=/home/chenxy/repos/workspace/research2/logs/eval_v4_v2_%j.log

source /home/chenxy/orcd/scratch/venvs/research2/bin/activate

echo "=== GPU INFO ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "=========================================="
echo "=== Flow v4 (28M params, epoch 22, AddCL) ==="
echo "=========================================="
python scripts/flow_matching_v4.py --mode eval --n_ensemble 10 --eval_batch_size 32 --ode_steps 10 --max_samples 2000 --split test --constraint addcl --base_channels 96 --channel_mults "1,2,4" --attn_heads 4

echo ""
echo "=========================================="
echo "=== Flow v4 (28M params, epoch 22, no constraint) ==="
echo "=========================================="
python scripts/flow_matching_v4.py --mode eval --n_ensemble 10 --eval_batch_size 32 --ode_steps 10 --max_samples 2000 --split test --constraint none --base_channels 96 --channel_mults "1,2,4" --attn_heads 4

echo ""
echo "=========================================="
echo "=== Flow v2 (13M params, epoch 39, AddCL) ==="
echo "=========================================="
python scripts/flow_matching_v2.py --mode eval --n_ensemble 10 --eval_batch_size 32 --ode_steps 10 --max_samples 2000 --split test --constraint addcl

echo ""
echo "ALL_EVAL_DONE"
