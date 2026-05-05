#!/bin/bash
#SBATCH --job-name=eval-full
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/chenxy/repos/workspace/research2
#SBATCH --output=/home/chenxy/repos/workspace/research2/logs/eval_v2_full_%j.log

source /home/chenxy/orcd/scratch/venvs/research2/bin/activate

echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
echo "=== Start: $(date) ==="
echo ""

# Full test set (no --max_samples), 10 ensemble, 10 Euler steps
COMMON="--mode eval --n_ensemble 10 --eval_batch_size 32 --ode_steps 10 --sampler euler --split test --save_dir models/flow_v2"

echo "=========================================="
echo "=== Full test: Euler 10, AddCL ==="
echo "=========================================="
python scripts/flow_matching_v2.py $COMMON --constraint addcl
echo ""

echo "=========================================="
echo "=== Full test: Euler 10, none ==="
echo "=========================================="
python scripts/flow_matching_v2.py $COMMON --constraint none
echo ""

echo "=== FULL EVAL DONE: $(date) ==="
