#!/bin/bash
#SBATCH --job-name=eval-tta
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/chenxy/repos/workspace/research2
#SBATCH --output=/home/chenxy/repos/workspace/research2/logs/eval_v2_tta_%j.log

source /home/chenxy/orcd/scratch/venvs/research2/bin/activate

echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
echo "=== Start: $(date) ==="
echo ""

COMMON="--mode eval --eval_batch_size 32 --ode_steps 10 --sampler euler --split test --save_dir models/flow_v2 --max_samples 2000"

echo "=========================================="
echo "=== Baseline: 10 ens, AddCL, no TTA ==="
echo "=========================================="
python scripts/flow_matching_v2.py $COMMON --n_ensemble 10 --constraint addcl
echo ""

echo "=========================================="
echo "=== TTA: 10 ens, AddCL, TTA ==="
echo "=========================================="
python scripts/flow_matching_v2.py $COMMON --n_ensemble 10 --constraint addcl --tta
echo ""

echo "=========================================="
echo "=== TTA: 20 ens, AddCL, TTA ==="
echo "=========================================="
python scripts/flow_matching_v2.py $COMMON --n_ensemble 20 --constraint addcl --tta
echo ""

echo "=========================================="
echo "=== More ens: 20 ens, AddCL, no TTA ==="
echo "=========================================="
python scripts/flow_matching_v2.py $COMMON --n_ensemble 20 --constraint addcl
echo ""

echo "=== ALL DONE: $(date) ==="
