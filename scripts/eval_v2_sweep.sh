#!/bin/bash
#SBATCH --job-name=eval-sweep
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/chenxy/repos/workspace/research2
#SBATCH --output=/home/chenxy/repos/workspace/research2/logs/eval_v2_sweep_%j.log

source /home/chenxy/orcd/scratch/venvs/research2/bin/activate

echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
echo "=== Start: $(date) ==="
echo ""

COMMON="--mode eval --n_ensemble 10 --eval_batch_size 32 --max_samples 2000 --split test --save_dir models/flow_v2"

# Baseline (already known: CRPS=0.0926)
echo "=========================================="
echo "=== [1/8] Euler 10 steps, AddCL ==="
echo "=========================================="
python scripts/flow_matching_v2.py $COMMON --sampler euler --ode_steps 10 --constraint addcl
echo ""

# Midpoint 10 steps = 20 NFE (vs Euler 10 = 10 NFE)
echo "=========================================="
echo "=== [2/8] Midpoint 10 steps, AddCL ==="
echo "=========================================="
python scripts/flow_matching_v2.py $COMMON --sampler midpoint --ode_steps 10 --constraint addcl
echo ""

# Euler 20 steps = 20 NFE (same NFE as midpoint 10)
echo "=========================================="
echo "=== [3/8] Euler 20 steps, AddCL ==="
echo "=========================================="
python scripts/flow_matching_v2.py $COMMON --sampler euler --ode_steps 20 --constraint addcl
echo ""

# Midpoint 20 steps = 40 NFE (high quality reference)
echo "=========================================="
echo "=== [4/8] Midpoint 20 steps, AddCL ==="
echo "=========================================="
python scripts/flow_matching_v2.py $COMMON --sampler midpoint --ode_steps 20 --constraint addcl
echo ""

# SmCL variants
echo "=========================================="
echo "=== [5/8] Euler 10 steps, SmCL ==="
echo "=========================================="
python scripts/flow_matching_v2.py $COMMON --sampler euler --ode_steps 10 --constraint smcl
echo ""

echo "=========================================="
echo "=== [6/8] Midpoint 10 steps, SmCL ==="
echo "=========================================="
python scripts/flow_matching_v2.py $COMMON --sampler midpoint --ode_steps 10 --constraint smcl
echo ""

# No constraint variants for reference
echo "=========================================="
echo "=== [7/8] Midpoint 10 steps, none ==="
echo "=========================================="
python scripts/flow_matching_v2.py $COMMON --sampler midpoint --ode_steps 10 --constraint none
echo ""

echo "=========================================="
echo "=== [8/8] Midpoint 20 steps, none ==="
echo "=========================================="
python scripts/flow_matching_v2.py $COMMON --sampler midpoint --ode_steps 20 --constraint none
echo ""

echo "=== ALL SWEEP DONE: $(date) ==="
