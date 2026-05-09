#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J fjrz-kbfy
#SBATCH -o /home/chenxy/orcd/pool/datasets/organize2/eval_output.log
#SBATCH -e /home/chenxy/orcd/pool/datasets/organize2/eval_error.log

export PYTHONUNBUFFERED=1

source /home/chenxy/orcd/scratch/venvs/organize2/bin/activate
cd /home/chenxy/repos/workspace/organize2

pip install -e . -q 2>&1 | tail -1

echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo "=== Running evaluation ==="
python scripts/run_eval.py \
    --max-samples 500 \
    --n-ensemble 10 \
    --ode-steps 10 \
    --constraint addcl \
    --sampler midpoint \
    --output /home/chenxy/orcd/pool/datasets/organize2/eval_results_500.json

echo "=== Done ==="
