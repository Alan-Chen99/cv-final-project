#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -J rxlt-ujbt-figs
#SBATCH -o /home/chenxy/orcd/pool/datasets/metrics-v2/figures_output.log
#SBATCH -e /home/chenxy/orcd/pool/datasets/metrics-v2/figures_error.log

export PYTHONUNBUFFERED=1

PROJECT_DIR=/home/chenxy/repos/workspace/metrics-v2
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"

echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo "=== Running make_figures.py (full, with ensemble) ==="
python scripts/make_figures.py \
    --pool-dir /home/chenxy/orcd/pool/datasets \
    --output-dir figures \
    --n-samples 5 \
    --n-ensemble 10

echo "=== Done ==="
echo "Exit code: $?"
