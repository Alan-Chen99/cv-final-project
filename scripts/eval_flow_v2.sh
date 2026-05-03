#!/bin/bash
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=flow-v2-eval
#SBATCH --output=/home/chenxy/repos/workspace/research2/logs/flow_v2_eval.log

source /home/chenxy/orcd/scratch/venvs/research2/bin/activate
cd /home/chenxy/repos/workspace/research2

echo "[$(date)] Starting flow v2 eval on $(hostname)"

# Eval 1: no constraint
python -u scripts/flow_matching_v2.py --mode eval --split test --max_samples 2000 \
    --n_ensemble 10 --ode_steps 10 --constraint none --save_dir models/flow_v2

echo "---"

# Eval 2: with AddCL
python -u scripts/flow_matching_v2.py --mode eval --split test --max_samples 2000 \
    --n_ensemble 10 --ode_steps 10 --constraint addcl --save_dir models/flow_v2

echo "[$(date)] Eval finished"
