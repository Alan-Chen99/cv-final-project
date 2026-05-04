#!/bin/bash
#SBATCH --job-name=eval-v3
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/flow_v3_eval.log
#SBATCH --chdir=/home/chenxy/repos/workspace/research2

source /home/chenxy/orcd/scratch/venvs/research2/bin/activate

echo "[$(date)] Starting flow v3 eval on $(hostname)"

# Evaluate multiple guidance scales
for SCALE in 1.0 1.5 2.0; do
    for CONST in none addcl; do
        echo "--- guidance=$SCALE, constraint=$CONST ---"
        python -u scripts/flow_matching_v3.py \
            --mode eval \
            --n_ensemble 10 \
            --ode_steps 10 \
            --max_samples 2000 \
            --guidance_scale $SCALE \
            --constraint $CONST \
            --save_dir models/flow_v3
    done
done

echo "[$(date)] Eval finished"
