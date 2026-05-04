#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -t 1:00:00
#SBATCH -J flow-eval-q
#SBATCH -o /home/chenxy/repos/workspace/research/logs/flow_eval_aug2.log
#SBATCH -e /home/chenxy/repos/workspace/research/logs/flow_eval_aug2.log

cd /home/chenxy/repos/workspace/research
source /home/chenxy/orcd/scratch/venvs/research/bin/activate

echo "=== Evaluating augmented model (latest checkpoint) with mult constraint ==="
python scripts/flow_downscale.py --mode eval \
  --data-dir external/constrained-downscaling/data/era5_sr_data \
  --save-dir models/flow_aug \
  --model-id flow_aug85_mult \
  --constraint mult \
  --euler-steps 20 --n-members 10 --eval-batch-size 128
