#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -t 4:00:00
#SBATCH -J flow-aug-train
#SBATCH -o /home/chenxy/repos/workspace/research/logs/flow_train_aug.log
#SBATCH -e /home/chenxy/repos/workspace/research/logs/flow_train_aug.log

cd /home/chenxy/repos/workspace/research
source /home/chenxy/orcd/scratch/venvs/research/bin/activate

python scripts/flow_downscale.py --mode train \
  --data-dir external/constrained-downscaling/data/era5_sr_data \
  --save-dir models/flow_aug \
  --channels 48,96,192 --attention \
  --lr-anchor --noise-std 0.3 --constraint-aware \
  --augment \
  --epochs 200 --lr 2e-4 --batch-size 256
