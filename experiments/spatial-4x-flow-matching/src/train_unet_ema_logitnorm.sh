#!/bin/bash
# Train UNet v2 with EMA + logit-normal timestep sampling, 200 epochs
# iter-003: testing training recipe improvements on UNet architecture

set -euo pipefail

SAVE_DIR="${1:-/home/chenxy/orcd/pool/datasets/research3/models/unet_ema_logitnorm}"
BASEDIR="external/constrained-downscaling"

echo "=== UNet v2 + EMA + Logit-Normal Training ==="
echo "Save dir: $SAVE_DIR"
echo "Start: $(date)"

cd /workspace
source .venv/bin/activate

python src/exp-spatial-4x-crps-v1/flow_matching_v2.py \
    --mode train \
    --basedir "$BASEDIR" \
    --save_dir "$SAVE_DIR" \
    --batch_size 64 \
    --epochs 26 \
    --lr 1e-4 \
    --base_channels 64 \
    --channel_mults "1,2,4" \
    --attn_heads 4 \
    --use_ema \
    --ema_decay 0.9999 \
    --t_sampling logit_normal \
    --t_logit_mean 0.0 \
    --t_logit_std 1.0 \
    --resume

echo "=== Training complete: $(date) ==="

# Evaluate both regular and EMA models
echo ""
echo "=== Evaluating regular model (10K test, 10 ensemble, Euler 10, AddCL) ==="
python src/exp-spatial-4x-crps-v1/flow_matching_v2.py \
    --mode eval \
    --basedir "$BASEDIR" \
    --save_dir "$SAVE_DIR" \
    --split test \
    --n_ensemble 10 \
    --ode_steps 10 \
    --constraint addcl \
    --eval_batch_size 32

echo ""
echo "=== Evaluating EMA model (10K test, 10 ensemble, Euler 10, AddCL) ==="
# EMA model is saved as best_flow_ema.pt — copy to best_flow.pt for eval
EMA_DIR="${SAVE_DIR}_ema_eval"
mkdir -p "$EMA_DIR"
cp "$SAVE_DIR/norm_stats.pt" "$EMA_DIR/"
cp "$SAVE_DIR/best_flow_ema.pt" "$EMA_DIR/best_flow.pt"

python src/exp-spatial-4x-crps-v1/flow_matching_v2.py \
    --mode eval \
    --basedir "$BASEDIR" \
    --save_dir "$EMA_DIR" \
    --split test \
    --n_ensemble 10 \
    --ode_steps 10 \
    --constraint addcl \
    --eval_batch_size 32

echo "=== All done: $(date) ==="
