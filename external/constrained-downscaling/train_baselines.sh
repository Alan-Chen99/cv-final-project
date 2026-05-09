#!/bin/bash
# Train Harder et al. baselines needed for figure comparison:
#   1. CNN (no constraint)
#   2. CNN + SmCL (softmax) — their best method
#   3. GAN + SmCL (softmax)
set -eo pipefail
cd "$(dirname "$0")"
source /workspace/.venv/bin/activate
export PYTHONUNBUFFERED=1

DATASET=era5_sr_data
BS=256
EPOCHS=200

models=(
  "cnn twc_cnn_none none mse"
  "cnn twc_cnn_softmax softmax mse"
  "gan twc_gan_softmax softmax mse"
)

for entry in "${models[@]}"; do
  read -r model model_id constraints loss <<< "$entry"
  if [ -f "models/${model_id}.pth" ]; then
    echo "=== Skipping $model_id (checkpoint exists) ==="
    continue
  fi
  echo "=== [$(date '+%H:%M:%S')] Training $model_id ($model, $constraints, $loss) ==="
  python main.py \
    --dataset "$DATASET" \
    --model "$model" \
    --model_id "$model_id" \
    --constraints "$constraints" \
    --batch_size "$BS" \
    --epochs "$EPOCHS"
  echo "--- [$(date '+%H:%M:%S')] Done: $model_id ---"
  echo
done

echo "=== All training complete ==="
# Copy checkpoints to pool for persistence
POOL_DIR="/home/chenxy/orcd/pool/datasets/organize2/models/harder"
mkdir -p "$POOL_DIR"
for f in models/twc_*.pth; do
  [ -f "$f" ] && cp -v "$f" "$POOL_DIR/"
done
echo "Checkpoints copied to $POOL_DIR"
