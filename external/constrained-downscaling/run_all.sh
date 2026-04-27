#!/bin/bash
# Reproduce Table 5 (TCW4 rows) from Harder et al.
# Sequential runs with batch_size=256 (matching paper)
# Skips models that already have a results CSV (preemption-safe with --requeue)
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
  "cnn twc_cnn_add add mse"
  "cnn twc_cnn_scadd scadd mse"
  "cnn twc_cnn_mult mult mse"
  "gan twc_gan_none none mse"
  "gan twc_gan_softmax softmax mse"
  "cnn twc_cnn_soft soft mass_constraints"
)

for entry in "${models[@]}"; do
  read -r model model_id constraints loss <<< "$entry"
  if [ -f "data/${model_id}.csv" ]; then
    echo "=== Skipping $model_id (already complete) ==="
    continue
  fi
  echo "=== [$(date '+%H:%M:%S')] Training $model_id ($model, $constraints, $loss) ==="
  extra_args=""
  if [ "$loss" = "mass_constraints" ]; then
    extra_args="--loss mass_constraints --alpha 0.99"
  fi
  python main.py \
    --dataset "$DATASET" \
    --model "$model" \
    --model_id "$model_id" \
    --constraints "$constraints" \
    --batch_size "$BS" \
    --epochs "$EPOCHS" \
    $extra_args
  echo "--- [$(date '+%H:%M:%S')] Done: $model_id ---"
  echo
done

echo
echo "=== All training complete ==="
echo "=== Results ==="
for f in data/twc_*.csv; do
  echo "--- $(basename "$f" .csv) ---"
  cat "$f"
  echo
done
