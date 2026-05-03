#!/bin/bash
# Run CNN baselines from Harder et al. — deterministic models
set -eo pipefail

REAL_WS=/home/chenxy/repos/workspace/research2
VENV=/home/chenxy/orcd/scratch/venvs/research2
cd $REAL_WS/external/constrained-downscaling

source $VENV/bin/activate
export PYTHONUNBUFFERED=1

DATASET=era5_sr_data
BS=256
EPOCHS=200

models=(
  "cnn none mse twc_cnn_none"
  "cnn softmax mse twc_cnn_softmax"
)

for entry in "${models[@]}"; do
  read -r model constraints loss model_id <<< "$entry"

  if [ -f "data/${model_id}.csv" ]; then
    echo "=== Skipping $model_id (already complete) ==="
    continue
  fi

  echo "=== [$(date '+%H:%M:%S')] Training $model_id ($model, $constraints, $loss) ==="

  python main.py \
    --dataset "$DATASET" \
    --model "$model" \
    --model_id "$model_id" \
    --constraints "$constraints" \
    --batch_size "$BS" \
    --epochs "$EPOCHS" \
    --test_val_train test

  echo "--- [$(date '+%H:%M:%S')] Done: $model_id ---"
  echo
done

echo "=== CNN Baselines Complete ==="
date
