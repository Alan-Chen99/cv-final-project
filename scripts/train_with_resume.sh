#!/bin/bash
# Train with auto-resume on preemption.
# Usage: train_with_resume.sh [extra args for flow_downscale.py]
set -e

MAX_ATTEMPTS=5
SAVE_DIR="models/flow_constrained"

for attempt in $(seq 1 $MAX_ATTEMPTS); do
    echo "=== Attempt $attempt/$MAX_ATTEMPTS ($(date)) ==="
    python scripts/flow_downscale.py \
        --mode train \
        --epochs 100 \
        --batch-size 256 \
        --lr 2e-4 \
        --channels 32,64,128 \
        --euler-steps 20 \
        --save-dir "$SAVE_DIR" \
        --constraint-aware \
        --constraint-weight 0.1 \
        "$@" && break
    echo "Training interrupted (attempt $attempt), resuming in 2s..."
    sleep 2
done

echo "=== Training finished at $(date) ==="
