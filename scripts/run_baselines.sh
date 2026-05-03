#!/bin/bash
# Run baseline models for constrained-downscaling CRPS evaluation
# Usage: srun --jobid=<JOBID> bash scripts/run_baselines.sh
set -euo pipefail

cd /workspace/external/constrained-downscaling
mkdir -p models data/prediction

EPOCHS=${1:-200}
BS=256

echo "=== Training baselines with $EPOCHS epochs, batch_size=$BS ==="
echo "Start time: $(date)"

# 1. GAN no constraints (needed for CRPS baseline)
echo ""
echo "=== [1/4] GAN no constraints ==="
python main.py \
    --dataset era5_sr_data --model gan --model_id gan_none \
    --constraints none --epochs $EPOCHS --batch_size $BS \
    --test_val_train test
echo "GAN none done at $(date)"

# 2. GAN with softmax constraints (best hard constraint from paper)
echo ""
echo "=== [2/4] GAN softmax constraints ==="
python main.py \
    --dataset era5_sr_data --model gan --model_id gan_softmax \
    --constraints softmax --epochs $EPOCHS --batch_size $BS \
    --test_val_train test
echo "GAN softmax done at $(date)"

# 3. CNN no constraints (deterministic baseline)
echo ""
echo "=== [3/4] CNN no constraints ==="
python main.py \
    --dataset era5_sr_data --model cnn --model_id cnn_none \
    --constraints none --epochs $EPOCHS --batch_size $BS \
    --test_val_train test
echo "CNN none done at $(date)"

# 4. CNN with softmax constraints
echo ""
echo "=== [4/4] CNN softmax constraints ==="
python main.py \
    --dataset era5_sr_data --model cnn --model_id cnn_softmax \
    --constraints softmax --epochs $EPOCHS --batch_size $BS \
    --test_val_train test
echo "CNN softmax done at $(date)"

echo ""
echo "=== All baselines complete ==="
echo "End time: $(date)"

# Run proper CRPS evaluation
echo ""
echo "=== CRPS Evaluation ==="
cd /workspace

DATA=external/constrained-downscaling/data
PRED=$DATA/prediction
TARGET=$DATA/era5_sr_data/test/target_test.pt
INPUT=$DATA/era5_sr_data/test/input_test.pt

for model_id in gan_none gan_softmax; do
    echo "--- $model_id ---"
    python scripts/eval_crps.py \
        --pred "$PRED/era5_sr_data_${model_id}_test_ensemble.pt" \
        --target "$TARGET" --input "$INPUT"
done

for model_id in cnn_none cnn_softmax; do
    echo "--- $model_id ---"
    python scripts/eval_crps.py \
        --pred "$PRED/era5_sr_data_${model_id}_test.pt" \
        --target "$TARGET" --input "$INPUT"
done
