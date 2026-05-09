# SwinIR Finetuning + Zero-Shot: Iteration 1

## Start
- **Time**: 2026-05-09 00:45 EDT
- **Commit**: 833d064 (ralph prompt)
- **Branch**: spatial-4x-add-v2
- **Prefix**: ipby-cdaw

## GPU Status
- 2 normal GPU jobs running (at limit of 2)
- 2 preemptable jobs running (limit 4, 2 available)
- Will use preemptable for this work

## Objective
Add SwinIR finetuning (train 2hr) and zero-shot to evaluation pipeline in src/.
Do hyperparameter tuning before committing to the 2hr training run.

## Concerns (required review)

### 1. Workflow: Prior SwinIR assumed hyperparameters without tuning
- research5 used: LR=2e-4, batch_size=32, L1 loss, no freeze, weight_decay=1e-4, cosine LR
- No systematic hyperparameter search was done
- **Fix**: Run a quick sweep over LR and freeze/unfreeze before 2hr train

### 2. Quality: Prior finetuning used suboptimal batch size
- research5 used batch_size=32 (L40S has enough memory for 64+)
- No AMP (mixed precision) was used - could 2x throughput
- **Fix**: Use AMP + larger batch size for more epochs in 2hr

### 3. Workflow: No SwinIR code exists in src/ yet
- All SwinIR code is in frozen experiments/pretrained-sr-downscaling/
- Need to write clean SwinIR evaluation + training code in src/
- Need to integrate into run_eval.py and make_figures.py
- **Fix**: Write src/downscaling/evaluation/swinir.py, training script, add to eval pipeline

## Plan for this iteration
1. Write SwinIR evaluation code in src/downscaling/evaluation/swinir.py
2. Write SwinIR training script as src/downscaling/training/swinir.py
3. Add SwinIR to scripts/run_eval.py
4. Commit code changes
5. Allocate GPU + start short hyperparameter tuning sweep
