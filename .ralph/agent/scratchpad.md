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
1. Write SwinIR evaluation code in src/downscaling/evaluation/swinir.py ✓
2. Write SwinIR training script as src/downscaling/training/swinir.py ✓
3. Add SwinIR to scripts/run_eval.py ✓
4. Commit code changes ✓ (99d3372)
5. Allocate GPU + run hyperparameter sweep ✓
6. Start 2hr training with best config ✓ (running on job 13613743, node3003)
7. Wait for training to finish, then run eval + update plots (next iteration)

## Sweep Results (5 epochs each, L40S GPU)
| Config | Val Loss | Epochs | Winner |
|--------|----------|--------|--------|
| LR=2e-4, unfrozen | **0.002294** | 3 | **YES** |
| LR=1e-4, unfrozen | 0.002319 | 4 | |
| LR=5e-5, unfrozen | 0.002420 | 4 | |
| LR=1e-3, frozen | 0.002537 | 5 | |
| LR=2e-4, frozen | 0.002619 | 5 | |

Key findings:
- Full backbone unfreezing much better than frozen (0.002294 vs 0.002537)
- LR=2e-4 best, LR=1e-4 close second
- Prior research5 assumption of LR=2e-4 was correct, but used batch_size=32 (slower)
- bfloat16 AMP works (float16 caused NaN)
- ~3.8 min/epoch at bs=64 → ~31 epochs in 2hr

## 2hr Training Run
- Job: 13613743, node3003 (preemptable, TIMEOUT at 3hr salloc limit)
- Config: LR=2e-4, bs=64, L1, unfrozen, bfloat16 AMP, cosine annealing (T_max=100)
- Save dir: pool/spatial-4x-add-v2/models/swinir_ft/
- Started: 02:24 EDT, killed: 03:53 EDT (~89 min training, 23 epochs)
- Best checkpoint: epoch 22, val_loss=0.002108
- Val loss trajectory (stable, slowly improving):
  - Ep 2: 0.002414
  - Ep 10: 0.002197
  - Ep 15: 0.002126
  - Ep 22: 0.002108 (best)
  - Ep 23: 0.002197 (spike, noisy)
- No final_swinir.pt saved (killed before wall limit)
- Checkpoint path: pool/spatial-4x-add-v2/models/swinir_ft/best_swinir.pt

## Notes
- float16 AMP causes NaN in SwinIR (attention overflow) → must use bfloat16
- ~3.8 min/epoch on L40S with bs=64, bfloat16
- Research5 got MAE=0.2504 (19 epochs, bs=32, no AMP). Our val_loss=0.002108
  in [0,1] space ≈ MAE 0.276 in physical units → need more epochs or
  eval may reveal a different story (val_loss isn't MAE directly)

## Ending state
- **Time**: 2026-05-09 ~04:00 EDT
- **Commit**: 99d3372 (add SwinIR evaluation and finetuning to src/)
- **GPU**: Released (salloc 13613743 TIMEOUT)

## Next iteration TODO
1. Allocate new GPU (preemptable)
2. Run scripts/run_eval.py with SwinIR models (needs GPU for zero-shot)
3. Optionally: continue training for more epochs if MAE worse than research5
4. Update figures with scripts/make_figures.py
5. Commit updated eval results + figures
