# Artifact Provenance: Plot-Generating Artifacts

All artifacts that feed into `scripts/make_figures.py`.

## Evaluation Result JSONs (direct inputs to make_figures.py)

### `eval_results_500.json` (ERA5 TCW 4x, 15 methods)

Created across 3 sessions, each adding methods:

| Version | Session | Branch | Command | Commit |
|---|---|---|---|---|
| v1 (8 methods) | `7e509c59-72ff-4a37-9665-9a08894827fb` | organize2 | `sbatch /workspace/scripts/sbatch_eval.sh` (job 13601979; params: `--max-samples 500 --n-ensemble 10 --ode-steps 10 --constraint addcl --sampler midpoint`) | `56f4ef6` |
| v2 (+Harder, 11 methods) | `1eb7efaa-92b0-4756-b925-14ae6b6331f4` | organize2 | `python3 scripts/gpu_run.py 13609961 'python scripts/run_eval.py --max-samples 500 --output eval_results_500.json'` | `ecd1e83` |
| v3 (+SwinIR, 15 methods) | `84a00048-762e-46e4-909a-b70934e6297b` | spatial-4x-add-v2 | `python3 scripts/gpu_run.py 13623783 python scripts/eval_swinir_only.py` | `ace332d` |

v1 output (8 methods):
```
=====================================================================================
Method                                         CRPS        MAE       RMSE   MassViol
=====================================================================================
flow-wide96-amp (28M)                      0.171877   0.251052   0.456348   0.000001
flow-v2-zscore (13M)                       0.175403   0.256022   0.466831   0.000001
flow-uniform-amp (13M)                     0.175552   0.256400   0.467005   0.000001
flow-logitnorm-ema (13M)                   0.181370   0.265576   0.498716   0.000001
bicubic+addcl                              0.362642   0.362642   0.740772   0.000001
bicubic                                    0.393946   0.393946   0.784946   0.149243
bilinear+addcl                             0.399124   0.399124   0.803993   0.000001
bilinear                                   0.519055   0.519055   0.963896   0.320263
=====================================================================================
```

v3 output (SwinIR merge):
```
Evaluating swinir-zeroshot...
  CRPS=0.325654  MAE=0.325654  RMSE=0.702455  MassViol=0.083542  (23.4s)
Evaluating swinir-zeroshot+addcl...
  CRPS=0.310593  MAE=0.310593  RMSE=0.684361  MassViol=0.000001  (12.6s)
Evaluating swinir-finetuned...
  CRPS=0.264911  MAE=0.264911  RMSE=0.510791  MassViol=0.024660  (3.1s)
Evaluating swinir-finetuned+addcl...
  CRPS=0.263245  MAE=0.263245  RMSE=0.509424  MassViol=0.000001  (1.9s)

Merged 4 SwinIR results into eval_results_500.json
```

Current file: `/workspace/eval_results_500.json` (2854 B, 2026-05-09).

### `noresm_eval_results_500.json` (NorESM TAS 2x, 12 methods)

| Session | Branch | Command | Commit |
|---|---|---|---|
| `f0508f7e-36d1-4d6d-8667-040498291bb0` | noresm-dataset | `python scripts/gpu_run.py 13661966 'python scripts/run_eval_noresm.py --max-samples 500 --constraint none --output noresm_eval_results_500.json'` | `fa63326` |

Output:
```
=====================================================================================
Method                                         CRPS        MAE       RMSE   MassViol
=====================================================================================
flow-wide96-amp (28M)                      0.648554   0.966938   1.513035   1.118547
swinir-finetuned                           0.988028   0.988028   1.533691   1.064584
harder-cnn                                 1.131490   1.131490   1.694460   0.942514
harder-cnn+smcl                            1.453476   1.453476   2.276664   0.000005
swinir-finetuned+addcl                     1.454977   1.454977   2.278966   0.000007
bilinear                                   1.472515   1.472515   2.307101   0.162353
swinir-zeroshot                            1.475288   1.475288   2.323859   0.067163
bicubic                                    1.476597   1.476597   2.318721   0.061015
bilinear+addcl                             1.477939   1.477939   2.323241   0.000007
swinir-zeroshot+addcl                      1.478364   1.478364   2.324362   0.000007
bicubic+addcl                              1.479416   1.479416   2.325272   0.000007
harder-gan+smcl                            1.480855   1.503450   2.344872   0.000012
=====================================================================================
```

Current file: `/workspace/noresm_eval_results_500.json` (2341 B, 2026-05-09).

## Figure Generation

| Figures | Session | Branch | Command | Commit |
|---|---|---|---|---|
| Metric bar charts (9 PNGs) | `a4338ecc-1c46-4160-a35c-f15dc98d72ec` | noresm-dataset | `uv run python scripts/make_figures.py --metrics-only --output-dir figures/` | `04e8172` |
| Sample predictions (26 PNGs) | `ebe4b7b6-a2fe-4ee9-87c9-e573a2475683` | noresm-dataset | `uv run python scripts/gpu_run.py 13662769 uv run python scripts/make_figures.py --pool-dir /home/chenxy/orcd/pool/datasets --n-samples 5 --n-ensemble 10` | `a681142` |

Output: `figures/era5/` (16 files), `figures/noresm/` (16 files), `figures/` (3 cross-dataset files).

---

## ERA5 Model Checkpoints

### `research3/models/unet_wide96_amp/best_flow.pt`

| Field | Value |
|---|---|
| Pool path | `/home/chenxy/orcd/pool/datasets/research3/models/unet_wide96_amp/best_flow.pt` |
| Session | `dbda10ea-8db3-4845-bd83-95973eafc71d` |
| Branch | research3 |
| Commit | `7718a9d` ("iter-007: Wider UNet (96ch, 28.4M) -- val_loss 0.243 beats baseline 0.252") |
| Norm stats | `research3/models/unet_wide96_amp/norm_stats.pt` (same session) |

**Command:**
```
python3 scripts/gpu_run.py 13435838 python src/exp-spatial-4x-crps-v1/flow_matching_v2.py \
  --mode train --basedir external/constrained-downscaling \
  --save_dir /home/chenxy/orcd/pool/datasets/research3/models/unet_wide96_amp \
  --batch_size 64 --epochs 40 --lr 1e-4 --base_channels 96 \
  --channel_mults "1,2,4" --attn_heads 4 --t_sampling uniform --amp
```

**Script output (quoted, abbreviated — full log in session):**
```
#params: 28,350,401

Epoch 1/40, Train: 0.466422, Val: 0.351467, LR: 0.000100, Time: 5.9min
Epoch 2/40, Train: 0.347454, Val: 0.327250, LR: 0.000099, Time: 12.0min
...
Epoch 24/40, Train: 0.246488, Val: 0.244358, LR: 0.000035, Time: 141.9min
Epoch 25/40, Train: 0.244780, Val: 0.243212, LR: 0.000031, Time: 147.8min

[2026-05-06T15:16:41.010] error: *** STEP 13435838.2 ON node3302 CANCELLED AT 2026-05-06T15:16:41 DUE TO TIME LIMIT ***
[train] Finished at Wed May  6 15:16:43 EDT 2026 with exit=0
```

No CRPS evaluation was run in this session. From session scratchpad: `CRPS evaluation NOT completed — cluster GPUs fully allocated`.

### `organize2/models/harder/*.pth` (ERA5 Harder CNN/GAN)

**Original training** — session `db7c9287-656e-43bb-8dae-b7bd77aaac67` (2026-04-26, master branch):

Sequential batch (SLURM job `12586890`, node3404): `twc_cnn_none`, `twc_cnn_softmax`, `twc_cnn_add` completed; `twc_cnn_scadd` killed by SLURM time limit. Parallel batch (5 `train_single.sbatch` jobs: `12609397`-`12609401`): remaining models completed.

Commit: `03adf1b` ("Replicate Harder et al. baseline: PyTorch 2.x compat, training scripts, visualization").

| Model | SLURM job | Node | Training command |
|---|---|---|---|
| `twc_cnn_none.pth` | 12586890 (sequential) | node3404 | `python main.py --dataset era5_sr_data --model cnn --model_id cnn_none --constraints none --epochs 200 --batch_size 256` |
| `twc_cnn_softmax.pth` | 12586890 (sequential) | node3404 | `python main.py --dataset era5_sr_data --model cnn --model_id cnn_softmax --constraints softmax --epochs 200 --batch_size 256` |
| `twc_gan_none.pth` | 12609399 | node4210 | `python main.py --dataset era5_sr_data --model gan --model_id gan_none --constraints none --epochs 200 --batch_size 256` |
| `twc_gan_softmax.pth` | 12609400 | node4211 | `python main.py --dataset era5_sr_data --model gan --model_id gan_softmax --constraints softmax --epochs 200 --batch_size 256` |

Training output not captured by extraction agents. Session log: `/home/chenxy/.claude/projects/-workspace/db7c9287-656e-43bb-8dae-b7bd77aaac67.jsonl`.

**Copied to pool** in session `1eb7efaa-92b0-4756-b925-14ae6b6331f4` (organize2 branch):
```
cp -v /home/chenxy/repos/workspace/main/external/constrained-downscaling/models/*.pth \
  /home/chenxy/orcd/pool/datasets/organize2/models/harder/
```
This glob copied all 16 `.pth` files in the directory, not only the 4 used in evaluation.

Commit: `d390f64` ("add Harder et al. CNN, GAN, CNN+SmCL baselines to eval and figures").

### `spatial-4x-add-v2/models/swinir_ft/best_swinir.pt`

| Field | Value |
|---|---|
| Pool path | `/home/chenxy/orcd/pool/datasets/spatial-4x-add-v2/models/swinir_ft/best_swinir.pt` |
| Session | `ec371f33-e763-4141-a75a-4cbc4b4fe00c` |
| Branch | spatial-4x-add-v2 |
| Commit | `99d3372` ("add SwinIR evaluation and finetuning to src/") |
| Pretrained weights | `research5/pretrained_weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth` |

**Command:**
```
python3 scripts/gpu_run.py 13613743 python -m downscaling.training.swinir \
  --pool-dir /home/chenxy/orcd/pool/datasets --epochs 100 --batch-size 64 --lr 2e-4 --wall-hours 2.0
```

**Script output (quoted, abbreviated):**

Hyperparameter sweep ran first (5 configs, 3-5 epochs each):
```
============================================================
SWEEP SUMMARY
============================================================
  sweep-lr2e4-bs64: val_loss=0.002294 (3 epochs, 18.4min)
  sweep-lr1e4-bs64: val_loss=0.002319 (4 epochs, 15.4min)
  sweep-lr5e5-bs64: val_loss=0.002420 (4 epochs, 15.4min)
  sweep-lr1e3-frozen: val_loss=0.002537 (5 epochs, 16.3min)
  sweep-lr2e4-frozen: val_loss=0.002619 (5 epochs, 16.3min)
```

Full training (best config lr=2e-4 unfrozen):
```
  Ep   1/100 | train: 0.004966 | val: 0.003609 | best: 0.003609 | lr: 2.00e-04 | 3.8min
  Ep   2/100 | train: 0.002498 | val: 0.002414 | best: 0.002414 | lr: 2.00e-04 | 7.7min
  ...
  Ep  21/100 | train: 0.002101 | val: 0.002111 | best: 0.002111 | lr: 1.79e-04 | 80.2min
  Ep  22/100 | train: 0.002093 | val: 0.002108 | best: 0.002108 | lr: 1.77e-04 | 84.1min
  Ep  23/100 | train: 0.002162 | val: 0.002197 | best: 0.002108 | lr: 1.75e-04 | 87.9min

[2026-05-09T03:53:27.956] error: *** STEP 13613743.6 ON node3003 CANCELLED AT 2026-05-09T03:53:27 DUE TO TIME LIMIT ***
```

SLURM `sacct` shows total allocation `03:00:01` (TIMEOUT), training step `.6` ran `01:28:49` before cancellation.

### `research3/models/unet_uniform_amp/best_flow.pt`

| Field | Value |
|---|---|
| Pool path | `/home/chenxy/orcd/pool/datasets/research3/models/unet_uniform_amp/best_flow.pt` |
| Session | `52eb744a-c93c-4418-b198-2e9d70418bf3` |
| Branch | research3 |
| Commit | `0fb25d3` ("iter-005: UNet v2 from scratch with uniform t + AMP -- CRPS 0.173 matches research2 baseline") |
| Norm stats | `research3/models/unet_uniform_amp/norm_stats.pt` (same session) |

**Command:**
```
python3 scripts/gpu_run.py 13413835 python src/exp-spatial-4x-crps-v1/flow_matching_v2.py \
  --mode train --basedir external/constrained-downscaling \
  --save_dir /home/chenxy/orcd/pool/datasets/research3/models/unet_uniform_amp \
  --batch_size 64 --epochs 40 --lr 1e-4 --base_channels 64 \
  --channel_mults "1,2,4" --attn_heads 4 --t_sampling uniform --amp
```

**Script output (quoted, abbreviated):**
```
#params: 13,074,561
Timestep sampling: uniform
AMP (mixed precision) enabled

Epoch 1/40, Train: 0.490532, Val: 0.370166, LR: 0.000100, Time: 3.5min
Epoch 2/40, Train: 0.366101, Val: 0.345536, LR: 0.000099, Time: 7.0min
...
Epoch 39/40, Train: 0.255869, Val: 0.253853, LR: 0.000000, Time: 135.4min
Epoch 40/40, Train: 0.256461, Val: 0.251844, LR: 0.000000, Time: 138.9min

Training complete. Best val loss: 0.251844
Total time: 138.9 min
```

**CRPS eval output (same session, 10K test):**
```
Evaluating 10000 samples, 10 ensemble, 10 euler steps, constraint=addcl...
Model epoch: 40, val_loss: 0.251844

Results (test, 10 ens, 10 euler steps, constraint=addcl):
  CRPS (paper): 0.093571
  CRPS (std):   0.173099
  MAE:          0.245027
  RMSE:         0.454637
  Mass viol:    0.000001
```

### `research3/models/unet_ema_logitnorm/best_flow.pt`

| Field | Value |
|---|---|
| Pool path | `/home/chenxy/orcd/pool/datasets/research3/models/unet_ema_logitnorm/best_flow.pt` |
| Session | `50c42ffe-cfd2-4237-8b4d-6fbc5643824b` |
| Branch | research3 |
| Commit | `8c386d9` ("iter-003: UNet v2 + EMA + logit-normal timestep sampling -- logit-normal doesn't help, EMA hurts with short training") |
| Norm stats | `research3/models/unet_ema_logitnorm/norm_stats.pt` (same session) |

**Command:**
```
python3 scripts/gpu_run.py 13392152 bash src/exp-spatial-4x-crps-v1/train_unet_ema_logitnorm.sh \
  /home/chenxy/orcd/pool/datasets/research3/models/unet_ema_logitnorm
```

**Script output (quoted, abbreviated):**

Training was initially started for 200 epochs, stopped after 3, resumed for 26:
```
#params: 13,074,561

Epoch 1/200, Train: 0.430229, Val: 0.374611, LR: 0.000100, Time: 4.5min
Epoch 2/200, Train: 0.312170, Val: 0.350547, LR: 0.000100, Time: 8.9min
Epoch 3/200, Train: 0.292932, Val: 0.338637, LR: 0.000100, Time: 13.4min
```

Resumed (26 epochs):
```
Epoch 4/26, Train: 0.282463, Val: 0.325660, LR: 0.000094, Time: 4.5min
...
Epoch 23/26, Train: 0.230496, Val: 0.267351, LR: 0.000003, Time: 89.2min
Epoch 24/26, Train: 0.230567, Val: 0.270019, LR: 0.000001, Time: 93.6min
Epoch 25/26, Train: 0.229719, Val: 0.268316, LR: 0.000000, Time: 98.1min
Epoch 26/26, Train: 0.229658, Val: 0.268874, LR: 0.000000, Time: 102.5min

Saved EMA model to /home/chenxy/orcd/pool/datasets/research3/models/unet_ema_logitnorm/best_flow_ema.pt

Training complete. Best val loss: 0.267351
Total time: 102.5 min
```

**CRPS eval output — regular model (same session, 10K test):**
```
=== Evaluating regular model (10K test, 10 ensemble, Euler 10, AddCL) ===
Evaluating 10000 samples, 10 ensemble, 10 euler steps, constraint=addcl...
Model epoch: 23, val_loss: 0.267351

Results (test, 10 ens, 10 euler steps, constraint=addcl):
  CRPS (paper): 0.097641
  CRPS (std):   0.179072
  MAE:          0.257364
  RMSE:         0.498021
  Mass viol:    0.000001
```

**CRPS eval output — EMA model (same session, 10K test):**
```
=== Evaluating EMA model (10K test, 10 ensemble, Euler 10, AddCL) ===
Evaluating 10000 samples, 10 ensemble, 10 euler steps, constraint=addcl...
Model epoch: 26, val_loss: 0.268874

Results (test, 10 ens, 10 euler steps, constraint=addcl):
  CRPS (paper): 0.121085
  CRPS (std):   0.227670
  MAE:          0.307132
  RMSE:         0.649675
  Mass viol:    0.000001
```

### `research6/models/flow_v2_zscore/best_flow.pt`

| Field | Value |
|---|---|
| Pool path | `/home/chenxy/orcd/pool/datasets/research6/models/flow_v2_zscore/best_flow.pt` |
| Session | `9289eca6-e2ac-4255-b737-691efbcf8d85` |
| Branch | research6 |
| Commit | `55119de` ("z-score normalization fix: CRPS 0.178 (was 0.232), false OT-coupling narrative corrected") |
| Norm stats | `research6/models/flow_v2_zscore/norm_stats.pt` (same session) |

**Command:**
```
scripts/gpu_run.sh 13444815 "python -u src/exp-spatial-4x-crps-v1/flow_matching_v2.py \
  --mode train --epochs 40 --batch_size 64 \
  --save_dir /orcd/pool/007/chenxy/datasets/research6/models/flow_v2_zscore"
```

**Script output (quoted, abbreviated):**
```
#params: 13,074,561

Epoch 1/40, Train: 0.486323, Val: 0.365564, LR: 0.000100, Time: 4.5min
Epoch 2/40, Train: 0.367666, Val: 0.340906, LR: 0.000099, Time: 8.9min
...
Epoch 39/40, Train: 0.254227, Val: 0.252294, LR: 0.000000, Time: 173.5min
Epoch 40/40, Train: 0.254782, Val: 0.250993, LR: 0.000000, Time: 178.0min

Training complete. Best val loss: 0.250993
Total time: 178.0 min
```

**CRPS eval output (same session, 50-sample CPU eval — the only eval that completed; a 10K eval was killed at 8032/10000 by SLURM time limit):**
```
Evaluating 50 samples, 10 ensemble, 10 euler steps, constraint=addcl...
Model epoch: 40, val_loss: 0.250993

Results (test, 10 ens, 10 euler steps, constraint=addcl):
  CRPS (paper): 0.096116
  CRPS (std):   0.177874
  MAE:          0.252113
  RMSE:         0.472997
  Mass viol:    0.000001
```

10K eval completed in a later session, commit `4409817`.

### `research5/pretrained_weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth` (SwinIR x4 pretrained)

| Field | Value |
|---|---|
| Session | `751af420-9f3b-458d-8934-7144c0e21a9e` |
| Branch | research5 |
| Commit | `e1b97ed` ("SwinIR pretrained SR: zero-shot eval + 2hr finetuning on ERA5 TCW 4x") |

**Command:**
```
python3 -c "import urllib.request; urllib.request.urlretrieve(
  'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth', ...)"
```

---

## NorESM Model Checkpoints

All trained during the ralph workflow on the `noresm-dataset` branch (2026-05-09).

### `noresm-dataset/models/flow-wide96-amp/best_flow.pt`

| Field | Value |
|---|---|
| Pool path | `/home/chenxy/orcd/pool/datasets/noresm-dataset/models/flow-wide96-amp/best_flow.pt` |
| Session | `19b0bff6-2910-414f-9ab9-bd4f43079460` |
| Ralph iteration | 1 |
| Commit | `982cbf1` ("add flow matching training CLI, train NorESM flow-wide96-amp model") |
| Norm stats | `noresm-dataset/models/flow-wide96-amp/norm_stats.pt` (same session) |

**Command:**
```
python3 scripts/gpu_run.py 13654342 "python scripts/train_flow.py --dataset noresm \
  --save-dir /home/chenxy/orcd/pool/datasets/noresm-dataset/models/flow-wide96-amp \
  --base-channels 96 --amp --batch-size 64 --epochs 40 --use-ema --ema-decay 0.9999"
```

**Script output (quoted, abbreviated):**
```
Dataset: noresm
Model params: 28,350,401 (28.4M)
Save dir: /home/chenxy/orcd/pool/datasets/noresm-dataset/models/flow-wide96-amp
Device: NVIDIA L40S

Epoch 1/40, Train: 0.449210, Val: 0.338677, LR: 0.000100, Time: 0.9min
Epoch 2/40, Train: 0.294811, Val: 0.253716, LR: 0.000099, Time: 1.8min
...
Epoch 38/40, Train: 0.130457, Val: 0.129882, LR: 0.000001, Time: 34.1min
Epoch 39/40, Train: 0.131228, Val: 0.130332, LR: 0.000000, Time: 35.0min
Epoch 40/40, Train: 0.132218, Val: 0.129887, LR: 0.000000, Time: 35.9min

Training complete. Best val loss: 0.129882
Total time: 35.9 min
```

### `noresm-dataset/models/harder/*.pth` (NorESM Harder CNN/GAN)

| Model | Session | Ralph iter | Commit |
|---|---|---|---|
| `twc_cnn_none.pth` | `8ff9bee6-c83e-44c5-ad78-14f7d8f63788` | 2 | `152957a` |
| `twc_cnn_softmax.pth` | same | 2 | same |
| `twc_gan_softmax.pth` | same | 2 | same |

**Commands:**
```
python3 scripts/gpu_run.py 13655663 python scripts/train_harder.py \
  --dataset noresm --model cnn --constraints none --model-id twc_cnn_none \
  --save-dir /home/chenxy/orcd/pool/datasets/noresm-dataset/models/harder --epochs 200

python3 scripts/gpu_run.py 13655663 python scripts/train_harder.py \
  --dataset noresm --model cnn --constraints softmax --model-id twc_cnn_softmax \
  --save-dir /home/chenxy/orcd/pool/datasets/noresm-dataset/models/harder --epochs 200

python3 scripts/gpu_run.py 13655663 python scripts/train_harder.py \
  --dataset noresm --model gan --constraints softmax --model-id twc_gan_softmax \
  --save-dir /home/chenxy/orcd/pool/datasets/noresm-dataset/models/harder --epochs 200
```

**Script output — twc_cnn_none (quoted, abbreviated):**
```
Loading noresm data...
  Train: torch.Size([24768, 1, 1, 32, 32]) -> torch.Size([24768, 1, 1, 64, 64])
  Val:   torch.Size([12384, 1, 1, 32, 32]) -> torch.Size([12384, 1, 1, 64, 64])
  Min-max range: [205.23, 320.33]

Model: cnn | constraints: none
  Params: 96,705 | upsampling: 2x

Epoch   1/200 | train: 0.012425 | val: 0.000599
...
Epoch 185/200 | train: 0.000328 | val: 0.000345
  -> Saved best model (val=0.000345)
...
Epoch 200/200 | train: 0.000324 | val: 0.000348

Training complete in 12.5 min
Best val loss: 0.000345
Checkpoint: /home/chenxy/orcd/pool/datasets/noresm-dataset/models/harder/twc_cnn_none.pth
```

**Script output — twc_cnn_softmax (quoted, final epochs):**
```
Epoch 199/200 | train: 0.000609 | val: 0.000649
  -> Saved best model (val=0.000649)
Epoch 200/200 | train: 0.000609 | val: 0.000649

Training complete in 13.0 min
Best val loss: 0.000649
Checkpoint: /home/chenxy/orcd/pool/datasets/noresm-dataset/models/harder/twc_cnn_softmax.pth
```

**Script output — twc_gan_softmax (quoted, abbreviated):**
```
Model: gan | constraints: softmax
  Params: 199,394 | upsampling: 2x
  Discriminator params: 286,561

Epoch   1/200 | train: 0.000746 | d_loss: 1.266587 | val: 0.000703
...
Epoch 187/200 | train: 0.007468 | d_loss: 0.000000 | val: 0.000696
...
Epoch 200/200 | train: 0.007931 | d_loss: 0.000000 | val: 0.000696

Training complete in 13.0 min
Best val loss: 0.000680
Checkpoint: /home/chenxy/orcd/pool/datasets/noresm-dataset/models/harder/twc_gan_softmax.pth
```

### `noresm-dataset/models/swinir_ft/best_swinir.pt`

| Field | Value |
|---|---|
| Pool path | `/home/chenxy/orcd/pool/datasets/noresm-dataset/models/swinir_ft/best_swinir.pt` |
| Session | `b08314e1-23b8-4bb5-895e-97528bf87525` |
| Ralph iteration | 3 |
| Commit | `0853673` ("add SwinIR training CLI, train NorESM swinir-finetuned model") |

**Command:**
```
python3 scripts/gpu_run.py 13658046 python scripts/train_swinir.py --dataset noresm \
  --pretrained-weights .../noresm-dataset/pretrained_weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth \
  --save-dir .../noresm-dataset/models/swinir_ft --epochs 100 --batch-size 64 --lr 2e-4 \
  --loss-fn l1 --wall-hours 1.5
```

**Script output (quoted, abbreviated):**
```
Device: cuda
GPU: NVIDIA L40S
Loading noresm data...
  Train: torch.Size([24768, 1, 32, 32]) -> torch.Size([24768, 1, 64, 64])
  Val:   torch.Size([12384, 1, 32, 32]) -> torch.Size([12384, 1, 64, 64])
  Upsampling factor: 2x
  Normalization range: [203.3445, 320.3327]
Loading pretrained SwinIR from .../001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth
  Parameters: 11,748,093 total, 11,748,093 trainable

  Ep   1/100 | train: 0.024293 | val: 0.014994 | best: 0.014994 | lr: 2.00e-04 | 2.5min
  ...
  Ep  31/100 | train: 0.008742 | val: 0.010771 | best: 0.010771 | lr: 1.56e-04 | 77.2min
  ...
  Ep  37/100 | train: 0.008205 | val: 0.010860 | best: 0.010771 | lr: 1.40e-04 | 92.1min

Wall time limit reached (1.54h). Stopping.

Training complete. 92.1 min, 37 epochs.
Best val loss: 0.010771
Saved to: /home/chenxy/orcd/pool/datasets/noresm-dataset/models/swinir_ft
```

**Checkpoint verification:**
```
Epoch: 30
Val loss: 0.010771
State dict params: 16,614,141
Forward: torch.Size([2, 1, 32, 32]) -> torch.Size([2, 1, 64, 64])
```

### `noresm-dataset/pretrained_weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth` (SwinIR x2 pretrained)

| Field | Value |
|---|---|
| Session | `b08314e1-23b8-4bb5-895e-97528bf87525` |
| Ralph iteration | 3 |
| Commit | `0853673` (same as SwinIR finetuned) |

**Command:**
```
curl -L -o .../noresm-dataset/pretrained_weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth \
  "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth"
```

---

## Datasets (inputs to evaluation)

| Dataset | Pool path | Source | Size |
|---|---|---|---|
| ERA5 TCW 4x | `era5_sr_data/` | [Google Drive 1IENhP1-aTYyqOkRcnmCIvxXkvUW2Qbdx](https://drive.google.com/file/d/1IENhP1-aTYyqOkRcnmCIvxXkvUW2Qbdx) | 4 GB |
| NorESM TAS 2x | `noresm-dataset/noresm/` | [Google Drive 1D5tLE7cGcvh3dap-P3VOLEOK_7FqdChF](https://drive.google.com/file/d/1D5tLE7cGcvh3dap-P3VOLEOK_7FqdChF) | 968 MB |

ERA5 downloaded in session `2947daee` (2026-05-03) via `gdown`. NorESM downloaded in session `9e2b798b-374f-4607-a147-7f9b39788f55` (ralph iteration 0, 2026-05-09).

---

## Dependency Graph

```
make_figures.py
├── eval_results_500.json (ERA5, 15 methods)
│   ├── research3/models/unet_wide96_amp/best_flow.pt     ← session dbda10ea, commit 7718a9d
│   ├── research3/models/unet_uniform_amp/best_flow.pt    ← session 52eb744a, commit 0fb25d3
│   ├── research3/models/unet_ema_logitnorm/best_flow.pt  ← session 50c42ffe, commit 8c386d9
│   ├── research6/models/flow_v2_zscore/best_flow.pt      ← session 9289eca6, commit 55119de
│   ├── organize2/models/harder/twc_cnn_none.pth           ← session db7c9287, commit 03adf1b
│   ├── organize2/models/harder/twc_cnn_softmax.pth        ← session db7c9287, commit 03adf1b
│   ├── organize2/models/harder/twc_gan_softmax.pth        ← session db7c9287, commit 03adf1b
│   ├── spatial-4x-add-v2/models/swinir_ft/best_swinir.pt ← session ec371f33, commit 99d3372
│   └── research5/pretrained_weights/SwinIR-M_x4.pth      ← session 751af420, commit e1b97ed
│
├── noresm_eval_results_500.json (NorESM, 12 methods)
│   ├── noresm-dataset/models/flow-wide96-amp/best_flow.pt ← session 19b0bff6, commit 982cbf1
│   ├── noresm-dataset/models/harder/twc_cnn_none.pth      ← session 8ff9bee6, commit 152957a
│   ├── noresm-dataset/models/harder/twc_cnn_softmax.pth   ← session 8ff9bee6, commit 152957a
│   ├── noresm-dataset/models/harder/twc_gan_softmax.pth   ← session 8ff9bee6, commit 152957a
│   ├── noresm-dataset/models/swinir_ft/best_swinir.pt     ← session b08314e1, commit 0853673
│   └── noresm-dataset/pretrained_weights/SwinIR-M_x2.pth  ← session b08314e1, commit 0853673
│
├── era5_sr_data/ (test split, live inference for sample figures)
└── noresm-dataset/noresm/ (test split, live inference for sample figures)
```

## Session Index

All sessions in `/home/chenxy/.claude/projects/-workspace/`.

| Session ID | Date | Branch | Role |
|---|---|---|---|
| `db7c9287-656e-43bb-8dae-b7bd77aaac67` | 2026-04-26 | master | Train all ERA5 Harder models (CNN + GAN, sequential + parallel sbatch) |
| `92eeedf1-3f62-480e-b334-7ab2f999eeb3` | 2026-05-03 | research2 | Re-train ERA5 Harder CNN models (twc_cnn_none, twc_cnn_softmax) |
| `2947daee-2d9c-48a1-a387-87d5d29cf647` | 2026-05-03 | research | Download ERA5 data |
| `50c42ffe-cfd2-4237-8b4d-6fbc5643824b` | 2026-05-05 | research3 | Train ERA5 flow logitnorm-ema model |
| `52eb744a-c93c-4418-b198-2e9d70418bf3` | 2026-05-06 | research3 | Train ERA5 flow uniform-amp model |
| `dbda10ea-8db3-4845-bd83-95973eafc71d` | 2026-05-06 | research3 | Train ERA5 flow wide96 model |
| `9289eca6-e2ac-4255-b737-691efbcf8d85` | 2026-05-06 | research6 | Train ERA5 flow v2-zscore model |
| `751af420-9f3b-458d-8934-7144c0e21a9e` | 2026-05-06 | research5 | Download SwinIR x4 pretrained weights |
| `7e509c59-72ff-4a37-9665-9a08894827fb` | 2026-05-08 | organize2 | First ERA5 eval (8 methods, via sbatch after failed srun attempts) |
| `1eb7efaa-92b0-4756-b925-14ae6b6331f4` | 2026-05-09 | organize2 | Copy Harder models to pool, re-eval with Harder (11 methods) |
| `ec371f33-e763-4141-a75a-4cbc4b4fe00c` | 2026-05-09 | spatial-4x-add-v2 | Train ERA5 SwinIR finetuned (sweep + full training) |
| `84a00048-762e-46e4-909a-b70934e6297b` | 2026-05-09 | spatial-4x-add-v2 | Re-eval SwinIR at 500 samples (was 10K), merge into eval_results_500.json |
| `9e2b798b-374f-4607-a147-7f9b39788f55` | 2026-05-09 | noresm-dataset | Ralph iter 0: download NorESM, write loader |
| `19b0bff6-2910-414f-9ab9-bd4f43079460` | 2026-05-09 | noresm-dataset | Ralph iter 1: train NorESM flow model |
| `8ff9bee6-c83e-44c5-ad78-14f7d8f63788` | 2026-05-09 | noresm-dataset | Ralph iter 2: train NorESM Harder CNN/GAN |
| `b08314e1-23b8-4bb5-895e-97528bf87525` | 2026-05-09 | noresm-dataset | Ralph iter 3: train NorESM SwinIR |
| `f0508f7e-36d1-4d6d-8667-040498291bb0` | 2026-05-09 | noresm-dataset | Ralph iter 4: evaluate all NorESM models |
| `a4338ecc-1c46-4160-a35c-f15dc98d72ec` | 2026-05-09 | noresm-dataset | Ralph iter 5: metric figures |
| `ebe4b7b6-a2fe-4ee9-87c9-e573a2475683` | 2026-05-09 | noresm-dataset | Ralph iter 6: sample prediction figures |
