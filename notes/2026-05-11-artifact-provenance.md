# Artifact Provenance: Plot-Generating Artifacts

All artifacts that feed into `scripts/make_figures.py` (metric bar charts, sample
comparisons, error maps, ensemble spread, dual-dataset comparisons, constraint impact).

## Evaluation Result JSONs (direct inputs to make_figures.py)

### `eval_results_500.json` (ERA5 TCW 4x, 11 methods)

Created across 3 sessions, each adding methods:

| Version | Session | Branch | Command | Commit |
|---|---|---|---|---|
| v1 (8 methods) | `7e509c59-72ff-4a37-9665-9a08894827fb` | organize2 | `srun --jobid=13601381 ... python scripts/run_eval.py --max-samples 500 --n-ensemble 10 --ode-steps 10 --constraint addcl --sampler midpoint --output eval_results_500.json` | `56f4ef6` |
| v2 (+Harder, 11 methods) | `1eb7efaa-92b0-4756-b925-14ae6b6331f4` | organize2 | `python3 scripts/gpu_run.py 13609961 'python scripts/run_eval.py --max-samples 500 --output eval_results_500.json'` | `ecd1e83` |
| v3 (+SwinIR, re-eval 500) | `84a00048-762e-46e4-909a-b70934e6297b` | spatial-4x-add-v2 | `python3 scripts/gpu_run.py 13623783 python scripts/eval_swinir_only.py` | `ace332d` |

Current file: `/workspace/eval_results_500.json` (2854 B, 2026-05-09). This is v3 (final).

### `noresm_eval_results_500.json` (NorESM TAS 2x, 12 variants)

| Session | Branch | Command | Commit |
|---|---|---|---|
| `f0508f7e-36d1-4d6d-8667-040498291bb0` | noresm-dataset | `python scripts/gpu_run.py 13661966 'python scripts/run_eval_noresm.py --max-samples 500 --constraint none --output noresm_eval_results_500.json'` | `fa63326` |

Current file: `/workspace/noresm_eval_results_500.json` (2341 B, 2026-05-09).

## Figure Generation

| Figures | Session | Branch | Command | Commit |
|---|---|---|---|---|
| Metric bar charts (9 PNGs) | `a4338ecc-1c46-4160-a35c-f15dc98d72ec` | noresm-dataset | `uv run python scripts/make_figures.py --metrics-only --output-dir figures/` | `04e8172` |
| Sample predictions (35 PNGs) | `ebe4b7b6-a2fe-4ee9-87c9-e573a2475683` | noresm-dataset | `uv run python scripts/gpu_run.py 13662769 uv run python scripts/make_figures.py --pool-dir /home/chenxy/orcd/pool/datasets --n-samples 5 --n-ensemble 10` | `a681142` |

Output: `figures/era5/` (16 files), `figures/noresm/` (16 files), `figures/` (3 cross-dataset files).

---

## ERA5 Model Checkpoints

### `research3/models/unet_wide96_amp/best_flow.pt` (ERA5 flow, 28.4M params)

The primary ERA5 flow model used in all plots.

| Field | Value |
|---|---|
| Pool path | `/home/chenxy/orcd/pool/datasets/research3/models/unet_wide96_amp/best_flow.pt` |
| Session | `dbda10ea-8db3-4845-bd83-95973eafc71d` |
| Branch | research3 |
| Command | `python3 scripts/gpu_run.py 13435838 python src/exp-spatial-4x-crps-v1/flow_matching_v2.py --mode train --basedir external/constrained-downscaling --save_dir /home/chenxy/orcd/pool/datasets/research3/models/unet_wide96_amp --batch_size 64 --epochs 40 --lr 1e-4 --base_channels 96 --channel_mults "1,2,4" --attn_heads 4 --t_sampling uniform --amp` |
| Commit | `7718a9d` ("iter-007: Wider UNet (96ch, 28.4M) -- val_loss 0.243 beats baseline 0.252") |
| Norm stats | `research3/models/unet_wide96_amp/norm_stats.pt` (same session) |
| Training | 25/40 epochs (preempted), val_loss 0.2432 |
| CRPS | 0.1676 on 10K test (midpoint, 10 ensemble, addcl) |

### `organize2/models/harder/*.pth` (ERA5 Harder CNN/GAN)

These were trained using the original Harder et al. codebase (`external/constrained-downscaling/main.py`), then copied to pool.

**CNN models** — trained in session `92eeedf1-3f62-480e-b334-7ab2f999eeb3` (2026-05-03, research2 branch):

| Model | Original path | Training |
|---|---|---|
| `twc_cnn_none.pth` | `research2/external/constrained-downscaling/models/` | `main.py --model cnn --constraints none --epochs 200` (via `scripts/run_cnn_baselines.sh`) |
| `twc_cnn_softmax.pth` | `research2/external/constrained-downscaling/models/` | `main.py --model cnn --constraints softmax --epochs 200` (via `scripts/run_cnn_baselines.sh`, re-trained in v2 run) |

Session `92eeedf1` wrote `scripts/run_cnn_baselines.sh` and executed it via `srun --jobid=13092600` (v1) and `srun --jobid=13098698` (v2). Log: `logs/cnn_baselines.log`, `logs/cnn_baselines_v2.log`. Commit: `e3cc488` ("iter-001: CNN baselines for 32x32->128x128 downscaling with CRPS").

**GAN models** — trained in session `2947daee-2d9c-48a1-a387-87d5d29cf647` (2026-05-03):

| Model | Command |
|---|---|
| `twc_gan_none.pth` | `python main.py --dataset era5_sr_data --model gan --model_id gan_none --constraints none --epochs 200 --batch_size 256 --test_val_train test` |
| `twc_gan_softmax.pth` | `python main.py --dataset era5_sr_data --model gan --model_id gan_softmax --constraints softmax --epochs 200 --batch_size 256 --test_val_train test` |

**Copied to pool** in session `1eb7efaa-92b0-4756-b925-14ae6b6331f4` (organize2 branch):
```
cp -v /home/chenxy/repos/workspace/main/external/constrained-downscaling/models/*.pth \
  /home/chenxy/orcd/pool/datasets/organize2/models/harder/
```
Commit: `d390f64` ("add Harder et al. CNN, GAN, CNN+SmCL baselines to eval and figures").

### `spatial-4x-add-v2/models/swinir_ft/best_swinir.pt` (ERA5 SwinIR finetuned)

| Field | Value |
|---|---|
| Pool path | `/home/chenxy/orcd/pool/datasets/spatial-4x-add-v2/models/swinir_ft/best_swinir.pt` |
| Session | `ec371f33-e763-4141-a75a-4cbc4b4fe00c` |
| Branch | spatial-4x-add-v2 |
| Command | `python3 scripts/gpu_run.py 13613743 python -m downscaling.training.swinir --pool-dir /home/chenxy/orcd/pool/datasets --epochs 100 --batch-size 64 --lr 2e-4 --wall-hours 2.0` |
| Commit | `99d3372` ("add SwinIR evaluation and finetuning to src/") |
| Training | 23/100 epochs (wall-clock limited), best val_loss 0.002108 |
| Pretrained weights | `research5/pretrained_weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth` |

### `research5/pretrained_weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth` (SwinIR x4 pretrained)

| Field | Value |
|---|---|
| Session | `751af420-9f3b-458d-8934-7144c0e21a9e` |
| Branch | research5 |
| Command | `python3 -c "import urllib.request; urllib.request.urlretrieve('https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth', ...)"` |
| Source | JingyunLiang/SwinIR GitHub release (DF2K, classical SR, medium, x4) |
| Commit | `e1b97ed` ("SwinIR pretrained SR: zero-shot eval + 2hr finetuning on ERA5 TCW 4x") |

---

## NorESM Model Checkpoints

All trained during the ralph workflow on the `noresm-dataset` branch (2026-05-09).

### `noresm-dataset/models/flow-wide96-amp/best_flow.pt` (NorESM flow, 28.4M params)

| Field | Value |
|---|---|
| Pool path | `/home/chenxy/orcd/pool/datasets/noresm-dataset/models/flow-wide96-amp/best_flow.pt` |
| Session | `19b0bff6-2910-414f-9ab9-bd4f43079460` |
| Ralph iteration | 1 |
| Command | `python3 scripts/gpu_run.py 13654342 "python scripts/train_flow.py --dataset noresm --save-dir /home/chenxy/orcd/pool/datasets/noresm-dataset/models/flow-wide96-amp --base-channels 96 --amp --batch-size 64 --epochs 40 --use-ema --ema-decay 0.9999"` |
| Commit | `982cbf1` ("add flow matching training CLI, train NorESM flow-wide96-amp model") |
| Training | 40 epochs, best val_loss 0.1299, 35.9 min |
| Norm stats | `noresm-dataset/models/flow-wide96-amp/norm_stats.pt` (same session) |

### `noresm-dataset/models/harder/*.pth` (NorESM Harder CNN/GAN)

| Model | Session | Ralph iter | Command | Commit |
|---|---|---|---|---|
| `twc_cnn_none.pth` | `8ff9bee6-c83e-44c5-ad78-14f7d8f63788` | 2 | `python3 scripts/gpu_run.py 13655663 python scripts/train_harder.py --dataset noresm --model cnn --constraints none --model-id twc_cnn_none --save-dir .../noresm-dataset/models/harder --epochs 200` | `152957a` |
| `twc_cnn_softmax.pth` | same | 2 | same script, `--constraints softmax --model-id twc_cnn_softmax` | same |
| `twc_gan_softmax.pth` | same | 2 | same script, `--model gan --constraints softmax --model-id twc_gan_softmax` | same |

### `noresm-dataset/models/swinir_ft/best_swinir.pt` (NorESM SwinIR finetuned)

| Field | Value |
|---|---|
| Pool path | `/home/chenxy/orcd/pool/datasets/noresm-dataset/models/swinir_ft/best_swinir.pt` |
| Session | `b08314e1-23b8-4bb5-895e-97528bf87525` |
| Ralph iteration | 3 |
| Command | `python3 scripts/gpu_run.py 13658046 python scripts/train_swinir.py --dataset noresm --pretrained-weights .../noresm-dataset/pretrained_weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth --save-dir .../noresm-dataset/models/swinir_ft --epochs 100 --batch-size 64 --lr 2e-4 --loss-fn l1 --wall-hours 1.5` |
| Commit | `0853673` ("add SwinIR training CLI, train NorESM swinir-finetuned model") |
| Training | 37/100 epochs (wall-clock limited), best val_loss 0.010771, 92 min on L40S |

### `noresm-dataset/pretrained_weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth` (SwinIR x2 pretrained)

| Field | Value |
|---|---|
| Session | `b08314e1-23b8-4bb5-895e-97528bf87525` |
| Ralph iteration | 3 |
| Command | `curl -L -o .../noresm-dataset/pretrained_weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth"` |
| Source | JingyunLiang/SwinIR GitHub release (DF2K, classical SR, medium, x2) |
| Commit | `0853673` (same as SwinIR finetuned) |

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
├── eval_results_500.json (ERA5)
│   ├── research3/models/unet_wide96_amp/best_flow.pt     ← session dbda10ea, commit 7718a9d
│   ├── organize2/models/harder/twc_cnn_none.pth           ← session 92eeedf1, commit e3cc488
│   ├── organize2/models/harder/twc_cnn_softmax.pth        ← session 92eeedf1, commit e3cc488
│   ├── organize2/models/harder/twc_gan_softmax.pth        ← session 2947daee, commit 03adf1b
│   ├── spatial-4x-add-v2/models/swinir_ft/best_swinir.pt ← session ec371f33, commit 99d3372
│   └── research5/pretrained_weights/SwinIR-M_x4.pth      ← session 751af420, commit e1b97ed
│
├── noresm_eval_results_500.json (NorESM)
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
| `92eeedf1-3f62-480e-b334-7ab2f999eeb3` | 2026-05-03 | research2 | Train ERA5 Harder CNN models (run_cnn_baselines.sh) |
| `2947daee-2d9c-48a1-a387-87d5d29cf647` | 2026-05-03 | research2 | Train ERA5 Harder GAN models, download ERA5 data |
| `dbda10ea-8db3-4845-bd83-95973eafc71d` | 2026-05-06 | research3 | Train ERA5 flow wide96 model |
| `751af420-9f3b-458d-8934-7144c0e21a9e` | 2026-05-06 | research5 | Download SwinIR x4 pretrained weights |
| `7e509c59-72ff-4a37-9665-9a08894827fb` | 2026-05-08 | organize2 | First ERA5 eval (8 methods) |
| `1eb7efaa-92b0-4756-b925-14ae6b6331f4` | 2026-05-09 | organize2 | Copy Harder models to pool, re-eval with Harder (11 methods) |
| `ec371f33-e763-4141-a75a-4cbc4b4fe00c` | 2026-05-09 | spatial-4x-add-v2 | Train ERA5 SwinIR finetuned |
| `84a00048-762e-46e4-909a-b70934e6297b` | 2026-05-09 | spatial-4x-add-v2 | Fix SwinIR eval (500 samples) |
| `9e2b798b-374f-4607-a147-7f9b39788f55` | 2026-05-09 | noresm-dataset | Ralph iter 0: download NorESM, write loader |
| `19b0bff6-2910-414f-9ab9-bd4f43079460` | 2026-05-09 | noresm-dataset | Ralph iter 1: train NorESM flow model |
| `8ff9bee6-c83e-44c5-ad78-14f7d8f63788` | 2026-05-09 | noresm-dataset | Ralph iter 2: train NorESM Harder CNN/GAN |
| `b08314e1-23b8-4bb5-895e-97528bf87525` | 2026-05-09 | noresm-dataset | Ralph iter 3: train NorESM SwinIR |
| `f0508f7e-36d1-4d6d-8667-040498291bb0` | 2026-05-09 | noresm-dataset | Ralph iter 4: evaluate all NorESM models |
| `a4338ecc-1c46-4160-a35c-f15dc98d72ec` | 2026-05-09 | noresm-dataset | Ralph iter 5: metric figures |
| `ebe4b7b6-a2fe-4ee9-87c9-e573a2475683` | 2026-05-09 | noresm-dataset | Ralph iter 6: sample prediction figures |

ERA5 CNN models trained in session `92eeedf1` on the research2 branch (2026-05-03), commit `e3cc488`. The session wrote `scripts/run_cnn_baselines.sh` and ran it via srun.
