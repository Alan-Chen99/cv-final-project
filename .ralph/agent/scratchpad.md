# NorESM Dataset Integration

## Run prefix: grbh-zbsi
## Starting commit: 9f86030
## Starting time: 2026-05-09T21:26:19Z

## Key facts
- NorESM: 32×32 → 64×64 (2x upsampling), surface temperature (tas), 24K/12K/12K splits
- ERA5: 32×32 → 128×128 (4x upsampling), total column water (TCW), 40K/10K/10K splits
- NorESM LR/HR from different simulation runs (not pooled), so constraints may hurt
- Data format: same as ERA5 — PyTorch .pt files with shape (N, 1, 1, H, W)

## Current models in figures (eval_results_500.json):
1. bilinear, bilinear+addcl
2. bicubic, bicubic+addcl
3. harder-cnn, harder-cnn+smcl, harder-gan+smcl
4. flow-wide96-amp (28M), flow-uniform-amp (13M), flow-logitnorm-ema (13M), flow-v2-zscore (13M)
5. swinir-zeroshot, swinir-zeroshot+addcl, swinir-finetuned, swinir-finetuned+addcl

## Top 3 concerns (iteration 1)

### 1. Workflow: Data not yet downloaded
NorESM dataset needs to be downloaded from Google Drive before anything else can happen.
Must go to pool/datasets/noresm-dataset/ (branch-specific write).

### 2. Quality: Code is tightly coupled to ERA5 4x
- `load_era5_tcw` is hardcoded in training loop, evaluation, and figure generation
- Shapes hardcoded to 128x128 output, factor=4
- Need to generalize data loading for multiple datasets with different resolutions

### 3. Workflow: Understanding which models to train
- Must match models in figures/eval_results
- Constraint models may not apply to NorESM (different physics)
- Need to decide which subset to train on NorESM

## Plan (across iterations)
1. ✅ Iteration 1: Download NorESM data + write data loader + generalize trainer
2. Iteration 2: Write NorESM training script + train flow model
3. Iteration 3: Train Harder CNN/GAN on NorESM
4. Iteration 4: Train SwinIR on NorESM (or evaluate zero-shot)
5. Iteration 5: Evaluate all NorESM models, write eval results JSON
6. Iteration 6: Update figures to show both datasets
7. Iteration 7: Visual review and polish

## Iteration 1 results
- Downloaded NorESM data to pool/datasets/noresm-dataset/noresm/ (741MB zip → 726MB extracted)
- Verified shapes: input=(N, 1, 1, 32, 32) → target=(N, 1, 1, 64, 64), 2x upsampling
- Constraint violation: ~1.82K (significant — constraints will degrade metrics)
- Residual stats: mean=-0.50K, std=2.83K (surface temperature)
- Wrote load_noresm_tas() data loader
- Added dataset='era5'|'noresm' to TrainConfig + DATASET_LOADERS dispatch
- Updated _compute_minmax_stats to accept dataset parameter
- Added NorESM tests: 14/14 pass
- Lint, format, typecheck all pass

### Key architecture note for next iteration
NorESM is 2x SR (32→64) vs ERA5 4x SR (32→128). The AttentionUNet and flow matching
training code are resolution-agnostic (use torch.randn_like for matching shapes).
But the Harder ResNet uses upsampling_factor in its architecture (PixelShuffle).
Training Harder models on NorESM requires upsampling_factor=2.

### Models to train on NorESM
From eval_results_500.json, the models plotted in figures:
- bilinear, bicubic (no training needed — just interpolation)
- harder-cnn (none constraint), harder-cnn+smcl, harder-gan+smcl
- flow-wide96-amp (28M) — best flow model
- swinir-zeroshot (no training), swinir-finetuned
- Constraint variants (+addcl, +smcl) — will likely hurt NorESM metrics

Skip: flow-uniform-amp, flow-logitnorm-ema, flow-v2-zscore (ablation variants, not main results)

## Ending commit: 90eeb59
## Ending time: 2026-05-09T21:40:00Z

---

## Iteration 2
### Run prefix: coral-maze
### Starting commit: 6dbf7c7
### Starting time: 2026-05-09T21:36:34Z

### Top 3 concerns (iteration 2)

#### 1. Workflow: No CLI entry point for training
FlowMatchingTrainer exists as a library class but there's no script to invoke it.
Need to write a reusable training CLI script that works with SLURM.

#### 2. Quality: upsampling_factor=4 hardcoded as defaults across evaluation code
evaluate_flow_model(), evaluate_ensemble(), evaluate_deterministic(), eval_bicubic(),
eval_bilinear(), evaluate_harder_cnn(), evaluate_harder_gan() all default to
upsampling_factor=4. NorESM needs 2. All callsites must explicitly pass the correct
factor. This is a correctness risk for future iterations.

#### 3. Quality: Model capacity for 2x SR
NorESM is 2x SR (64x64 output) vs ERA5 4x (128x128 output). The 28M flow-wide96-amp
model may be overparameterized for the simpler 2x task. But the goal is to match the
existing figures, so train with same architecture (base_channels=96).

### Plan for this iteration
- Write training CLI script (scripts/train_flow.py)
- Allocate GPU node
- Train flow-wide96-amp on NorESM (base_channels=96, amp=True, uniform timesteps, 40 epochs)
- Save checkpoint to pool/datasets/noresm-dataset/models/flow-wide96-amp/

### Iteration 2 results
- Wrote scripts/train_flow.py — reusable CLI for flow matching training on any dataset
- Allocated GPU (L40S on node3008, job 13654342)
- Trained flow-wide96-amp (28.4M params) on NorESM for 40 epochs in 35.9 min
- Best val loss: 0.129882 at epoch 38
- Checkpoint: pool/datasets/noresm-dataset/models/flow-wide96-amp/best_flow.pt (454MB)
- Norm stats: res_mean=-0.495, res_std=2.830, lr_mean=279.87, lr_std=21.65
- Has EMA weights: yes
- Lint/format/typecheck: all pass
- GPU allocation released

### Next iteration
- Train Harder CNN/GAN on NorESM (requires upsampling_factor=2)
- Need to check if Harder training code exists or if we need to write it

### Ending commit: dea7b3b
### Ending time: 2026-05-09T22:20:00Z
