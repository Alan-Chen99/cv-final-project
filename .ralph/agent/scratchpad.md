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

---

## Iteration 3
### Run prefix: yfll-mvph
### Starting commit: 982cbf1
### Starting time: 2026-05-09T22:21:34Z

### Top 3 concerns (iteration 3)

#### 1. Workflow: flow-wide96-amp checkpoint never evaluated
Iteration 2 reported val loss 0.1299 but never ran evaluation metrics (CRPS, MAE, RMSE).
The checkpoint exists (454MB) but we have no evidence it produces correct predictions.
This is deferred risk — evaluation comes in iteration 5 per the plan.

#### 2. Quality: Harder training code in external/ is messy and tightly coupled
The external/constrained-downscaling/ code uses:
- Global mutable state (max_val, min_val in utils.py)
- Deprecated `torchgeometry` import
- Relative paths (`./models/`, `./data/`)
- No arg for data directory
Must write a clean standalone training script instead of calling external code directly.

#### 3. Quality: Discriminator architecture has 6 stride-2 convs designed for 128x128
For NorESM 64x64 HR output, the discriminator still works (adaptive final pooling)
but has fewer meaningful feature map resolutions. This may affect GAN quality.
Verified: 64→32→16→8→4→2→conv1(padding=1)→4→avgpool→1. Acceptable.

### Plan for this iteration
- Write scripts/train_harder.py — clean reusable CLI for Harder CNN/GAN training
- Allocate GPU
- Train all 3 Harder models on NorESM: harder-cnn, harder-cnn+smcl, harder-gan+smcl
- Save checkpoints to pool/datasets/noresm-dataset/models/harder/

### Iteration 3 results
- Wrote scripts/train_harder.py — clean CLI for Harder CNN/GAN training on any dataset
- Allocated GPU (L40S on node3005, job 13655663)
- Verified ResNet architecture works with upsampling_factor=2: CNN=96,705 params, GAN=199,394 params
- Trained 3 models on NorESM (200 epochs each, ~13 min each):

| Model | Constraints | Best Val | Time |
|-------|-------------|----------|------|
| harder-cnn | none | 0.000345 | 12.5 min |
| harder-cnn+smcl | softmax | 0.000649 | 13.0 min |
| harder-gan+smcl | softmax | 0.000680 | 13.0 min |

- Note: constrained models have worse val loss — confirms NorESM constraint violation
- GAN discriminator collapsed (d_loss→0 by epoch ~50) but generator still learned
- All checkpoints verified: load + forward pass produces (B, 1, 1, 64, 64)
- Checkpoints at pool/datasets/noresm-dataset/models/harder/twc_{cnn_none,cnn_softmax,gan_softmax}.pth
- GPU released
- Lint/format pass

### Next iteration
- Train SwinIR on NorESM (zero-shot + finetuned) — or skip if SwinIR doesn't support 2x
- Then evaluate all models and produce eval_results JSON

### Ending commit: 152957a
### Ending time: 2026-05-09T23:09:47Z

---

## Iteration 4
### Run prefix: jade-flux
### Starting commit: a798861
### Starting time: 2026-05-09T23:10:56Z

### Top 3 concerns (iteration 4)

#### 1. Architecture: SwinIR pretrained weights are x4 only — cannot do NorESM 2x SR
The existing pretrained weights are `001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth`.
SwinIR embeds the upsampling factor in PixelShuffle layers — a x4 model cannot produce
x2 outputs. Need to download x2 pretrained weights from SwinIR GitHub releases.
Spandrel auto-detects architecture from weights, so loading code should work unchanged.

#### 2. Quality: SwinIR training code is hardcoded for ERA5
`_load_splits()` in `src/downscaling/training/swinir.py` hardcodes ERA5 data paths.
Need to write a new `scripts/train_swinir.py` CLI (like train_flow.py and train_harder.py)
that takes --dataset parameter and loads the correct data.

#### 3. Quality: No verification of NorESM data loader for SwinIR shapes
SwinIR expects input shape (N, 1, 32, 32) and produces (N, 1, 64, 64) for x2.
Need to verify that the NorESM data loader produces tensors compatible with SwinIR's
expected format (not the (N, 1, 1, H, W) 5D format used elsewhere).

### Plan for this iteration
1. Download SwinIR x2 pretrained weights
2. Write scripts/train_swinir.py — generalized SwinIR training CLI
3. Allocate GPU, train SwinIR finetuned on NorESM
4. Verify checkpoint produces correct output shapes
5. Commit

### Iteration 4 results
- Downloaded SwinIR x2 pretrained weights (65MB) to pool/datasets/noresm-dataset/pretrained_weights/
- Wrote scripts/train_swinir.py — generalized SwinIR training CLI (works for era5/noresm)
- Allocated GPU (L40S on node4504, job 13658046)
- Trained SwinIR finetuned on NorESM: 37 epochs in 92.1 min (wall limit 1.5h)
  - 11.7M trainable parameters
  - Best val loss: 0.010771 at epoch 31
  - Norm stats: vmin=203.3445, vmax=320.3327
- Checkpoint verified: load + forward pass produces (B, 1, 64, 64)
- Checkpoint at pool/datasets/noresm-dataset/models/swinir_ft/best_swinir.pt (65MB)
- GPU released
- Lint/format pass

### NorESM models status (all trained)
| Model | Checkpoint | Params |
|-------|-----------|--------|
| flow-wide96-amp | pool/.../flow-wide96-amp/best_flow.pt | 28.4M |
| harder-cnn | pool/.../harder/twc_cnn_none.pth | 97K |
| harder-cnn+smcl | pool/.../harder/twc_cnn_softmax.pth | 97K |
| harder-gan+smcl | pool/.../harder/twc_gan_softmax.pth | 199K |
| swinir-finetuned | pool/.../swinir_ft/best_swinir.pt | 11.7M |
| swinir-zeroshot | (uses x2 pretrained weights directly) | 11.7M |
| bilinear/bicubic | (no training needed) | - |

### Next iteration
- Evaluate ALL NorESM models, produce eval_results JSON (iteration 5)
- Need to write eval script that handles NorESM 2x + x2 pretrained weights

### Ending commit: 0853673
### Ending time: 2026-05-09T20:55:00Z

---

## Iteration 5
### Run prefix: pine-reef
### Starting commit: 8abfd26
### Starting time: 2026-05-10T00:51:48Z

### Top 3 concerns (iteration 5)

#### 1. Workflow: evaluate_flow_model returns EvalMetrics dataclass, not dict
run_eval.py wraps EvalMetrics into a dict via eval_flow_matching_model helper.
Writing run_eval_noresm.py, I initially called evaluate_flow_model directly and
tried to subscript the result as a dict — TypeError crash. Fixed by converting
EvalMetrics fields to dict after the call.

#### 2. Quality: Constraint flag applies globally to flow model eval
The --constraint flag defaults to "addcl" but NorESM LR/HR violate the mass
constraint (~1.8K mean violation). With addcl, flow model CRPS=1.383 (worse
than SwinIR finetuned). With constraint=none, CRPS=0.649 (best model).
Ran evaluation with constraint=none for the final JSON.

#### 3. Workflow: GPU nodes can't access /workspace mount
Compute nodes use NFS paths, not /workspace bind mount. Must use scripts/gpu_run.py
wrapper which resolves the real NFS path and runs inside the Apptainer container.

### Plan for this iteration
- Write scripts/run_eval_noresm.py — NorESM-specific evaluation script
- Allocate GPU, run evaluation with 500 test samples
- Save noresm_eval_results_500.json

### Iteration 5 results
- Wrote scripts/run_eval_noresm.py — complete NorESM evaluation script
- Allocated GPU (L40S on node4210, job 13661966, mit_preemptable)
- Evaluated all 12 model variants on 500 NorESM test samples
- Total eval time: ~100s (baselines ~1s, SwinIR ~30s, Harder ~2s, flow ~62s)
- GPU released

#### NorESM evaluation results (500 test samples, constraint=none for flow)

| Rank | Method | CRPS | MAE | RMSE | MassViol |
|------|--------|------|-----|------|----------|
| 1 | flow-wide96-amp (28M) | 0.649 | 0.967 | 1.513 | 1.119 |
| 2 | swinir-finetuned | 0.988 | 0.988 | 1.534 | 1.065 |
| 3 | harder-cnn | 1.131 | 1.131 | 1.694 | 0.943 |
| 4 | harder-cnn+smcl | 1.453 | 1.453 | 2.277 | 0.000 |
| 5 | swinir-finetuned+addcl | 1.455 | 1.455 | 2.279 | 0.000 |
| 6 | bilinear | 1.473 | 1.473 | 2.307 | 0.162 |
| 7 | swinir-zeroshot | 1.475 | 1.475 | 2.324 | 0.067 |
| 8 | bicubic | 1.477 | 1.477 | 2.319 | 0.061 |
| 9-12 | (addcl/smcl variants) | 1.478-1.481 | — | — | ~0.000 |

#### Key findings
- **Flow model is best** (CRPS=0.649) when unconstrained, beating SwinIR-FT by 34%
- **Constraints universally hurt NorESM** — LR/HR from different simulations
- **Unconstrained models** (flow, swinir-ft, harder-cnn) have high mass violation (~1K)
  but much better CRPS/MAE/RMSE
- **Zero-shot methods useless** — SwinIR-ZS barely beats bicubic
- Results saved to noresm_eval_results_500.json

### Bug fixed
- evaluate_flow_model returns EvalMetrics, not dict — must convert fields

### Next iteration
- Update figures to show both ERA5 and NorESM results (iteration 6)
- Generalize plotting code for multi-dataset comparison

### Ending commit: 7d4fae9
### Ending time: 2026-05-10T01:08:00Z

---

## Iteration 6
### Run prefix: 6439-5659
### Starting commit: fa63326
### Starting time: 2026-05-10T01:07:14Z

### Top 3 concerns (iteration 6)

#### 1. Quality: metrics.py is single-dataset only
plot_crps_comparison, plot_metrics_panel, plot_flow_vs_baseline all take one results
dict and produce ERA5-titled charts. Cannot show ERA5 vs NorESM comparison without
new plotting functions. Titles are hardcoded to "ERA5 TCW 4x Downscaling".

#### 2. Quality: sample visualization hardcodes ERA5 shapes
generate_baseline_predictions uses scale_factor=4, plot_sample_comparison hardcodes
(128, 128) for LR upsampling. NorESM needs scale_factor=2 and 64x64 output.
→ Defer to next iteration (needs GPU).

#### 3. Quality: NorESM eval has constraint=none but JSON metadata says constraint=addcl
The run_eval_noresm.py defaults to --constraint addcl but iteration 5 ran with --constraint none.
Check: noresm_eval_results_500.json says "constraint": "none" — correct, the override was applied.
No bug.

### Plan for this iteration
- Extend metrics.py: add dual-dataset comparison functions
- Generalize sample visualization for variable scale factors (no GPU, code only)
- Update make_figures.py to support both datasets
- Generate metrics-only figures on CPU
- Commit

### Iteration 6 results
- Extended metrics.py with 3 new dual-dataset functions:
  - plot_dual_crps: side-by-side CRPS bar charts (shared methods only)
  - plot_dual_metrics_panel: 4x2 grid (CRPS/MAE/RMSE/MassViol × ERA5/NorESM)
  - plot_constraint_impact: grouped delta-CRPS showing constraint effects per dataset
- Generalized existing code:
  - plot_crps_comparison and plot_metrics_panel now accept title parameter
  - generate_baseline_predictions now accepts upsampling_factor parameter
  - plot_sample_comparison derives HR size from data (not hardcoded 128x128)
- Reorganized figure output: figures/era5/, figures/noresm/, figures/ (cross-dataset)
- Updated make_figures.py:
  - Accepts --era5-results and --noresm-results
  - Generates per-dataset + dual-dataset comparison figures
  - make_noresm_sample_figures for NorESM-specific predictions (GPU, deferred)
  - generate_flow_predictions accepts hr_size and apply_constraint parameters
- Generated all 9 metrics-only figures (no GPU)
- Removed old top-level ERA5-only figures (replaced by era5/ subdirectory)
- Lint, format, typecheck: all pass

### Figure inventory
| Path | Description |
|------|-------------|
| figures/era5/crps_comparison.png | ERA5 CRPS bar chart (15 methods) |
| figures/era5/metrics_panel.png | ERA5 2x2 metrics panel |
| figures/era5/flow_vs_baseline.png | ERA5 flow vs best baseline |
| figures/noresm/crps_comparison.png | NorESM CRPS bar chart (12 methods) |
| figures/noresm/metrics_panel.png | NorESM 2x2 metrics panel |
| figures/noresm/flow_vs_baseline.png | NorESM flow vs best baseline |
| figures/dual_crps.png | Side-by-side ERA5 vs NorESM CRPS |
| figures/dual_metrics_panel.png | 4x2 all-metrics comparison |
| figures/constraint_impact.png | Delta CRPS from constraints |

### Next iteration
- GPU: generate sample prediction figures for both datasets
- Visual review of all sample figures
