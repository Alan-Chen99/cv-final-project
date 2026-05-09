# Dev

## Package management

Use `uv add <pkg>` to add dependencies. Do not use `pip install`.

## Skills

Project skills live in `skills/` (symlinked from `.claude/skills`). Use them — check the trigger conditions and invoke when relevant.

| Skill                   | Trigger                                     |
| ----------------------- | ------------------------------------------- |
| `long-running-commands` | About to run a command expected to take >1m, or runtime is uncertain (queue waits, network transfers) |
| `slurm-preemptable`     | If requested by user                        |

## Directory structure

```
.
├── .claude/
│   └── skills -> ../skills       # symlink
├── src/downscaling/               # persistent, maintained library code
│   ├── models/                    #   UNet, building blocks
│   ├── data/                      #   ERA5 data loading
│   ├── metrics/                   #   CRPS (energy + paper), MAE, RMSE
│   ├── constraints/               #   AddCL, SmCL constraint layers
│   ├── sampling/                  #   ODE solvers, timestep strategies
│   ├── training/                  #   Flow matching trainer, EMA
│   ├── evaluation/                #   Model evaluation pipelines
│   └── plotting/                  #   Visualization code
├── tests/                         # integration tests (GPU required)
├── experiments/                   # frozen experiment code (read-only after merge)
│   ├── spatial-4x-flow-matching/  # research2-3: core flow matching + ablations
│   ├── flow-matching-alternatives/ # research4: DDPM, DiT, CFG variants
│   ├── flow-matching-zscore-normalization/ # research6: z-score norm + zero-shot
│   └── pretrained-sr-downscaling/ # research5: pretrained SwinIR approaches
├── skills/                        # project-wide Claude skills
│   ├── long-running-commands/
│   └── slurm-preemptable/
├── external/
│   ├── constrained-downscaling/   # Harder et al. baseline code
│   ├── ClimateDiffuse/            # GenDiff (robbiewatt1) — git subtree
│   └── WassDiff/                  # WassDiff (Way-Yuhao) — git subtree
├── papers/                        # arXiv paper notes (markdown)
├── notes/                         # research session notes
├── scripts/                       # project scripts
│   ├── container.sh               # Apptainer instance (CPU node)
│   └── gpu_run.sh                 # srun + singularity exec wrapper
├── CLAUDE.md
├── pyproject.toml
└── uv.lock
```

### src/ vs experiments/

- **`src/downscaling/`**: Maintained library. Canonical implementations of models, metrics,
  constraints, data loading. All new code goes here. Must pass ruff, basedpyright, tests.
- **`experiments/`**: Frozen experiment snapshots from research branches. Each subdirectory
  was active during a research iteration and is frozen after merge. Do not modify — use as
  reference only. New experiments should import from `src/downscaling/`.

### Dev tools

```bash
ruff check src/ tests/       # lint
ruff format src/ tests/      # format
basedpyright src/            # type check
pytest tests/ -v             # integration tests (GPU)
pytest --cov=downscaling     # coverage
```

# Project

## Goal

Improve on Harder et al. (2208.05424) — hard-constrained deep learning for climate downscaling — using modern CV methods, particularly diffusion models. The baseline work enforces physical conservation laws (mass, energy) via constraint layers (AddCL, MultCL, SmCL) appended to CNN/GAN/RNN architectures. We want to bring these guarantees into stronger generative frameworks.

## Background

Harder et al. showed that hard constraints (architectural enforcement of conservation between LR and HR) outperform soft constraints (loss penalties) across architectures and datasets. Their constraint layers are architecture-agnostic — they renormalize the final output so the mean of each HR super-pixel exactly equals the corresponding LR pixel. SmCL (softmax-based) is the best default: handles any input range, enforces non-negativity, and performs competitively. Key limitation: constraints hurt when LR-HR divergence is large (NorESM case, constraint RMSE 2.48).

Since that work, diffusion models have become the state of the art for both image SR and climate downscaling:

- **CorrDiff** (2309.15214): Two-step approach — UNet predicts deterministic mean, diffusion corrects residuals. 25km to 2km over Taiwan. 22x faster than numerical models on single GPU.
- **STVD** (2312.06071): Spatiotemporal video diffusion for precipitation. Deterministic downscaler + conditional diffusion for stochastic residuals. Factorized attention for spatial and temporal dims. NeurIPS 2024.
- **GenDiff** (2404.17752): Generative diffusion-based downscaling across climate variables. Ensemble generation for uncertainty quantification.
- **WassDiff** (2410.00381): Score-based diffusion with Wasserstein distance regularization. Better extreme value capture than standard score matching.

## Research Directions

### Bridging foundation models and task-specific downscaling

These directions exploit the gap between the two disconnected lineages (see [notes/2026-04-27](notes/2026-04-27-foundation-models-vs-downscaling.md)).

6. **Pretrained ViT backbone + diffusion head + constraints**: Use frozen/LoRA'd Prithvi WxC as encoder in CorrDiff-style two-stage setup. SmCL on deterministic mean, diffusion for stochastic residuals. No existing work combines all three.
7. **DiT for climate downscaling**: All downscaling diffusion papers use UNet backbones. Transformer-based score networks (DiT) are untested in climate downscaling and are a natural fit with foundation model pretraining.
8. **Foundation model on real GCM→RCM distribution shift**: Prithvi WxC only demonstrated perfect-model downscaling (coarsen-then-recover). Testing on actual GCM→RCM pairs measures whether pretrained representations transfer across the distribution gap.

### Constraints + modern generative models

1. **Constraint layers + diffusion**: Can SmCL/AddCL be applied to the output of a diffusion denoising step or to the final sample? The constraint is a differentiable projection — it should compose with any generator.
2. **Latent diffusion for efficiency**: Harder et al. used 128x128 patches. Latent diffusion (2112.10752) compresses to a lower-dim latent space, enabling higher resolution and faster sampling. Constraints could be applied after decoding.
3. **Residual diffusion with constraints**: CorrDiff's two-step (mean + stochastic residual) naturally separates the conservation-satisfying component (mean) from fine detail (residual). Constraining only the mean prediction and letting diffusion handle texture may be cleaner than constraining the full output.
4. **Constrained stochastic interpolants**: CDSI outperforms CorrDiff, starts from LR field instead of noise. SmCL on the final output is straightforward. Probably lowest-hanging fruit for constraints + modern generative model.
5. **Multi-variable constraints in generative framework**: 1EMD shows multi-variable ViT downscaling works but has no constraints. Multi-variable constraints paper (2308.01868) only handles Tmin/Tmean/Tmax in UNet. Combining both in a generative model is open.

### Extensions

6. **Spatiotemporal extension**: Harder et al.'s FlowConvGRU was a first attempt at joint spatial-temporal SR. Video diffusion (STVD) is a more capable framework for this. Conservation constraints across time frames remain unexplored.
7. **Extreme events**: WassDiff shows standard diffusion underestimates extremes. Wasserstein regularization or tail-aware losses could combine with hard constraints that already help in coastal/mountainous regions.

# Pool data (not in git)

Located on `/home/chenxy/orcd/pool/datasets/` (pool disk). Data MUST go here, never in `/workspace` or worktrees. Symlink into worktrees as needed.

## Pool layout and access rules

```
pool/datasets/
  era5_sr_data/        # 4 GB  — shared, read-only to agents
  era5_sr_temporal/    # 12 GB — shared, read-only to agents
  gendiff/             # shared, read-only to agents
  wassdiff/            # shared, read-only to agents
  corrdiff/            # shared, read-only to agents
  <branch>/            # per-branch workspace (write-only to that branch)
    predictions/
    models/
    ...
```

**Rules:**
- **Read**: any agent can read from any path under `pool/datasets/`.
- **Write**: agents may only write to `pool/datasets/<own-branch>/`. Create the directory if it doesn't exist.
- **Missing shared data**: if a dataset is not in pool, download it to `pool/datasets/<own-branch>/` — not to the top-level shared dirs. Two branches downloading the same dataset concurrently will produce two copies; this is accepted overhead.
- **After merge**: branch-specific paths persist in pool and remain accessible from master or other branches.

## Shared datasets

| Path | Size | Contents |
|---|---|---|
| `era5_sr_data/` | 4 GB | ERA5 TCW 4x SR dataset (40K/10K/10K, 32x32->128x128). Source: [Google Drive](https://drive.google.com/file/d/1IENhP1-aTYyqOkRcnmCIvxXkvUW2Qbdx) |
| `era5_sr_temporal/` | 12 GB | ERA5 TCW temporal SR dataset |
| `gendiff/weights/diffusion.pt` | 372 MB | GenDiff EDM diffusion checkpoint (from Git LFS) |
| `gendiff/weights/unet.pt` | 372 MB | GenDiff UNet checkpoint (from Git LFS) |
| `wassdiff/weights/wassdiff.pth` | 938 MB | WassDiff NCSN++ checkpoint (from HuggingFace YuhaoL/WassDiff) |
| `wassdiff/data/mini_dataset/` | 21 GB | CPC precipitation (1990-1999) + ERA5 single/pressure level vars (from HuggingFace) |

# Notes

IMPORTANT: prior agent notes MUST be taken as CONTEXT ONLY. Verify all claims before using.

| Date | Title | File |
|---|---|---|
| 2026-04-27 | Foundation Models vs Task-Specific Downscaling: Two Disconnected Lineages | [notes/2026-04-27-foundation-models-vs-downscaling.md](notes/2026-04-27-foundation-models-vs-downscaling.md) |
| 2026-04-29 | Paper Code, Weights & Compute Survey | [notes/2026-04-29-paper-code-weights-compute.md](notes/2026-04-29-paper-code-weights-compute.md) |
| 2026-05-02 | CV Final Project Outline | [notes/2026-05-02-cv-project-outline.md](notes/2026-05-02-cv-project-outline.md) |
| 2026-05-02 | Constrained Flow Matching for Climate Downscaling (research branch report) | [notes/2026-05-02-flow-matching-downscaling.md](notes/2026-05-02-flow-matching-downscaling.md) |
| 2026-05-05 | Spatial 4x CRPS Experiment (research2 branch report) | [notes/2026-05-05-spatial-4x-crps-experiment.md](notes/2026-05-05-spatial-4x-crps-experiment.md) |
| 2026-05-05 | Cross-Comparison: Flow Matching Experiments | [notes/2026-05-05-cross-comparison.md](notes/2026-05-05-cross-comparison.md) |
| 2026-05-05 | Ralph Workflow Hang Diagnosis: research3/research4 | [notes/2026-05-05-ralph-hang-diagnosis.md](notes/2026-05-05-ralph-hang-diagnosis.md) |
| 2026-05-06 | research3 Branch Report: Architecture & Training Optimization | [experiments/spatial-4x-flow-matching/2026-05-06-research3-report.md](experiments/spatial-4x-flow-matching/2026-05-06-research3-report.md) |
| 2026-05-06 | Research6: Flow Matching for TCW 4x Downscaling (final report) | [experiments/flow-matching-zscore-normalization/2026-05-06-research6-report.md](experiments/flow-matching-zscore-normalization/2026-05-06-research6-report.md) |
| 2026-05-07 | Research5: Pretrained Image SR for Climate Downscaling (research5 branch report) | [experiments/pretrained-sr-downscaling/2026-05-07-research5-pretrained-sr-report.md](experiments/pretrained-sr-downscaling/2026-05-07-research5-pretrained-sr-report.md) |
| 2026-05-08 | Organize2: Project Structure & Evaluation Report | [notes/2026-05-08-organize2-report.md](notes/2026-05-08-organize2-report.md) |

# Papers

## Diffusion & Flow Foundations

| Date | Title | arXiv | File |
|---|---|---|---|
| 2020-06-19 | Denoising Diffusion Probabilistic Models (DDPM) | [2006.11239](https://arxiv.org/abs/2006.11239) | [papers/2020-06-19 Denoising Diffusion Probabilistic Models.md](papers/2020-06-19%20Denoising%20Diffusion%20Probabilistic%20Models.md) |
| 2022-06-01 | Elucidating the Design Space of Diffusion-Based Generative Models (EDM) | [2206.00364](https://arxiv.org/abs/2206.00364) | [papers/2022-06-01 Elucidating the Design Space of Diffusion-Based Generative Models.md](papers/2022-06-01%20Elucidating%20the%20Design%20Space%20of%20Diffusion-Based%20Generative%20Models.md) |
| 2022-07-26 | Classifier-Free Diffusion Guidance | [2207.12598](https://arxiv.org/abs/2207.12598) | [papers/2022-07-26 Classifier-Free Diffusion Guidance.md](papers/2022-07-26%20Classifier-Free%20Diffusion%20Guidance.md) |
| 2022-10-06 | Flow Matching for Generative Modeling | [2210.02747](https://arxiv.org/abs/2210.02747) | [papers/2022-10-06 Flow Matching for Generative Modeling.md](papers/2022-10-06%20Flow%20Matching%20for%20Generative%20Modeling.md) |
| 2023-03-02 | Consistency Models | [2303.01469](https://arxiv.org/abs/2303.01469) | [papers/2023-03-02 Consistency Models.md](papers/2023-03-02%20Consistency%20Models.md) |
| 2024-03-21 | Physics-Informed Diffusion Models (ICLR 2025) | [2403.14404](https://arxiv.org/abs/2403.14404) | [papers/2024-03-21 Physics-Informed Diffusion Models.md](papers/2024-03-21%20Physics-Informed%20Diffusion%20Models.md) |

## Image SR & Restoration

| Date | Title | arXiv | File |
|---|---|---|---|
| 2021-04-15 | Image Super-Resolution via Iterative Refinement (SR3) | [2104.07636](https://arxiv.org/abs/2104.07636) | [papers/2021-04-15 Image Super-Resolution via Iterative Refinement.md](papers/2021-04-15%20Image%20Super-Resolution%20via%20Iterative%20Refinement.md) |
| 2021-08-23 | SwinIR: Image Restoration Using Swin Transformer | [2108.10257](https://arxiv.org/abs/2108.10257) | [papers/2021-08-23 SwinIR Image Restoration Using Swin Transformer.md](papers/2021-08-23%20SwinIR%20Image%20Restoration%20Using%20Swin%20Transformer.md) |
| 2021-11-10 | Palette: Image-to-Image Diffusion Models | [2111.05826](https://arxiv.org/abs/2111.05826) | [papers/2021-11-10 Palette Image-to-Image Diffusion Models.md](papers/2021-11-10%20Palette%20Image-to-Image%20Diffusion%20Models.md) |
| 2021-12-20 | High-Resolution Image Synthesis with Latent Diffusion Models | [2112.10752](https://arxiv.org/abs/2112.10752) | [papers/2021-12-20 High-Resolution Image Synthesis with Latent Diffusion Models.md](papers/2021-12-20%20High-Resolution%20Image%20Synthesis%20with%20Latent%20Diffusion%20Models.md) |
| 2023-09-11 | HAT: Hybrid Attention Transformer for Image Restoration | [2309.05239](https://arxiv.org/abs/2309.05239) | [papers/2023-09-11 HAT Hybrid Attention Transformer for Image Restoration.md](papers/2023-09-11%20HAT%20Hybrid%20Attention%20Transformer%20for%20Image%20Restoration.md) |

## Video Diffusion & Temporal

| Date | Title | arXiv | File |
|---|---|---|---|
| 2022-04-07 | Video Diffusion Models | [2204.03458](https://arxiv.org/abs/2204.03458) | [papers/2022-04-07 Video Diffusion Models.md](papers/2022-04-07%20Video%20Diffusion%20Models.md) |
| 2023-04-18 | Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models | [2304.08818](https://arxiv.org/abs/2304.08818) | [papers/2023-04-18 Align your Latents High-Resolution Video Synthesis with Latent Diffusion Models.md](papers/2023-04-18%20Align%20your%20Latents%20High-Resolution%20Video%20Synthesis%20with%20Latent%20Diffusion%20Models.md) |
| 2024-04-01 | Video Interpolation with Diffusion Models (VIDIM) | [2404.01203](https://arxiv.org/abs/2404.01203) | [papers/2024-04-01 Video Interpolation with Diffusion Models.md](papers/2024-04-01%20Video%20Interpolation%20with%20Diffusion%20Models.md) |

## Climate Downscaling: Constraints

| Date | Title | arXiv | File |
|---|---|---|---|
| 2022-08-08 | Hard-Constrained Deep Learning for Climate Downscaling | [2208.05424](https://arxiv.org/abs/2208.05424) | [papers/2022-08-08 Hard-Constrained Deep Learning for Climate Downscaling.md](papers/2022-08-08%20Hard-Constrained%20Deep%20Learning%20for%20Climate%20Downscaling.md) |
| 2023-08-02 | Multi-variable Hard Physical Constraints for Climate Model Downscaling | [2308.01868](https://arxiv.org/abs/2308.01868) | [papers/2023-08-02 Multi-variable Hard Physical Constraints for Climate Model Downscaling.md](papers/2023-08-02%20Multi-variable%20Hard%20Physical%20Constraints%20for%20Climate%20Model%20Downscaling.md) |

## Climate Downscaling: Diffusion

| Date | Title | arXiv | File |
|---|---|---|---|
| 2023-09-24 | Residual Corrective Diffusion Modeling for Km-scale Atmospheric Downscaling (CorrDiff) | [2309.15214](https://arxiv.org/abs/2309.15214) | [papers/2023-09-24 Residual Corrective Diffusion Modeling for Km-scale Atmospheric Downscaling.md](papers/2023-09-24%20Residual%20Corrective%20Diffusion%20Modeling%20for%20Km-scale%20Atmospheric%20Downscaling.md) |
| 2023-12-11 | Precipitation Downscaling with Spatiotemporal Video Diffusion (STVD) | [2312.06071](https://arxiv.org/abs/2312.06071) | [papers/2023-12-11 Precipitation Downscaling with Spatiotemporal Video Diffusion.md](papers/2023-12-11%20Precipitation%20Downscaling%20with%20Spatiotemporal%20Video%20Diffusion.md) |
| 2024-04-05 | Conditional Diffusion Models for Downscaling & Bias Correction of ESM Precip | [2404.14416](https://arxiv.org/abs/2404.14416) | [papers/2024-04-05 Conditional Diffusion Models for Downscaling and Bias Correction of ESM Precipitation.md](papers/2024-04-05%20Conditional%20Diffusion%20Models%20for%20Downscaling%20and%20Bias%20Correction%20of%20ESM%20Precipitation.md) |
| 2024-04-27 | Generative Diffusion-Based Downscaling for Climate (GenDiff) | [2404.17752](https://arxiv.org/abs/2404.17752) | [papers/2024-04-27 Generative Diffusion-Based Downscaling for Climate.md](papers/2024-04-27%20Generative%20Diffusion-Based%20Downscaling%20for%20Climate.md) |
| 2024-10-01 | Downscaling Extreme Precipitation with Wasserstein Regularized Diffusion (WassDiff) | [2410.00381](https://arxiv.org/abs/2410.00381) | [papers/2024-10-01 Downscaling Extreme Precipitation with Wasserstein Regularized Diffusion.md](papers/2024-10-01%20Downscaling%20Extreme%20Precipitation%20with%20Wasserstein%20Regularized%20Diffusion.md) |
| 2024-12-19 | Downscaling Precipitation with Bias-informed Conditional Diffusion Model | [2412.14539](https://arxiv.org/abs/2412.14539) | [papers/2024-12-19 Downscaling Precipitation with Bias-informed Conditional Diffusion Model.md](papers/2024-12-19%20Downscaling%20Precipitation%20with%20Bias-informed%20Conditional%20Diffusion%20Model.md) |
| 2026-02-13 | High-Resolution Climate Projections Using Diffusion-Based Downscaling | [2602.13416](https://arxiv.org/abs/2602.13416) | [papers/2026-02-13 High-Resolution Climate Projections Using Diffusion-Based Downscaling of a Lightweight Climate Emulator.md](papers/2026-02-13%20High-Resolution%20Climate%20Projections%20Using%20Diffusion-Based%20Downscaling%20of%20a%20Lightweight%20Climate%20Emulator.md) |

## Climate Foundation Models

Deterministic ViT-based regressors for weather/climate. Separate lineage from downscaling papers — see [notes/2026-04-27-foundation-models-vs-downscaling.md](notes/2026-04-27-foundation-models-vs-downscaling.md).

| Date | Title | arXiv | File |
|---|---|---|---|
| 2023-01-24 | ClimaX: A Foundation Model for Weather and Climate | [2301.10343](https://arxiv.org/abs/2301.10343) | [papers/2023-01-24 ClimaX A Foundation Model for Weather and Climate.md](papers/2023-01-24%20ClimaX%20A%20Foundation%20Model%20for%20Weather%20and%20Climate.md) |
| 2024-05-20 | A Foundation Model for the Earth System (Aurora) | [2405.13063](https://arxiv.org/abs/2405.13063) | [papers/2024-05-20 A Foundation Model for the Earth System.md](papers/2024-05-20%20A%20Foundation%20Model%20for%20the%20Earth%20System.md) |
| 2024-09-20 | Prithvi WxC: Foundation Model for Weather and Climate (ICLR 2025) | [2409.13598](https://arxiv.org/abs/2409.13598) | [papers/2024-09-20 Prithvi WxC Foundation Model for Weather and Climate.md](papers/2024-09-20%20Prithvi%20WxC%20Foundation%20Model%20for%20Weather%20and%20Climate.md) |
| 2024-11-08 | WeatherGFM: Learning A Weather Generalist Foundation Model via In-context Learning | [2411.05420](https://arxiv.org/abs/2411.05420) | [papers/2024-11-08 WeatherGFM Learning A Weather Generalist Foundation Model via In-context Learning.md](papers/2024-11-08%20WeatherGFM%20Learning%20A%20Weather%20Generalist%20Foundation%20Model%20via%20In-context%20Learning.md) |

## Climate Downscaling: Other Methods

| Date | Title | arXiv | File |
|---|---|---|---|
| 2023-05-23 | Fourier Neural Operators for Arbitrary Resolution Climate Data Downscaling | [2305.14452](https://arxiv.org/abs/2305.14452) | [papers/2023-05-23 Fourier Neural Operators for Arbitrary Resolution Climate Data Downscaling.md](papers/2023-05-23%20Fourier%20Neural%20Operators%20for%20Arbitrary%20Resolution%20Climate%20Data%20Downscaling.md) |
| 2024-05-31 | Climate Variable Downscaling with Conditional Normalizing Flows | [2405.20719](https://arxiv.org/abs/2405.20719) | [papers/2024-05-31 Climate Variable Downscaling with Conditional Normalizing Flows.md](papers/2024-05-31%20Climate%20Variable%20Downscaling%20with%20Conditional%20Normalizing%20Flows.md) |
| 2025-06-12 | Vision Transformers for Multi-Variable Climate Downscaling (1EMD) | [2506.22447](https://arxiv.org/abs/2506.22447) | [papers/2025-06-12 Vision Transformers for Multi-Variable Climate Downscaling.md](papers/2025-06-12%20Vision%20Transformers%20for%20Multi-Variable%20Climate%20Downscaling.md) |
| 2025-12-16 | An Intercomparison of Generative ML Methods for Downscaling Precipitation | [2512.13987](https://arxiv.org/abs/2512.13987) | [papers/2025-12-16 An intercomparison of generative machine learning methods for downscaling precipitation at fine spatial scales.md](papers/2025-12-16%20An%20intercomparison%20of%20generative%20machine%20learning%20methods%20for%20downscaling%20precipitation%20at%20fine%20spatial%20scales.md) |
| 2026-03-04 | Climate Downscaling with Stochastic Interpolants (CDSI) | [2603.03838](https://arxiv.org/abs/2603.03838) | [papers/2026-03-04 Climate Downscaling with Stochastic Interpolants.md](papers/2026-03-04%20Climate%20Downscaling%20with%20Stochastic%20Interpolants.md) |

## Cross-Domain Diffusion Adaptation

| Date | Title | arXiv | File |
|---|---|---|---|
| 2021-10-08 | Score-based Diffusion Models for Accelerated MRI | [2110.05243](https://arxiv.org/abs/2110.05243) | [papers/2021-10-08 Score-based Diffusion Models for Accelerated MRI.md](papers/2021-10-08%20Score-based%20Diffusion%20Models%20for%20Accelerated%20MRI.md) |
| 2023-12-06 | DiffusionSat: A Generative Foundation Model for Satellite Imagery | [2312.03606](https://arxiv.org/abs/2312.03606) | [papers/2023-12-06 DiffusionSat A Generative Foundation Model for Satellite Imagery.md](papers/2023-12-06%20DiffusionSat%20A%20Generative%20Foundation%20Model%20for%20Satellite%20Imagery.md) |
| 2024-09-18 | Denoising Diffusion Models for High-Resolution Microscopy Image Restoration | [2409.12078](https://arxiv.org/abs/2409.12078) | [papers/2024-09-18 Denoising Diffusion Models for High-Resolution Microscopy Image Restoration.md](papers/2024-09-18%20Denoising%20Diffusion%20Models%20for%20High-Resolution%20Microscopy%20Image%20Restoration.md) |
| 2025-02-03 | A Generative Foundation Model for All-in-One Seismic Processing (GSFM) | [2502.01111](https://arxiv.org/abs/2502.01111) | [papers/2025-02-03 A Generative Foundation Model for All-in-One Seismic Processing.md](papers/2025-02-03%20A%20Generative%20Foundation%20Model%20for%20All-in-One%20Seismic%20Processing.md) |
