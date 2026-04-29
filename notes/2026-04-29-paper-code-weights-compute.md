# Paper Code, Weights & Compute Survey

Date: 2026-04-29

Each paper was independently researched by 2 subagents (reading the local markdown + web search). Results were cross-verified; discrepancies noted inline.

## Diffusion & Flow Foundations

| Paper | Code | Params | Weights | Compute |
|---|---|---|---|---|
| DDPM (2006.11239) | [hojonathanho/diffusion](https://github.com/hojonathanho/diffusion) | 35.7M (CIFAR-10), 114M (LSUN/CelebA-HQ 256), ~256M (LSUN Bedroom large) | Yes | TPU v3-8. CIFAR-10: 10.6h/800k steps. CelebA-HQ: 0.5M steps. LSUN Bedroom: 2.4M steps |
| EDM (2206.00364) | [NVlabs/edm](https://github.com/NVlabs/edm) | ~56M (DDPM++/CIFAR-10), ~296M (ADM/ImageNet-64) | Yes | ~250 MWh total on V100 cluster. CIFAR-10: ~2d on 8xV100. ImageNet-64: ~13d on 32xA100 |
| Classifier-Free Guidance (2207.12598) | No official repo | ~554M (inherits ADM architecture) | No | 64x64: 400k steps, 128x128: 2.7M steps. Hardware not reported (Google internal) |
| Flow Matching (2210.02747) | [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching) (library); [atong01/conditional-flow-matching](https://github.com/atong01/conditional-flow-matching) | ADM UNet architecture; exact counts not reported | No | ImageNet-128: 500k iters, batch 1.5k (33% less throughput than ADM) |
| Consistency Models (2303.01469) | [openai/consistency_models](https://github.com/openai/consistency_models) | ~56M (CIFAR-10), ADM-scale (~296M) for ImageNet-64/LSUN-256 | Yes | CIFAR-10: 8 GPUs, 800k iters. ImageNet-64/LSUN: 64 A100s, 600k-1M iters |
| Physics-Informed DM (2403.14404) | [jhbastek/PhysicsInformedDiffusionModels](https://github.com/jhbastek/PhysicsInformedDiffusionModels) | Not reported (small-scale: Darcy flow 64x64, topology opt) | Yes (ETHZ Research Collection) [^1] | 1x Quadro RTX 6000. Darcy: 13-22h/300k iters. Topology: 48-54h |

[^1]: Agent A found weights on ETHZ Research Collection; Agent B reported no weights. The GitHub repo links to the ETHZ collection for reproducibility artifacts.

## Image SR & Restoration

| Paper | Code | Params | Weights | Compute |
|---|---|---|---|---|
| SR3 (2104.07636) | No official repo. Unofficial: [Janspiry/Image-Super-Resolution-via-Iterative-Refinement](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement) | 550M (16x->128x), 625M (64x->256x/512x), 150M (256x->1024x) | No (Google internal) | 1M steps, batch 256. Hardware not reported |
| SwinIR (2108.10257) | [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR) | 11.8M (classical SR), 0.9M (lightweight), ~12M (denoise/JPEG) | Yes | Not reported in paper. Community estimates: 1.8-5d on 8x GPUs, 500k iters |
| Palette (2111.05826) | No official repo. Unofficial: [Janspiry/Palette-Image-to-Image-Diffusion-Models](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models) | 552M (256x256 U-Net) | No (Google internal) | 1M steps, batch 1024. TPUv4 likely |
| LDM (2112.10752) | [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion) | 274M (unconditional), 400M (class-cond ImageNet), 1.45B (text-to-image), 169-552M (SR/inpainting) | Yes | CelebA-HQ: 14.4 V100-days. Bedrooms: 60. ImageNet: 271. Churches: 18. FFHQ: 26 |
| HAT (2309.05239) | [XPixelGroup/HAT](https://github.com/XPixelGroup/HAT) | 20.8M (base), ~40M (HAT-L), ~14M (HAT-S) | Yes | 500k iters on DF2K + 800k iters ImageNet pre-training. Hardware not reported |

## Video Diffusion & Temporal

| Paper | Code | Params | Weights | Compute |
|---|---|---|---|---|
| Video Diffusion Models (2204.03458) | No official repo. Unofficial: [lucidrains/video-diffusion-pytorch](https://github.com/lucidrains/video-diffusion-pytorch) | Not reported (3D U-Net, base channels 128-256) | No (Google internal) | 128-256 TPU-v4 chips. UCF101: 60k steps. Kinetics: 220k steps. Text-to-video: 700-800k steps |
| Align your Latents (2304.08818) | No official repo. Unofficial: [srpkdyy/VideoLDM](https://github.com/srpkdyy/VideoLDM) | ~3.1B total (~2.2B trained): 865M spatial LDM (frozen) + 656M temporal (trained) + 1,509M interpolation LDM | No (NVIDIA proprietary) | Not reported |
| VIDIM (2404.01203) | No official repo. Project page: [vidim-interpolation.github.io](https://vidim-interpolation.github.io/) | Base: 441M (medium) / 1.6B (large). SR: 644M (medium) / 1.01B (large). Full cascade: ~2.6B | No | Base 500k steps, SR 200k steps. Hardware not reported (Google internal) |

## Climate Downscaling: Constraints

| Paper | Code | Params | Weights | Compute |
|---|---|---|---|---|
| Hard-Constrained Downscaling (2208.05424) | [RolnickLab/constrained-downscaling](https://github.com/RolnickLab/constrained-downscaling) | Not reported (small SR-CNNs, GANs, ConvGRUs on 128x128 patches) | No | 3-6h per model on 1x A100, 200 epochs, batch 256 |
| Multi-variable Constraints (2308.01868) | [RolnickLab/constrained-downscaling](https://github.com/RolnickLab/constrained-downscaling) (same repo) | Not reported (UNet, DeepESD) | No | Not reported |

## Climate Downscaling: Diffusion

| Paper | Code | Params | Weights | Compute |
|---|---|---|---|---|
| CorrDiff (2309.15214) | [NVIDIA/physicsnemo/.../corrdiff](https://github.com/NVIDIA/physicsnemo/tree/main/examples/weather/corrdiff) | 80M per UNet (regression + diffusion), ~160M total | Yes (NGC + [HuggingFace](https://huggingface.co/nvidia/corrdiff-cmip6-era5)) | 128 H100s (16 DGX nodes), 7 days, ~21,504 GPU-hours. Regression: 2M steps, Diffusion: 28M steps |
| STVD (2312.06071) | [mandt-lab/STVD](https://github.com/mandt-lab/STVD) | Not reported (2x UNet, base ch 64, mults [1,1,2,2,3,4]) | No | ~7d on 1x RTX A6000, 1M steps, batch 1 |
| Cond. Diffusion ESM Precip (2404.14416) | [aim56009/ESM_cdifffusion_downscaling_bc](https://github.com/aim56009/ESM_cdifffusion_downscaling_bc) | ~730M (Efficient U-Net) | Yes ([Zenodo](https://zenodo.org/records/18069119)) | 100 epochs, batch 2. Hardware not reported |
| GenDiff (2404.17752) | [robbiewatt1/ClimateDiffuse](https://github.com/robbiewatt1/ClimateDiffuse) | Not reported (U-Net, EDM-style) | No | Not reported ("spare time", no HPC) |
| WassDiff (2410.00381) | Not found (paper claims open-source, no repo located) | 61.4M (WassDiff/SBDM), 157M (CorrDiff reimpl) | No | 120k iters on 1x A100, batch 12 |
| Bias-informed Cond. Diffusion (2412.14539) | [RoseLV/research_super-resolution](https://github.com/RoseLV/research_super-resolution) | Not reported (Improved DDPM U-Net, 128x128) | No | ~1d on 1x A100, 1000 epochs |
| High-Res Climate Projections (2602.13416) | Emulator: [ISCLPennState/LUCIE](https://github.com/ISCLPennState/LUCIE). SR models: not found | Not reported (U-Net, base width 32, mults [1,2,4,8,16]) | No | ~23 GPU-hours per EDM model on A100s. SFNO-SR: 12.5 GPU-hours |

## Climate Foundation Models

| Paper | Code | Params | Weights | Compute |
|---|---|---|---|---|
| ClimaX (2301.10343) | [microsoft/ClimaX](https://github.com/microsoft/ClimaX) | ~109M (ViT-L: D=1024, 8 layers, 16 heads) | Yes | Pretrain: 80 V100s, 100 epochs. Finetune: ~15h on 8 V100s per task |
| Aurora (2405.13063) | [microsoft/aurora](https://github.com/microsoft/aurora) | 117M, 290M, 660M, 1.3B (4 configs) | Yes ([HuggingFace](https://huggingface.co/microsoft/aurora)) | Pretrain: 32 A100s, 150k steps, ~2.5 weeks (~13,440 A100-hours). Finetune: 8 GPUs, 4-8 weeks/task |
| Prithvi WxC (2409.13598) | [NASA-IMPACT/Prithvi-WxC](https://github.com/NASA-IMPACT/Prithvi-WxC) | 2.3B | Yes ([HuggingFace](https://huggingface.co/Prithvi-WxC)) | Phase 1: 64 A100s, 100k steps. Phase 2: 16-48 GPUs |
| WeatherGFM (2411.05420) | [xiangyu-mm/WeatherGFM](https://github.com/xiangyu-mm/WeatherGFM) | 110M (base), 330M (large) | Yes (HuggingFace via repo) | 16 A100s, 50 epochs, fp16 |

## Climate Downscaling: Other Methods

| Paper | Code | Params | Weights | Compute |
|---|---|---|---|---|
| FNO Downscaling (2305.14452) | [qy707/DSFNO](https://github.com/qy707/DSFNO) | Not reported (ResNet + FNO) | No | Not reported |
| Cond. Normalizing Flows (2405.20719) | [christina-winkler/clim-var-ds-cnf](https://github.com/christina-winkler/clim-var-ds-cnf) | Not reported (3-scale CNF, 2 flow steps/scale) | No | Not reported (35 epochs) |
| 1EMD (2506.22447) | Not found | 11.63M (single-var ViT), 15.39M (1E1D), 32.25M (1EMD), 5.20M (U-Net baseline) | No | 1x A100, 400 epochs x 500 steps, batch 1. ~3h (single-var) to ~17h (1EMD) |
| Precip Intercomparison (2512.13987) | [tukib/An-intercomparison-of-generative-machine-learning-methods-for-downscaling-precipitation](https://github.com/tukib/An-intercomparison-of-generative-machine-learning-methods-for-downscaling-precipitation); data: [Zenodo](https://zenodo.org/records/13755688) | Not reported (U-Net GAN + diffusion denoiser) | No | 1x A100. GAN: 70h/200 epochs. DPM: 44h/200 epochs |
| CDSI (2603.03838) | Not found | ~62.5M (U-Net, 128/256 ch, 4 levels) | No | 1-4 A100s, 25 epochs, batch 2 |

## Cross-Domain Diffusion Adaptation

| Paper | Code | Params | Weights | Compute |
|---|---|---|---|---|
| Score-based MRI (2110.05243) | [HJ-harry/score-MRI](https://github.com/HJ-harry/score-MRI) | Not reported (~60-100M est., ncsnpp architecture) | No | 1x RTX 3090, 100 epochs, ~3 weeks (~504 GPU-hours) |
| DiffusionSat (2312.03606) | [samar-khanna/DiffusionSat](https://github.com/samar-khanna/DiffusionSat) | ~860M (init from SD 2.1) | Yes (Zenodo) | 8x A100s, 100k iters, batch 128. ControlNets: 4x A100s, 40-50k iters |
| Microscopy Diffusion (2409.12078) | [kaschube-lab/ddpm_highres_microscopy](https://github.com/kaschube-lab/ddpm_highres_microscopy) | Not reported (U-Net, self-attn at res 32) | No | 1x A100, batch 8, T=200. Duration not reported |
| GSFM Seismic (2502.01111) | [DeepWave-KAUST/GSFM](https://github.com/DeepWave-KAUST/GSFM) | Not reported (U-Net, ch 64->512) | No | 200k iters pretrain, 30k finetune, batch 5. Hardware not specified |

## Cross-Verification Notes

Each paper was researched by 2 independent subagents. Discrepancies found:

1. **Physics-Informed DM weights**: Agent A found pretrained models hosted on ETHZ Research Collection (linked from GitHub repo). Agent B missed this. Resolved: weights available.
2. **SwinIR training time**: Agent A cited ~5d on 8x TITAN RTX, Agent B cited ~1.8d on 8x RTX 2080Ti. Both are community estimates; the paper does not report training time.
3. **VIDIM code**: Agent A found a GitHub org ([vidim-interpolation](https://github.com/vidim-interpolation)); Agent B found only the project page. Neither confirmed substantive code release.

All other entries agreed between paired agents.
