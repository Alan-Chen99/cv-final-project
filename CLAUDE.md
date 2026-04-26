# Dev

## Package management

Use `uv add <pkg>` to add dependencies. Do not use `pip install`.

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

1. **Constraint layers + diffusion**: Can SmCL/AddCL be applied to the output of a diffusion denoising step or to the final sample? The constraint is a differentiable projection — it should compose with any generator.
2. **Latent diffusion for efficiency**: Harder et al. used 128x128 patches. Latent diffusion (2112.10752) compresses to a lower-dim latent space, enabling higher resolution and faster sampling. Constraints could be applied after decoding.
3. **Residual diffusion with constraints**: CorrDiff's two-step (mean + stochastic residual) naturally separates the conservation-satisfying component (mean) from fine detail (residual). Constraining only the mean prediction and letting diffusion handle texture may be cleaner than constraining the full output.
4. **Spatiotemporal extension**: Harder et al.'s FlowConvGRU was a first attempt at joint spatial-temporal SR. Video diffusion (STVD) is a more capable framework for this. Conservation constraints across time frames remain unexplored.
5. **Extreme events**: WassDiff shows standard diffusion underestimates extremes. Wasserstein regularization or tail-aware losses could combine with hard constraints that already help in coastal/mountainous regions.

## Paper Categories

- **Baseline**: 2208.05424 (Harder et al.) — hard-constrained downscaling with CNN/GAN/RNN
- **CV foundations**: 2006.11239 (DDPM), 2104.07636 (SR3), 2112.10752 (Latent Diffusion)
- **Climate + diffusion**: 2309.15214 (CorrDiff), 2312.06071 (STVD), 2404.17752 (GenDiff), 2410.00381 (WassDiff)

# Papers

| Date | Title | arXiv | File |
|------|-------|-------|------|
| 2020-06-19 | Denoising Diffusion Probabilistic Models | [2006.11239](https://arxiv.org/abs/2006.11239) | [papers/2020-06-19 Denoising Diffusion Probabilistic Models.md](papers/2020-06-19%20Denoising%20Diffusion%20Probabilistic%20Models.md) |
| 2021-04-15 | Image Super-Resolution via Iterative Refinement | [2104.07636](https://arxiv.org/abs/2104.07636) | [papers/2021-04-15 Image Super-Resolution via Iterative Refinement.md](papers/2021-04-15%20Image%20Super-Resolution%20via%20Iterative%20Refinement.md) |
| 2021-12-20 | High-Resolution Image Synthesis with Latent Diffusion Models | [2112.10752](https://arxiv.org/abs/2112.10752) | [papers/2021-12-20 High-Resolution Image Synthesis with Latent Diffusion Models.md](papers/2021-12-20%20High-Resolution%20Image%20Synthesis%20with%20Latent%20Diffusion%20Models.md) |
| 2022-08-08 | Hard-Constrained Deep Learning for Climate Downscaling | [2208.05424](https://arxiv.org/abs/2208.05424) | [papers/2022-08-08 Hard-Constrained Deep Learning for Climate Downscaling.md](papers/2022-08-08%20Hard-Constrained%20Deep%20Learning%20for%20Climate%20Downscaling.md) |
| 2023-09-24 | Residual Corrective Diffusion Modeling for Km-scale Atmospheric Downscaling | [2309.15214](https://arxiv.org/abs/2309.15214) | [papers/2023-09-24 Residual Corrective Diffusion Modeling for Km-scale Atmospheric Downscaling.md](papers/2023-09-24%20Residual%20Corrective%20Diffusion%20Modeling%20for%20Km-scale%20Atmospheric%20Downscaling.md) |
| 2023-12-11 | Precipitation Downscaling with Spatiotemporal Video Diffusion | [2312.06071](https://arxiv.org/abs/2312.06071) | [papers/2023-12-11 Precipitation Downscaling with Spatiotemporal Video Diffusion.md](papers/2023-12-11%20Precipitation%20Downscaling%20with%20Spatiotemporal%20Video%20Diffusion.md) |
| 2024-04-27 | Generative Diffusion-Based Downscaling for Climate | [2404.17752](https://arxiv.org/abs/2404.17752) | [papers/2024-04-27 Generative Diffusion-Based Downscaling for Climate.md](papers/2024-04-27%20Generative%20Diffusion-Based%20Downscaling%20for%20Climate.md) |
| 2024-10-01 | Downscaling Extreme Precipitation with Wasserstein Regularized Diffusion | [2410.00381](https://arxiv.org/abs/2410.00381) | [papers/2024-10-01 Downscaling Extreme Precipitation with Wasserstein Regularized Diffusion.md](papers/2024-10-01%20Downscaling%20Extreme%20Precipitation%20with%20Wasserstein%20Regularized%20Diffusion.md) |
