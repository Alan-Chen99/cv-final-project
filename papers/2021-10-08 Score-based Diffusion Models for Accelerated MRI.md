# Score-based Diffusion Models for Accelerated MRI

- **arXiv**: [2110.05243](https://arxiv.org/abs/2110.05243)
- **Published**: 2021-10-08 (Medical Image Analysis)
- **Authors**: Hyungjin Chung, Jong Chul Ye (KAIST)
- **Code**: [https://github.com/HJ-harry/score-MRI](https://github.com/HJ-harry/score-MRI)

## Summary

Uses score-based diffusion models (specifically VE-SDE) as learned priors for accelerated MRI reconstruction. A single score function trained on magnitude-only DICOM images can reconstruct complex-valued single-coil and multi-coil data across arbitrary sub-sampling patterns, without retraining. The method alternates between reverse SDE sampling steps and data consistency projections, achieving state-of-the-art performance while also providing uncertainty quantification and strong out-of-distribution generalization.

## Key Contributions

1. **Magnitude-only training for complex reconstruction**: Train score function s_theta on magnitude images only; at inference, apply predictor-corrector steps separately to real and imaginary parts. This avoids requiring raw k-space data for training.

2. **Parallel imaging (PI) integration**: Two multi-coil strategies:
   - **SSOS-type**: Reconstruct each coil independently with the same score function, then merge via sum-of-squares
   - **Hybrid-type**: Coil-wise reconstruction with periodic SENSE-type multi-coil data consistency every m steps

3. **Sampling pattern agnostic**: Unlike supervised methods that must be retrained per sampling pattern, the score-based approach works with any sub-sampling scheme (1D uniform, 1D Gaussian, 2D Gaussian, variable density Poisson disk).

4. **Uncertainty quantification**: Multiple stochastic samples from the posterior enable pixel-wise uncertainty maps without special treatment (e.g., MC dropout).

5. **Strong OOD generalization**: Score function trained on knee PD/PDFS data generalizes to brain, cardiac, ankle, abdomen, and other anatomies/contrasts.

## Method

### Background: Score-based SDE

The forward process perturbs data x(0) ~ p_data to x(T) ~ p_T via SDE:

dx = f(x, t)dt + g(t)dw

For VE-SDE: f = 0, g = sqrt(d[sigma^2(t)]/dt), where sigma(t) is a monotonically increasing geometric series.

The reverse SDE requires the score function nabla_x log p_t(x), estimated by a neural network s_theta(x(t), t) trained with denoising score matching:

min_theta E_t [ lambda(t) E_{x(0)} E_{x(t)|x(0)} [ ||s_theta(x(t), t) - nabla_x log p_{0t}(x(t)|x(0))||^2 ] ]

Sampling uses predictor-corrector (PC) algorithm: predictor solves discretized reverse SDE, corrector applies Langevin MC steps.

### MRI Forward Model

y = Ax where A = P_Omega F S (sub-sampling operator, Fourier transform, sensitivity maps). Sensitivity maps normalized so S*S = I.

### Conditional Sampling via Data Consistency

At each iteration, alternate between:
1. **Score update** (predictor/corrector step): x_i <- unconditional SDE update
2. **Data consistency projection**: x_i <- x_i + lambda A*(y - Ax_i)

**Non-expansive mapping guarantee**: With sensitivity normalization, (I - lambda A*A) is non-expansive for lambda in [0,1], ensuring stable iteration.

### Algorithms

- **Algorithm 1 (PC sampling)**: Standard unconditional predictor-corrector
- **Algorithm 2 (Real)**: PC + data consistency for real-valued images
- **Algorithm 3 (SENSE-type)**: Separate real/imaginary PC updates + multi-coil consistency
- **Algorithm 4 (SSOS-type)**: Independent per-coil complex reconstruction, SSOS merge
- **Algorithm 5 (Hybrid-type)**: Per-coil updates with periodic multi-coil SENSE consistency every m steps

### CCDF Acceleration

Come-Closer-Diffuse-Faster (CCDF): Initialize from a forward-diffused U-Net reconstruction instead of pure noise, enabling as few as 40 iterations (vs. 2000) with similar or better performance. Backed by stochastic contraction theory.

## Architecture

- **ncsnpp** from Song et al. (2021): U-Net backbone with BigGAN residual blocks
- 4 scale levels, 4 residual blocks per level
- Skip connections scaled by 1/sqrt(2)
- Anti-aliasing pooling
- Time conditioning via Gaussian Fourier Projection (GFP): random Fourier features of time t, passed through MLP, added to encoder features at each level

## Training Details

- Dataset: fastMRI knee, ~25k slices of 320x320 magnitude images (DICOM)
- VE-SDE with sigma_min = 0.01, sigma_max = 378 (geometric series)
- Likelihood weighting: lambda(t) = sigma^2(t)
- Adam optimizer (lr 2e-4 with 5k step warmup), gradient clipping at 1.0
- EMA rate 0.999, batch size 1, 100 epochs (~3 weeks on single RTX 3090)
- Langevin step size: epsilon_i = 2r ||z||_2 / ||s_theta(x_i, sigma_i)||_2, r = 0.16

## Inference

- Default: N = 2000 predictor steps, M = 1 corrector step
- ~10 min for real-valued, ~20 min for complex-valued reconstruction
- Hybrid PI: lambda linearly decreases from 1.0 to 0.2 over iterations
- SSOS for 2D patterns, Hybrid for 1D patterns (empirical best)

## Results

### Single-coil (Real Simulation)

On fastMRI knee with various sub-sampling patterns:

| Pattern | Acc. | TV | U-Net | DuDoRNet | Proposed |
|---------|------|-----|-------|----------|----------|
| Gaussian 1D | x4 | 30.77 | 32.85 | 33.01 | **33.32** |
| Gaussian 1D | x8 | 28.87 | 30.81 | 30.46 | **30.94** |
| Gaussian 2D | x8 | 23.19 | 21.92 | 25.29 | **29.95** |
| VD Poisson | x15 | 19.56 | 20.97 | 22.84 | **30.46** |

Massive advantage on 2D patterns and high acceleration factors where supervised methods fail due to training bias toward 1D masks.

### Complex Single-coil

Same score function (trained on magnitude only) reconstructs complex data:

| Pattern | Acc. | TV | U-Net | DuDoRNet | Proposed |
|---------|------|-----|-------|----------|----------|
| Gaussian 1D | x4 | 28.39 | 32.86 | 33.46 | **33.96** |
| Gaussian 2D | x8 | 20.09 | 19.99 | 21.53 | **29.45** |
| VD Poisson | x15 | 22.13 | 21.67 | 23.95 | **30.66** |

### Multi-coil Parallel Imaging

Compared against E2E-VarNet (state-of-the-art supervised PI method):

| Pattern | Acc. | TV | U-Net | E2E-VarNet | Proposed |
|---------|------|-----|-------|------------|----------|
| Gaussian 1D | x4 | 30.55 | 32.66 | 33.15 | **34.25** |
| Gaussian 2D | x8 | 29.20 | 24.51 | 20.97 | **31.43** |
| VD Poisson | x8 | 29.52 | 20.89 | 20.70 | **31.98** |

E2E-VarNet catastrophically fails on 2D patterns (sensitivity map estimation breaks).

### Pathology Detection (Downstream Task)

Using YOLOv5 trained on fully-sampled fastMRI+ annotations:
- Proposed method matches fully-sampled data on mAP, precision, recall
- Supervised methods degrade diagnostic capability
- Bland-Altman analysis confirms proposed method has least variance from fully-sampled reference

### CCDF Acceleration

| Method | TV | U-Net | DuDoRNet | Proposed (2000) | CCDF (40) |
|--------|-----|-------|----------|-----------------|-----------|
| x4 Gaussian 1D | 30.77 | 32.85 | 33.01 | 33.32 | **34.11** |
| x8 Gaussian 1D | 28.87 | 30.81 | 30.46 | 30.94 | **32.08** |

40 iterations with CCDF initialization outperforms 2000-step full diffusion.

### Uncertainty Quantification

- Low acceleration (x2): minimal pixel-wise variance, high confidence
- High acceleration (x8): increased variance in specific regions
- Variance maps can inform clinical decision-making

### Out-of-Distribution Generalization

Score function trained only on knee PD/PDFS successfully reconstructs:
- Brain (axial, coronal), cardiac, foot, ankle, abdomen, lower extremity
- Both 1D and 2D sub-sampling patterns work, though 1D occasionally shows mild aliasing artifacts on OOD data

## Limitations

1. **OOD + 1D patterns**: Mild aliasing-like artifacts at local edges when reconstructing OOD data with 1D sub-sampling (not observed with 2D patterns)
2. **Extreme acceleration**: At very high factors (x15 uniform 1D), occasional poor posterior samples with high structural variance
3. **Inference speed**: 10-20 min per reconstruction (mitigated by CCDF to ~40 iterations)
4. **Multi-coil averaging**: SSOS aggregation introduces slight averaging effect compared to single-coil

## Connections to Related Work

- **vs. Jalal et al. (2021)**: Both use score-based priors for MRI; key differences: this work uses DICOM-only training (no raw k-space), continuous SDE (vs. discrete), PC sampler, no additional hyperparameters for annealing
- **vs. Song et al. (2022)**: Same VE-SDE + ncsnpp architecture; this work additionally handles complex-valued and multi-coil data
- **vs. Ramzi et al. (2020)**: Earlier score-matching for MRI that fell behind supervised methods and required complex-valued training data
- **vs. GAN priors**: Score-based approach avoids difficult GAN training, enables iterative refinement rather than single forward pass, handles arbitrary forward models
- **vs. Energy-based models (Guan et al.)**: Score matching training is simpler than contrastive divergence (no negative sample generation), and only needs DICOM data
