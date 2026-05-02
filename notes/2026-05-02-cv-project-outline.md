# CV Final Project Outline

## 1. Introduction & Motivation

Statistical downscaling - learning a mapping from coarse-resolution climate fields to their fine-scale counterparts - is essential for bridging the gap between what global climate models can computationally afford and what regional impact assessments require. Traditional dynamical approaches are prohibitively expensive, and classical statistical methods assume stationarity, making them unreliable under shifting climate regimes.

Deep learning has offered a path forward, but early CNN-based and GAN-based methods suffer from two well-documented failure modes. First, supervised learning encourages mode averaging, which blurs fine-scale structure and systematically underestimates extreme events precisely the signals most relevant to climate science. Second, these models typically ignore known physical laws (e.g., conservation of mass), producing outputs that are visually plausible but physically inconsistent. Hard constraints, introduced by Harder et al. (2022), address the second issue but were demonstrated only on simple CNN and GAN architectures for univariate fields.

Diffusion models represent a promising alternative: by learning to reverse a stochastic noising process, they capture the full multimodal distribution of high-resolution outputs rather than collapsing to a mean - making them particularly well-suited for precipitation and other heavy-tailed climate variables. However, even diffusion-based approaches have not been systematically evaluated with respect to spectral fidelity - their ability to reproduce the correct distribution of energy across spatial and temporal scales. This matters because fine-scale phenomena (convective cells, orographic precipitation) manifest as high-frequency content that blurry or mode-averaged predictions suppress.

## 2. Related Work

### 2.1 Generative Models for Climate Downscaling

### 2.2 Physics Constrained Deep Learning

Doc with notes from all the papers that can be categorized into these sections

## 3. Key Contributions / Project Summary:

This project proposes a unified study that combines (1) hard physical constraints from Harder et al. with (2) diffusion model backbones (UNet and Vision Transformer) and (3) spectral regularization losses (Fourier- and wavelet-based) for both spatial and spatiotemporal downscaling. We further introduce spectral evaluation metrics alongside standard pixel-space metrics to expose failure modes invisible to RMSE alone.

## 3. Datasets

Directly following Harder et al. (2022) to allow fair comparison against their baselines:

| Task | Input Resolution | Output Resolution | Variable | Notes |
|---|---|---|---|---|
| Spatial | 8×8 | 128×128 | Atmospheric moisture (ERA5) | 16× upscaling |
| Spatial | 16×16 | 128×128 | Atmospheric moisture (ERA5) | 8× upscaling |
| Spatial | 32×32 | 128×128 | Atmospheric moisture (ERA5) | 4× upscaling |
| Spatiotemporal | 3×32×32 | 3×128×128 | Atmospheric moisture (ERA5) | Temporal + spatial SR |
| Spatiotemporal | 2×32×32 | 3×128×128 | Atmospheric moisture (ERA5) | Temporal upsampling included |

## 4. Methods and Metrics

### 4.1 Models (8 in total, but need to train with various loss functions)

- For spatial:
    - CNN
        - Vanilla (RMSE)
        - RMSE + softmax hard constraint
        - RMSE + FFT / power spectrum
    - GAN
        - Vanilla (RMSE)
        - RMSE + softmax hard constraint
        - RMSE + FFT / power spectrum
    - Diffusion with UNet
        - Vanilla (RMSE)
        - RMSE + softmax hard constraint
        - RMSE + FFT / power spectrum
    - Diffusion with ViT
        - Vanilla (RMSE)
        - RMSE + softmax hard constraint
        - RMSE + FFT / power spectrum

- For spatiotemporal:
    - SR-ConvGRU (TCWT1)
        - Vanilla (RMSE)
        - RMSE + softmax hard constraint
        - RMSE + FFT / power spectrum
    - SR-FlowConvGRU (optical flow perspective) (TCWT2)
        - Vanilla (RMSE)
        - RMSE + softmax hard constraint
        - RMSE + FFT / power spectrum
    - Diffusion with UNet
        - Vanilla (RMSE)
        - RMSE + softmax hard constraint
        - RMSE + FFT / power spectrum
    - Diffusion with ViT
        - Vanilla (RMSE)
        - RMSE + softmax hard constraint
        - RMSE + FFT / power spectrum

- Latent diffusion (LOLA) - finetune lola
    - Vanilla (RMSE)
    - RMSE + softmax hard constraint
    - RMSE + FFT / power spectrum

During diffusion training, at each noise step $t$, the model receives a noisy high-resolution sample $x_t$ concatenated with the low-resolution conditioning input (upsampled to match spatial dims) and must predict the noise $\epsilon$. The backbone is the network doing that prediction. The two choices are:

- UNet backbone: encoder-decoder with skip connections; strong spatial inductive bias; standard in SR3, Palette, and the precipitation downscaling papers you cite
- ViT backbone (DiT-style): patchify the input, apply transformer blocks with cross-attention or concatenated conditioning; better at capturing long-range dependencies; motivated by the Vision Transformer downscaling paper you cite

### 4.2 Loss Functions

- **MSE**: standard pixel-space L2
- **FFT spectral loss**: $\mathcal{L}_{FFT} = \||\mathcal{F}(\hat{x})| - |\mathcal{F}(x)|\|_2$ — penalizes mismatch in the power spectrum directly
- **Wavelet loss**: $\mathcal{L}_{Wav} = \sum_{j,k} \|W_j(\hat{x})_k - W_j(x)_k\|_2$ where $j$ indexes scale and $k$ orientation — better captures localized multiscale structure than global FFT
- **Soft constraint**: a penalty term added to the loss that penalizes violation of conservation of mass (differentiable relaxation)
- **Hard constraint**: the Harder et al. softmax layer; non-negotiable enforcement at output

### 4.3 Evaluation Metrics

- Power spectrum
- MSE (take the mean of the diffusion model outputs and use this)
- Think of some other metric that accounts for the stochasticity of the diffusion model
- Inference runtime

## Results (we're aiming to answer these key questions)

- Does the diffusion framework itself help over CNN/GAN baselines? (yes, expected)
- Does the ViT backbone help over UNet inside diffusion? (probably yes at high SR ratios)
- Does spectral regularization improve high-frequency fidelity independent of backbone? (your main contribution)
- Do hard constraints improve physical consistency without hurting spectral quality? (the tension to resolve)

## Training details (from original paper):

Our models were trained with the Adam optimizer, a learning rate of 0.001, and a batch size of 256. We trained for 200 epochs, which took about 3-6 hours on a single NVIDIA A100 Tensor Core GPU, depending on the architecture. All models use the MSE as their criterion, the GAN additionally uses its discriminator loss term. All the data are normalized between 0 and 1 for training, except for the cases where the ScAddCL is applied. In the case of this constraint layer we scale the data between -1 and 1 as proposed in Geiss and Hardin (2023). For our time-dependent models though, ConvGRU and FlowConvGRU, we are scaling between 0 and 1, because the original scaling led to NaN-values during training.
