# Precipitation Downscaling with Spatiotemporal Video Diffusion

arXiv: [2312.06071](https://arxiv.org/abs/2312.06071)

## Authors

Prakhar Srivastava, Ruihan Yang, Gavin Kerrigan (University of California, Irvine); Gideon Dresdner, Jeremy McGibbon, Christopher Bretherton (Allen Institute for AI); Stephan Mandt (University of California, Irvine)

## Abstract

In climate science and meteorology, high-resolution local precipitation (rain and snowfall) predictions are limited by the computational costs of simulation-based methods. Statistical downscaling, or super-resolution, is a common workaround where a low-resolution prediction is improved using statistical approaches. Unlike traditional computer vision tasks, weather and climate applications require capturing the accurate conditional distribution of high-resolution given low-resolution patterns to assure reliable ensemble averages and unbiased estimates of extreme events, such as heavy rain. This work extends recent video diffusion models to precipitation super-resolution, employing a deterministic downscaler followed by a temporally-conditioned diffusion model to capture noise characteristics and high-frequency patterns. We test our approach on FV3GFS output, an established large-scale global atmosphere model, and compare it against six state-of-the-art baselines. Our analysis, capturing CRPS, MSE, precipitation distributions, and qualitative aspects using California and the Himalayas as examples, establishes our method as a new standard for data-driven precipitation downscaling.

## 1. Introduction

Precipitation patterns are central to human and natural life. In a rapidly warming climate, reliable simulations of changing precipitation patterns can help adapt to climate change. However, these simulations are challenging due to the multi-scale variability of weather systems and the influence of complex surface features (like mountains and coastlines) on precipitation trends and extremes [Salathe et al., 2008]. For many purposes, such as estimating flood hazards, precipitation must be estimated at spatial resolutions of only a few kilometers. Fluid-dynamical models of the global atmosphere are too expensive to run routinely at such fine scales [Stevens et al., 2019], so the climate adaptation community relies on "downscaling" (the climate science terminology for super-resolution) of coarse-grid simulations to a finer grid. Traditional downscaling methods are either "dynamical" (running a fine-grid fluid-dynamical model limited to the region of interest, which requires specialized knowledge and computational resources) or "statistical" (typically restricted to simple univariate methods) [Tang et al., 2016]. This work builds on vision-based super-resolution methods to improve statistical downscaling and is a natural follow-up to recent deep-learning-based weather/climate prediction methods, which have revolutionized data-driven forecasting. These approaches boast improvements of orders of magnitude in runtime without sacrificing accuracy [Pathak et al., 2022; Lam et al., 2023].

The authors address the downscaling problem for a sequence. The objective is to transform a sequence ("video") of low-resolution precipitation frames into a sequence of high-resolution frames. Despite differences from natural videos, precipitation's hourly temporal continuity allows the use of video super-resolution techniques to leverage multiple context frames for stochastic downscaling [Rota et al., 2022; Liu et al., 2022].

Recent efforts to enhance the resolution of climate states like precipitation have relied on deterministic regression methods using convolutions or transformers. However, super-resolution is a one-to-many mapping with a continuum of "correct" answers. Supervised learning for these problems often leads to visual artifacts from mode averaging, where the network predicts an average of incompatible solutions, causing blurriness in visual data [Ledig et al., 2017; Zhao et al., 2017]. Besides visual artifacts, mode averaging can have even more dramatic implications in climate and weather modeling, such as the underestimation of extreme precipitation, which is mainly induced by regional weather patterns on the unresolved scale. A natural alternative to supervised super-resolution methods [Dong et al., 2015; Kim et al., 2016; Zhang et al., 2018; Dai et al., 2019; Kappeler et al., 2016] to prevent mode averaging is conditional generative modeling, which captures multimodal conditional distributions.

Recent works propose using generative adversarial networks (GANs) for precipitation downscaling. These methods often face challenges, tending to converge on specific modes of the data distribution and occasionally fixating on isolated points in extreme cases. Despite their perceptual appeal, the scientific utility of super-resolution requires accurate modeling of the statistical distribution of high-resolution data given low-resolution input, which GANs typically fail to capture.

In this paper, the authors propose SpatioTemporal Video Diffusion (STVD), for spatio-temporal precipitation downscaling. It uses a downscaler for a coarse prediction. The residual error is modeled with conditional diffusion, adding finer details to the coarse prediction. Both modules rely on spatio-temporal factorized attention to process the input sequence. Diffusion models are well-suited for precipitation downscaling as they successfully capture high dimensional and multimodal distributions, alleviating a key drawback of GAN-based methods for climate science applications.

### Key Contributions

1. A novel framework for temporal precipitation downscaling using diffusion models. The model combines a deterministic downscaling module with a diffusion-based residual module. It leverages spatio-temporal factorized attention to process information from multiple low-resolution frames.

2. The model outperforms six strong super-resolution baselines across multiple criteria, including MSE and several distributional metrics. Comparisons are made against two image super-resolution models and four video super-resolution models using the FV3GFS global atmosphere simulation dataset [Zhou et al., 2019; UFS Community, 2021].

3. The approach captures key characteristics of precipitation, including extreme precipitation probabilities and spatial patterns of annual precipitation in mountainous regions, which are crucial for domain science applications.

## 2. Downscaling via Spatiotemporal Video Diffusion

### Problem Statement

At training time, the model assumes access to a collection of high-resolution precipitation frame sequences y^{0:T} and their corresponding low-resolution precipitation frame sequences x^{0:T}. Such a low-resolution sequence can be obtained through area-weighted coarsening [Mahecha et al., 2020] of the corresponding high-resolution sequence. The objective is to train a model to effectively downscale (super-resolve) a given sequence x^{0:T} with y^{0:T} serving as the target.

More formally, let x^t in R^{C x H x W} and y^t in R^{1 x sH x sW} represent individual low-resolution and high-resolution frames. Here, s denotes the downscaling factor, C is the number of channels (quantities used as input to the model to characterize the atmospheric state in each low-resolution grid cell), and H, W indicate the height and width of the low-resolution frame. The study adopts a downscaling factor of s=8 and has C=12 total low-resolution channels. In addition to the low-resolution precipitation state, eleven channels of information are provided to the model, such as topography, wind velocity, and surface temperature.

### Solution Sketch

The approach treats the downscaling problem as a conditional generative modeling task. The model learns the conditional distribution of high-resolution precipitation frames, incorporating contextual information from the low-resolution precipitation frame sequence.

The proposed solution, SpatioTemporal Video Diffusion (STVD), relies on two modules: a deterministic downscaler and a stochastic component based on conditional diffusion models [Ho et al., 2020; Song and Ermon, 2019], both using spatio-temporal factorized attention. The first module uses a UNet with factorized attention to integrate information from a low-resolution frame sequence, resulting in an initial prediction frame sequence. The second module is a conditional diffusion model that stochastically generates a sequence of additive residual frames r^{0:T} which serves to add fine-grained details to the initial prediction. Together, these two modules produce a high-resolution frame sequence y_hat^{0:T} = y_bar^{0:T} + r^{0:T}. Both modules are trained end-to-end.

Decomposing the prediction into a deterministic mean and a stochastic residual is inspired by predictive-coding-based video decompression. This approach aims to predict a sequence of video frames while compressing the sparse residuals [Agustsson et al., 2020; Yang et al., 2023] which are easier to model than dense frames. Similarly, it is easier to generate residuals than dense images when using diffusion models [Yang et al., 2023; Mardani et al., 2023].

### 2.1 Probabilistic Modeling of Downscaling

Given a sequence of low-resolution frames x^{0:T} and the corresponding high-resolution frames y^{0:T}, the aim is to learn a parametric approximation p_theta of the conditional distribution:

p(y^{0:T} | x^{0:T}) ≈ p_theta(y^{0:T} | x^{0:T})

Importantly, independence across time is not assumed; each generated frame y^t can depend on all other generated frames. The generated high-resolution frame sequence is conditioned on the entire low-resolution frame sequence, capturing long-range temporal correlations and enhancing the fidelity and cohesion of the high-resolution reconstruction.

The likelihood p_theta(y^{0:T} | x^{0:T}) is modeled using a deterministic downscaler and a residual diffusion model. The model parameters theta = (phi, psi) decompose into those for a downscaler (phi) and a diffusion model (psi).

#### Deterministic Downscaling

The first module is a deterministic downscaler that predicts an initial high-resolution frame sequence:

y_bar^{0:T} = mu_phi(x^{0:T})

where mu_phi is a network generating a deterministic high-resolution prediction with parameters phi. Bicubic interpolation is performed on each frame of x^{0:T} before passing the sequence through the network mu_phi. Since the diffusion network operates on high-resolution inputs (i.e. denoising the high-resolution residuals), this choice allows the use of the same UNet [Ronneberger et al., 2015] architecture (with different weights) for both the downscaling module mu_phi and the residual diffusion module. This enables easy sharing of features across the modules via concatenation.

Importantly, mu_phi incorporates a temporal attention mechanism that allows any frame at time t, or its corresponding feature map, to attend to all context frames from 0 to T. This architecture enables the concurrent inference of all frames within the sequence y_bar^{0:T}. The attention weights differ for each frame, allowing for the flexible incorporation of information across time.

#### Stochastic Residual Modeling via Diffusion

After computing the initial prediction y_bar^{0:T}, finer details are modeled by residuals learned from a conditional diffusion model. The final stochastic high-resolution frame sequence y_hat^{0:T} is generated by sampling an additive residual sequence r^{0:T} from this model:

y_hat^{0:T} = y_bar^{0:T} + r^{0:T}

Thus the goal is to model the residuals r^{0:T} = y^{0:T} - y_bar^{0:T}. The diffusion model generates the entire residual sequence r^{0:T} concurrently, with the generation of each residual r^t dependent on the others. This is achieved via a UNet architecture with spatio-temporal attention, similar to the mechanism used for the deterministic downscaling module.

To model the distribution of r^{0:T}, DDPM [Ho et al., 2020] is used. A collection of latent variables r_{0:N}^{0:T} is introduced, where the lower subscripts indicate the denoising diffusion step. In the forward process, the latent variable r_n^{0:T} is created from r_{n-1}^{0:T} via additive noise. In the reverse process for generation, a denoising model (with parameters psi) is trained to predict r_{j-1}^{0:T} from r_j^{0:T}. N denotes the total number of denoising steps. Note that r^{0:T} = r_0^{0:T}, i.e. the first diffusion step corresponds to the true residual. Additionally, r_0^{0:T} implicitly depends on the downscaler parameters phi, allowing simultaneous optimization of all model parameters theta = (phi, psi) within the context of diffusion modeling.

As is standard in diffusion models [Ho et al., 2020], the reverse process is parameterized via a Gaussian distribution with a mean determined by a neural network M_psi:

p_psi(r_{n-1}^{0:T} | r_n^{0:T}, c) = N(r_{n-1}^{0:T} | M_psi(r_n^{0:T}, n, c), gamma * I)

where M_psi is a denoising network and gamma is a hyperparameter for variance. The diffusion model directly accesses the context c = (x^{0:T}, y_bar^{0:T}), and is implicitly conditioned on x^{0:T} via concatenation of feature maps from the downscaler module. As in the downscaler, x^{0:T} is bicubically upsampled before channel-wise concatenation with y_bar^{0:T} to match the dimensions when forming c.

### Loss Function

To train the model, the angular parametrization suggested by Salimans and Ho [2022] is used. Specifically, this results in the diffusion loss of the form:

L(psi, phi) = E_{x^{0:T}, y^{0:T}, n, epsilon} sum_{t=0}^{T} ||v - M_psi(r_n^{0:T}, n, c)||^2

where epsilon ~ N(0, I), n is sampled uniformly from {1, 2, ..., N}, and the sequences x^{0:T}, y^{0:T} are sampled from the training distribution. Here, c = (x^{0:T}, y_bar^{0:T}) where y_bar^{0:T} = mu_phi(x^{0:T}). The scalars alpha_n^2 = prod_{i=1}^{n} (1 - beta_i) and sigma_n^2 = 1 - alpha_n^2 are used to define v = alpha_n * epsilon - sigma_n * r_0^{0:T}. Training and inference are concurrent across multiple frames due to spatio-temporal attention. DDIM sampling [Song et al., 2020] is used to generate frame residuals with fewer diffusion steps.

### Training Algorithm

1. While not converged:
   - Sample x^{0:T} and y^{0:T}
   - n ~ U(0, 1, 2, ..., N)
   - epsilon ~ N(0, I)
   - y_bar^{0:T} = mu_phi(x^{0:T})
   - r_0^{0:T} = y^{0:T} - y_bar^{0:T}
   - v = alpha_n * epsilon - sigma_n * r_0^{0:T}
   - r_n^{0:T} = alpha_n * r_0^{0:T} + sigma_n * epsilon
   - c = (x^{0:T}, y_bar^{0:T})
   - v_hat = M_psi(r_n^{0:T}, n, c)
   - L = ||v - v_hat||^2
   - (psi, phi) = (psi, phi) - nabla_{psi,phi} L

### Sampling Algorithm

1. Get an equally spaced increasing sub-sequence tau of length K << N
2. y_bar^{0:T} = mu_phi(x^{0:T})
3. c = (x^{0:T}, y_bar^{0:T})
4. r_K^{0:T} ~ N(0, I)
5. For n in reversed(tau):
   - v_hat = M_psi(r_n^{0:T}, n, c)
   - r_hat = alpha_n * r_n^{0:T} - sigma_n * v_hat
   - epsilon_hat = (sigma_n / alpha_n) * (r_n^{0:T} - r_hat)
   - r_{n-1}^{0:T} = alpha_{n-1} * r_hat + sigma_{n-1} * epsilon_hat
6. y_hat^{0:T} = y_bar^{0:T} + r_0^{0:T}

### Network Architecture

Both the downscaler and the conditional diffusion model employ a UNet backbone with similar architectures and key adaptations to the attention mechanism. The downscaler takes the multi-channel input frames (x^{0:T}), yielding an initial estimate (y_bar^{0:T}). The diffusion UNet conditions on diffusion step n and concatenates feature maps from the downscaler with its own. The concatenated input to the diffusion UNet (x^{0:T}, y_bar^{0:T}, and r_n^{0:T}), along with the conditioning variables (diffusion step n and the feature maps from downscaler), yields the output v.

Computing full attention for temporal coherence across the entire video data cube is very expensive for processing long sequences or high-resolution inputs. To optimize efficiency, several strategies are employed:
- Decoupling attention between spatial and temporal dimensions
- Using a linear variant of self-attention [Katharopoulos et al., 2020] for non-bottleneck layers (where the effective number of "tokens" for attention is relatively large)
- Focusing spatial attention on localized patches (instead of the entire feature map)
- Calculating per-channel temporal attention in large spatial dimensions (namely, the ultimate and penultimate expansion and contraction layers of UNet)

These modifications dramatically reduce the time complexity and memory footprint of the transformer blocks.

## 3. Experiments

### Baselines

The evaluation compares STVD against six contemporary state-of-the-art baselines:

1. **Swin-IR** [Liang et al., 2021]: Image super-resolution model based on the Swin Vision Transformer
2. **Swin-IR-Diff** [Mardani et al., 2023; Saharia et al., 2022]: Residual diffusion variant of Swin-IR
3. **VRT** [Liang et al., 2022]: Video super-resolution model grounded in vision transformer architecture
4. **RVRT** [Liang et al., 2022]: Recurrent variant of VRT incorporating guided deformable attention for clip alignment
5. **PSRT** [Shi et al., 2022]: Video super-resolution using multi-frame attention groups
6. **VDM** [Ho et al., 2022]: Video diffusion model

### Ablation Studies

Three ablation configurations are tested:
- **STVD-3**: Uses 3 context frames instead of 5
- **STVD-1**: Uses 1 context frame (ablates for the temporal attention block)
- **STVD-Single**: Removes additional input channels (only precipitation state as input)

### Dataset

The dataset derives from an 11-member initial condition ensemble of 13-month simulations using a global atmosphere model, FV3GFS, run at 25 km resolution and forced by climatological sea surface temperatures and sea ice. The first month of each simulation is discarded to allow spin-up, effectively providing 11 years of reference data (first 10 years for training, last year for validation). FV3GFS was developed by the National Oceanic and Atmospheric Administration (NOAA) [Zhou et al., 2019; UFS Community, 2021].

Three-hourly average data were saved from this entire simulation using a 25 km horizontal "fine grid." The data were further coarsened by a factor of 8 to create a 200 km horizontal "coarse grid," resulting in paired data (x_t, y_t). The goal is to apply video downscaling to the coarse-grid precipitation field to obtain temporally smooth fine-grid precipitation estimates. FV3GFS uses a cubed-sphere grid, where the surface of the globe is divided into six tiles, each covered by an S x S array of points with S=48 for the 200 km coarse grid and S=384 for the 25 km fine grid.

12 coarse-grid input fields are used, including precipitation, topography, and horizontal vector wind at various levels.

#### Additional Variables in FV3GFS Dataset

| Short Name | Long Name | Units |
|---|---|---|
| CPRATsfc | Surface convective precip. rate | kg/m^2/s |
| DSWRFtoa | Top of atmos. down shortwave flux | W/m^2 |
| TMPsfc | Surface temperature | K |
| UGRD10m | 10-meter eastward wind | m/s |
| VGRD10m | 10-meter northward wind | m/s |
| ps | Surface pressure | Pa |
| u700 | 700-mb eastward wind | m/s |
| v700 | 700-mb northward wind | m/s |
| liq_wat | Vert. integral of cloud water mix ratio | kg/kg kg/m^2 |
| sphum | Vert. integral of specific humidity | kg/kg kg/m^2 |
| zsurf | Topography | -- |

### Training and Testing Details

- Downscaling factor: 8x
- Context length: 5 consecutive frames
- Optimizer: Adam with initial learning rate 1e-4, decaying to 5e-7 with cosine annealing
- GPU: NVidia RTX A6000
- Diffusion parametrization: v-parametrization [Salimans and Ho, 2022]
- Fixed diffusion depth: N = 1400
- Training: 1 million steps (~7 days on a single node)
- Batch size: 1
- Preprocessing: logarithmic transformation of precipitation states, normalized to [-1, 1]
- Inference: DDIM sampling with 30 steps on EMA variant (decay rate 0.995)
- Tile dimensions: 384x384 high-resolution, 48x48 low-resolution

### Evaluation Metrics

- **MSE**: Mean Square Error -- average squared difference between predicted and actual values
- **CRPS** (Continuous Ranked Probability Score) [Brown, 1974; Taillardat et al., 2023]: Assesses the discrepancy between the predicted cumulative distribution function and the observed data. Computed over 10 stochastic realizations.
- **EMD** (Earth Mover Distance) [Rubner et al., 1998]: Quantifies the agreement between the target and predicted global precipitation distributions.
- **PE** (99.999th Percentile Error): Focuses on tail events and extreme precipitation.
- **SAE** (Spatial Autocorrelation Error) [Teufel et al., 2023]: Mean absolute error between the spatial autocorrelation of the predictions and ground truth.

### Quantitative Results

| Method | CRPS (10^-5) | MSE (10^-8) | EMD (10^-6) | PE (10^-3) | SAE (10^-6) |
|---|---|---|---|---|---|
| **STVD (ours)** | **1.85** | **0.59** | **2.49** | **1.2** | **4.00** |
| PSRT | 2.15 | 0.66 | 4.21 | 3.8 | 6.24 |
| RVRT | 3.55 | 1.73 | 4.33 | 3.6 | 7.39 |
| VRT | 3.58 | 1.74 | 4.61 | 4.0 | 7.39 |
| Swin-IR-Diff | 2.29 | 1.94 | 6.38 | 4.4 | 7.70 |
| VDM | 2.21 | 0.73 | 12.70 | 6.4 | 8.84 |
| Swin-IR | 2.36 | 2.29 | 17.40 | 23.40 | 18.9 |
| STVD-single (ablation) | 1.81 | 0.62 | 4.64 | 2.3 | 6.09 |
| STVD-3 (ablation) | 1.96 | 0.68 | 4.94 | 2.6 | 4.99 |
| STVD-1 (ablation) | 2.05 | 0.72 | 7.19 | 4.1 | 6.87 |

### Qualitative and Quantitative Analysis

STVD performs strongly across all metrics, outperforming all baselines. Swin-IR overestimates precipitation, while all other baselines underestimate it. This discrepancy is undesirable, as poor performance on rare and extreme precipitation events can negatively impact disaster mitigation policies. The proposed method closely matches the precipitation distribution, as measured by PE and EMD.

Using only precipitation as an input (STVD-single) results in slightly worse performance across all metrics, indicating the predictive value of additional inputs. The ablation model STVD-1, which lacks full sequence information, performs significantly worse, highlighting the importance of temporal attention.

The model generates high quality results which preserve most patterns with a high degree of similarity. PSRT and RVRT produce slightly more diffuse precipitation features, while Swin-IR produces slightly more pixelated features.

Annually-averaged precipitation from the model effectively replicates the ground truth, including the strength and narrow spatial structure of high precipitation bands along the Northern California coastal mountains and the Sierras -- features not resolved by the coarse-grid inputs.

### Realism-Distortion Tradeoff

Distortion metrics such as MSE often conflict with perceptual quality, wherein reducing distortion typically degrades perceptual realism [Blau and Michaeli, 2018]. In this context, the tradeoff translates to balancing MSE and PE. While MSE captures the average accuracy of predictions, PE represents the model's ability to reproduce extreme events, serving as a proxy for realism. Realism in climate modeling refers to the accurate representation of extreme weather patterns, crucial for applications like flood forecasting and disaster mitigation. As the number of sampling steps increases, MSE tends to rise slightly, but PE decreases significantly.

## 4. Related Work

### Diffusion Models

Diffusion models [Sohl-Dickstein et al., 2015; Ho et al., 2020; Song et al., 2021; Pandey and Mandt, 2023; Pandey et al., 2024; Manduchi et al., 2024] are a class of generative models based on an iterative denoising process. Related video diffusion models generate deterministic next-frame predictions autoregressively with additional residuals generated by a diffusion model [Yang et al., 2023], or generate videos directly in pixel space [Harvey et al., 2022; Voleti et al., 2022; Ho et al., 2022] or in a latent space [Blattmann et al., 2023a, 2023b]. While some works on video diffusion employ video super-resolution as a step in the overall modeling process, STVD focuses exclusively on the video super-resolution task within the context of precipitation downscaling.

### Super-Resolution

Within the computer vision community, the paradigm for single image super-resolution has shifted from classical approaches [Bascle et al., 1996; Farsiu et al., 2004] to deep learning based methods [Wang et al., 2020]. Generative approaches, like cascaded diffusion [Ramesh et al., 2022; Saharia et al., 2022a], SR3 [Saharia et al., 2022b], and DiffPIR [Zhu et al., 2023] employ diffusion models for image super-resolution but are unable to leverage temporal context. Many approaches for video super-resolution have been proposed [Chan et al., 2021; Fuoli et al., 2019; Huang et al., 2015]. Recent models of note include the transformer-based models PSRT [Shi et al., 2022] and VRT [Liang et al., 2022], as well as the recurrent variant RVRT [Liang et al., 2022]. These state-of-the-art approaches are deterministic, whereas STVD is generative, allowing it to prevent mode averaging and produce more realistic samples.

### Data-driven Weather and Climate Modeling

Recent years have seen advancements in data-driven climate and weather modeling [Reichstein et al., 2019; Mooers et al., 2023], with models like GraphCast [Lam et al., 2023], GenCast [Price et al., 2023], and FourCastNet [Pathak et al., 2022] providing forecasts competitive with meteorological methods while being significantly faster. Rather than replacing numerical forecasting methods, STVD seeks to augment their capabilities by downscaling coarse-grid predictions.

Previous data-driven downscaling approaches include: iterative methods inspired by FRVSR [Teufel et al., 2023; Sajjadi et al., 2018], Fourier neural operators for arbitrary resolution downscaling [Yang et al., 2023], and physically consistent downscaling using softmax layers for conservation laws [Harder et al., 2022]. These approaches are deterministic and lack the realism and uncertainty quantification provided by a generative approach.

In terms of generative approaches, concurrent work [Mardani et al., 2023] employs diffusion models for downscaling climate states. GANs have also been used for downscaling and precipitation prediction [Leinonen et al., 2020; Price and Rasp, 2022; Harris et al., 2022; Ravuri et al., 2021; Gong et al., 2023; Vosper et al., 2023]. However, GAN-based approaches inherit mode collapse and training difficulties [Arora et al., 2018], and are applied at each frame individually without incorporating temporal information.

## 5. Conclusion

STVD is a video super-resolution method for probabilistic precipitation downscaling. It deterministically super-resolves a given low-resolution frame sequence and then stochastically models the residual details via diffusion. The model effectively resolves how fine-grid precipitation features, generated as weather systems, interact with complex topography based on temporally coherent coarse-grid information. The method outperforms several competitive baselines on a range of quantitative metrics. This is an important step towards designing effective statistical downscaling methods, providing highly localized information for planning against extreme weather events, such as floods or hurricanes in a warming climate, using tractable coarse-grid atmospheric models.

### Limitations and Broader Impacts

A limitation of the approach is the necessity of paired low-resolution and high-resolution images for training. While this can be done once prior to training, designing methods that only require the low-resolution states is an interesting challenge. In terms of broader impacts, the approach could potentially have harmful consequences if adopted blindly to a new dataset, where distribution shift could cause model performance to degrade, potentially leading to underestimation of extreme weather risks such as droughts or floods. To mitigate these risks, the model should be re-trained and rigorously evaluated on the dataset of interest.

## Appendix

### A.1 Model Architecture

The architecture is a conditional extension of the DDPM [Ho et al., 2020] and SR3 [Saharia et al., 2022] models. The architecture consists of denoising and downscaling networks built on UNet backbones.

**Key components:**

- **ChannelDim**: ResBlock channel dimension for first contractive layer of UNet (set to 64)
- **ChannelMultipliers**: 1, 1, 2, 2, 3, 4 for subsequent contractive layers
- **TileSize**: 384 x 384

**ResBlock**: Standard implementation consisting of two blocks, each with a weight-standardized convolution using a 3x3 kernel, Group Normalization over groups of 8 and SiLU activation, followed by a channel-adjusting 1x1 convolution.

**Attention variants:**
- **Q-Spatial** (quadratic): Applied in bottleneck layer; self-attends between every pixel of the feature map. Rearrangement: [b*t, h*c, x, y] -> [b*t, h, c, x*y]
- **Q-Temporal** (quadratic): Applied in bottleneck layer; self-attends between feature maps across time. Rearrangement: [b*t, h*c, x, y] -> [b, h, c*x*y, t]
- **L-Spatial** (linear): Applied in expansive and contractive layers; self-attends between every pixel within a patch. Rearrangement: [b*t, h*c, x*p, y*p] -> [b*t, h*x*y, c, p*p] where p is patch size, starting with 192, halving at each contractive layer.
- **L-Temporal** (linear): Applied in expansive and contractive layers; self-attends between feature maps across time in a channel-factorized manner. Rearrangement: [b*t, h*c, x, y] -> [b, h*x*y, c, t]

**MLP**: Conditioning on the denoising step n is achieved using 32-dimensional random Fourier features, followed by a linear layer, GELU activation and another linear layer.

The denoising network is conditioned in three ways:
1. On the bicubically downscaled low-resolution frames x^{0:T}, concatenated with both the noisy residual r_n^{0:T}, the downscaler output, and y_bar^{0:T} along the channel dimension
2. On the feature maps generated by the downscaler network (connecting contractive layers of both networks)
3. On the denoising step n via MLP embedding received by both ResBlocks

### A.2 On Swin-IR-Diff and Multiple Channels

Swin-IR-Diff expands on the Swin-IR baseline. It opts to downscale each precipitation state individually, akin to an image super-resolution model. Resembling SR3 in its foundation of a conditional diffusion model, Swin-IR-Diff adopts a residual pipeline with a deterministic prediction corrected by a residual from the conditional diffusion model, with Swin-IR serving as the deterministic downscaler.

An ablation focusing on the incorporation of additional climate states shows that the introduction of additional channels yields a notable improvement in performance (comparing STVD with multiple input states versus STVD-single with only precipitation state as input).

### A.3 Spectra

Analysis of the squared-magnitude of the complex-valued FFT applied to evaluation images shows that samples from STVD closely match the ground-truth high resolution spectrum. The baselines RVRT, VRT, and PSRT demonstrate a spectrum which decays too rapidly, placing too little energy in the high-frequency components, with banding indicating over-smoothness. The Swin-IR baseline shows outliers of large magnitudes leading to a checkerboard pattern in the spectrum. Swin-IR-Diff and VDM also decay the spectra more rapidly than STVD.

## References

- Agustsson, E., Minnen, D., Johnston, N., Balle, J., Hwang, S.J., Toderici, G. (2020). Scale-space flow for end-to-end optimized video compression. CVPR.
- Arora, S., Risteski, A., Zhang, Y. (2018). Do GANs learn the distribution? Some theory and empirics. ICLR.
- Bascle, B., Blake, A., Zisserman, A. (1996). Motion deblurring and super-resolution from an image sequence. ECCV.
- Blattmann, A., Rombach, R., Ling, H., Dockhorn, T., Kim, S.W., Fidler, S., Kreis, K. (2023). Align your latents: High-resolution video synthesis with latent diffusion models. CVPR.
- Blattmann, A., Dockhorn, T., Kulal, S., et al. (2023). Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv:2311.15127.
- Blau, Y., Michaeli, T. (2018). The perception-distortion tradeoff. CVPR.
- Brown, T.A. (1974). Admissible scoring systems for continuous distributions.
- Chan, K.C.K., Wang, X., Yu, K., Dong, C., Loy, C.C. (2021). BasicVSR: The search for essential components in video super-resolution and beyond. CVPR.
- Dai, T., Cai, J., Zhang, Y., Xia, S.T., Zhang, L. (2019). Second-order attention network for single image super-resolution. CVPR.
- Dong, C., Loy, C.C., He, K., Tang, X. (2015). Image super-resolution using deep convolutional networks. IEEE TPAMI.
- Farsiu, S., Robinson, M.D., Elad, M., Milanfar, P. (2004). Fast and robust multiframe super resolution. IEEE TIP.
- Fuoli, D., Gu, S., Timofte, R. (2019). Efficient video super-resolution through recurrent latent space propagation. ICCVW.
- Gong, A., Li, R., Pan, B., Chen, H., Ni, G., Chen, M. (2023). Enhancing spatial variability representation of radar nowcasting with generative adversarial networks. Remote Sensing.
- Harder, P., Yang, Q., Ramesh, V., Sattigeri, P., Hernandez-Garcia, A., Watson, C., Szwarcman, D., Rolnick, D. (2022). Generating physically-consistent high-resolution climate data with hard-constrained neural networks. arXiv:2208.05424.
- Harris, L., McRae, A.T.T., Chantry, M., Dueben, P.D., Palmer, T.N. (2022). A generative deep learning approach to stochastic downscaling of precipitation forecasts. JAMES.
- Harvey, W., Naderiparizi, S., Masrani, V., Weilbach, C.D., Wood, F. (2022). Flexible diffusion modeling of long videos. NeurIPS.
- Ho, J., Jain, A., Abbeel, P. (2020). Denoising diffusion probabilistic models. NeurIPS.
- Ho, J., Salimans, T., Gritsenko, A., Chan, W., Norouzi, M., Fleet, D.J. (2022). Video diffusion models. arXiv:2204.03458.
- Huang, Y., Wang, W., Wang, L. (2015). Bidirectional recurrent convolutional networks for multi-frame super-resolution. NeurIPS.
- Kappeler, A., Yoo, S., Dai, Q., Katsaggelos, A.K. (2016). Video super-resolution with convolutional neural networks. IEEE TCI.
- Katharopoulos, A., Vyas, A., Pappas, N., Fleuret, F. (2020). Transformers are RNNs: Fast autoregressive transformers with linear attention. ICML.
- Kim, J., Lee, J.K., Lee, K.M. (2016). Accurate image super-resolution using very deep convolutional networks. CVPR.
- Kingma, D.P., Ba, J. (2014). Adam: A method for stochastic optimization. arXiv:1412.6980.
- Lam, R., Sanchez-Gonzalez, A., Willson, M., et al. (2023). Learning skillful medium-range global weather forecasting. Science.
- Ledig, C., Theis, L., Huszar, F., et al. (2017). Photo-realistic single image super-resolution using a generative adversarial network. CVPR.
- Leinonen, J., Nerini, D., Berne, A. (2020). Stochastic super-resolution for downscaling time-evolving atmospheric fields with a generative adversarial network. IEEE TGRS.
- Liang, J., Cao, J., Sun, G., Zhang, K., Van Gool, L., Timofte, R. (2021). SwinIR: Image restoration using Swin Transformer. ICCV.
- Liang, J., Cao, J., Fan, Y., Zhang, K., Ranjan, R., Li, Y., Timofte, R., Van Gool, L. (2022). VRT: A video restoration transformer. arXiv:2201.12288.
- Liang, J., Fan, Y., Xiang, X., Ranjan, R., Ilg, E., Green, S., Cao, J., Zhang, K., Timofte, R., Van Gool, L. (2022). Recurrent video restoration transformer with guided deformable attention. NeurIPS.
- Liu, H., Ruan, Z., Zhao, P., Dong, C., Shang, F., Liu, Y., Yang, L., Timofte, R. (2022). Video super-resolution based on deep learning: A comprehensive survey. AIR.
- Mahecha, M.D., et al. (2020). Earth system data cubes unravel global multivariate dynamics. Earth System Dynamics.
- Manduchi, L., Pandey, K., Bamler, R., et al. (2024). On the challenges and opportunities in generative AI. arXiv:2403.00025.
- Mardani, M., Brenowitz, N., Cohen, Y., et al. (2023). Generative residual diffusion modeling for km-scale atmospheric downscaling. arXiv:2309.15214.
- Mooers, G., Pritchard, M., Beucler, T., Srivastava, P., et al. (2023). Comparing storm resolving models and climates via unsupervised machine learning. Scientific Reports.
- Pandey, K., Mandt, S. (2023). A complete recipe for diffusion generative models. ICCV.
- Pandey, K., Rudolph, M., Mandt, S. (2024). Efficient integrators for diffusion generative models. ICLR.
- Pathak, J., Subramanian, S., Harrington, P., et al. (2022). FourCastNet: A global data-driven high-resolution weather model using adaptive Fourier neural operators. arXiv:2202.11214.
- Price, I., Rasp, S. (2022). Increasing the accuracy and resolution of precipitation forecasts using deep generative models. AISTATS.
- Price, I., Sanchez-Gonzalez, A., Alet, F., et al. (2023). GenCast: Diffusion-based ensemble forecasting for medium-range weather. arXiv:2312.15796.
- Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., Chen, M. (2022). Hierarchical text-conditional image generation with CLIP latents. arXiv:2204.06125.
- Ravuri, S., Lenc, K., Willson, M., et al. (2021). Skilful precipitation nowcasting using deep generative models of radar. Nature.
- Reichstein, M., Camps-Valls, G., Stevens, B., et al. (2019). Deep learning and process understanding for data-driven Earth system science. Nature.
- Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. MICCAI.
- Rota, C., Buzzelli, M., Bianco, S., Schettini, R. (2022). Video restoration based on deep learning: A comprehensive survey. AIR.
- Rubner, Y., Tomasi, C., Guibas, L.J. (1998). A metric for distributions with applications to image databases. ICCV.
- Saharia, C., Ho, J., Chan, W., Salimans, T., Fleet, D.J., Norouzi, M. (2022). Image super-resolution via iterative refinement. IEEE TPAMI.
- Saharia, C., Chan, W., Saxena, S., et al. (2022). Photorealistic text-to-image diffusion models with deep language understanding. NeurIPS.
- Sajjadi, M.S.M., Vemulapalli, R., Brown, M. (2018). Frame-recurrent video super-resolution. CVPR.
- Salathe, E.P., Steed, R., Mass, C.F., Zahn, P.H. (2008). A high-resolution climate model for the US Pacific Northwest. J. Climate.
- Salimans, T., Ho, J. (2022). Progressive distillation for fast sampling of diffusion models. arXiv:2202.00512.
- Shi, S., Gu, J., Xie, L., Wang, X., Yang, Y., Dong, C. (2022). Rethinking alignment in video super-resolution transformers. NeurIPS.
- Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., Ganguli, S. (2015). Deep unsupervised learning using nonequilibrium thermodynamics. ICML.
- Song, J., Meng, C., Ermon, S. (2020). Denoising diffusion implicit models. arXiv:2010.02502.
- Song, Y., Ermon, S. (2019). Generative modeling by estimating gradients of the data distribution. NeurIPS.
- Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S., Poole, B. (2021). Score-based generative modeling through stochastic differential equations. ICLR.
- Stevens, B., et al. (2019). DYAMOND: The DYnamics of the Atmospheric general circulation Modeled On Non-hydrostatic Domains. PEPS.
- Taillardat, M., Fougeres, A.L., Naveau, P., De Fondeville, R. (2023). Evaluating probabilistic forecasts of extremes using continuous ranked probability score distributions. IJF.
- Tang, J., Niu, X., Wang, S., Gao, H., Wang, X., Wu, J. (2016). Statistical downscaling and dynamical downscaling of regional climate in China. JGR Atmospheres.
- Teufel, B., Carmo, F., Sushama, L., et al. (2023). Physics-informed deep learning framework to model intense precipitation events at super resolution. Geoscience Letters.
- UFS Community. (2021). UFS Weather Model v1.1.0. Zenodo.
- Voleti, V., Jolicoeur-Martineau, A., Pal, C. (2022). MCVD: Masked conditional video diffusion for prediction, generation, and interpolation. NeurIPS.
- Vosper, E., Watson, P., Harris, L., McRae, A., Santos-Rodriguez, R., Aitchison, L., Mitchell, D. (2023). Deep learning for downscaling tropical cyclone rainfall to hazard-relevant spatial scales. JGR Atmospheres.
- Wang, Z., Chen, J., Hoi, S.C.H. (2020). Deep learning for image super-resolution: A survey. IEEE TPAMI.
- Yang, Q., Hernandez-Garcia, A., Harder, P., et al. (2023). Fourier neural operators for arbitrary resolution climate data downscaling. arXiv:2305.14452.
- Yang, R., Srivastava, P., Mandt, S. (2023). Diffusion probabilistic modeling for video generation. Entropy.
- Yang, R., Yang, Y., Marino, J., Mandt, S. (2023). Insights from generative modeling for neural video compression. IEEE TPAMI.
- Zhang, Y., Li, K., Li, K., Wang, L., Zhong, B., Fu, Y. (2018). Image super-resolution using very deep residual channel attention networks. ECCV.
- Zhao, S., Song, J., Ermon, S. (2017). Towards deeper understanding of variational autoencoding models. arXiv:1702.08658.
- Zhou, L., Lin, S.J., Chen, J.H., Harris, L.M., Chen, X., Rees, S.L. (2019). Toward convective-scale prediction within the next generation global prediction system. BAMS.
- Zhu, Y., Zhang, K., Liang, J., Cao, J., Wen, B., Timofte, R., Van Gool, L. (2023). Denoising diffusion models for plug-and-play image restoration. CVPR.
