# Abstract

Latent Diffusion Models (LDMs) enable high-quality image synthesis while avoiding excessive compute demands by training a diffusion model in a compressed lower-dimensional latent space. Here, we apply the LDM paradigm to high-resolution video generation, a particularly resource-intensive task. We first pre-train an LDM on images only; then, we turn the image generator into a video generator by introducing a temporal dimension to the latent space diffusion model and fine-tuning on encoded image sequences, i.e., videos. Similarly, we temporally align diffusion model upsamplers, turning them into temporally consistent video super resolution models. We focus on two relevant real-world applications: Simulation of in-the-wild driving data and creative content creation with text-to-video modeling. In particular, we validate our **Video LDM** on real driving videos of resolution ```latex $512\times1024$ ```, achieving state-of-the-art performance. Furthermore, our approach can easily leverage off-the-shelf pre-trained image LDMs, as we only need to train a temporal alignment model in that case. Doing so, we turn the publicly available, state-of-the-art text-to-image LDM Stable Diffusion into an efficient and expressive text-to-video model with resolution up to ```latex $1280\times2048$ ```. We show that the temporal layers trained in this way generalize to different fine-tuned text-to-image LDMs. Utilizing this property, we show the first results for personalized text-to-video generation, opening exciting directions for future content creation.

# Introduction

Generative models of images have received unprecedented attention, owing to recent breakthroughs in the underlying modeling methodology. The most powerful models today are built on generative adversarial networks [goodfellow2014generative; karras2019style; karras2020analyzing; karras2021aliasfree; sauer2021styleganxl], autoregressive transformers [esser2020taming; ramesh2021dalle; yu2022parti], and most recently diffusion models [sohl2015deep; ho2020ddpm; song2020score; nichol2021improved; dhariwal2021diffusion; ho2021cascaded; nichol2021glide; rombach2021highresolution; ramesh2022dalle2; saharia2022imagen]. Diffusion models (DMs) in particular have desirable advantages; they offer a robust and scalable training objective and are typically less parameter intensive than their transformer-based counterparts. However, while the image domain has seen great progress, *video* modeling has lagged behind---mainly due to the significant computational cost associated with training on video data, and the lack of large-scale, general, and publicly available video datasets. While there is a rich literature on video synthesis [babaeizadeh2018stochastic; svg; lee2018savp; hvrnn; lsvg; Weissenborn2020Scaling; yan2021videogpt; hong2022cogvideo; wu2021godiva; wu2022nuwa; ge2022longvideo; Gupta_2022_CVPR; scene_dyn; yu2022generating; tian2021a; villegas17mcnet; Luc2020TransformationbasedAV; TGAN2020; brooks2022generating; Skorokhodov_2022_CVPR; kahembwe2020lower; mittal2017sync; Pan2017ToCW; marwah2017attentive; li2017video; gupta2018imagine], most works, including previous video DMs [yang2022video; ho2022video; hoeppe2022diffusion; voleti2022mcvd; harvey2022flexible], only generate relatively low-resolution, often short, videos. Here, we apply video models to real-world problems and generate high-resolution, long videos. Specifically, we focus on two relevant real-world video generation problems: (i) video synthesis of high-resolution real-world driving data, which has great potential as a simulation engine in the context of autonomous driving, and (ii) text-guided video synthesis for creative content generation.

To this end, we build on latent diffusion models (LDMs), which can reduce the heavy computational burden when training on high-resolution images [rombach2021highresolution]. We propose *Video LDMs* and extend LDMs to high-resolution *video* generation, a particularly compute-intensive task. In contrast to previous work on DMs for video generation [yang2022video; ho2022video; hoeppe2022diffusion; voleti2022mcvd; harvey2022flexible], we first pre-train our Video LDMs on images only (or use available pre-trained image LDMs), thereby allowing us to leverage large-scale image datasets. We then transform the LDM image generator into a video generator by introducing a temporal dimension into the latent space DM and training only these temporal layers on encoded image sequences, i.e., videos, while fixing the pre-trained spatial layers. We similarly fine-tune LDM's decoder to achieve temporal consistency in pixel space. To further enhance the spatial resolution, we also temporally align pixel-space and latent DM upsamplers [ho2021cascaded], which are widely used for image super resolution [saharia2021image; li2022srdiff; saharia2022imagen; rombach2021highresolution], turning them into temporally consistent video super resolution models. Building on LDMs, our method can generate globally coherent and long videos in a computationally and memory efficient manner. For synthesis at very high resolutions, the video upsampler only needs to operate locally, keeping training and computational requirements low. We ablate our method and test on ```latex $512\times1024$ ``` real driving scene videos, achieving state-of-the-art video quality, and synthesize videos of several minutes length. We also video fine-tune a powerful, publicly available text-to-image LDM, *Stable Diffusion* [rombach2021highresolution], and turn it into an efficient and powerful text-to-video generator with resolution up to ```latex $1280\times2048$ ```. Since we only need to train the temporal alignment layers in that case, we can use a relatively small training set of captioned videos. By transferring the trained temporal layers to differently fine-tuned text-to-image LDMs, we demonstrate personalized text-to-video generation for the first time. We hope our work opens new avenues for efficient digital content creation and autonomous driving simulation.

**Contributions.** *(i)* We present an efficient approach for training high-resolution, long-term consistent video generation models based on LDMs. Our key insight is to leverage pre-trained image DMs and turn them into video generators by inserting temporal layers that learn to align images in a temporally consistent manner. *(ii)* We further temporally fine-tune super resolution DMs, which are ubiquitous in the literature. *(iii)* We achieve state-of-the-art high-resolution video synthesis performance on real driving scene videos, and we can generate multiple minute long videos. *(iv)* We transform the publicly available *Stable Diffusion* text-to-image LDM into a powerful and expressive text-to-video LDM, and *(v)* show that the learned temporal layers can be combined with different image model checkpoints (e.g., *DreamBooth* [ruiz2022dreambooth]).

# Background

DMs [sohl2015deep; ho2020ddpm; song2020score] learn to model a data distribution ```latex $p_{\text{data}}(\mathbf{x})$ ``` via *iterative denoising* and are trained with *denoising score matching* [hyvarinen2005scorematching; lyu2009scorematching; vincent2011; sohl2015deep; song2019generative; ho2020ddpm; song2020score]: Given samples ```latex $\mathbf{x} \sim p_{\text{data}}$ ```, *diffused* inputs ```latex $\mathbf{x}_\tau = \alpha_\tau \mathbf{x} + \sigma_\tau \boldsymbol{\epsilon}, \; \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ ``` are constructed; ```latex $\alpha_{\tau}$ ``` and ```latex $\sigma_\tau$ ``` define a *noise schedule*, parameterized via a diffusion-time ```latex $\tau$ ```, such that the logarithmic signal-to-noise ratio ```latex $\lambda_\tau = \log(\alpha_{\tau}^2/\sigma_\tau^2)$ ``` monotonically decreases. A denoiser model ```latex $\mathbf{f}_\theta$ ``` (parameterized with learnable parameters ```latex $\theta$ ```) receives the diffused ```latex $\mathbf{x}_\tau$ ``` as input and is optimized minimizing the denoising score matching objective

```latex
$$\mathbb{E}_{\mathbf{x} \sim p_{\text{data}}, \tau \sim p_{\tau}, \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left[\Vert \mathbf{y} - \mathbf{f}_\theta(\mathbf{x}_\tau; \mathbf{c}, \tau) \Vert_2^2 \right]$$
```

where ```latex $\mathbf{c}$ ``` is optional conditioning information, such as a text prompt, and the target vector ```latex $\mathbf{y}$ ``` is either the random noise ```latex $\boldsymbol{\epsilon}$ ``` or ```latex $\mathbf{v} = \alpha_\tau \boldsymbol{\epsilon} - \sigma_\tau \mathbf{x}$ ```. The latter objective (often referred to as *v-prediction*) has been introduced in the context of progressive distillation [salimans2022progressive] and empirically often yields faster convergence of the model (here, we use both objectives). Furthermore, ```latex $p_\tau$ ``` is a uniform distribution over the diffusion time ```latex $\tau$ ```. The forward diffusion as well as the reverse generation process in diffusion models can be described via stochastic differential equations in a continuous-time framework [song2020score], but in practice a fixed discretization can be used [ho2020ddpm]. The maximum diffusion time is generally chosen such that the input data is entirely perturbed into Gaussian random noise and an iterative generative denoising process that employs the learned denoiser ```latex $\mathbf{f}_\theta$ ``` can be initialized from such Gaussian noise to synthesize novel data. Here, we use ```latex $p_\tau \sim \mathcal{U}\{0,1000\}$ ``` and rely on a *variance-preserving* noise schedule [song2020score], for which ```latex $\sigma_\tau^2 = 1 - \alpha_\tau^2$ ```.

**Latent Diffusion Models (LDMs)** [rombach2021highresolution] improve in computational and memory efficiency over pixel-space DMs by first training a compression model to transform input images ```latex $\mathbf{x} \sim p_{\text{data}}$ ``` into a spatially lower-dimensional latent space of reduced complexity, from which the original data can be reconstructed at high fidelity. In practice, this approach is implemented with a regularized autoencoder, which reconstructs inputs ```latex $\mathbf{x}$ ``` via an encoder module ```latex $\mathcal{E}$ ``` and a decoder ```latex $\mathcal{D}$ ```, such that the reconstruction ```latex $\hat{\mathbf{x}} = \mathcal{D}(\mathcal{E}(\mathbf{x})) \approx \mathbf{x}$ ```. To ensure photorealistic reconstructions, an adversarial objective can be added to the autoencoder training [rombach2021highresolution], which is implemented using a patch-based discriminator [isola2017image]. A DM can then be trained in the compressed latent space and ```latex $\mathbf{x}$ ``` is replaced by its latent representation ```latex $\mathbf{z} = \mathcal{E}(\mathbf{x})$ ```. This latent space DM can be typically smaller in terms of parameter count and memory consumption compared to corresponding pixel-space DMs of similar performance.

# Latent Video Diffusion Models

Here we describe how we *video fine-tune* pre-trained image LDMs (and DM upsamplers) for high-resolution video synthesis. We assume access to a dataset ```latex $p_{\text{data}}$ ``` of videos, such that ```latex $\mathbf{x} \in \mathbb{R}^{T \times 3 \times \tilde{H} \times \tilde{W}}, \; \mathbf{x} \sim p_{\text{data}}$ ``` is a sequence of ```latex $T$ ``` RGB frames, with height and width ```latex $\tilde{H}$ ``` and ```latex $\tilde{W}$ ```.

## Turning Latent Image into Video Generators

Our key insight for efficiently training a video generation model is to re-use a pre-trained, fixed image generation model; an LDM parameterized by parameters ```latex $\theta$ ```. Formally, let us denote the neural network layers that comprise the image LDM and process inputs over the pixel dimensions as *spatial* layers ```latex $l_\theta^i$ ```, with layer index ```latex $i$ ```. However, although such a model is able to synthesize individual frames at high quality, using it directly to render a video of ```latex $T$ ``` consecutive frames will fail, as the model has no temporal awareness. We thus introduce additional *temporal* neural network layers ```latex $l_\phi^i$ ```, which are interleaved with the existing *spatial* layers ```latex $l_\theta^i$ ``` and learn to align individual frames in a temporally consistent manner. These ```latex $L$ ``` additional temporal layers ```latex $\{l_\phi^i\}_{i=1}^L$ ``` define the *video-aware* temporal backbone of our model, and the full model ```latex $\mathbf{f}_{\theta, \phi}$ ``` is thus the combination of the spatial and temporal layers.

We start from a frame-wise encoded input video ```latex $\mathcal{E}(\mathbf{x}) = \mathbf{z} \in \mathbb{R}^{T \times C \times H \times W}$ ```, where ```latex $C$ ``` is the number of latent channels and ```latex $H$ ``` and ```latex $W$ ``` are the spatial latent dimensions. The spatial layers interpret the video as a batch of independent images (by shifting the temporal axis into the batch dimension), and for each *temporal mixing layer* ```latex $l_\phi^i$ ```, we reshape back to video dimensions as follows (using `einops` [rogozhnikov2022einops] notation):

```latex
$$\begin{aligned}
\mathbf{z}' &\leftarrow \texttt{rearrange}(\mathbf{z}, \; \texttt{(b t) c h w} \rightarrow \texttt{b c t h w}) \\
\mathbf{z}' &\leftarrow l_\phi^i(\mathbf{z}', \mathbf{c}) \\
\mathbf{z}' &\leftarrow \texttt{rearrange}(\mathbf{z}', \; \texttt{b c t h w} \rightarrow \texttt{(b t) c h w})
\end{aligned}$$
```

where we added the batch dimension `b` for clarity. In other words, the spatial layers treat all ```latex $B \cdot T$ ``` encoded video frames independently in the batch dimension `b`, while the temporal layers ```latex $l_\phi^i(\mathbf{z}', \mathbf{c})$ ``` process entire videos in a new temporal dimension `t`. Furthermore, ```latex $\mathbf{c}$ ``` is (optional) conditioning information such as a text prompt. After each temporal layer, the output ```latex $\mathbf{z}'$ ``` is combined with ```latex $\mathbf{z}$ ``` as ```latex $\alpha_\phi^i \mathbf{z} + (1 - \alpha_\phi^i) \mathbf{z}'$ ```; ```latex $\alpha_\phi^i \in [0,1]$ ``` denotes a (learnable) parameter.

In practice, we implement two different kinds of temporal mixing layers: (i) temporal attention and (ii) residual blocks based on 3D convolutions. We use sinusoidal embeddings [vaswani2017attention; ho2020ddpm] to provide the model with a positional encoding for time.

Our video-aware temporal backbone is then trained using the same noise schedule as the underlying image model, and, importantly, we fix the spatial layers ```latex $l_\theta^i$ ``` and *only* optimize the temporal layers ```latex $l_\phi^i$ ``` via

```latex
$$\arg\min_{\phi} \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}, \tau \sim p_{\tau}, \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left[\Vert \mathbf{y} - \mathbf{f}_{\theta, \phi}(\mathbf{z}_\tau; \mathbf{c}, \tau) \Vert_2^2 \right]$$
```

where ```latex $\mathbf{z}_\tau$ ``` denotes diffused encodings ```latex $\mathbf{z} = \mathcal{E}(\mathbf{x})$ ```. This way, we retain the native image generation capabilities by simply skipping the temporal blocks, e.g., by setting ```latex $\alpha_\phi^i = 1$ ``` for each layer. A crucial advantage of our strategy is that huge image datasets can be used to pre-train the spatial layers, while the video data, which is often less widely available, can be utilized for focused training of the temporal layers.

### Temporal Autoencoder Finetuning

Our video models build on pre-trained image LDMs. While this increases efficiency, the autoencoder of the LDM is trained on images only, causing flickering artifacts when encoding and decoding a temporally coherent sequence of images. To counteract this, we introduce additional temporal layers for the autoencoder's decoder, which we finetune on video data with a (patch-wise) temporal discriminator built from 3D convolutions. Note that the encoder remains unchanged from image training such that the image DM that operates in latent space on encoded video frames can be re-used. As demonstrated by computing reconstruction FVD [unterthiner2018towards] scores, this step is critical for achieving good results.

## Prediction Models for Long-Term Generation

Although the approach described above is efficient for generating short video sequences, it reaches its limits when it comes to synthesizing very long videos. Therefore, we also train models as *prediction models* given a number of (first) ```latex $S$ ``` context frames. We implement this by introducing a temporal binary mask ```latex $\mathbf{m}_S$ ``` which masks the ```latex $T-S$ ``` frames the model has to predict, where ```latex $T$ ``` is the total sequence length. We feed this mask and the masked encoded video frames into the model for conditioning. Specifically, the frames are encoded with LDM's image encoder ```latex $\mathcal{E}$ ```, multiplied by the mask, and then fed (channel-wise concatenated with the masks) into the temporal layers ```latex $l_\phi^i$ ``` after being processed with a learned downsampling operation. Let ```latex $\mathbf{c}_S = (\mathbf{m}_S \circ \mathbf{z}, \mathbf{m}_S)$ ``` denote the concatenated spatial conditioning of masks and masked (encoded) images. Then, the objective reads

```latex
$$\mathbb{E}_{\mathbf{x} \sim p_{\text{data}}, \mathbf{m}_S \sim p_S, \tau \sim p_{\tau}, \boldsymbol{\epsilon}} \left[\Vert \mathbf{y} - \mathbf{f}_{\theta, \phi}(\mathbf{z}_\tau; \mathbf{c}_S, \mathbf{c}, \tau) \Vert_2^2 \right]$$
```

where ```latex $p_S$ ``` represents the (categorical) mask sampling distribution. In practice, we learn prediction models that condition either on 0, 1 or 2 context frames, allowing for classifier-free guidance as discussed below.

During inference, for generating long videos, we can apply the sampling process iteratively, re-using the latest predictions as new context. The first initial sequence is generated by synthesizing a single context frame from the base image model and generating a sequence based on that; afterwards, we condition on two context frames to encode movement. To stabilize this process, we found it beneficial to use *classifier-free diffusion guidance* [ho2021classifierfree], where we guide the model during sampling via

```latex
$$\mathbf{f}_{\theta, \phi}'(\mathbf{z}_\tau; \mathbf{c}_S) = \mathbf{f}_{\theta, \phi}(\mathbf{z}_\tau) + s \cdot \left(\mathbf{f}_{\theta, \phi}(\mathbf{z}_\tau; \mathbf{c}_S) - \mathbf{f}_{\theta, \phi}(\mathbf{z}_\tau) \right)$$
```

where ```latex $s \geq 1$ ``` denotes the guidance scale. We refer to this guidance as *context guidance*.

## Temporal Interpolation for High Frame Rates

High-resolution video is characterized not only by high spatial resolution, but also by high temporal resolution, i.e., a high frame rate. To achieve this, we divide the synthesis process for high-resolution video into two parts: The first is the process described above, which can generate *key frames* with large semantic changes, but (due to memory constraints) only at a relatively low frame rate. For the second part, we introduce an additional model whose task is to interpolate between given key frames. To implement this, we use the masking-conditioning mechanism introduced above. However, unlike the prediction task, we now mask the frames to be interpolated---otherwise, the mechanism remains the same, i.e., the image model is refined into a video interpolation model. In our experiments, we predict three frames between two given key frames, thereby training a ```latex $T \rightarrow 4T$ ``` interpolation model. To achieve even larger frame rates, we train the model simultaneously in the ```latex $T \rightarrow 4T$ ``` and ```latex $4T \rightarrow 16T$ ``` regimes (using videos with different fps), specified by binary conditioning.

Our training approach for prediction and interpolation models is inspired by recent works [voleti2022mcvd; harvey2022flexible; hoeppe2022diffusion] that use similar masking techniques.

## Temporal Fine-tuning of SR Models

Although the LDM mechanism already provides a good native resolution we aim to push this towards the megapixel range. We take inspiration from cascaded DMs [ho2021cascaded] and use a DM to further scale up the Video LDM outputs by ```latex $4\times$ ```. For our driving video synthesis experiments, we use a pixel-space DM [ho2021cascaded] and scale to ```latex $512\times1024$ ```; for our text-to-video models, we use an LDM upsampler [rombach2021highresolution] and scale to ```latex $1280\times2048$ ```. We use noise augmentation with noise level conditioning [ho2021cascaded; saharia2022imagen] and train the super resolution (SR) model ```latex $\mathbf{g}_{\theta, \phi}$ ``` (on images or latents) via

```latex
$$\mathbb{E}_{\mathbf{x} \sim p_{\text{data}}, (\tau,\tau_{\gamma}) \sim p_{\tau}, \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left[\Vert \mathbf{y} - \mathbf{g}_{\theta, \phi}(\mathbf{x}_\tau; \mathbf{c}_{\tau_{\gamma}}, \tau_{\gamma}, \tau) \Vert_2^2 \right]$$
```

where ```latex $\mathbf{c}_{\tau_{\gamma}} = \alpha_{\tau_\gamma} \mathbf{x} + \sigma_{\tau_\gamma} \boldsymbol{\epsilon}, \; \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ ```, denotes a noisy low-resolution image given to the model via concatenation, and ```latex $\tau_\gamma$ ``` the amount of noise added to the low-resolution image following the noise schedule ```latex $\alpha_\tau$ ```, ```latex $\sigma_\tau$ ```.

Since upsampling video frames independently would result in poor temporal consistency, we also make this SR model video-aware. We follow the mechanism introduced above with spatial layers ```latex $l_\theta^i$ ``` and temporal layers ```latex $l_\phi^i$ ``` and similarly video fine-tune the upscaler, conditioning on a low-resolution sequence of length ```latex $T$ ``` and concatenating low-resolution video images frame-by-frame. Since the upscaler operates locally, we conduct all upscaler training efficiently on patches only and later apply the model convolutionally.

Overall, we believe that the combination of an LDM with an upsampler DM is ideal for efficient high-resolution video synthesis. On the one hand, the main LDM component of our Video LDM leverages a computationally efficient, compressed latent space to perform all video modeling. This allows us to use large batch sizes and jointly encode more video frames, which benefits long-term video modeling, without excessive memory demands, as all video predictions and interpolations are carried out in latent space. On the other hand, the upsampler can be trained in an efficient patch-wise manner, therefore similarly saving computational resources and reducing memory consumption, and it also does not need to capture long-term temporal correlations due to the low-resolution conditioning. Therefore, no prediction and interpolation framework is required for this component.

# Experiments

**Datasets.** Since we focus on driving scene video generation as well as text-to-video, we use two corresponding datasets/models: *(i)* An in-house dataset of real driving scene (RDS) videos. The dataset consists of 683,060 videos of 8 seconds each at resolution ```latex $512\times1024$ ``` (```latex $H\times W$ ```) and frame rate up to 30 fps. Furthermore, the videos have binary night/day labels, annotations for the number of cars in a scene ("crowdedness"), and a subset of the data also has car bounding boxes. *(ii)* We use the WebVid-10M [bain21frozen] dataset to turn the publicly available *Stable Diffusion* Image LDM [rombach2021highresolution] into a Video LDM. WebVid-10M consists of 10.7M video-caption pairs with a total of 52K video hours. We resize the videos into resolution ```latex $320\times512$ ```. *(iii)* Moreover, we show experiments on the Mountain Biking dataset by Brooks et al. [brooks2022generating].

**Evaluation Metrics.** To evaluate our models, we use frame-wise Frechet Inception Distance (FID) [heusel2017gans] as well as Frechet Video Distance (FVD) [unterthiner2018towards]. Since FVD can be unreliable (discussed, for instance, by Brooks et al. [brooks2022generating]), we additionally perform human evaluation. For our text-to-video experiments, we also evaluate CLIP similarity (CLIPSIM) [wu2021godiva] and (video) inception score (IS).

**Model Architectures and Sampling.** Our Image LDMs are based on Rombach et al. [rombach2021highresolution]. They use convolutional encoders and decoders, and their latent space DM architecture builds on the U-Net by Dhariwal et al. [dhariwal2021diffusion]. Our pixel-space upsampler DMs use the same Image DM backbone [dhariwal2021diffusion]. DM sampling is performed using DDIM [song2021denoising] in all experiments.

## High-Resolution Driving Video Synthesis

We train our Video LDM pipeline, including a ```latex $4\times$ ``` pixel-space video upsampler, on the real driving scene (RDS) data. We condition on day/night labels and crowdedness, and randomly drop these labels during training to allow for classifier-free guidance and unconditional synthesis (we do not condition on bounding boxes here). Following the proposed training strategy above, we first train the image backbone LDM (spatial layers) on video frames independently, before we then train the temporal layers on videos. We also train Long Video GAN (LVG) [brooks2022generating], the previous state-of-the-art in long-term high-resolution video synthesis, on the RDS data to serve as main baseline. Our Video LDM generally outperforms LVG and adding conditioning further reduces FVD. Our human evaluation shows that our samples are generally preferred over LVG in terms of realism, and samples from our conditional model are also preferred over unconditional samples.

**Main Results (Driving, 128x256 resolution):**

| Method | FVD | FID |
|---|---|---|
| LVG | 478 | 53.5 |
| Ours | 389 | **31.6** |
| Ours (cond.) | **356** | 51.9 |

**User Study (Driving):**

| Method | Pref. A | Pref. B | Equal |
|---|---|---|---|
| Ours (cond.) vs. Ours (uncond.) | **49.33** | 42.67 | 8.0 |
| Ours (uncond.) vs. LVG | **54.02** | 40.23 | 5.74 |
| Ours (cond.) vs. LVG | **62.03** | 31.65 | 6.33 |

Next, we compare our video fine-tuned pixel-space upsampler with independent frame-wise image upsampling, using ```latex $128\times256$ ``` 30 fps ground truth videos for conditioning. We find that temporal alignment of the upsampler is crucial for high performance. FVD degrades significantly if the video frames are upsampled independently, indicating loss of temporal consistency.

**Upsampler and Decoder Ablation (RDS):**

| Method | FVD | FID |
|---|---|---|
| Ours Image Upsampler | 165.98 | **19.71** |
| Ours Video Upsampler | **45.39** | 19.85 |

| Decoder | image-only | finetuned |
|---|---|---|
| FVD | 390.88 | **32.94** |
| FID | **7.61** | 9.17 |

### Ablation Studies

To show the efficacy of our design choices, we compare a smaller version of our Video LDM with various baselines on the RDS dataset. First, using the exact same architecture as for our Video LDM, we apply our temporal finetuning strategy to a pre-trained pixel-space image diffusion model, which is clearly outperformed by ours. Further, we train an End-to-End LDM, whose entire set of parameters is learned on RDS videos without image pre-training, leading to heavy degradations both in FID and FVD. Another important architectural choice is the introduction of 3D convolutional temporal layers. This model achieves both lower FVD and FID scores than an attention-only temporal model. Finally, we see that we can further lower FVD scores by applying *context guidance* while sacrificing a bit of visual quality.

**Ablation Results:**

| Method | FVD | FID |
|---|---|---|
| Pixel-baseline | 639.56 | 59.70 |
| End-to-end LDM | 1155.10 | 71.26 |
| Attention-only | 704.41 | 50.01 |
| Ours | 534.17 | **48.26** |
| Ours (context-guided) | **508.82** | 54.16 |

### Driving Scenario Simulation

A high-resolution video generator trained on in-the-wild driving scenes can potentially serve as a powerful simulation engine. Given an initial frame, our video model can generate several different plausible future predictions. Furthermore, we also trained a separate, bounding box-conditioned image LDM on our data (only for image synthesis). A user can now manually create a scene composition of interest by specifying the bounding boxes of different cars, generate a corresponding image, and then use this image as initialization for our Video LDM, which can then predict different scenarios in a multimodal fashion.

## Text-to-Video with Stable Diffusion

Instead of first training our own Image LDM backbone, our Video LDM approach can also leverage existing Image LDMs and turn them into video generators. To demonstrate this, we turn the publicly available text-to-image LDM *Stable Diffusion* into a text-to-video generator. Specifically, using the WebVid-10M text-captioned video dataset, we train a temporally aligned version of Stable Diffusion for text-conditioned video synthesis. We briefly fine-tune Stable Diffusion's spatial layers on frames from WebVid, and then insert the temporal alignment layers and train them (at resolution ```latex $320\times512$ ```). We also add text-conditioning in those alignment layers. Moreover, we further video fine-tune the publicly available latent *Stable Diffusion upsampler*, which enables ```latex $4\times$ ``` upscaling and allows us to generate videos at resolution ```latex $1280\times 2048$ ```. We generate videos consisting of 113 frames, which we can render, for instance, into clips of 4.7 seconds length at 24 fps or into clips of 3.8 seconds length at 30 fps. While WebVid-10M consists of photo-quality real-life videos, we are able to generate highly expressive and artistic videos beyond the video training data. This demonstrates that the general image generation capabilities of the Image LDM backbone readily translate to video generation, even though the video dataset we trained on is much smaller and limited in diversity and style.

We evaluate zero-shot text-to-video generation on UCF-101 [soomro2012ucf101] and MSR-VTT [xu2016msr-vtt]. We significantly outperform all baselines except Make-A-Video [singer2022make], which we still surpass in IS on UCF-101. However, Make-A-Video is concurrent work, focuses entirely on text-to-video and trains with more video data than we do. We use only WebVid-10M; Make-A-Video also uses HD-VILA-100M [xue2022hdvila].

**UCF-101 Zero-Shot Text-to-Video Generation:**

| Method | Zero-Shot | IS | FVD |
|---|---|---|---|
| CogVideo (Chinese) | Yes | 23.55 | 751.34 |
| CogVideo (English) | Yes | 25.27 | 701.59 |
| MagicVideo | Yes | - | 699.00 |
| Make-A-Video | Yes | 33.00 | 367.23 |
| Video LDM (Ours) | Yes | **33.45** | 550.61 |

**MSR-VTT Zero-Shot Text-to-Video Generation:**

| Method | Zero-Shot | CLIPSIM |
|---|---|---|
| GODIVA | No | 0.2402 |
| NUWA | No | 0.2439 |
| CogVideo (Chinese) | Yes | 0.2614 |
| CogVideo (English) | Yes | 0.2631 |
| Make-A-Video | Yes | **0.3049** |
| Video LDM (Ours) | Yes | 0.2929 |

### Personalized Text-to-Video with DreamBooth

Since we have separate spatial and temporal layers in our Video LDM, the question arises whether the temporal layers trained on one Image LDM backbone transfer to other model checkpoints (e.g., fine-tuned). We test this for personalized text-to-video generation: Using DreamBooth [ruiz2022dreambooth], we fine-tune our Stable Diffusion spatial backbone on small sets of images of certain objects, tying their identity to a rare text token ("*sks*"). We then insert the temporal layers from the previously video-tuned Stable Diffusion (without DreamBooth) into the new DreamBooth version of the original Stable Diffusion model and generate videos using the token tied to the training images for DreamBooth. We find that we can generate personalized coherent videos that correctly capture the identity of the DreamBooth training images. This validates that our temporal layers generalize to other Image LDMs. To the best of our knowledge, we are the first to demonstrate personalized text-to-video generation.

# Conclusions

We presented *Video Latent Diffusion Models* for efficient high-resolution video generation. Our key design choice is to build on pre-trained image diffusion models and to turn them into video generators by temporally video fine-tuning them with temporal alignment layers. To maintain computational efficiency, we leverage LDMs, optionally combined with a super resolution DM, which we also temporally align. Our Video LDM can synthesize high-resolution and temporally coherent driving scene videos of many minutes. We also turn the publicly available *Stable Diffusion* text-to-image LDM into an efficient text-to-video LDM and show that the learned temporal layers transfer to different model checkpoints. We leverage this for personalized text-to-video generation. We hope that our work can benefit simulators in the context of autonomous driving research and help democratize high quality video content creation.

# Broader Impact and Limitations

Powerful video generative models, like our Video LDM, have the potential to enable important content creation applications in the future and streamline and improve the creative workflow of digital artists, thereby democratizing artistic expression. Moreover, when applied for instance on videos captured from vehicles, as in our main validation experiments, generative models like ours may also serve as simulators in autonomous driving research.

Our synthesized videos are not indistinguishable from real content yet. However, enhanced versions of our model may in the future reach an even higher quality, potentially being able to generate videos that appear to be deceptively real. This has important ethical and safety implications, as state-of-the-art deep generative models can also be used for malicious purposes, and therefore generative models like ours generally need to be applied with an abundance of caution. Moreover, the data sources cited in this paper are for research purposes only and not intended for commercial application or use, and the text-to-image backbone LDMs used in this research project have been trained on large amounts of internet data. Consequently, our model is not suitable for productization. An important direction for future work is training large-scale generative models with ethically sourced, commercially viable data.

# Related Work

Here, we present an extended discussion about related work.

**Diffusion Models for Image Synthesis.** Diffusion models (DMs) [sohl2015deep; ho2020ddpm; song2020score] have proven to be powerful image generators, yielding state-of-the art results in both unconditional and class-conditional synthesis [nichol2021improved; rombach2021highresolution; dhariwal2021diffusion] as well as text-to-image generation [nichol2021glide; rombach2021highresolution; ramesh2022dalle2; saharia2022imagen; balaji2022eDiffi]. They have also been successfully used for various image editing and processing tasks [meng2022sdedit; lugmayr2022repaint; saharia2021image; li2022srdiff; sasaki2021unitddpm; saharia2021palette; su2022dual; kawar2022restoration; hertz2022prompt; ruiz2022dreambooth; gal2022animage].

However, despite advances in model distillation [salimans2022progressive; luhman2021knowledge; meng2022distillation] and accelerated sampling [song2021denoising; jolicoeur2021gotta; dockhorn2022score; liu2022pseudo; xiao2022DDGAN; zhang2022Fast; lu2022dpm; dockhorn2022genie; watson2022learning; bao2022analyticdpm], DMs generally require repeated evaluations of a computationally demanding large U-Net. Thus, DMs are computationally expensive during both training and inference, especially when applied at high resolutions. To address this, *cascaded* [ho2021cascaded] and *latent* [vahdat2021score; rombach2021highresolution] diffusion models have been introduced. Both approaches divide the synthesis (and training) process into multiple stages and move the resource-intensive training and evaluation to a space of lower computational complexity. Cascaded diffusion models start out as low-resolution models and apply a series of super resolution diffusion models. Latent space models first compress the image data using an autoencoder and learn the DMs on the resulting latent space. We combine the best of these approaches for video synthesis. Our main video generator is a latent diffusion model. Additionally, some of our models use a video upsampler like in cascaded models to further increase the resolution.

**Video Synthesis.** Video generation has been tackled with recurrent neural networks [babaeizadeh2018stochastic; svg; lee2018savp; hvrnn; lsvg], autoregressive transformers [Weissenborn2020Scaling; yan2021videogpt; hong2022cogvideo; wu2021godiva; wu2022nuwa; ge2022longvideo; Gupta_2022_CVPR], Normalizing Flows [si2v; ipoke], and generative adversarial networks (GANs) [scene_dyn; yu2022generating; tian2021a; villegas17mcnet; Luc2020TransformationbasedAV; TGAN2020; brooks2022generating; Skorokhodov_2022_CVPR; kahembwe2020lower; TGAN2017; Wang_2020_CVPR; fox2021stylevideogan]. In particular LongVideoGAN [brooks2022generating] achieves high-resolution video synthesis over relatively long time intervals, combining a low- and a high-resolution model. Moreover, the idea to insert temporal layers into pre-trained generators has been explored by the GAN-based methods MoCoGAN-HD [tian2021a] and StyleVideoGAN [fox2021stylevideogan] before, but at a much smaller scale for simple object-centric videos. Another important work is CogVideo [hong2022cogvideo], which video fine-tunes a text-to-image model. However, it is a strictly autoregressive architecture building on transformers, trained in a discrete latent space. Our method, in contrast, relies on continuous DMs, is much less parameter intensive and outperforms CogVideo in text-to-video synthesis.

Most related to our work are previous DMs for video synthesis: Ho et al. [ho2022video] model low-resolution videos with DMs in pixel space and train jointly on image and video data. Yang et al. [yang2022video] parameterize autoregressive video generation using a deterministic frame predictor together with a probabilistic DM. Voleti et al. [voleti2022mcvd] introduce a general DM framework that simultaneously enables video generation, prediction, and interpolation. Harvey et al. [harvey2022flexible] synthesize long videos by generating sparse frames first, and then adding the missing frames. However, they model only low-resolution videos from small simulated toy worlds, whereas we train exclusively on diverse high-resolution real-world data.

**Concurrent Work.** Concurrently with us, Make-A-Video [singer2022make] leveraged a DALL-E 2-like text-to-image DM [ramesh2022dalle2] and temporally aligned its decoder as well as one super resolution DM for consistent video generation. Furthermore, Imagen Video [ho2022imagenvideo] trained a large-scale cascaded text-to-video model building on the Imagen [saharia2022imagen] architecture. It constructs a highly demanding spatial and temporal super resolution pipeline consisting of in total 7 DMs with a total of >11B parameters. In contrast to our Video LDM, both Imagen Video and Make-A-Video operate entirely in pixel space, which is less efficient, and exclusively target text-to-video, whereas we also demonstrate high-resolution driving scene generation. Furthermore, we are building on *Stable Diffusion* to train our text-to-video model, and construct an overall significantly more efficient pipeline. Phenaki [villegas2022phenaki] is another strong concurrent text-to-video model that compresses videos into discrete tokens and models the token distribution via bidirectional transformers. MagicVideo [zhou2022magicvideo] is a concurrent method that also leverages latent diffusion models for video generation.

# Using Video LDM "Convolutional in Time" and "Convolutional in Space"

An intriguing property of image LDMs is their ability to generalize to spatial resolutions much larger than the ones they are trained on. This is realized by increasing the spatial size of the sampled noise and leveraging the convolutional nature of the U-Net backbone [rombach2021highresolution]. Since we use the Stable Diffusion LDM as fixed generative image backbone for our text-to-video model, our approach naturally preserves this property, which enables us to increase the spatial resolution at inference time without significant loss of image quality. Note that our model was trained on resolution ```latex $320\times512$ ```. This convolutional application of our model for spatially extended generation essentially comes for free.

Furthermore, to extend convolutional sampling to the temporal dimension and, thus, to be able to generate videos much longer than those our model has been trained on, we make the following design choices regarding our temporal layers. First, we use relative sinusoidal positional encodings for our temporal attention layers. Second, we parameterize the learned mixing factors ```latex $\alpha_{\phi}^{i}$ ``` with scalars for our text-to-video model. These choices ensure that our model can generate longer sequences by simply increasing the number of frames. Note that when applying our model convolutionally in time, we mask the temporal attention layers such that we only attend over a maximum frame distance of 8 frames, similar to training.

We find that our Video LDM generalizes to longer sequences, although some degradation in quality can be observed. Furthermore, we can combine convolutional sampling in space and time leading to high-resolution videos of lengths up to 30 seconds, although our model has been trained only on sequences of 4 seconds. That said, convolutional-in-time generation can be fragile, in particular when targeting long videos. Hence, we advocate training of prediction models for more robust long-term generation.

# Datasets

## Real Driving Scene (RDS) Video

Our RDS dataset consists of 683,060 real driving videos of 8 seconds length each at resolution ```latex $512\times1024$ ``` (```latex $H\times W$ ```). 85,841 of these video have a frame rate of 30 fps, the rest 10 fps. Furthermore, all videos have binary night/day labels (1 for night, 0 for day) and annotations for the number of cars in a scene ("crowdedness"). Moreover, the data comes with an additional 100k independent frames that have car bounding box annotations.

## WebVid-10M

When turning Stable Diffusion into a text-to-video generator, we rely on WebVid-10M [bain21frozen]. WebVid-10M is a large-scale dataset of short videos with textual descriptions sourced from stock footage sites. The videos are diverse and rich in their content. There are 10.7M video-caption pairs with a total of 52k video hours.

## Mountain Bike

We report additional results on a first-person mountain biking video dataset, originally introduced by Brooks et al. in Long Video GAN [brooks2022generating]. The dataset consists of 1,202 clips of varying, but at least 5 seconds length. The videos have a frame rate of 30 fps. The dataset is available in different resolutions, with a maximum resolution of ```latex $576\times1024$ ```.

# Architecture, Training and Sampling Details

Our Image LDMs are based on Rombach et al. [rombach2021highresolution]. They use convolutional encoders and decoders, and their latent space DM architecture builds on the U-Net by Dhariwal et al. [dhariwal2021diffusion]. Our pixel-space upsampler DMs use the same Image DM backbone [dhariwal2021diffusion]. We generally work with discretized diffusion time steps ```latex $t \in \{0, 1000\}$ ```, and use a linear noise schedule, following the formulation of Ho et al. [ho2020ddpm].

To sample from our models, we generally use the sampler from *Denoising Diffusion Implicit Models* (DDIM) [song2021denoising].

# Quantitative Evaluation

We perform quantitative evaluations on all datasets. In particular, we compute Frechet Inception Distance (FID) and Frechet Video Distance (FVD) metrics. Since FVD can be unreliable, we additionally perform human evaluation. For text-to-video evaluation, we also compute (video) Inception Scores (IS) and CLIP Similarly scores (CLIPSIM).

**FVD:** The FVD metric measures similarity between real and generated videos [unterthiner2018towards]. We generally generate 2,048 videos (16 frames at 30 fps), except for the FVD score evaluation of our SD 2.1-based text-to-video Video LDM for the UCF-101 benchmark, where we used 10k model samples, following Make-A-Video [singer2022make]. We then extract features from a pre-trained I3D action classification model.

**FID:** To compute FID [heusel2017gans], we randomly extract 10k frames (except for mountain biking for which we extract 50k frames) from the 2,048 generated videos as well as from dataset videos. We then extract features from a pre-trained Inception model.

**Human evaluation:** We conduct human evaluation (user study) on Amazon Mechanical Turk to evaluate the realism of generated videos by our method in comparison to LVG [brooks2022generating]. For our user study, we create 100 videos, each of length 4 seconds. Each video pair was shown to four participants resulting in 400 responses per dataset.

**Inception Score:** In our text-to-video experiments on UCF-101, we also evaluated inception scores (IS) [salimans2016inceptionscore]. Following previous work on video synthesis [hong2022cogvideo; singer2022make], we used a C3D [tran2015c3d] model trained on UCF-101 to calculate a video version of the inception score.

**CLIP Similarity (CLIPSIM):** In our text-to-video experiments on MSR-VTT, we also evaluated CLIP similarity (CLIPSIM) [wu2021godiva]. The MSR-VTT test set contains 2990 examples and 20 descriptions/prompts per example. We generate 2990 videos (16 frames at 30 fps) by using one random prompt per example. We use the *ViT-B/32* [radford2021learning] model to compute the CLIP score.

# Experiment Details

## Details: High-Resolution Driving Video Synthesis

We initially train our base Image LDM (both autoencoder and latent space diffusion model) on 1 fps videos from the RDS dataset at resolution ```latex $128\times256$ ```. We then train the temporal layers for sparse key frame prediction with 2 fps.

The pixel-space ```latex $4\times$ ``` upsampler that scales the resolution to ```latex $512\times1024$ ``` is trained using the same data, but at a correspondingly higher resolution. The upsampler is trained with noise augmentation and conditioning on the noise level [ho2021cascaded; saharia2022imagen].

The temporal interpolation model is trained using 30 fps video data. We train the temporal interpolation model to first scale from 1.875 fps to 7.5 fps, and then to scale from 7.5 fps to 30 fps. We are using one interpolation model with shared parameters for that, providing a conditioning label to indicate to the model which of the two temporal upsampling operations is desired.

**Video Generation.** For video synthesis, we first generate a single frame using the image LDM, then we run the prediction model, conditioning on the single frame, to generate a sequence of key frames. When extending the video, we again call the prediction model, but condition on two frames (which captures directional information) to produce consistent motion. Next, we optionally perform two steps of the temporal interpolation, going from 1.875 to 7.5 fps and from 7.5 to 30 fps, respectively. Also optionally, the video fine-tuned upsampler is then run over portions of 8 video frames.

## Details: Text-to-Video with Stable Diffusion

We ran experiments with three of the publicly available Stable Diffusion (SD) checkpoints as image LDM backbones: 1.4, 2.0, and 2.1. Most of the research project was conducted with the SD 1.4-based model and the SD 2.0- and SD 2.1-based Video LDMs were trained primarily for exploration purposes and additional qualitative results.

Since SD is trained on images at resolution ```latex $512\times512$ ```, naively applying it to the smaller-sized videos of the WebVid-10M dataset would lead to severe degradations in image quality. We therefore first fine-tune the Stable Diffusion image backbone (spatial layers) on the WebVid-10M data. Specifically, we resize and center-crop the WebVid-10M videos to ```latex $320\times512$ ``` resolution and then fine-tune the SD latent space diffusion model on independent encoded frames from the videos.

Overall, we train our pipeline for generation of videos consisting of 113 frames. To reach high frame rates and enable smooth video generation, we again train an interpolation model that can temporally upsample videos from 1.875 fps to 7.5 fps as well as 7.5 fps to 30 fps.

**Upsampler Training:** We also video fine-tune the publicly available text-guided Stable Diffusion ```latex $4\times$ ```-upscaler, which is itself a latent diffusion model. We train the upsampler for temporal alignment in a patch-wise manner on ```latex $320\times320$ ``` cropped videos (WebVid-10M). We apply the model at extended resolution during inference, providing ```latex $320\times512$ ``` resolution videos as low resolution input, predicting ```latex $320\times512$ ``` resolution latents, and decoding to ```latex $1280\times2048$ ``` resolution videos.

**Extended Results (UCF-101):**

| Method | Zero-Shot | IS | FVD |
|---|---|---|---|
| CogVideo (Chinese) | Yes | 23.55 | 751.34 |
| CogVideo (English) | Yes | 25.27 | 701.59 |
| MagicVideo | Yes | - | 699.00 |
| Make-A-Video | Yes | 33.00 | 367.23 |
| Video LDM (SD 1.4) (Ours) | Yes | 29.49 | 656.49 |
| Video LDM (SD 2.1) (Ours) | Yes | **33.45** | 550.61 |

**Extended Results (MSR-VTT):**

| Method | Zero-Shot | CLIPSIM |
|---|---|---|
| GODIVA | No | 0.2402 |
| NUWA | No | 0.2439 |
| CogVideo (Chinese) | Yes | 0.2614 |
| CogVideo (English) | Yes | 0.2631 |
| Make-A-Video | Yes | **0.3049** |
| Video LDM (SD 1.4) (Ours) | Yes | 0.2848 |
| Video LDM (SD 2.1) (Ours) | Yes | 0.2929 |

### Number of Model Parameters

Our text-to-video LDMs that are based on Stable Diffusion have:

- 84 million parameters in the autoencoder (decoder is fine-tuned).
- 860 (SD 1.4) / 865 (SD 2.0/2.1) million parameters in the image backbone LDM (not trained).
- 649 (SD 1.4) / 656 (SD 2.0/2.1) million parameters in the temporal layers (trained).
- 123 (SD 1.4 uses CLIP ViT-L/14) / 354 (SD 2.0/2.1 uses OpenCLIP-ViT/H) million parameters in the text encoder (not trained).
- 1,509 million parameters in the interpolation latent diffusion model (trained).

Combined, for the SD-2.0/2.1-based Video LDMs this sums to around 3.1B parameters in the autoencoder and diffusion model components (excluding the CLIP text embedders). Out of this, only around 2.2B parameters are actually trained.

We see that compared to Imagen Video [ho2022imagenvideo], which has 11.6B parameters, our model is much smaller. CogVideo [hong2022cogvideo] also has much more parameters, around 9B, than our Video LDM.

## Details: Personalized Text-to-Video with DreamBooth

Using DreamBooth [ruiz2022dreambooth], we fine-tune our Stable Diffusion spatial backbone (after fine-tuning on WebVid-10M) on small sets of images of certain objects (using SD 1.4). We use 256 regularization images and train for 800 iterations using a learning rate ```latex $10^{-6}$ ```. We found it greatly beneficial to train both the U-Net as well as the CLIP text encoder. After training, we insert the temporal layers from the previously video-tuned Stable Diffusion (without DreamBooth) into the new DreamBooth version. Importantly, for video generation, the spatial layers use the DreamBooth-fine-tuned CLIP text encoder whereas the temporal layers use the standard CLIP text encoder they were trained on.

# Additional Results

## Text-to-Video

### Video-Finetuning of Decoders

Video fine-tuning the decoder allows for a significant performance boost for our text-to-video model, similar to what we observed for the driving model.

**Decoder Fine-tuning Effects:**

| Dataset | WebVid (image-only) | WebVid (fine-tuned) | Mountain Biking (image-only) | Mountain Biking (fine-tuned) |
|---|---|---|---|---|
| FVD | 35.82 | **18.66** | 73.78 | **25.55** |
| FID | 13.89 | **11.68** | 20.76 | **18.65** |

## Mountain Biking Video Synthesis

We conducted additional experiments on the Mountain Biking dataset [brooks2022generating] downsampled and center-cropped to resolution ```latex $256 \times 128$ ```. We compare our model with the publicly available model from Long Video GAN (LVG) [brooks2022generating].

**Mountain Biking Results:**

| Method | FVD | FID |
|---|---|---|
| LVG | **85.3** | 21.1 |
| Video LDM (ours) | 118 | **7.73** |

**Mountain Biking User Study:**

| Method | Pref. A | Pref. B | Equal |
|---|---|---|---|
| Video LDM (ours) vs. LVG | **54.2** | 42.2 | 3.6 |

We outperform LVG both in FID and human evaluation, but slightly underperform on FVD. The first-person mountain biking videos have very rapidly changing background details. LVG cannot create these single-frame realistic details, "smoothening out" the background and therefore resulting in worse FID. Our method has more realistic single frames but slightly struggles to keep the temporal consistency of these details. The FVD metric favors short-term "smoothness" over photorealism.

## Driving Video Synthesis

### Ablation on Additional Image Discriminator for Decoder Fine-Tuning

We found that image-level quality, as measured by FID, barely changed, while video quality, as measured by FVD, suffered considerably when an additional image discriminator was used alongside the video discriminator. Consequently, we resorted to using only the video discriminator.

| Method | Reconstruction FVD | Reconstruction FID |
|---|---|---|
| Video discriminator only | **32.94** | 9.17 |
| Additional image discriminator | 51.01 | **9.04** |

### Ablation on Image-level Quality Degradation after Temporal Video Fine-Tuning

Does the image-level quality degrade when the model is fine-tuned for video synthesis? With ```latex $\alpha_\phi = 1$ ```, we obtain 47.00 FID; with the learnt parameters, we get 48.26 FID. We observe only a tiny degradation and conclude that image-level quality is affected only slightly when training the temporal layers for video generation.
