# Abstract

Diffusion models have achieved state-of-the-art results on many modalities including images, speech, and video. However, existing models are not tailored to support remote sensing data, which is widely used in important applications including environmental monitoring and crop-yield prediction. Satellite images are significantly different from natural images -- they can be multi-spectral, irregularly sampled across time -- and existing diffusion models trained on images from the Web do not support them. Furthermore, remote sensing data is inherently spatio-temporal, requiring conditional generation tasks not supported by traditional methods based on captions or images. In this paper, we present DiffusionSat, to date the largest generative foundation model trained on a collection of publicly available large, high-resolution remote sensing datasets. As text-based captions are sparsely available for satellite images, we incorporate the associated metadata such as geolocation as conditioning information. Our method produces realistic samples and can be used to solve multiple generative tasks including temporal generation, superresolution given multi-spectral inputs and in-painting. Our method outperforms previous state-of-the-art methods for satellite image generation and is the first large-scale *generative* foundation model for satellite imagery. Project website: https://samar-khanna.github.io/DiffusionSat/

# Introduction

Diffusion models have achieved state of the art results in image generation [sohl2015deep, ho2020denoising, dhariwal2021diffusion, kingma2021variational, song2019generative, song2020improved]. Large scale models such as Stable Diffusion [stablediffusion] (SD) have been trained on Internet-scale image-text datasets to generate high-resolution images from user-provided captions. These diffusion-based foundation models, used as priors, have led to major improvements in a variety of inverse problems like inpainting, colorization, deblurring [luo2023refusion], medical image reconstruction [khader2023denoising, xie2022measurement], and video generation [videoldm].

Similarly, there are a variety of high-impact ML tasks involving the analysis of satellite images, such as disaster response, environmental monitoring, poverty prediction, crop-yield estimation, urban planning and others [gupta2019xbd, burke2021using, ayush2021efficient, ayush2020generating, jean2016combining, you2017deep, wang2018deep, russwurm2020selfatt, martinez2021fullyconvrec, semseg2019africa, yeh2021sustainbench]. These tasks consist of important inverse problems, such as super-resolution (from frequent low resolution images to high resolution ones), cloud removal, temporal in-painting and more. However, satellite images fundamentally differ from natural images in terms of perspective, resolutions, additional spectral bands, and temporal regularity. While foundation models have been recently developed for discriminative learning on satellite images [satmae, ayush2021geography, bastani2022satlas], they are not designed to and cannot solve the inverse problems (e.g., super-resolution) described above.

To fill this gap, we propose **DiffusionSat**, a generative foundation model for satellite imagery inspired from SD. Using commonly associated metadata with satellite images including latitude, longitude, timestamp, and ground-sampling distance (GSD), we train our model for single-image generation on a collection of publicly available satellite image data sets. Further, inspired from ControlNets [controlnet], we design conditioning models that can easily be trained for specific generative tasks or inverse problems including super-resolution, in-painting, and temporal generation. Specifically, our contributions include:

1. We propose a novel generative foundation model for satellite image data with the ability to generate high-resolution satellite imagery from numerical metadata as well as text.
2. We design a novel 3D-conditioning extension which enables DiffusionSat to demonstrate state-of-the-art performance on super-resolution, temporal generation, and in-painting.
3. We collect and compile a global generative pre-training dataset from large, publicly available satellite image datasets.

[IMAGE: Conditioning on freely available metadata and using large, publicly available satellite imagery datasets shows DiffusionSat is a powerful generative foundation model for remote sensing data.]

# Background

**Diffusion Models.** Diffusion models are generative models that aim to learn a data distribution $p_{\text{Data}}$ from samples [sohl2015deep, ho2020denoising, song2019generative, song2020score, song2020improved]. Given an input image $x \sim p_{\text{Data}}$, we add noise to create a *noisy* input $x_t = \alpha_t x + \sigma_t \epsilon$, where $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ is Gaussian noise. $\alpha_t$ and $\sigma_t$ denote a noise schedule parameterized by diffusion time $t$ (higher $t$ leads to more added noise). The diffusion model $\epsilon_{\theta}$ then aims to *denoise* $x_t$, and is optimized using the score-matching objective:

$$\mathbb{E}_{x \sim p_{\text{Data}}, \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})}\left[ \| y - \epsilon_{\theta}(x_t ; t, c) \|_2^2 \right]$$

where the target $y$ can be the input noise $\epsilon$, the input image $x$ or the "velocity" $v = \alpha_t \epsilon - \sigma_t x$. We can additionally condition the denoising model with side information $\mathbf{c} \in \mathbb{R}^D$, which can be a class embedding, text, or other images etc.

Latent diffusion models (LDMs) [vahdat2021score_latent, sinha2021d2c, stablediffusion] first downsample the input $x$ using a VAE with an encoder $\mathcal{E}$ and a decoder $\mathcal{D}$, such that $\tilde{x} = \mathcal{D}(\mathcal{E}(x))$ is a reconstructed image. Instead of denoising the input image $x$, the diffusion process is used on a downsampled latent representation $z = \mathcal{E}(x)$. This approach reduces computational and memory cost and has formed the basis for the popularly used StableDiffusion (SD) model [stablediffusion].

# Method

First, we describe our method for the following tasks of interest: single-image generation, conditioned on text and metadata, multi-spectral superresolution, temporal prediction, and temporal inpainting.

## Single Image Generation

Our first goal is to pre-train DiffusionSat to be able to generate *single* images given an input text prompt and/or metadata. Concretely, we begin by considering datasets such that each image $\mathbf{x} \in \mathbb{R}^{C \times H \times W}$ is paired with an associated text-caption $\tau$. Our goal is to learn the conditional data distribution $p(\mathbf{x} | \tau)$ such that we can sample new images $\tilde{\mathbf{x}} \sim p(\cdot | \tau)$.

LDMs are popularly used for text-to-image generation primarily for their strong ability to use text prompts. An associated text prompt $\tau$ is tokenized, encoded via CLIP [clip], and then passed to the DM $\epsilon_\theta(\mathbf{x}_t ; t, \tau)$ via cross-attention [vaswani2017attention] at each layer. However, while text prompts are widely available for image datasets such as LAION-5B [schuhmann2022laion], satellite images typically either do not have such captions, or are accompanied by object-detection boxes, segmentation masks, or classification labels. Moreover, requiring such labels precludes the use of vast amounts of unlabeled satellite imagery. Ideally, we would like to pretrain DiffusionSat on existing labelled and unlabelled datasets, without necessarily curating expensive labels.

To solve this challenge, we note that satellite images are commonly associated with metadata including their timestamp, latitude, longitude, and various other numerical information that are correlated with the image [fmow]. We thus consider datasets where each image $\mathbf{x} \in \mathbb{R}^{C \times H \times W}$ is paired with a text-caption $\tau$, as well as cheaply available numerical metadata $\mathbf{k} \in \mathbb{R}^M$, where $M$ is the number of metadata items. We thus wish to learn the data distribution $p(\mathbf{x} | \tau, \mathbf{k})$. With good enough metadata $\mathbf{k}$, we want to still sample an image of high quality even if $\tau$ is poor or missing.

We now turn to conditioning on $\mathbf{k}$. One option is to naively incorporate each numerical metadata item $k_j$, $j \in \{1, \dots, M\}$, into the text caption with a short description. However, this approach unnecessarily discretizes continuous-valued covariates and can suffer from text-encoders' known shortcomings related to encoding numerical information [clip]. Instead, we choose to encode the metadata using the same sinusoidal timestep embedding used in diffusion models:

$$\texttt{Project}(k, 2i) = \sin\left(k \Omega^{-\frac{2i}{d}}\right), \quad \texttt{Project}(k, 2i+1) = \cos\left(k \Omega^{-\frac{2i}{d}}\right)$$

where $k$ is the metadata or timestep value, $i$ is the index of feature dimension in the encoding, $d$ is the dimension, and $\Omega = 10000$ is a large constant. Each metadata value $k_j$ is first normalized to a value between 0 and 1000 (since the diffusion timestep $t \in \{0, \dots, 1000\}$), and is then projected via the sinusoidal encoding. A different MLP for each metadatum encodes the projected metadata value identically to the diffusion timestep $t$ [ho2020denoising] as follows:

$$f_{\theta_j}(k_j) = \texttt{MLP}\left(\left[\texttt{Project}(k_j, 0), \dots, \texttt{Project}(k_j, d)\right]\right)$$

where $f_{\theta_j}$ represents the learned MLP embedding for metadata value $k_j$, corresponding to metadata type $j$ (e.g., longitude). Our embedding is then $f_{\theta_j}(k_j) \in \mathbb{R}^D$, where $D$ is the embedding dimension. The $M$ metadata vectors are then added together $\mathbf{m} = f_{\theta_1}(k_1) + \dots + f_{\theta_M}(k_M)$, where $\mathbf{m} \in \mathbb{R}^D$, which is then also added with the embedded timestep $\mathbf{t} = f_{\theta}(t) \in \mathbb{R}^D$, so that the final conditioning vector is $\mathbf{c} = \mathbf{m} + \mathbf{t}$.

To summarize, we first encode an image $\mathbf{x} \in \mathbb{R}^{C \times H \times W}$ using the SD variational autoencoder (VAE) [stablediffusion, vqgan] to a latent representation $\mathbf{z} = \mathcal{E}(\mathbf{x}) \in \mathbb{R}^{C' \times H' \times W'}$. Gaussian noise is then added to the latent image features to give us $\mathbf{z}_t = \alpha_t \mathbf{z} + \sigma_t \epsilon$. The conditioning vector $\mathbf{c}$, created from embedding metadata and the diffusion timestep, as well as the CLIP-embedded text caption $\tau' = \mathcal{T}_\theta(\tau)$, are passed through a DM $\epsilon_{\theta}(\mathbf{z}_t ; \tau', \mathbf{c})$ to predict the added noise. Finally, the VAE decoder $\mathcal{D}$ upsamples the denoised latents to full resolution.

Lastly, we initialize the encoder $\mathcal{E}$, the decoder $\mathcal{D}$, the CLIP text encoder $\mathcal{T}_\theta$, and the denoising UNet $\epsilon_\theta$ with SD 2.1's weights. We only update the denoising UNet $\epsilon_\theta$, the metadata and the timestep embeddings $f_{\theta_j}$ during training to speed up convergence using the rich semantic information in the pretrained SD weights. During training, we also randomly zero out the metadata vector $\mathbf{m}$ with a probability of 0.1 to allow the model to generate images when metadata might be unavailable or inaccurate. A similar strategy is employed to learn unconditional generation by [ho2020denoising].

**Single Image-Text-Metadata Datasets.** There is no equivalent of a large, text-image dataset (e.g., LAION [schuhmann2022laion]) for satellite images. Instead, we compile publicly available annotated satellite data and contribute a large, high-resolution generative dataset for satellite images. Detailed descriptions on how the caption is generated for each dataset are in the appendix.

- **fMoW**: Function Map of the World (fMoW) [fmow] consists of global, high-resolution (GSD ~0.3m-1.5m) DigitalGlobe satellite images, each belonging to one of 62 categories. We crop each image to 512x512 pixels. The metadata we consider include longitude, latitude, GSD (in meters), cloud cover (as a fraction), year, month, and day. To generate a caption, we consider the semantic class and the country code.
- **Satlas**: Satlas [bastani2022satlas] is a large-scale, multi-task dataset of NAIP and Sentinel-2 satellite images. For our dataset, we use the NAIP imagery in Satlas-small, roughly of the same size as fMoW. We use the same metadata as fMoW.
- **SpaceNet**: Spacenet [van2018spacenet, van2021spacenet7] is a collection of satellite image datasets for tasks including object detection, semantic segmentation and road network mapping. We consider a subset of Spacenet datasets, namely Spacenet v1, Spacenet v2, and Spacenet v5. We use the same metadata as earlier.

[IMAGE: DiffusionSat flexibly extends to a variety of conditional generation tasks. We design a 3D version of a ControlNet which can accept a sequence of images. Like regular ControlNets, our 3D ControlNet keeps a trainable copy of SD weights for the downsampling and middle blocks. Latent image features are reshaped to combine the batch and temporal dimensions before being input to these layers. The output of each SD block is then passed through a temporal layer (top right), which re-expands the temporal dimension before passing the latent features through a 3D convolution (initialized with zeros) and a temporal, pixel-wise transformer. The metadata associated with each input image is projected as in the single-image generation figure.]

## Control Signal Conditional Generation

Single-image DiffusionSat can generate a high-resolution satellite image given input prompt and metadata, but it cannot yet solve the inverse problems described above. To leverage its pretrained weights, we can use it as a prior for conditional generation tasks which do encompass inverse problems such as super-resolution and in-painting. Thus, we now consider generative tasks where we can additionally condition on control signals (e.g., sequences of satellite images) $\mathbf{s} \in \mathbb{R}^{T \times C' \times H' \times W'}$, with associated metadata $\mathbf{k}_s \in \mathbb{R}^{T \times M}$, a single caption $\tau$ and target metadata $\mathbf{k} \in \mathbb{R}^M$. Here, $C'$, $H'$, and $W'$ reflect the possible difference in the number of channels, height, and width, respectively, between the conditioning images and the target image. The goal is to sample $\tilde{\mathbf{x}} \sim p(\cdot | \mathbf{s}; \mathbf{k}_s; \tau; \mathbf{k})$, where $\tilde{\mathbf{x}}$ is a sample conditioned on the control signal $\mathbf{s}$ for a given caption $\tau$ and given metadata $\mathbf{k}$.

**Temporal Generation.** Recent works for video diffusion have proposed using 3D convolutions and temporal attention [videoldm, tune_a_video, zhou2022magicvideo], while others propose using existing 2D UNets and concatenating temporal frames in the channel dimension [masked_conditional_video_diffusion, latent_shift]. However, sequences of satellite images differ in a few key ways from frames of images in a video:

1. There is high variance in the length of time separating images in the sequence, while frames in video data are usually separated by a fixed amount of time (fixed frame rate).
2. The length of time between images can be on the order of months or years, therefore capturing a wider range of semantic information than consecutively placed frames in video (e.g., season, human development, land cover).
3. There is a sense of "global time" across locations. Even if one compares satellite images across different countries or terrains, patterns may be similar if the year is known to be 2012 as opposed to 2020 (especially for urban landscapes). This is not the case for video data, where "local" time across frames is sufficient to provide semantic meaning.

Usually, sequences of satellite images have fewer images than frames in natural image videos. As such, generating long sequences of images is less useful than conditioning on existing satellite imagery to predict the future or interpolate in the past [he2021spatial, satlas_superres]. Thus, we introduce our novel conditioning framework to solve the inverse problem of frame-by-frame conditional temporal prediction. Unlike 2D ControlNet, we use 3D zero-convolutions between each StableDiffusion block [controlnet]. Our temporal attention layers, similar to VideoLDM [videoldm], further enable the model to condition on temporal control signals. We introduce a learned parameter $\alpha_i$ for each block $i$ to "mix" in the output of the temporal attention layer to prevent noise from early stages in training from affecting our pre-trained weights.

A key advantage of our approach is the ability to provide each item in the control sequence $\mathbf{s}$ with its own associated metadata. This is done similarly to single-image generation: we project each metadatum individually and embed it with an MLP. The embedded metadata for each image is then concatenated with its image and passed through the 2D layers of the ControlNet. DiffusionSat is thus invariant to the ordering of images in the control sequence $\mathbf{s}$, since the timestamp in the metadata of each image solely determines its temporal position. A single DiffusionSat model can then be trained to predict images in the past and future, or interpolate within the temporal range of the sequence.

**Super-resolution with multi-spectral input.** Unlike in temporal generation, our input is a sequence $\mathbf{s}$ of lower resolution (GSD) images than the target image and can contain a differing number of channels. The output of the model is still a high-resolution RGB image, as before.

**Temporal Inpainting.** The task is functionally equivalent to temporal generation, except the goal is to in-paint corrupted pixels (e.g., from cloud cover, flooding, fire-damage) rather than *predict* a new frame in $\mathbf{s}$.

# Experiments

[IMAGE: Here we generate samples from single-image DiffusionSat. We see that changing the coordinates from a location in Paris to one in USA changes the type of stadium generated, with American football and baseball more likely to appear in the latter location. Additionally, for locations that receive snow, DiffusionSat accurately captures the correlation between location and season. However, naively incorporating the metadata into the text caption results in poorer conditioning flexibility across geography and season (e.g., with winter and summer time images produced for both August and January, or a lack of "zooming in" when lowering the GSD).]

We describe the experiments for the tasks described in Method. Implementation details are in the Appendix.

For single image generation, we report standard visual-quality metrics such as FID [fid], Inception Score (IS), and CLIP-score [clip]. For conditional generation, given a reference ground-truth image, we report pixel-quality metrics including SSIM [ssim], PSNR, LPIPS [lpips] with VGG [vgg] features. As noted in [gong2021enlighten] and [he2021spatial], LPIPS is a more relevant perceptual quality metric used in evaluating satellite images. Our metrics are reported on a sample size of 10,000 images.

## Single Image Generation

We first consider single-image generation, as the task that DiffusionSat is pre-trained on. We compare against a pre-trained SD 2.1 model [stablediffusion], a SD 2.1 model finetuned on our dataset with our captions, but without metadata, and finally a SD 2.1 model finetuned on our dataset with the metadata included in the caption. We find that including the metadata, even within the caption, is better than a caption formed from just the labels of satellite images. This is reflected in better FID scores, which measure visual quality. We expect that the text-metadata model does better in terms of CLIP score given its more highly descriptive caption. However, treating metadata numerically, as in DiffusionSat, further improves generation quality and control.

**Table: Single-image 512x512 generation on the validation set of fMoW.**

| Method | FID (down) | IS (up) | CLIP (up) |
|---|---|---|---|
| SD 2.1 | 117.74 | 6.42 | 17.23 |
| SD 2.1 (finetuned, no metadata) | 37.99 | 7.42 | 16.59 |
| SD 2.1 (metadata in caption) | 24.23 | **7.60** | **18.62** |
| DiffusionSat (Ours) | **15.80** | 6.69 | 17.20 |

**Table: Image sample quality on fMoW superresolution.**

| Method | SSIM (up) | PSNR (up) | LPIPS (down) | MSE (down) |
|---|---|---|---|---|
| Pix2Pix | 0.1374 | 8.2722 | 0.6895 | 0.1492 |
| DBPN | 0.1518 | **11.8568** | 0.6826 | **0.0680** |
| SD | 0.1671 | 10.2417 | 0.6403 | 0.0962 |
| SD + CN | 0.1626 | 10.0098 | 0.6506 | 0.1009 |
| DiffusionSat (Ours) | **0.1703** | 10.3924 | **0.6221** | 0.0928 |

## Control Signal Conditional Generation

We now use single-image DiffusionSat as an effective prior for the conditional generation tasks of super-resolution, temporal generation/prediction, and in-painting. We describe the dataset for each task and demonstrate results using our 3D conditioning approach on Texas-housing super-resolution, fMoW super-resolution using fMoW-Sentinel multispectral inputs, temporal generation on the fMoW-temporal dataset, and temporal inpainting on the xBD natural disaster dataset. DiffusionSat achieves state of the art LPIPS and close to optimal performance on the SSIM and PSNR metrics as well.

**fMoW Superresolution.** Using the dataset provided in [satmae], we create a fMoW-Sentinel-fMoW-RGB dataset with paired Sentinel-2 (10m-60m GSD) and fMoW (0.3-1.5m GSD) images at each of the original fMoW-RGB locations. Given all 13 multi-spectral bands of the Sentinel-2 image (here $T=1$), we aim to reconstruct the corresponding high resolution RGB image. Super-resolution given low-resolution (10m-60m), multi-spectral input is especially difficult, since most fMoW-RGB images are <1m GSD. We find that DiffusionSat once again outperforms strong super-resolution baselines, such as SD. We further note that while methods such as DBPN [dbpn_superres] yield strong PSNR/SSIM, these metrics don't reflect human perception and favor blurriness over sharp detail [lpips, sr3].

[IMAGE: Generated samples from fMoW-Sentinel superresolution validation set. The conditioning image is the Sentinel-2 multispectral (MS) image represented here as SWIR, NIR, RGB. The desired output is the high-resolution (HR) fMoW-RGB image. Our method is able to capture fine-grained details better than other baselines, even when the low-resolution MS image lacks detail. SD tends to "hallucinate" details.]

**Texas Housing Superresolution.** The dataset for this task is introduced by Spatial Temporal Superresolution (STSR) [he2021spatial] and contains 286717 houses built between 2014 and 2017 in Texas. Each location consists of two high-resolution images from NAIP (GSD 1m) and 2 low-resolution images from Sentinel-2 (GSD 10m). A high resolution image at a time $t$ and corresponding low-resolution images at times $t$ and $t'$ form the control signal $\mathbf{s}$, and the task is to reconstruct the other high resolution image $\mathbf{x}$ at time $t'$.

We also perform an ablation on the efficacy of pretraining on our single image datasets against finetuning directly on SD weights. We find a significant improvement from DiffusionSat pretraining, and from using the 3D ControlNet (across all metrics) over simply stacking the images in the channel dimension and using a 2D ControlNet.

**Table: Sample quality results on Texas housing validation data.** $t' > t$ represents generating an image in the past given a future HR image, and $t' < t$ is the task for generating a future image given a past HR image.

| Model | SSIM (up) t'>t | PSNR (up) t'>t | LPIPS (down) t'>t | SSIM (up) t'<t | PSNR (up) t'<t | LPIPS (down) t'<t |
|---|---|---|---|---|---|---|
| Pix2Pix | 0.5432 | 20.8420 | 0.4243 | 0.3909 | 17.9528 | 0.4909 |
| cGAN Fusion | 0.5976 | 21.5226 | 0.3936 | 0.4220 | 17.8763 | 0.4726 |
| DBPN | 0.5781 | 21.4716 | 0.5101 | 0.4572 | 18.9330 | 0.5910 |
| SRGAN | 0.5361 | 21.1968 | 0.5261 | 0.4221 | 18.9772 | 0.5694 |
| STSR (EAD) | 0.6470 | 22.4906 | 0.3695 | 0.5225 | 19.7675 | 0.4275 |
| STSR (EA64) | **0.6570** | **22.5552** | 0.3764 | **0.5338** | **19.8547** | 0.4342 |
| SD + 3D ControlNet | 0.4747 | 17.8023 | 0.4166 | 0.3458 | 16.1467 | 0.4351 |
| Ours + ControlNet | 0.5403 | 20.3982 | 0.3874 | 0.4657 | 18.1007 | 0.3652 |
| Ours + 3D ControlNet | 0.5982 | 21.0299 | **0.3247** | 0.4825 | 18.4604 | **0.3534** |

**fMoW Temporal Generation.** Many locations in fMoW [fmow] contain multiple images across time. For our experiments, if $T < 4$, we add copies of the latest image to pad the sequence $\mathbf{s}$ to 4 images. Given a sequence $\mathbf{s}$ of conditioning images, DiffusionSat can predict another image at any desired target time by appropriately adjusting the target metadata $\mathbf{k}_s$. Since prior works aren't designed to predict an image at any given target time, we consider tasks where the target image is chronologically prior to or later than the first image in $\mathbf{s}$.

**Table: Sample quality results on fMoW-temporal validation data.** $t' > t$ represents generating an image in the past given a future image, and $t' < t$ is the task for generating a future image given a past image.

| Model | SSIM (up) t'>t | PSNR (up) t'>t | LPIPS (down) t'>t | SSIM (up) t'<t | PSNR (up) t'<t | LPIPS (down) t'<t |
|---|---|---|---|---|---|---|
| STSR (EAD) | 0.3657 | 13.5191 | 0.4898 | 0.3654 | 13.7425 | 0.4940 |
| MCVD | 0.3110 | 9.6330 | 0.6058 | 0.2721 | 9.5559 | 0.6124 |
| SD + 3D CN | 0.2027 | 11.0536 | 0.5523 | 0.2218 | 11.3094 | 0.5342 |
| DiffusionSat + CN | 0.3297 | 13.6938 | 0.5062 | 0.2862 | 12.4990 | 0.5307 |
| DiffusionSat + 3D CN | **0.3983** | **13.7886** | **0.4304** | **0.4293** | **14.8699** | **0.3937** |

Our experiments show that DiffusionSat outperforms STSR and MCVD [masked_conditional_video_diffusion], as well as regular SD with our 3D ControlNet. These reveal DiffusionSat's improved ability over the baselines to capture the target date's season (e.g., snow, terrain color, crop maturity) as well as development of roads and buildings. Other models, lacking the ability to reason about metadata covariates, often simply copy an input image in the conditioning sequence as their generated output.

[IMAGE: Generated samples from the fMoW-temporal validation set, for temporal prediction. The 4 columns in the center are ground-truth images from the temporal sequence. DiffusionSat leverages pretrained weights to capture seasonal changes and predict human development better than the baselines.]

**In-painting.** Rather than artificially corrupt input images, we use the xBD dataset [gupta2019xbd] which is a subset of the xView-2 [lam2018xview] challenge to assess damage caused by natural disasters. Since each location carries a pre- and post-disaster satellite image, we consider the in-painting task of reconstructing damaged areas in the post-disaster image, or introducing destruction to the pre-disaster image, where $T=1$. DiffusionSat's capability to reconstruct damaged roads and houses for a variety of disasters including floods, wind, fire, earthquakes etc will be important for disaster response teams to identify access routes and assess damage. We also show that DiffusionSat can *add* damage from different natural disasters, which can be useful for forecasting or preparing areas for evacuation.

[IMAGE: Inpainting results. The two columns marked "before" and "after" represent ground truth images. DiffusionSat successfully reconstructs damaged roads and houses from floods, fires, and wind, even when large portions of the conditioning image are masked by clouds or damage.]

# Related Work

**Diffusion Models.** Diffusion models [ho2020denoising, song2020score, kingma2021variational] have recently dominated the field of generative modeling, including application areas such as speech [kong2020diffwave, popov2021grad], 3D geometry [xu2022geodiff, luo2021diffusion, zhou20213d], and graphics [chan2023generative, poole2022dreamfusion, shue20233d]. Besides advancements in the theoretical foundation, large-scale variants built on latent space [stablediffusion, saharia2022photorealistic, ho2022imagen] have arguably been the most influential. With these foundation models came a slew of novel applications such as subject customization [ruiz2023dreambooth, liu2023cones, kumari2023multi] and text-to-3D generation [poole2022dreamfusion, lin2023magic3d, wang2023prolificdreamer]. Many works have also demonstrated these models' impressive adaptation capabilities through finetuning. For example, ControlNet [controlnet], T2IAdapter [mou2023t2i], and InstructPix2Pix [brooks2023instructpix2pix], which add additional trainable parameters, have proven to be highly successful in adding control signals to the pre-trained diffusion networks.

**Generative Models for Remote Sensing.** Image super-resolution is well studied for natural image datasets [srcnn, srgan, sr3, dbpn_superres, stablediffusion]. Generative Adversarial Networks (GANs) [goodfellow2014generative] such as SR-GAN [srgan] are among the most popular remote-sensing super-resolution methods [wang2020ultra, ma2019super, gong2021enlighten, worldstrat, satlas_superres, eesrgan]. Other methods have tailor-made convolutional architectures for Sentinel-2 image-input superresolution [razzak2023multi, tarasiewicz2023multitemporal]. More recently, Spatial-Temporal Super Resolution (STSR) [he2021spatial] uses a conditional-pixel synthesis approach to condition on a combination of high and low resolution images to generate a high-resolution image at an earlier or later date. In general, these models lack the flexibility and generality of latent-diffusion models across a variety of tasks and datasets, and can suffer from unstable training [kodali2017convergence]. Our work aims to address these shortcomings by proposing a single approach based on pretrained LDMs that can flexibly translate to downstream generative tasks via our novel conditioning mechanism.

# Conclusion

In this work, we provide DiffusionSat, the first *generative* foundation model for remote sensing data based on the latent-diffusion model architecture of StableDiffusion [stablediffusion]. Our approach consists of two components: (i) a single-image generation model that can generate high-resolution satellite data conditioned on numerical metadata and text captions, and (ii) a novel 3D control signal conditioning module that generalizes to inverse problems such as multi-spectral input super-resolution, temporal prediction, and in-painting.

For future work, we would like to explore expanding DiffusionSat to even larger and more diverse satellite imagery datasets. Testing the feasibility of DiffusionSat on generating synthetic data [le2023mask] might also augment existing discriminative methods to scale to larger datasets. Another relevant area of future research is reducing variance in the generated samples from DiffusionSat, which can sometimes hallucinate details when producing outputs for inverse problems. Lastly, investigating faster sampling methods or more efficient architectures will enable easier deployment or use of DiffusionSat in resource-constrained settings.

We hope that DiffusionSat spurs future investigation into solving inverse problems posed by remote-sensing data. Doing so would unlock societal benefits to important applications including object detection given super-resolved Sentinel-2 images [shermeyer2019effects], crop-phenotyping [zhang2020high], ecological conservation efforts [boyle2014high, johansen2007application], natural disaster hazard assessment [nichol2006application], archaeological prospection [beck2007evaluation], urban planning [li2019generating, xiao2006evaluating, piyoosh2017semi], and precise agricultural applications [gevaert2015generation].

# Appendix

## Training Details

All models are trained on half-precision and with gradient checkpointing, borrowing from the Diffusers [von-platen-etal-2022-diffusers] library.

**Single-Image DiffusionSat.** We use 8 NVIDIA A100 GPUs. The text-to-image models are trained with a batch size of 128 for 100000 iterations, which we determined was sufficient for convergence. We choose a constant learning rate of 2e-6 with the AdamW optimizer. We train two variants -- one for images of resolution 512x512 pixels, and one for 256x256 pixels. For sampling, we use the DDIM [song2020denoising] sampler with 100 steps and a guidance scale of 1.0. We generate 10000 samples on the validation sets of fMoW-RGB.

**Super-resolution.** We use the 512 single-image DiffusionSat model as our prior. We train our ControlNet [controlnet] by upsampling the conditional multi-spectral image to 256x256 pixels, which we found to work better than conditioning on 64x64 conditioning images. We use 4 NVIDIA A100 GPUs, and train the model for 50000 iterations with a learning rate of 5e-5 using the AdamW optimizer. We drop Sentinel bands B1, B9, B10, which we find to not be useful, similar to [satmae]. We use the same sampling configuration as above.

**Texas Housing.** We use the 256 single-image DiffusionSat model as our prior. We train our 3D ControlNet on sequences of the HR image and the two LR Sentinel-2 images. We use 4 NVIDIA A100 GPUs, and train the model for 50000 iterations with a learning rate of 5e-5 using the AdamW optimizer. The sampling configuration is the same as above.

**fMoW Temporal.** We use the 256 single-image DiffusionSat model as our prior. We train our 3D ControlNet on sequences of at-most 3 conditioning images on the fMoW-temporal dataset. If the location has less than 3 images, we pick one of the conditioning images and copy it over until the sequence is padded to length. We avoid samples where there is only 1 image per location. We train using 4 NVIDIA A100 GPUs, for 40000 iterations with a learning rate of 4e-4 using the AdamW optimizer. The sampling configuration matches the ones above.

## Datasets

### Captions and metadata

The text captions are dependent on the metadata fields available for each dataset. Fields denoted in angle brackets are filled in using the metadata for each example. Some sections of each caption, denoted by square brackets, are randomly and independently dropped out of caption instances at a 10% rate. Datasets from the same satellite sources are labelled with the same image type (e.g., both Texas Housing and Satlas use NAIP images, so both are labelled as "satlas" images).

| Dataset | Caption |
|---|---|
| fMoW | `"a [fmow] satellite image [of a <object>] [in <country>]"` |
| SpaceNet | `"a [spacenet] satellite image [of <object>] [in <city>]"` |
| Satlas | `"a [satlas] satellite image [of <object>]"` |
| Texas Housing | `"a [satlas] satellite image [of houses] [built in <year_built>] [covering <num_acres> acres]"` |
| xBD | `"a [fmow] satellite image [<before/after>] being affected by a <disaster_type> natural disaster"` |

Besides the captions, we also incorporate numerical metadata from 7 fields. Each field was normalized based on high and low reference values: $m_{norm} = m / (high - low) \times scale$, where $scale$ is a scaling factor of 1000, such that $low$ maps to 0 and $high$ maps to $scale$.

| Field | Description | Min | Max |
|---|---|---|---|
| `lon` | longitude, in degrees | -180 | 180 |
| `lat` | latitude, in degrees | -90 | 90 |
| `gsd` | ground sampling distance | 0 | 10 |
| `cloud_cover` | proportion of pixels with cloud cover | 0 | 1 |
| `year` | year of the satellite image | 1980 | 2100 |
| `month` | month of the year | 0 | 12 |
| `day` | day of the month | 0 | 31 |

## Temporal Generation (Appendix)

### Sequence Generation

We first demonstrate how we can generate temporal sequences of satellite images *unconditionally* (i.e., without any prior conditioning image). To do so, we first generate a satellite image using single-image DiffusionSat given a caption and desired metadata for our image. We then apply our novel 3D-conditioning ControlNet, already trained for temporal generation, on the first image to generate the next image in the sequence, given some desired metadata (e.g., how many years/months/days into the future or past). We now re-apply the 3D-conditioning ControlNet on the first 2 generated images to get the third image of the sequence. Repeating this procedure, we are able to auto-regressively sample sequences of satellite images given desired metadata properties.

[IMAGE: Auto-regressively generated sequences of satellite images given a caption (for the sequence) and desired metadata (per image). The model generates realistic sequences that reflect trends in detail and development both forwards and backwards in time.]

Our results demonstrate a novel way of generating arbitrarily long sequences of satellite images -- our conditioning mechanism can flexibly handle both conditional and unconditional generation. The generated samples reflect season and trends in development (e.g., past images have fewer structures, future images usually have more detail).

## Geographical Bias

Concerns about bias for the outputs of machine learning models are natural given the large, potentially biased datasets they are trained on [huang2021sensing]. We perform an evaluation of the generation quality of single-image DiffusionSat across latitude and longitude around the globe.

[IMAGE: FID scores of single-image DiffusionSat prompted on 10k samples of the fMoW-RGB validation set for coordinates around the world.]

[IMAGE: LPIPS scores of super-resolution DiffusionSat prompted on 10k samples of the fMoW-Sentinel-fMoW-RGB validation set for coordinates around the world.]

Our results show no particular favoritism for location, even though one would expect better generation quality for regions in North America and Europe. We would still like to point out a few caveats:

1. FID or LPIPS scores may not be the most informative metric towards estimating bias in sample quality. We use it as a measure of generation quality given a lack of better alternatives for the novel problem of estimating geographical bias in generative remote sensing models.
2. The FID scores are dependent on sample size, and so while the scores might be evenly distributed, it still remains the case that there are far more dataset samples from developed regions of the world, and a dearth of images for large swaths (e.g., across Africa). Even so, for a severely biased model we would expect poorer generation quality for data-poor regions of the world.
3. We estimate only one angle of bias. Bias may still exist along different axes, such as generating types of buildings, roads, trees, crops, and understanding the effects of season. We leave this investigation to future work.
