# Abstract

Generating temporally coherent high fidelity video is an important milestone in generative modeling research. We make progress towards this milestone by proposing a diffusion model for video generation that shows very promising initial results. Our model is a natural extension of the standard image diffusion architecture, and it enables jointly training from image and video data, which we find to reduce the variance of minibatch gradients and speed up optimization. To generate long and higher resolution videos we introduce a new conditional sampling technique for spatial and temporal video extension that performs better than previously proposed methods. We present the first results on a large text-conditioned video generation task, as well as state-of-the-art results on established benchmarks for video prediction and unconditional video generation. Supplementary material is available at https://video-diffusion.github.io/.

# Introduction

Diffusion models have recently been producing high quality results in image generation and audio generation [e.g. kingma2021variational; saharia2021palette; saharia2021image; dhariwal2021diffusion; ho2021cascaded; nichol2021glide; song2020score; whang2021deblurring; salimans2021progressive; chen2020wavegrad; kong2021diffwave], and there is significant interest in validating diffusion models in new data modalities. In this work, we present first results on video generation using diffusion models, for both unconditional and conditional settings.

We show that high quality videos can be generated using essentially the standard formulation of the Gaussian diffusion model [sohl2015deep], with little modification other than straightforward architectural changes to accommodate video data within the memory constraints of deep learning accelerators. We train models that generate a fixed number of video frames using a 3D U-Net diffusion model architecture, and we enable generating longer videos by applying this model autoregressively using a new method for conditional generation. We additionally show the benefits of joint training on video and image modeling objectives. We test our methods on video prediction and unconditional video generation, where we achieve state-of-the-art sample quality scores, and we also show promising first results on text-conditioned video generation.

# Background

A diffusion model [sohl2015deep; song2019generative; ho2020denoising] specified in continuous time [tzen2019neural; song2020score; chen2020wavegrad; kingma2021variational] is a generative model with latents ```latex $\mathbf{z}= \{\mathbf{z}_t \,|\, t \in [0,1]\}$ ``` obeying a forward process ```latex $q(\mathbf{z}|\mathbf{x})$ ``` starting at data ```latex $\mathbf{x}\sim p(\mathbf{x})$ ```. The forward process is a Gaussian process that satisfies the Markovian structure:

```latex
$$q(\mathbf{z}_t|\mathbf{x}) = \mathcal{N}(\mathbf{z}_t; \alpha_t \mathbf{x}, \sigma_t^2 \mathbf{I}), \quad q(\mathbf{z}_t | \mathbf{z}_s) = \mathcal{N}(\mathbf{z}_t; (\alpha_t/\alpha_s)\mathbf{z}_s, \sigma_{t|s}^2\mathbf{I})$$
```

where ```latex $0 \leq s < t \leq 1$ ```, ```latex $\sigma^2_{t|s} = (1-e^{\lambda_t-\lambda_s})\sigma_t^2$ ```, and ```latex $\alpha_t, \sigma_t$ ``` specify a differentiable noise schedule whose log signal-to-noise-ratio ```latex $\lambda_t = \log[\alpha_t^2/\sigma_t^2]$ ``` decreases with ```latex $t$ ``` until ```latex $q(\mathbf{z}_1) \approx \mathcal{N}(\mathbf{0}, \mathbf{I})$ ```.

#### Training

Learning to reverse the forward process for generation can be reduced to learning to denoise ```latex $\mathbf{z}_t\sim q(\mathbf{z}_t|\mathbf{x})$ ``` into an estimate ```latex $\hat\mathbf{x}_\theta(\mathbf{z}_t, \lambda_t) \approx \mathbf{x}$ ``` for all ```latex $t$ ``` (we will drop the dependence on ```latex $\lambda_t$ ``` to simplify notation). We train this denoising model ```latex $\hat\mathbf{x}_\theta$ ``` using a weighted mean squared error loss

```latex
$$\mathbb{E}_{{\boldsymbol{\epsilon}},t}\!\left[w(\lambda_t) \|\hat\mathbf{x}_\theta(\mathbf{z}_t) - \mathbf{x}\|^2_2\right]$$
```

over uniformly sampled times ```latex $t \in [0,1]$ ```. This reduction of generation to denoising can be justified as optimizing a weighted variational lower bound on the data log likelihood under the diffusion model, or as a form of denoising score matching [vincent2011connection; song2019generative; ho2020denoising; kingma2021variational]. In practice, we use the ```latex ${\boldsymbol{\epsilon}}$ ```-prediction parameterization, defined as ```latex $\hat\mathbf{x}_\theta(\mathbf{z}_t) = (\mathbf{z}_t - \sigma_t {\boldsymbol{\epsilon}}_\theta(\mathbf{z}_t))/\alpha_t$ ```, and train ```latex ${\boldsymbol{\epsilon}}_\theta$ ``` using a mean squared error in ```latex ${\boldsymbol{\epsilon}}$ ``` space with ```latex $t$ ``` sampled according to a cosine schedule [nichol2021improvedddpm]. This corresponds to a particular weighting ```latex $w(\lambda_t)$ ``` for learning a scaled score estimate ```latex ${\boldsymbol{\epsilon}}_\theta(\mathbf{z}_t) \approx -\sigma_t \nabla_{\mathbf{z}_t}\log p(\mathbf{z}_t)$ ```, where ```latex $p(\mathbf{z}_t)$ ``` is the true density of ```latex $\mathbf{z}_t$ ``` under ```latex $\mathbf{x}\sim p(\mathbf{x})$ ``` [ho2020denoising; kingma2021variational; song2020score]. We also train using the ```latex $\mathbf{v}$ ```-prediction parameterization for certain models [salimans2021progressive].

#### Sampling

We use a variety of diffusion model samplers in this work. One is the discrete time ancestral sampler [ho2020denoising] with sampling variances derived from lower and upper bounds on reverse process entropy [sohl2015deep; ho2020denoising; nichol2021improvedddpm]. To define this sampler, first note that the forward process can be described in reverse as ```latex $q(\mathbf{z}_s|\mathbf{z}_t,\mathbf{x}) = \mathcal{N}(\mathbf{z}_s; \tilde{\boldsymbol{\mu}}_{s|t}(\mathbf{z}_t,\mathbf{x}), \tilde\sigma^2_{s|t}\mathbf{I})$ ``` (noting ```latex $s < t$ ```), where

```latex
$$\tilde{\boldsymbol{\mu}}_{s|t}(\mathbf{z}_t,\mathbf{x}) = e^{\lambda_t-\lambda_s}(\alpha_s/\alpha_t)\mathbf{z}_t + (1-e^{\lambda_t-\lambda_s})\alpha_s\mathbf{x}\quad \text{and} \quad \tilde\sigma^2_{s|t} = (1-e^{\lambda_t-\lambda_s})\sigma_s^2.$$
```

Starting at ```latex $\mathbf{z}_1 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ ```, the ancestral sampler follows the rule

```latex
$$\mathbf{z}_s = \tilde{\boldsymbol{\mu}}_{s|t}(\mathbf{z}_t, \hat\mathbf{x}_\theta(\mathbf{z}_t)) + \sqrt{(\tilde\sigma^2_{s|t})^{1-\gamma} (\sigma^2_{t|s})^\gamma}{\boldsymbol{\epsilon}}$$
```

where ```latex ${\boldsymbol{\epsilon}}$ ``` is standard Gaussian noise, ```latex $\gamma$ ``` is a hyperparameter that controls the stochasticity of the sampler [nichol2021improvedddpm], and ```latex $s,t$ ``` follow a uniformly spaced sequence from 1 to 0.

Another sampler, which we found especially effective with our new method for conditional generation (Section 3.1), is the predictor-corrector sampler [song2020score]. Our version of this sampler alternates between the ancestral sampler step and a Langevin correction step of the form

```latex
$$\mathbf{z}_s \leftarrow \mathbf{z}_s - \frac{1}{2}\delta \sigma_s {\boldsymbol{\epsilon}}_\theta(\mathbf{z}_s) + \sqrt{\delta}\sigma_s{\boldsymbol{\epsilon}}'$$
```

where ```latex $\delta$ ``` is a step size which we fix to ```latex $0.1$ ``` here, and ```latex ${\boldsymbol{\epsilon}}'$ ``` is another independent sample of standard Gaussian noise. The purpose of the Langevin step is to help the marginal distribution of each ```latex $\mathbf{z}_s$ ``` generated by the sampler to match the true marginal under the forward process starting at ```latex $\mathbf{x}\sim p(\mathbf{x})$ ```.

In the conditional generation setting, the data ```latex $\mathbf{x}$ ``` is equipped with a conditioning signal ```latex $\mathbf{c}$ ```, which may represent a class label, text caption, or other type of conditioning. To train a diffusion model to fit ```latex $p(\mathbf{x}|\mathbf{c})$ ```, the only modification that needs to be made is to provide ```latex $\mathbf{c}$ ``` to the model as ```latex $\hat\mathbf{x}_\theta(\mathbf{z}_t, \mathbf{c})$ ```. Improvements to sample quality can be obtained in this setting by using *classifier-free guidance* [ho2021classifier]. This method samples using adjusted model predictions ```latex $\tilde{{\boldsymbol{\epsilon}}}_\theta$ ```, constructed via

```latex
$$\tilde{{\boldsymbol{\epsilon}}}_\theta(\mathbf{z}_t, \mathbf{c}) = (1+w){\boldsymbol{\epsilon}}_\theta(\mathbf{z}_t, \mathbf{c}) - w{\boldsymbol{\epsilon}}_{\theta}(\mathbf{z}_t),$$
```

where ```latex $w$ ``` is the *guidance strength*, ```latex ${\boldsymbol{\epsilon}}_\theta(\mathbf{z}_t, \mathbf{c}) = \frac{1}{\sigma_t}(\mathbf{z}_t - \hat\mathbf{x}_\theta(\mathbf{z}_t, \mathbf{c}))$ ``` is the regular conditional model prediction, and ```latex ${\boldsymbol{\epsilon}}_{\theta}(\mathbf{z}_t)$ ``` is a prediction from an unconditional model jointly trained with the conditional model (if ```latex $\mathbf{c}$ ``` consists of embedding vectors, unconditional modeling can be represented as ```latex $\mathbf{c}=\mathbf{0}$ ```). For ```latex $w > 0$ ``` this adjustment has the effect of over-emphasizing the effect of conditioning on the signal ```latex $\mathbf{c}$ ```, which tends to produce samples of lower diversity but higher quality compared to sampling from the regular conditional model [ho2021classifier]. The method can be interpreted as a way to guide the samples towards areas where an implicit classifier ```latex $p(\mathbf{c}| \mathbf{z}_t)$ ``` has high likelihood, and is an adaptation of the explicit classifier guidance method proposed by [dhariwal2021diffusion].

# Video diffusion models

Our approach to video generation using diffusion models is to use the standard diffusion model formalism described in Section 2 with a neural network architecture suitable for video data. Each of our models is trained to jointly model a fixed number of frames at a fixed spatial resolution. To extend sampling to longer sequences of frames or higher spatial resolutions, we will repurpose our models with a conditioning technique described later in Section 3.1.

In prior work on image modeling, the standard architecture for ```latex $\hat\mathbf{x}_\theta$ ``` in an image diffusion model is a U-Net [ronneberger2015unet; salimans2017pixelcnn++], which is a neural network architecture constructed as a spatial downsampling pass followed by a spatial upsampling pass with skip connections to the downsampling pass activations. The network is built from layers of 2D convolutional residual blocks, for example in the style of the Wide ResNet [zagoruyko2016wide], and each such convolutional block is followed by a spatial attention block [vaswani2017attention; wang2018non; chen2018pixelsnail]. Conditioning information, such as ```latex $\mathbf{c}$ ``` and ```latex $\lambda_t$ ```, is provided to the network in the form of an embedding vector added into each residual block (we find it helpful for our models to process these embedding vectors using several MLP layers before adding).

We propose to extend this image diffusion model architecture to video data, given by a block of a fixed number of frames, using a particular type of 3D U-Net [cciccek20163d] that is factorized over space and time. First, we modify the image model architecture by changing each 2D convolution into a space-only 3D convolution, for instance, we change each 3x3 convolution into a 1x3x3 convolution (the first axis indexes video frames, the second and third index the spatial height and width). The attention in each spatial attention block remains as attention over space; i.e., the first axis is treated as a batch axis. Second, after each spatial attention block, we insert a temporal attention block that performs attention over the first axis and treats the spatial axes as batch axes. We use relative position embeddings [shaw2018self] in each temporal attention block so that the network can distinguish ordering of frames in a way that does not require an absolute notion of video time. We visualize the model architecture in Figure 1.

[IMAGE: The 3D U-Net architecture for the diffusion model. Each block represents a 4D tensor with axes labeled as frames x height x width x channels, processed in a space-time factorized manner. The input is a noisy video z_t, conditioning c, and the log SNR lambda_t. The downsampling/upsampling blocks adjust the spatial input resolution height x width by a factor of 2 through each of the K blocks. The channel counts are specified using channel multipliers M_1, M_2, ..., M_K, and the upsampling pass has concatenation skip connections to the downsampling pass.]

The use of factorized space-time attention is known to be a good choice in video transformers for its computational efficiency [arnab2021vivit; bertasius2021space; ho2019axial]. An advantage of our factorized space-time architecture, which is unique to our video generation setting, is that it is particularly straightforward to mask the model to run on independent images rather than a video, simply by removing the attention operation inside each time attention block and fixing the attention matrix to exactly match each key and query vector at each video timestep. The utility of doing so is that it allows us to jointly train the model on both video and image generation. We find in our experiments that this joint training is important for sample quality (Section 4).

## Reconstruction-guided sampling for improved conditional generation

The videos we consider modeling typically consist of hundreds to thousands of frames, at a frame rate of at least 24 frames per second. To manage the computational requirements of training our models, we only train on a small subset of say 16 frames at a time. However, at test time we can generate longer videos by extending our samples. For example, we could first generate a video ```latex $\mathbf{x}^{\text{a}} \sim p_{\theta}(\mathbf{x})$ ``` consisting of 16 frames, and then extend it with a second sample ```latex $\mathbf{x}^{\text{b}} \sim p_{\theta}(\mathbf{x}^{\text{b}} | \mathbf{x}^{\text{a}})$ ```. If ```latex $\mathbf{x}^{\text{b}}$ ``` consists of frames following ```latex $\mathbf{x}^{\text{a}}$ ```, this allows us to autoregressively extend our sampled videos to arbitrary lengths, which we demonstrate in Section 4.3.3. Alternatively, we could choose ```latex $\mathbf{x}^{\text{a}}$ ``` to represent a video of lower frame rate, and then define ```latex $\mathbf{x}^{\text{b}}$ ``` to be those frames in between the frames of ```latex $\mathbf{x}^{\text{a}}$ ```. This allows one to then to upsample a video temporally, similar to how [menick2018generating] generate high resolution images through spatial upsampling.

Both approaches require one to sample from a conditional model, ```latex $p_{\theta}(\mathbf{x}^{\text{b}} | \mathbf{x}^{\text{a}})$ ```. This conditional model could be trained explicitly, but it can also be derived approximately from our unconditional model ```latex $p_{\theta}(\mathbf{x})$ ``` by imputation, which has the advantage of not requiring a separately trained model. For example, [song2020score] present a general method for conditional sampling from a jointly trained diffusion model ```latex $p_{\theta}(\mathbf{x}=[\mathbf{x}^{\text{a}}, \mathbf{x}^{\text{b}}])$ ```: In their approach to sampling from ```latex $p_{\theta}(\mathbf{x}^{\text{b}} | \mathbf{x}^{\text{a}})$ ```, the sampling procedure for updating ```latex $\mathbf{z}^{\text{b}}_s$ ``` is unchanged from the standard method for sampling from ```latex $p_{\theta}(\mathbf{z}_s | \mathbf{z}_t)$ ```, with ```latex $\mathbf{z}_s=[\mathbf{z}^{\text{a}}_s, \mathbf{z}^{\text{b}}_s]$ ```, but the samples for ```latex $\mathbf{z}^{\text{a}}_s$ ``` are replaced by exact samples from the forward process, ```latex $q(\mathbf{z}^{\text{a}}_s | \mathbf{x}^{\text{a}})$ ```, at each iteration. The samples ```latex $\mathbf{z}^{\text{a}}_s$ ``` then have the correct marginal distribution by construction, and the samples ```latex $\mathbf{z}^{\text{b}}_s$ ``` will conform with ```latex $\mathbf{z}^{\text{a}}_s$ ``` through their effect on the denoising model ```latex $\hat\mathbf{x}_\theta([\mathbf{z}^{\text{a}}_t, \mathbf{z}^{\text{b}}_t])$ ```. Similarly, we could sample ```latex $\mathbf{z}^{\text{a}}_s$ ``` from ```latex $q(\mathbf{z}^{\text{a}}_s | \mathbf{x}^{\text{a}}, \mathbf{z}^{\text{a}}_t)$ ```, which follows the correct conditional distribution in addition to the correct marginal. We will refer to both of these approaches as the *replacement* method for conditional sampling from diffusion models.

When we tried the replacement method to conditional sampling, we found it to not work well for our video models: Although samples ```latex $\mathbf{x}^{\text{b}}$ ``` looked good in isolation, they were often not coherent with ```latex $\mathbf{x}^{\text{a}}$ ```. This is caused by a fundamental problem with this replacement sampling method. That is, the latents ```latex $\mathbf{z}^{\text{b}}_s$ ``` are updated in the direction provided by ```latex $\hat\mathbf{x}^{\text{b}}_\theta(\mathbf{z}_t) \approx \mathbb{E}_{q}[\mathbf{x}^{b} | \mathbf{z}_t]$ ```, while what is needed instead is ```latex $\mathbb{E}_{q}[\mathbf{x}^{b} | \mathbf{z}_t, \mathbf{x}^{a}]$ ```. Writing this in terms of the score of the data distribution, we get ```latex $\mathbb{E}_{q}[\mathbf{x}^{b} | \mathbf{z}_t, \mathbf{x}^{a}] = \mathbb{E}_{q}[\mathbf{x}^{b} | \mathbf{z}_t] + (\sigma^{2}_t / \alpha_t)\nabla_{\mathbf{z}^{b}_t}\log q(\mathbf{x}^{a} | \mathbf{z}_t)$ ```, where the second term is missing in the replacement method. Assuming a perfect denoising model, plugging in this missing term would make conditional sampling exact. Since ```latex $q(\mathbf{x}^{a} | \mathbf{z}_t)$ ``` is not available in closed form, however, we instead propose to approximate it using a Gaussian of the form ```latex $q(\mathbf{x}^{a} | \mathbf{z}_t) \approx \mathcal{N}[\hat\mathbf{x}^{\text{a}}_\theta(\mathbf{z}_t), (\sigma^{2}_t/\alpha^{2}_t)\text{I}]$ ```, where ```latex $\hat\mathbf{x}^{\text{a}}_\theta(\mathbf{z}_t)$ ``` is a reconstruction of the conditioning data ```latex $\mathbf{x}^{\text{a}}$ ``` provided by our denoising model. Assuming a perfect model, this approximation becomes exact as ```latex $t \rightarrow 0$ ```, and empirically we find it to be good for larger ```latex $t$ ``` also. Plugging in the approximation, and adding a weighting factor ```latex $w_{r}$ ```, our proposed method to conditional sampling is a variant of the replacement method with an adjusted denoising model, ```latex $\tilde\mathbf{x}^{b}_\theta$ ```, defined by

```latex
$$\tilde\mathbf{x}^{b}_\theta(\mathbf{z}_t) = \hat\mathbf{x}^{b}_\theta(\mathbf{z}_t) - \frac{w_{r} \alpha_t}{2}\nabla_{\mathbf{z}^{b}_t} \lVert \mathbf{x}^{a} - \hat\mathbf{x}^{a}_\theta(\mathbf{z}_t) \rVert_{2}^{2}$$
```

The additional gradient term in this expression can be interpreted as a form of *guidance* [dhariwal2021diffusion; ho2021classifier] based on the model's reconstruction of the conditioning data, and we therefore refer to this method as *reconstruction-guided sampling*, or simply *reconstruction guidance*. Like with other forms of guidance, we find that choosing a larger weighting factor, ```latex $w_{r} > 1$ ```, tends to improve sample quality. We empirically investigate reconstruction guidance in Section 4.3.3, where we find it to work surprisingly well, especially when combined with predictor-corrector samplers using Langevin diffusion [song2020score].

Reconstruction guidance also extends to the case of spatial interpolation (or super-resolution), in which the mean squared error loss is imposed on a downsampled version of the model prediction, and backpropagation is performed through this downsampling. In this setting, we have low resolution ground truth videos ```latex $\mathbf{x}^{a}$ ``` (e.g. at the 64x64 spatial resolution), which may be generated from a low resolution model, and we wish to upsample them into high resolution videos (e.g. at the 128x128 spatial resolution) using an unconditional high resolution diffusion model ```latex $\hat\mathbf{x}_\theta$ ```. To accomplish this, we adjust the high resolution model as follows:

```latex
$$\tilde\mathbf{x}_\theta(\mathbf{z}_t) = \hat\mathbf{x}_\theta(\mathbf{z}_t) - \frac{w_{r}\alpha_t}{2}\nabla_{\mathbf{z}_t} \lVert \mathbf{x}^{a} - \hat\mathbf{x}^{a}_\theta(\mathbf{z}_t) \rVert_{2}^{2}$$
```

where ```latex $\hat\mathbf{x}^{a}_\theta(\mathbf{z}_t)$ ``` is our model's reconstruction of the low-resolution video from ```latex $\mathbf{z}_t$ ```, which is obtained by downsampling the high resolution output of the model using a differentiable downsampling algorithm such as bilinear interpolation. Note that it is also possible to simultaneously condition on low resolution videos while autoregressively extending samples at the high resolution using the same reconstruction guidance method. In Figure 2, we show samples of this approach for extending 16x64x64 low resolution samples at frameskip 4 to 64x128x128 samples at frameskip 1 using a 9x128x128 diffusion model.

# Experiments

We report our results on video diffusion models for unconditional video generation (Section 4.1), conditional video generation (video prediction) (Section 4.2), and text-conditioned video generation (Section 4.3). We evaluate our models using standard metrics such as FVD [unterthiner2018towards], FID [heusel2017gans], and IS [salimans2016improved]; details on evaluation are provided below alongside each benchmark. Samples and additional results are provided at https://video-diffusion.github.io/. Architecture hyperparameters, training details, and compute resources are listed in Appendix A.

## Unconditional video modeling

To demonstrate our approach on unconditional generation, we use a popular benchmark of [soomro2012ucf101] for unconditional modeling of video. The benchmark consists of short clips of people performing one of 101 activities, and was originally collected for the purpose of training action recognition models. We model short segments of 16 frames from this dataset, downsampled to a spatial resolution of 64x64. In Table 1 we present perceptual quality scores for videos generated by our model, and we compare against methods from the literature, finding that our method strongly improves upon the previous state-of-the-art.

We use the data loader provided by TensorFlow Datasets [TFDS] without further processing, and we train on all 13,320 videos. Similar to previous methods, we use the C3D network [tran2015learning] for calculating FID and IS, using 10,000 samples generated from our model. C3D internally resizes input data to the 112x112 spatial resolution, so perceptual scores are approximately comparable even when the data is sampled at a different resolution originally. As discussed by [yushchenko2019markov], methods in the literature are unfortunately not always consistent in the data preprocessing that is used, which may lead to small differences in reported scores between papers. The Inception Score we calculate for real data (approximately 60) is consistent with that reported by [kahembwe2020lower], who also report a higher real data Inception score of approximately 90 for data sampled at the 128x128 resolution, which indicates that our 64x64 model might be at a disadvantage compared to works that generate at a higher resolution. Nevertheless, our model obtains the best perceptual quality metrics that we could find in the literature.

**Table 1: Unconditional video modeling results on UCF101.**

| Method | Resolution | FID (lower is better) | IS (higher is better) |
|---|---|---|---|
| MoCoGAN [tulyakov2018mocogan] | 16x64x64 | 26998 +/- 33 | 12.42 |
| TGAN-F [kahembwe2020lower] | 16x64x64 | 8942.63 +/- 3.72 | 13.62 |
| TGAN-ODE [gordon2021latent] | 16x64x64 | 26512 +/- 27 | 15.2 |
| TGAN-F [kahembwe2020lower] | 16x128x128 | 7817 +/- 10 | 22.91 +/- .19 |
| VideoGPT [yan2021videogpt] | 16x128x128 | | 24.69 +/- 0.30 |
| TGAN-v2 [saito2020train] | 16x64x64 | 3431 +/- 19 | 26.60 +/- 0.47 |
| TGAN-v2 [saito2020train] | 16x128x128 | 3497 +/- 26 | 28.87 +/- 0.47 |
| DVD-GAN [clark2019adversarial] | 16x128x128 | | 32.97 +/- 1.7 |
| **Video Diffusion (ours)** | **16x64x64** | **295 +/- 3** | **57 +/- 0.62** |
| real data | 16x64x64 | | 60.2 |

## Video prediction

A common benchmark task for evaluating generative models of video is *video prediction*, where the model is given the first frame(s) of a video and is asked to generate the remainder. Models that do well on this *conditional generation* task are usually trained explicitly for this conditional setting, for example by being autoregressive across frames. Although our models are instead only trained unconditionally, we can adapt them to the video prediction setting by using the guidance method proposed in Section 3.1. Here we evaluate this method on two popular video prediction benchmarks, obtaining state-of-the-art results.

#### BAIR Robot Pushing

We evaluate video prediction performance on BAIR Robot Pushing [ebert2017self], a standard benchmark in the video literature consisting of approximately 44000 videos of robot pushing motions at the 64x64 spatial resolution. Methods for this benchmark are conditioned on 1 frame and generate the next 15. Following the evaluation protocol of [babaeizadeh2021fitvid] and others, we calculate FVD [unterthiner2018towards] using the I3D network [carreira2017quo] by comparing 100 x 256 model samples against the 256 examples in the evaluation set.

#### Kinetics-600

We additionally evaluate video prediction performance on the Kinetics-600 benchmark [kay2017kinetics; carreira2018short]. Kinetics-600 contains approximately 400 thousand training videos depicting 600 different activities. We train unconditional models on this dataset at the 64 x 64 resolution and evaluate on 50 thousand randomly sampled videos from the test set, where we condition on a randomly sampled subsequence of 5 frames and generate the next 11 frames. Like previous works, we calculate FVD and Inception Score using the I3D network [carreira2017quo]. See Table 3 for results. In our reported results we sample test videos without replacement, and we use the same randomly selected subsequences for generating model samples and for defining the ground truth, since this results in the lowest bias and variance in the reported FVD metric. However, from personal communication we learned that [luc2020transformation; clark2019adversarial] instead sampled *with replacement*, and used a different random seed when sampling the ground truth data. We find that this way of evaluating raises the FVD obtained by our model slightly, from 16.2 to 16.9. Inception Score is unaffected.

**Table 2: Video prediction on Kinetics-600 (FVD only).**

| Method | FVD (lower is better) |
|---|---|
| DVD-GAN [clark2019adversarial] | 109.8 |
| VideoGPT [yan2021videogpt] | 103.3 |
| TrIVD-GAN-FP [luc2020transformation] | 103.3 |
| Transframer [nash2022transframer] | 100 |
| CCVS [le2021ccvs] | 99 |
| VideoTransformer [weissenborn2019scaling] | 94 |
| FitVid [babaeizadeh2021fitvid] | 93.6 |
| NUWA [wu2021n] | 86.9 |
| Video Diffusion (ours), ancestral sampler, 512 steps | 68.19 |
| Video Diffusion (ours), Langevin sampler, 256 steps | **66.92** |

**Table 3: Video prediction on Kinetics-600 (FVD and IS).**

| Method | FVD (lower is better) | IS (higher is better) |
|---|---|---|
| Video Transformer [weissenborn2019scaling] | 170 +/- 5 | |
| DVD-GAN-FP [clark2019adversarial] | 69.1 +/- 0.78 | |
| Video VQ-VAE [walker2021predicting] | 64.3 +/- 2.04 | |
| CCVS [le2021ccvs] | 55 +/- 1 | |
| TrIVD-GAN-FP [luc2020transformation] | 25.74 +/- 0.66 | 12.54 |
| Transframer [nash2022transframer] | 25.4 | |
| Video Diffusion (ours), ancestral, 256 steps | 18.6 | 15.39 |
| Video Diffusion (ours), Langevin, 128 steps | **16.2 +/- 0.34** | **15.64** |

## Text-conditioned video generation

The remaining experiments reported are on text-conditioned video generation. In this text-conditioned video generation setting, we employ a dataset of 10 million captioned videos, and we condition the diffusion model on captions in the form of BERT-large embeddings [devlin2019bert] processed using attention pooling. We consider two model sizes: a small model for the joint training ablation, and a large model for generating the remaining results (both architectures are described in detail in Appendix A), and we explore the effects of joint video-image training, classifier-free guidance, and our newly proposed reconstruction guidance method for autoregressive extension and simultaneous spatial and temporal super-resolution. We report the following metrics in this section on 4096 samples: the video metric FVD, and the Inception-based image metrics FID and IS measured by averaging activations across frames (FID/IS-avg) and by measuring the first frame only (FID/IS-first). For FID and FVD, we report two numbers which are measured against the training and validation sets, respectively. For IS, we report two numbers which are averaged scores across 1 split and 10 splits of samples, respectively.

[IMAGE: Text-conditioned video samples from a cascade of two models. First samples are generated from a 16x64x64 frameskip 4 model. Then those samples are treated as ground truth for simultaneous super-resolution and autoregressive extension to 64x128x128 using a 9x128x128 frameskip 1 model. Both models are conditioned on the text prompt.]

### Joint training on video and image modeling

As described in Section 3, one of the main advantages of our video architecture is that it allows us to easily train the model jointly on video and image generative modeling objectives. To implement this joint training, we concatenate random independent image frames to the end of each video sampled from the dataset, and we mask the attention in the temporal attention blocks to prevent mixing information across video frames and each individual image frame. We choose these random independent images from random videos within the same dataset; in future work we plan to explore the effect of choosing images from other larger image-only datasets.

Table 4 reports results for an experiment on text-conditioned 16x64x64 videos, where we consider training on an additional 0, 4, or 8 independent image frames per video. One can see clear improvements in video and image sample quality metrics as more independent image frames are added. Adding independent image frames has the effect of reducing variance of the gradient at the expense of some bias for the video modeling objective, and thus it can be seen as a memory optimization to fit more independent examples in a batch.

**Table 4: Improved sample quality due to image-video joint training on text-to-video generation.**

| Image frames | FVD (lower is better) | FID-avg (lower is better) | IS-avg (higher is better) | FID-first (lower is better) | IS-first (higher is better) |
|---|---|---|---|---|---|
| 0 | 202.28/205.42 | 37.52/37.40 | 7.91/7.58 | 41.14/40.87 | 9.23/8.74 |
| 4 | 68.11/70.74 | 18.62/18.42 | 9.02/8.53 | 22.54/22.19 | 10.58/9.91 |
| 8 | 57.84/60.72 | 15.57/15.44 | 9.32/8.82 | 19.25/18.98 | 10.81/10.12 |

### Effect of classifier-free guidance

Table 5 reports results that verify the effectiveness of classifier-free guidance [ho2021classifier] on text-to-video generation. As expected, there is clear improvement in the Inception Score-like metrics with higher guidance weight, while the FID-like metrics improve and then degrade with increasing guidance weight. Similar findings have been reported on text-to-image generation [nichol2021glide].

Figure 3 shows the effect of classifier-free guidance [ho2021classifier] on a text-conditioned video model. Similar to what was observed in other work that used classifier-free guidance on text-conditioned image generation [nichol2021glide] and class-conditioned image generation [ho2021classifier; dhariwal2021diffusion], adding guidance increases the sample fidelity of each individual image and emphasizes the effect of the conditioning signal.

[IMAGE: Example frames from a random selection of videos generated by the 16x64x64 text-conditioned model. Left: unguided samples, right: guided samples using classifier-free guidance.]

**Table 5: Effect of classifier-free guidance on text-to-video generation (large models). Sample quality is reported for 16x64x64 models trained on frameskip 1 and 4 data. The model was jointly trained on 8 independent image frames per 16-frame video.**

| Frameskip | Guidance weight | FVD (lower is better) | FID-avg (lower is better) | IS-avg (higher is better) | FID-first (lower is better) | IS-first (higher is better) |
|---|---|---|---|---|---|---|
| 1 | 1.0 | 41.65/43.70 | 12.49/12.39 | 10.80/10.07 | 16.42/16.19 | 12.17/11.22 |
| | 2.0 | 50.19/48.79 | 10.53/10.47 | 13.22/12.10 | 13.91/13.75 | 14.81/13.46 |
| | 5.0 | 163.74/160.21 | 13.54/13.52 | 14.80/13.46 | 17.07/16.95 | 16.40/14.75 |
| 4 | 1.0 | 56.71/60.30 | 11.03/10.93 | 9.40/8.90 | 16.21/15.96 | 11.39/10.61 |
| | 2.0 | 54.28/51.95 | 9.39/9.36 | 11.53/10.75 | 14.21/14.04 | 13.81/12.63 |
| | 5.0 | 185.89/176.82 | 11.82/11.78 | 13.73/12.59 | 16.59/16.44 | 16.24/14.62 |

### Autoregressive video extension for longer sequences

In Section 3.1 we proposed the *reconstruction guidance method* for conditional sampling from diffusion models, an improvement over the *replacement method* of [song2020score]. In Table 6 we present results on generating longer videos using both techniques, and find that our proposed method indeed improves over the replacement method in terms of perceptual quality scores.

Figure 4 shows the samples of our reconstruction guidance method for conditional sampling compared to the replacement method (Section 3.1) for the purposes of generating long samples in a block-autoregressive manner. The samples from the replacement method clearly show a lack of temporal coherence, since frames from different blocks throughout the generated videos appear to be uncorrelated samples. The samples from the reconstruction guidance method, by contrast, are clearly temporally coherent over the course of the entire autoregressive generation process. Figure 2 additionally shows samples of using the reconstruction guidance method to simultaneously condition on low frequency, low resolution videos while autoregressively extending temporally at a high resolution.

[IMAGE: Comparing the replacement method (left) vs the reconstruction guidance method (right) for conditioning for block-autoregressive generation of 64 frames from a 16 frame model. Video frames are displayed over time from left to right; each row is an independent sample. The replacement method suffers from a lack of temporal coherence, unlike the reconstruction guidance method.]

**Table 6: Generating 64x64x64 videos using autoregressive extension of 16x64x64 models.**

| Guidance weight | Conditioning method | FVD (lower is better) | FID-avg (lower is better) | IS-avg (higher is better) | FID-first (lower is better) | IS-first (higher is better) |
|---|---|---|---|---|---|---|
| 2.0 | reconstruction guidance | 136.22/134.55 | 13.77/13.62 | 10.30/9.66 | 16.34/16.46 | 14.67/13.37 |
| | replacement | 451.45/436.16 | 25.95/25.52 | 7.00/6.75 | 16.33/16.46 | 14.67/13.34 |
| 5.0 | reconstruction guidance | 133.92/133.04 | 13.59/13.58 | 10.31/9.65 | 16.28/16.53 | 15.09/13.72 |
| | replacement | 456.24/441.93 | 26.05/25.69 | 7.04/6.78 | 16.30/16.54 | 15.11/13.69 |

# Related work

Prior work on video generation has usually employed other types of generative models, notably, autoregressive models, VAEs, GANs, and normalizing flows [e.g. babaeizadeh2017stochastic; babaeizadeh2021fitvid; lee2018stochastic; kumar2019videoflow; clark2019adversarial; weissenborn2019scaling; yan2021videogpt; walker2021predicting]. Related work on model classes similar to diffusion models includes [kadkhodaie2020solving; kadkhodaie2021stochastic]. Concurrent work [yang2022diffusion] proposes a diffusion-based approach to video generation that uses an image diffusion model to predict each individual frame within a RNN temporal autoregressive model. Our video diffusion model, by contrast, jointly models entire videos (blocks of frames) using a 3D video architecture with interleaved spatial and temporal attention, and we extend to long sequence lengths by filling in frames or autoregressive temporal extension.

# Conclusion

We have introduced diffusion models for video modeling, thus bringing recent advances in generative modeling using diffusion models to the video domain. We have shown that with straightforward extensions of conventional U-Net architectures for 2D image modeling to 3D space-time, with factorized space-time attention blocks, one can learn effective generative models for video data using the standard formulation of the diffusion model. This includes unconditional models, text-conditioned models, and video prediction models.

We have additionally demonstrated the benefits of joint image-video training and classifier-free guidance for video diffusion models on both video and image sample quality metrics, and we also introduced a new reconstruction-guided conditional sampling method that outperforms existing replacement or imputation methods for conditional sampling from unconditionally trained models. Our reconstruction guidance method can generate long sequences using either frame interpolation (or temporal super-resolution) or extrapolation in an auto-regressive fashion, and also can perform spatial super-resolution. We look forward to investigating this method in a wider variety of conditioning settings.

Our goal with this work is to advance research on methods in generative modeling, and our methods have the potential to positively impact creative downstream applications. As with prior work in generative modeling, however, our methods have the potential for causing harmful impact and could enhance malicious or unethical uses of generative models, such as fake content generation, harassment, and misinformation spread, and thus we have decided not to release our models. Like all generative models, our models reflect the biases of their training datasets and thus may require curation to ensure fair results from sampling. In particular, our text-to-video models inherit the challenges faced by prior work on text-to-image models, and our future work will involve auditing for forms of social bias. We see our work as only a starting point for further investigation on video diffusion models and investigation into their societal implications, and we will aim to explore benchmark evaluations for social and cultural bias in the video generation setting and make the necessary research advances to address them.

# Details and hyperparameters

[IMAGE: More samples accompanying Figure 2.]

Here, we list the hyperparameters, training details, and compute resources used for each model.

## UCF101

| Parameter | Value |
|---|---|
| Base channels | 256 |
| Channel multipliers | 1, 2, 4, 8 |
| Blocks per resolution | 2 |
| Attention resolutions | 8, 16, 32 |
| Attention head dimension | 64 |
| Conditioning embedding dimension | 1024 |
| Conditioning embedding MLP layers | 4 |
| Diffusion noise schedule | cosine |
| Noise schedule log SNR range | [-20, 20] |
| Video resolution | 16x64x64 frameskip 1 |
| Weight decay | 0.0 |
| Optimizer | Adam (beta_1=0.9, beta_2=0.99) |
| Learning rate | 0.0003 |
| Batch size | 128 |
| EMA | 0.9999 |
| Dropout | 0.1 |
| Training hardware | 128 TPU-v4 chips |
| Training steps | 60000 |
| Joint training independent images per video | 8 |
| Sampling timesteps | 256 |
| Sampling log-variance interpolation | gamma=0.1 |
| Prediction target | epsilon |

## BAIR Robot Pushing

| Parameter | Value |
|---|---|
| Base channels | 128 |
| Channel multipliers | 1, 2, 3, 4 |
| Blocks per resolution | 3 |
| Attention resolutions | 8, 16, 32 |
| Attention head dimension | 64 |
| Conditioning embedding dimension | 1024 |
| Conditioning embedding MLP layers | 2 |
| Diffusion noise schedule | cosine |
| Noise schedule log SNR range | [-20, 20] |
| Video resolution | 16x64x64 frameskip 1 |
| Weight decay | 0.01 |
| Reconstruction guidance weight | 50 |
| Optimizer | Adam (beta_1=0.9, beta_2=0.999) |
| Learning rate | 0.0002 |
| Batch size | 128 |
| EMA | 0.999 |
| Dropout | 0.1 |
| Training hardware | 128 TPU-v4 chips |
| Training steps | 660000 |
| Joint training independent images per video | 8 |
| Sampling timesteps | 256 (+256 Langevin cor.) |
| Sampling log-variance interpolation | gamma=0.0 |
| Prediction target | v |
| Data augmentation | left-right flips |

## Kinetics

| Parameter | Value |
|---|---|
| Base channels | 256 |
| Channel multipliers | 1, 2, 4, 8 |
| Blocks per resolution | 2 |
| Attention resolutions | 8, 16, 32 |
| Attention head dimension | 64 |
| Conditioning embedding dimension | 1024 |
| Conditioning embedding MLP layers | 2 |
| Diffusion noise schedule | cosine |
| Noise schedule log SNR range | [-20, 20] |
| Video resolution | 16x64x64 frameskip 1 |
| Weight decay | 0.0 |
| Reconstruction guidance weight | 9 |
| Optimizer | Adam (beta_1=0.9, beta_2=0.99) |
| Learning rate | 0.0002 |
| Batch size | 256 |
| EMA | 0.9999 |
| Dropout | 0.1 |
| Training hardware | 256 TPU-v4 chips |
| Training steps | 220,000 |
| Joint training independent images per video | 8 |
| Sampling timesteps | 128 (+128 Langevin cor.) |
| Sampling log-variance interpolation | gamma=0.0 |
| Prediction target | v |

## Text-to-video

**Small 16x64x64 model**

| Parameter | Value |
|---|---|
| Base channels | 128 |
| Channel multipliers | 1, 2, 4, 8 |
| Blocks per resolution | 2 |
| Attention resolutions | 8, 16, 32 |
| Attention head dimension | 64 |
| Conditioning embedding dimension | 1024 |
| Conditioning embedding MLP layers | 4 |
| Diffusion noise schedule | cosine |
| Noise schedule log SNR range | [-20, 20] |
| Video resolution | 16x64x64 frameskip 1 |
| Weight decay | 0.0 |
| Optimizer | Adam (beta_1=0.9, beta_2=0.99) |
| Learning rate | 0.0003 |
| Batch size | 128 |
| EMA | 0.9999 |
| Dropout | 0.0 |
| Training hardware | 64 TPU-v4 chips |
| Training steps | 200000 |
| Joint training independent images per video | 0, 4, 8 |
| Sampling timesteps | 256 |
| Sampling log-variance interpolation | gamma=0.3 |
| Prediction target | epsilon |

**Large 16x64x64 model**

| Parameter | Value |
|---|---|
| Base channels | 256 |
| Channel multipliers | 1, 2, 4, 8 |
| Blocks per resolution | 2 |
| Attention resolutions | 8, 16, 32 |
| Attention head dimension | 64 |
| Conditioning embedding dimension | 1024 |
| Conditioning embedding MLP layers | 4 |
| Diffusion noise schedule | cosine |
| Noise schedule log SNR range | [-20, 20] |
| Video resolution | 16x64x64 frameskip 1,4 |
| Weight decay | 0.0 |
| Optimizer | Adam (beta_1=0.9, beta_2=0.99) |
| Learning rate | 0.0003 |
| Batch size | 128 |
| EMA | 0.9999 |
| Dropout | 0.0 |
| Training hardware | 128 TPU-v4 chips |
| Training steps | 700000 |
| Joint training independent images per video | 8 |
| Sampling timesteps | 256 |
| Sampling log-variance interpolation | gamma=0.3 |
| Prediction target | epsilon |

**Large 9x128x128 model**

| Parameter | Value |
|---|---|
| Base channels | 128 |
| Channel multipliers | 1, 2, 4, 8, 16 |
| Blocks per resolution | 2 |
| Attention resolutions | 8, 16, 32 |
| Attention head dimension | 128 |
| Conditioning embedding dimension | 1024 |
| Conditioning embedding MLP layers | 4 |
| Diffusion noise schedule | cosine |
| Noise schedule log SNR range | [-20, 20] |
| Video resolution | 9x128x128 frameskip 1 |
| Weight decay | 0.0 |
| Optimizer | Adam (beta_1=0.9, beta_2=0.99) |
| Learning rate | 0.0002 |
| Batch size | 128 |
| EMA | 0.9999 |
| Dropout | 0.0 |
| Training hardware | 128 TPU-v4 chips |
| Training steps | 800000 |
| Joint training independent images per video | 7 |
| Sampling timesteps | 256 |
| Sampling log-variance interpolation | gamma=0.3 |
| Prediction target | epsilon |
