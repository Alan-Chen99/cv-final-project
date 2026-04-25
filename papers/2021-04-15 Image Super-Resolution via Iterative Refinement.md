# Image Super-Resolution via Iterative Refinement

**Authors:** Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J. Fleet, Mohammad Norouzi

**Published:** 2021-04-15 (arXiv: 2104.07636)

**Venue:** ICCV 2021

---

## Abstract

We present SR3, an approach to image Super-Resolution via Repeated Refinement. SR3 adapts denoising diffusion probabilistic models (Ho et al., 2020; Sohl-Dickstein et al., 2015) to conditional image generation and performs super-resolution through a stochastic iterative denoising process. Output generation starts with pure Gaussian noise and iteratively refines the noisy output using a U-Net model trained on denoising at various noise levels. SR3 exhibits strong performance on super-resolution tasks at different magnification factors, on faces and natural images. We conduct human evaluation on a standard 8x face super-resolution task on CelebA-HQ, comparing with SOTA GAN methods. SR3 achieves a fool rate close to 50%, suggesting photo-realistic outputs, while GANs do not exceed a fool rate of 34%. We further show the effectiveness of SR3 in cascaded image generation, where generative models are chained with super-resolution models, yielding a competitive FID score of 11.3 on ImageNet.

## 1 Introduction

Single-image super-resolution is the process of generating a high-resolution image that is consistent with an input low-resolution image. It falls under the broad family of image-to-image translation tasks, including colorization, in-painting, and de-blurring. Like many such inverse problems, image super-resolution is challenging because multiple output images may be consistent with a single input image, and the conditional distribution of output images given the input typically does not conform well to simple parametric distributions, e.g., a multivariate Gaussian. Accordingly, while simple regression-based methods with feedforward convolutional nets may work for super-resolution at low magnification ratios, they often lack the high-fidelity details needed for high magnification ratios.

Deep generative models have seen success in learning complex empirical distributions of images (e.g., Sutskever et al., 2014; Vaswani et al., 2017). Autoregressive models (van den Oord et al., 2016a,b), variational autoencoders (VAEs) (Kingma and Welling, 2013; Vahdat and Kautz, 2020), Normalizing Flows (NFs) (Dinh et al., 2016; Kingma and Dhariwal, 2018), and GANs (Goodfellow et al., 2014; Karras et al., 2018; Radford et al., 2015) have shown convincing image generation results and have been applied to conditional tasks such as image super-resolution (Chen et al., 2018; Dahl et al., 2017; Ledig et al., 2017; Menon et al., 2020; Parmar et al., 2018). However, these approaches often suffer from various limitations; e.g., autoregressive models are prohibitively expensive for high-resolution image generation, NFs and VAEs often yield sub-optimal sample quality, and GANs require carefully designed regularization and optimization tricks to tame optimization instability (Arjovsky et al., 2017; Gulrajani et al., 2017) and mode collapse (Metz et al., 2016; Ravuri and Vinyals, 2019).

We propose SR3 (Super-Resolution via Repeated Refinement), a new approach to conditional image generation, inspired by recent work on Denoising Diffusion Probabilistic Models (DDPM) (Ho et al., 2020; Sohl-Dickstein et al., 2015), and denoising score matching (Ho et al., 2020; Song and Ermon, 2019). SR3 works by learning to transform a standard normal distribution into an empirical data distribution through a sequence of refinement steps, resembling Langevin dynamics. The key is a U-Net architecture (Ronneberger et al., 2015) that is trained with a denoising objective to iteratively remove various levels of noise from the output. We adapt DDPMs to *conditional* image generation by proposing a simple and effective modification to the U-Net architecture. In contrast to GANs that require inner-loop maximization, we minimize a well-defined loss function. Unlike autoregressive models, SR3 uses a constant number of inference steps regardless of output resolution.

SR3 works well across a range of magnification factors and input resolutions. SR3 models can also be cascaded, e.g., going from 64x64 to 256x256, and then to 1024x1024. Cascading models allows one to independently train a few small models rather than a single large model with a high magnification factor. We find that chained models enable more efficient inference, since directly generating a high-resolution image requires more iterative refinement steps for the same quality. We also find that one can chain an unconditional generative model with SR3 models to unconditionally generate high-fidelity images. Unlike existing work that focuses on specific domains (e.g., faces), we show that SR3 is effective on both faces and natural images.

Automated image quality scores like PSNR and SSIM do not reflect human preference well when the input resolution is low and the magnification ratio is large (e.g., Berthelot et al., 2020; Chen et al., 2018; Dahl et al., 2017; Menon et al., 2020). These quality scores often penalize synthetic high-frequency details, such as hair texture, because synthetic details do not perfectly align with the reference details. We resort to human evaluation to compare the quality of super-resolution methods. We adopt a 2-alternative forced-choice (2AFC) paradigm in which human subjects are shown a low-resolution input and are required to select between a model output and a ground truth image (cf. Zhang et al., 2016). Based on this study, we calculate *fool rate* scores that capture both image quality and the consistency of model outputs with low-resolution inputs. Experiments demonstrate that SR3 achieves a significantly higher fool rate than SOTA GAN methods (Chen et al., 2018; Menon et al., 2020) and a strong regression baseline.

Our key contributions are summarized as:
- We adapt denoising diffusion models to conditional image generation. Our method, *SR3*, is an approach to image super-resolution via iterative refinement.
- SR3 proves effective on face and natural image super-resolution at different magnification factors. On a standard 8x face super-resolution task, SR3 achieves a human fool rate close to 50%, outperforming FSRGAN (Chen et al., 2018) and PULSE (Menon et al., 2020) that achieve fool rates of at most 34%.
- We demonstrate unconditional and class-conditional generation by cascading a 64x64 image synthesis model with SR3 models to progressively generate 1024x1024 unconditional faces in 3 stages, and 256x256 class-conditional ImageNet samples in 2 stages. Our class conditional ImageNet samples attain competitive FID scores.

## 2 Conditional Denoising Diffusion Model

We are given a dataset of input-output image pairs, denoted $\mathcal{D} = \{\mathbf{x}_i, \mathbf{y}_i\}_{i=1}^N$, which represent samples drawn from an unknown conditional distribution $p(\mathbf{y} \mid \mathbf{x})$. This is a one-to-many mapping in which many target images may be consistent with a single source image. We are interested in learning a parametric approximation to $p(\mathbf{y} \mid \mathbf{x})$ through a *stochastic* iterative refinement process that maps a source image $\mathbf{x}$ to a target image $\mathbf{y} \in \mathbb{R}^d$. We approach this problem by adapting the denoising diffusion probabilistic (DDPM) model of Ho et al. (2020) and Sohl-Dickstein et al. (2015) to *conditional* image generation.

The conditional DDPM model generates a target image $\mathbf{y}_0$ in $T$ refinement steps. Starting with a pure noise image $\mathbf{y}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, the model iteratively refines the image through successive iterations $(\mathbf{y}_{T-1}, \mathbf{y}_{T-2}, \dotsc, \mathbf{y}_{0})$ according to learned conditional transition distributions $p_\theta(\mathbf{y}_{t-1} \mid \mathbf{y}_t, \mathbf{x})$ such that $\mathbf{y}_{0} \sim p(\mathbf{y} \mid \mathbf{x})$.

The distributions of intermediate images in the inference chain are defined in terms of a *forward* diffusion process that gradually adds Gaussian noise to the signal via a fixed Markov chain, denoted $q(\mathbf{y}_t \mid \mathbf{y}_{t-1})$. The goal of our model is to reverse the Gaussian diffusion process by iteratively recovering signal from noise through a reverse Markov chain conditioned on $\mathbf{x}$. In principle, each forward process step can be conditioned on $\mathbf{x}$ too, but we leave that to future work. We learn the reverse chain using a neural denoising model $f_\theta$ that takes as input a source image and a noisy target image and estimates the noise. We first give an overview of the forward diffusion process, and then discuss how our denoising model $f_\theta$ is trained and used for inference.

### 2.1 Gaussian Diffusion Process

Following Ho et al. (2020) and Sohl-Dickstein et al. (2015), we first define a *forward* Markovian diffusion process $q$ that gradually adds Gaussian noise to a high-resolution image $\mathbf{y}_0$ over $T$ iterations:

$$q(\mathbf{y}_{1:T} \mid \mathbf{y}_0) = \prod_{t=1}^{T} q(\mathbf{y}_{t} \mid \mathbf{y}_{t-1})$$

$$q(\mathbf{y}_{t} \mid \mathbf{y}_{t-1}) = \mathcal{N}(\mathbf{y}_{t} \mid \sqrt{\alpha_t}\, \mathbf{y}_{t-1}, (1 - \alpha_t) \mathbf{I})$$

where the scalar parameters $\alpha_{1:T}$ are hyper-parameters, subject to $0 < \alpha_t < 1$, which determine the variance of the noise added at each iteration. Note that $\mathbf{y}_{t-1}$ is attenuated by $\sqrt{\alpha_t}$ to ensure that the variance of the random variables remains bounded as $t \to \infty$. For instance, if the variance of $\mathbf{y}_{t-1}$ is $1$, then the variance of $\mathbf{y}_{t}$ is also $1$.

Importantly, one can characterize the distribution of $\mathbf{y}_t$ given $\mathbf{y}_0$ by marginalizing out the intermediate steps as

$$q(\mathbf{y}_t \mid \mathbf{y}_0) = \mathcal{N}(\mathbf{y}_t \mid \sqrt{\gamma_t}\, \mathbf{y}_0, (1-\gamma_t) \mathbf{I})$$

where $\gamma_t = \prod_{i=1}^t \alpha_i$. Furthermore, with some algebraic manipulation and completing the square, one can derive the posterior distribution of $\mathbf{y}_{t-1}$ given $(\mathbf{y}_0, \mathbf{y}_t)$ as

$$q(\mathbf{y}_{t-1} \mid \mathbf{y}_0, \mathbf{y}_t) = \mathcal{N}(\mathbf{y}_{t-1} \mid \boldsymbol{\mu}, \sigma^2 \mathbf{I})$$

$$\boldsymbol{\mu} = \frac{\sqrt{\gamma_{t-1}}\,(1-\alpha_t)}{1-\gamma_t}\, \mathbf{y}_0 + \frac{\sqrt{\alpha_t}\,(1-\gamma_{t-1})}{1-\gamma_t}\mathbf{y}_t$$

$$\sigma^2 = \frac{(1-\gamma_{t-1})(1-\alpha_t)}{1-\gamma_t}$$

This posterior distribution is helpful when parameterizing the reverse chain and formulating a variational lower bound on the log-likelihood of the reverse chain. We next discuss how one can learn a neural network to reverse this Gaussian diffusion process.

### 2.2 Optimizing the Denoising Model

**Algorithm 1: Training a denoising model $f_\theta$**
1. Repeat:
   - $(\mathbf{x}, \mathbf{y}_0) \sim p(\mathbf{x}, \mathbf{y})$
   - $\gamma \sim p(\gamma)$
   - $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
   - Take a gradient descent step on $\nabla_\theta \left\lVert f_\theta(\mathbf{x}, \sqrt{\gamma} \mathbf{y}_0 + \sqrt{1-\gamma} \boldsymbol{\epsilon}, \gamma) - \boldsymbol{\epsilon} \right\rVert_p^p$
2. Until converged

To help reverse the diffusion process, we take advantage of additional side information in the form of a source image $\mathbf{x}$ and optimize a neural denoising model $f_{\theta}$ that takes as input this source image $\mathbf{x}$ and a noisy target image $\tilde{\mathbf{y}}$,

$$\tilde{\mathbf{y}} = \sqrt{\gamma}\, \mathbf{y}_0 + \sqrt{1-\gamma} \,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I})$$

and aims to recover the noiseless target image $\mathbf{y}_0$. This definition of a noisy target image $\tilde{\mathbf{y}}$ is compatible with the marginal distribution of noisy images at different steps of the forward diffusion process.

In addition to a source image $\mathbf{x}$ and a noisy target image $\tilde{\mathbf{y}}$, the denoising model $f_\theta(\mathbf{x}, \tilde{\mathbf{y}}, \gamma)$ takes as input the sufficient statistics for the variance of the noise $\gamma$, and is trained to predict the noise vector $\boldsymbol{\epsilon}$. We make the denoising model aware of the level of noise through conditioning on a scalar $\gamma$, similar to Song and Ermon (2019) and Chen et al. (2021). The proposed objective function for training $f_{\theta}$ is

$$\mathbb{E}_{(\mathbf{x}, \mathbf{y})} \mathbb{E}_{\boldsymbol{\epsilon}, \gamma} \left\lVert f_\theta\!\left(\mathbf{x}, \underbrace{\sqrt{\gamma} \,\mathbf{y}_0 + \sqrt{1-\gamma}\, \boldsymbol{\epsilon}}_{\tilde{\mathbf{y}}}, \gamma\right) - \boldsymbol{\epsilon}\, \right\rVert^{p}_p$$

where $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, $(\mathbf{x}, \mathbf{y})$ is sampled from the training dataset, $p \in \{1, 2\}$, and $\gamma \sim p(\gamma)$. The distribution of $\gamma$ has a big impact on the quality of the model and the generated outputs. We discuss our choice of $p(\gamma)$ in Section 2.4.

Instead of regressing the output of $f_{\theta}$ to $\boldsymbol{\epsilon}$, one can also regress the output of $f_{\theta}$ to $\mathbf{y}_0$. Given $\gamma$ and $\tilde{\mathbf{y}}$, the values of $\boldsymbol{\epsilon}$ and $\mathbf{y}_0$ can be derived from each other deterministically, but changing the regression target has an impact on the scale of the loss function. We expect both of these variants to work reasonably well if $p(\gamma)$ is modified to account for the scale of the loss function. Further investigation of the loss function used for training the denoising model is an interesting avenue for future research in this area.

### 2.3 Inference via Iterative Refinement

**Algorithm 2: Inference in $T$ iterative refinement steps**
1. $\mathbf{y}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
2. For $t = T, \dotsc, 1$:
   - $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ if $t > 1$, else $\mathbf{z} = \mathbf{0}$
   - $\mathbf{y}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left( \mathbf{y}_t - \frac{1-\alpha_t}{\sqrt{1-\gamma_t}} f_\theta(\mathbf{x}, \mathbf{y}_t, \gamma_t) \right) + \sqrt{1 - \alpha_t} \mathbf{z}$
3. Return $\mathbf{y}_0$

Inference under our model is defined as a *reverse* Markovian process, which goes in the reverse direction of the forward diffusion process, starting from Gaussian noise $\mathbf{y}_T$:

$$p_\theta(\mathbf{y}_{0:T} | \mathbf{x}) = p(\mathbf{y}_T) \prod_{t=1}^T p_\theta(\mathbf{y}_{t-1} | \mathbf{y}_t, \mathbf{x})$$

$$p(\mathbf{y}_T) = \mathcal{N}(\mathbf{y}_T \mid \mathbf{0}, \mathbf{I})$$

$$p_\theta(\mathbf{y}_{t-1} | \mathbf{y}_{t}, \mathbf{x}) = \mathcal{N}(\mathbf{y}_{t-1} \mid \mu_{\theta}(\mathbf{x}, {\mathbf{y}}_{t}, \gamma_t), \sigma_t^2\mathbf{I})$$

We define the inference process in terms of isotropic Gaussian conditional distributions, $p_\theta(\mathbf{y}_{t-1} | \mathbf{y}_{t}, \mathbf{x})$, which are learned. If the noise variance of the forward process steps are set as small as possible, i.e., $\alpha_{1:T} \approx 1$, the optimal reverse process $p(\mathbf{y}_{t-1} | \mathbf{y}_{t}, \mathbf{x})$ will be approximately Gaussian (Sohl-Dickstein et al., 2015). Accordingly, our choice of Gaussian conditionals in the inference process can provide a reasonable fit to the true reverse process. Meanwhile, $1 - \gamma_T$ should be large enough so that $\mathbf{y}_T$ is approximately distributed according to the prior $p(\mathbf{y}_T) = \mathcal{N}(\mathbf{y}_T | \mathbf{0}, \mathbf{I})$, allowing the sampling process to start at pure Gaussian noise.

Recall that the denoising model $f_{\theta}$ is trained to estimate $\boldsymbol{\epsilon}$, given any noisy image $\tilde{\mathbf{y}}$ including $\mathbf{y}_t$. Thus, given $\mathbf{y}_t$, we approximate $\mathbf{y}_0$ by rearranging the terms as

$$\hat{\mathbf{y}}_0 = \frac{1}{\sqrt{\gamma_t}} \left( \mathbf{y}_t - \sqrt{1 - \gamma_t}\, f_{\theta}(\mathbf{x}, \mathbf{y}_{t}, \gamma_t) \right)$$

Following the formulation of Ho et al. (2020), we substitute our estimate $\hat{\mathbf{y}}_0$ into the posterior distribution of $q(\mathbf{y}_{t-1} | \mathbf{y}_0, \mathbf{y}_t)$ to parameterize the mean of $p_\theta(\mathbf{y}_{t-1} | \mathbf{y}_t, \mathbf{x})$ as

$$\mu_{\theta}(\mathbf{x}, {\mathbf{y}}_{t}, \gamma_t) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{y}_t - \frac{1-\alpha_t}{ \sqrt{1 - \gamma_t}} f_{\theta}(\mathbf{x}, \mathbf{y}_{t}, \gamma_t) \right)$$

and we set the variance of $p_\theta(\mathbf{y}_{t-1}|\mathbf{y}_t, \mathbf{x})$ to $(1 - \alpha_t)$, a default given by the variance of the forward process (Ho et al., 2020).

Following this parameterization, each iteration of iterative refinement under our model takes the form,

$$\mathbf{y}_{t-1} \leftarrow \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{y}_t - \frac{1-\alpha_t}{ \sqrt{1 - \gamma_t}} f_{\theta}(\mathbf{x}, \mathbf{y}_{t}, \gamma_t) \right) + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_t$$

where $\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. This resembles one step of Langevin dynamics with $f_{\theta}$ providing an estimate of the gradient of the data log-density. We justify the choice of the training objective for the probabilistic model from a variational lower bound perspective and a denoising score-matching perspective in Appendix B.

### 2.4 SR3 Model Architecture and Noise Schedule

The SR3 architecture is similar to the U-Net found in DDPM (Ho et al., 2020), with modifications adapted from Song et al. (2021); we replace the original DDPM residual blocks with residual blocks from BigGAN (Brock et al., 2018), and we re-scale skip connections by $\frac{1}{\sqrt{2}}$. We also increase the number of residual blocks, and the channel multipliers at different resolutions (see Appendix A for details). To condition the model on the input $\mathbf{x}$, we up-sample the low-resolution image to the target resolution using bicubic interpolation. The result is concatenated with $\mathbf{y}_t$ along the channel dimension. We experimented with more sophisticated methods of conditioning, such as using FiLM (Perez et al., 2018), but we found that the simple concatenation yielded similar generation quality.

For our training noise schedule, we follow Chen et al. (2021), and use a piecewise distribution for $\gamma$, $p(\gamma) = \sum_{t=1}^T \frac{1}{T} U(\gamma_{t-1}, \gamma_t)$. Specifically, during training, we first uniformly sample a time step $t \sim \{0, ..., T\}$ followed by sampling $\gamma \sim U(\gamma_{t-1}, \gamma_t)$. We set $T = 2000$ in all our experiments.

Prior work of diffusion models (Ho et al., 2020; Song et al., 2021) require 1-2k diffusion steps during inference, making generation slow for large target resolution tasks. We adapt techniques from Chen et al. (2021) to enable more efficient inference. Our model conditions on $\gamma$ directly (vs $t$ as in Ho et al., 2020), which allows us flexibility in choosing number of diffusion steps, and the noise schedule during inference. This has been demonstrated to work well for speech synthesis (Chen et al., 2021), but has not been explored for images. For efficient inference we set the maximum inference budget to 100 diffusion steps, and hyper-parameter search over the inference noise schedule. This search is inexpensive as we only need to train the model once (Chen et al., 2021). We use FID on held out data to choose the best noise schedule, as we found PSNR did not correlate well with image quality.

## 3 Related Work

SR3 is inspired by recent work on deep generative models and recent learning-based approaches to super-resolution.

**Generative Models.** Autoregressive models (ARs) (van den Oord et al., 2016; Salimans et al., 2017) can model exact data log likelihood, capturing rich distributions. However, their sequential generation of pixels is expensive, limiting application to low-resolution images. Normalizing flows (Rezende and Mohamed, 2015; Dinh et al., 2016; Kingma and Dhariwal, 2018) improve on sampling speed while modelling the exact data likelihood, but the need for invertible parameterized transformations with a tractable Jacobian determinant limits their expressiveness. VAEs (Kingma and Welling, 2013; Rezende et al., 2014) offer fast sampling, but tend to underperform GANs and ARs in image quality (Vahdat and Kautz, 2020). Generative Adversarial Networks (GANs) (Goodfellow et al., 2014) are popular for class conditional image generation and super-resolution. Nevertheless, the inner-outer loop optimization often requires tricks to stabilize training (Arjovsky et al., 2017; Gulrajani et al., 2017), and conditional tasks like super-resolution usually require an auxiliary consistency-based loss to avoid mode collapse (Ledig et al., 2017). Cascades of GAN models have been used to generate higher resolution images (Denton et al., 2015).

Score matching (Hyvarinen and Dayan, 2005) models the gradient of the data log-density with respect to the image. Score matching on noisy data, called denoising score matching (Vincent, 2011), is equivalent to training a denoising autoencoder, and to DDPMs (Ho et al., 2020). Denoising score matching over multiple noise scales with Langevin dynamics sampling from the learned score functions has recently been shown to be effective for high quality unconditional image generation (Song and Ermon, 2019; Ho et al., 2020). These models have also been generalized to continuous time (Song et al., 2021). Denoising score matching and diffusion models have also found success in shape generation (Cai et al., 2020), and speech synthesis (Chen et al., 2021). We extend this method to super-resolution, with a simple learning objective, a constant number of inference generation steps, and high quality generation.

**Super-Resolution.** Numerous super-resolution methods have been proposed in the computer vision community (Dong et al., 2014; Ahn et al., 2018; Kim et al., 2016; Tai et al., 2017; Ledig et al., 2017; Sajjadi et al., 2017). Much of the early work on super-resolution is regression based and trained with an MSE loss (Dong et al., 2014; Ahn et al., 2018; Wang et al., 2015; Dong et al., 2016; Kim et al., 2016). As such, they effectively estimate the posterior mean, yielding blurry images when the posterior is multi-modal (Ledig et al., 2017; Sajjadi et al., 2017; Menon et al., 2020). Our regression baseline defined below is also a one-step regression model trained with MSE (cf. Ahn et al., 2018; Kim et al., 2016), but with a large U-Net architecture. SR3, by comparison, relies on a series of iterative refinement steps, each of which is trained with a regression loss. This difference permits our iterative approach to capture richer distributions. Further, rather than estimating the posterior mean, SR3 generates samples from the target posterior.

Autoregressive models have been used successfully for super-resolution and cascaded up-sampling (Dahl et al., 2017; Menick and Kalchbrenner, 2019; van den Oord et al., 2016; Parmar et al., 2018). Nevertheless, the expense of inference limits their applicability to low-resolution images. SR3 can generate high-resolution images, e.g., 1024x1024, but with a constant number of refinement steps (often no more than 100).

Normalizing flows have been used for super-resolution with a multi-scale approach (Yu et al., 2020). They are capable of generating 1024x1024 images due in part to their efficient inference process. But SR3 uses a series of reverse diffusion steps to transform a Gaussian distribution to an image distribution while flows require a deep and invertible network.

GAN-based super-resolution methods have also found considerable success (Karras et al., 2018; Ledig et al., 2017; Menon et al., 2020; Yang et al., 2020; Sajjadi et al., 2017). FSRGAN (Chen et al., 2018) and PULSE (Menon et al., 2020) in particular have demonstrated high quality face super-resolution results. However, many such GAN based methods are generally difficult to optimize, and often require auxiliary objective functions to ensure consistency with the low resolution inputs.

## 4 Experiments

We assess the effectiveness of SR3 models in super-resolution on faces, natural images, and synthetic images obtained from a low-resolution generative model. The latter enables high-resolution image synthesis using model cascades. We compare SR3 with recent methods such as FSRGAN (Chen et al., 2018) and PULSE (Menon et al., 2020) using human evaluation, and report FID for various tasks. We also compare to a regression baseline model that shares the same architecture as SR3, but is trained with a MSE loss. Our experiments include:
- Face super-resolution at 16x16 -> 128x128 and 64x64 -> 512x512 trained on FFHQ and evaluated on CelebA-HQ.
- Natural image super-resolution at 64x64 -> 256x256 pixels on ImageNet (Russakovsky et al., 2015).
- Unconditional 1024x1024 face generation by a cascade of 3 models, and class-conditional 256x256 ImageNet image generation by a cascade of 2 models.

**Datasets:** We follow previous work (Menon et al., 2020), training face super-resolution models on Flickr-Faces-HQ (FFHQ) (Karras et al., 2019) and evaluating on CelebA-HQ (Karras et al., 2018). For natural image super-resolution, we train on ImageNet 1K (Russakovsky et al., 2015) and use the dev split for evaluation. We train unconditional face and class-conditional ImageNet generative models using DDPM on the same datasets discussed above. For training and testing, we use low-resolution images that are down-sampled using bicubic interpolation with anti-aliasing enabled. For ImageNet, we discard images where the shorter side is less than the target resolution. We use the largest central crop like Brock et al. (2018), which is then resized to the target resolution using area resampling as our high resolution image.

**Training Details:** We train all of our SR3 and regression models for 1M training steps with a batch size of 256. We choose a checkpoint for the regression baseline based on peak-PSNR on the held out set. We do not perform any checkpoint selection on SR3 models and simply select the latest checkpoint. Consistent with Ho et al. (2020), we use the Adam optimizer with a linear warmup schedule over 10k training steps, followed by a fixed learning rate of 1e-4 for SR3 models and 1e-5 for regression models. We use 625M parameters for our 64x64 -> {256x256, 512x512} models, 550M parameters for the 16x16 -> 128x128 models, and 150M parameters for 256x256 -> 1024x1024 model. We use a dropout rate of 0.2 for 16x16 -> 128x128 models super-resolution, but otherwise, we do not use dropout. (See Appendix A for task specific architectural details.)

### 4.1 Qualitative Results

**Natural Images:** Figure 2 gives examples of super-resolution natural images for 64x64 -> 256x256 on the ImageNet dev set, along with enlarged patches for finer inspection. The baseline Regression model generates images that are faithful to the inputs, but are blurry and lack detail. By comparison, SR3 produces sharp images with more detail; this is most evident in the enlarged patches.

**Face Images:** Figure 3 shows outputs of a face super-resolution model (64x64 -> 512x512) on two test images, again with selected patches enlarged. With the 8x magnification factor one can clearly see the detailed structure inferred. Note that, because of the large magnification factor, there are many plausible outputs, so we do not expect the output to exactly match the reference image.

### 4.2 Benchmark Comparison

#### 4.2.1 Automated Metrics

Table 1 shows the PSNR, SSIM (Wang et al., 2004) and Consistency scores for 16x16 -> 128x128 face super-resolution. SR3 outperforms PULSE and FSRGAN on PSNR and SSIM while underperforming the regression baseline. Previous work (Chen et al., 2018; Dahl et al., 2017; Menon et al., 2020) observed that these conventional automated evaluation measures do not correlate well with human perception when the input resolution is low and the magnification factor is large. This is not surprising because these metrics tend to penalize any synthetic high-frequency detail that is not perfectly aligned with the target image. Since generating perfectly aligned high-frequency details is almost impossible, PSNR and SSIM tend to prefer MSE regression-based techniques that are extremely conservative with high-frequency details.

**Table 1: PSNR & SSIM on 16x16 -> 128x128 face super-resolution. Consistency measures MSE (x10^-5) between the low-resolution inputs and the down-sampled super-resolution outputs.**

| Metric | PULSE | FSRGAN | Regression | SR3 |
|---|---|---|---|---|
| **PSNR** | 16.88 | 23.01 | **23.96** | 23.04 |
| **SSIM** | 0.44 | 0.62 | **0.69** | 0.65 |
| **Consistency** | 161.1 | 33.8 | 2.71 | **2.68** |

**Consistency:** As a measure of the consistency of the super-resolution outputs, we compute MSE between the downsampled outputs and the low resolution inputs. Table 1 shows that SR3 achieves the best consistency error beating PULSE and FSRGAN by a significant margin slightly outperforming even the regression baseline. This result demonstrates the key advantage of SR3 over state of the art GAN based methods as they do not require any auxiliary objective function in order to ensure consistency with the low resolution inputs.

**Table 2: Performance comparison between SR3 and Regression baseline on natural image super-resolution using standard metrics computed on the ImageNet validation set.**

| Model | FID | IS | PSNR | SSIM |
|---|---|---|---|---|
| Reference | 1.9 | 240.8 | - | - |
| Regression | 15.2 | 121.1 | **27.9** | **0.801** |
| SR3 | **5.2** | **180.1** | 26.4 | 0.762 |

This is further confirmed in Table 2 for ImageNet super-resolution (64x64 -> 256x256) where the outputs of SR3 achieve higher sample quality scores (FID and IS), but worse PSNR and SSIM than regression.

**Classification Accuracy:** Table 3 compares our 4x natural image super-resolution models with previous work in terms of object classification on low-resolution images. We mirror the evaluation setup of Sajjadi et al. (2017) and Zhang et al. (2018) and apply 4x super-resolution models to 56x56 center crops from the validation set of ImageNet. Then, we report classification error based on a pre-trained ResNet-50 (He et al., 2016). SR3 outperforms existing methods by a large margin on top-1 and top-5 classification errors, demonstrating high perceptual quality of SR3 outputs.

**Table 3: Comparison of classification accuracy scores for 4x natural image super-resolution on the first 1K images from the ImageNet Validation set.**

| Method | Top-1 Error | Top-5 Error |
|---|---|---|
| Baseline | 0.252 | 0.080 |
| DRCN (Kim et al., 2016) | 0.477 | 0.242 |
| FSRCNN (Dong et al., 2016) | 0.437 | 0.196 |
| PsyCo (Perez-Pellitero et al., 2016) | 0.454 | 0.224 |
| ENet-E (Sajjadi et al., 2017) | 0.449 | 0.214 |
| RCAN (Zhang et al., 2018) | 0.393 | 0.167 |
| Regression | 0.383 | 0.173 |
| SR3 | **0.317** | **0.120** |

#### 4.2.2 Human Evaluation (2AFC)

In this work, we are primarily interested in photo-realistic super-resolution with large magnification factors. Accordingly, we resort to direct human evaluation. While mean opinion score (MOS) is commonly used to measure image quality in this context, forced choice pairwise comparison has been found to be a more reliable method for such subjective quality assessments (Mantiuk et al., 2012). Furthermore, standard MOS studies do not capture consistency between low-resolution inputs and high-resolution outputs.

We use a 2-alternative forced-choice (2AFC) paradigm to measure how well humans can discriminate true images from those generated from a model. In Task-1 subjects were shown a low resolution input in between two high-resolution images, one being the real image (ground truth), and the other generated from the model. Subjects were asked *"Which of the two images is a better high quality version of the low resolution image in the middle?"* This task takes into account both image quality and consistency with the low resolution input. Task-2 is similar to Task-1, except that the low-resolution image was not shown, so subjects only had to select the image that was more photo-realistic. They were asked *"Which image would you guess is from a camera?"* Subjects viewed images for 3 seconds before responding, in both tasks.

The subject *fool rate* is the fraction of trials on which a subject selects the model output over ground truth. Our fool rates for each model are based on 50 subjects, each of whom were shown 50 of the 100 images in the test set. In both experiments, the fool rate of SR3 is close to 50%, indicating that SR3 produces images that are both photo-realistic and faithful to the low-resolution inputs. We find similar fool rates over a wide range of viewing durations up to 12 seconds.

The fool rates for FSRGAN and PULSE in Task-1 are lower than the Regression baseline and SR3. We speculate that the PULSE optimization has failed to converge to high resolution images sufficiently close to the inputs. Indeed, when asked solely about image quality in Task-2, the PULSE fool rate increases significantly.

The fool rate for the Regression baseline is lower in Task-2 than Task-1. The regression model tends to generate images that are blurry, but nevertheless faithful to the low resolution input. We speculate that in Task-1, given the inputs, subjects are influenced by consistency, while in Task-2, ignoring consistency, they instead focus on image sharpness.

We conduct similar human evaluation studies on natural images comparing SR3 and the regression baseline on ImageNet. In both tasks with natural images, SR3 achieves a human subject fool rate close to 40%. Like the face image experiments, here again we find that the Regression baseline yields a lower fool rate in Task-2, where the low resolution image is not shown.

To further appreciate the experimental results it is useful to visually compare outputs of different models on the same inputs. FSRGAN exhibits distortion in face region and struggles with generating glasses properly. It also fails to recover texture details in the hair region. PULSE often produces images that differ significantly from the input image, both in the shape of the face and the background, and sometimes in gender too, presumably due to failure of the optimization to find a sufficiently good minimum. The Regression baseline produces results consistent to the input, however they are typically quite blurry. By comparison, the SR3 results are consistent with the input and contain more detailed image structure.

### 4.3 Cascaded High-Resolution Image Synthesis

We study *cascaded* image generation, where SR3 models at different scales are chained together with unconditional generative models, enabling high-resolution image synthesis. Cascaded generation allows one to train different models in parallel, and each model in the cascade solves a simpler task, requiring fewer parameters and less computation for training. Inference with cascaded models is also more efficient, especially for iterative refinement models. With cascaded generation we found it effective to use more refinement steps at low-resolutions, and fewer steps at higher resolutions. This was much more efficient than generating directly at high resolution without sacrificing image quality.

We train a DDPM (Ho et al., 2020) model for unconditional 64x64 face generation. Samples from this model are then fed to two 4x SR3 models, up-sampling to 256x256 and then to 1024x1024 pixels. In addition, we train an Improved DDPM (Nichol and Dhariwal, 2021) model on class-conditional 64x64 ImageNet, and we pass its generated samples to a 4x SR3 model yielding 256x256 pixels. The 4x SR3 model is not conditioned on the class label.

**Table 4: FID scores for class-conditional 256x256 ImageNet.**

| Model | FID-50k |
|---|---|
| VQ-VAE-2 (Razavi et al., 2019) | 38.1 |
| BigGAN (Truncation 1.0) (Brock et al., 2018) | 7.4 |
| BigGAN (Truncation 1.5) (Brock et al., 2018) | 11.8 |
| SR3 (Two Stage) | 11.3 |

Our 2-stage model improves on VQ-VAE-2, is comparable to deep BigGANs at truncation factor of 1.5 but underperforms them at truncation factor of 1.0. Unlike BigGAN, our diffusion models do not provide a knob to control sample quality vs. sample diversity, and finding ways to do so is an interesting avenue for future research. Nichol and Dhariwal (2021) concurrently trained cascaded generation models using super-resolution conditioned on class labels (our super-resolution is not conditioned on class labels), and observed a similar trend in FID scores. The effectiveness of cascaded image generation indicates that SR3 models are robust to the precise distribution of inputs (i.e., the specific form of anti-aliasing and downsampling).

**Ablation Studies:** Table 5 shows ablation studies on our 64x64 -> 256x256 ImageNet SR3 model. In order to improve the robustness of the SR3 model, we experiment with use of data augmentation while training. Specifically, we trained the model with varying amounts of Gaussian Blurring noise added to the low resolution input image. No blurring is applied during inference. We find that this has a significant impact, improving the FID score roughly by 2 points. We also explore the choice of $L_p$ norm for the denoising objective. We find that $L_1$ norm gives slightly better FID scores than $L_2$.

**Table 5: Ablation study on SR3 model for class-conditional 256x256 ImageNet.**

| Model | FID-50k |
|---|---|
| **Training with Augmentation** | |
| SR3 | 13.1 |
| SR3 (w/ Gaussian Blur) | 11.3 |
| **Objective L_p Norm** | |
| SR3 (L2) | 11.8 |
| SR3 (L1) | 11.3 |

## 5 Discussion and Conclusion

Bias is an important problem in all generative models. SR3 is no different, and suffers from bias issues. While in theory, our log-likelihood based objective is mode covering (e.g., unlike some GAN-based objectives), we believe it is likely our diffusion-based models drop modes. We observed some evidence of mode dropping: the model consistently generates nearly the same image output during sampling (when conditioned on the same input). We also observed the model to generate very continuous skin texture in face super-resolution, dropping moles, pimples and piercings found in the reference. SR3 should not be used for any real world super-resolution tasks, until these biases are thoroughly understood and mitigated.

In conclusion, SR3 is an approach to image super-resolution via iterative refinement. SR3 can be used in a cascaded fashion to generate high resolution super-resolution images, as well as unconditional samples when cascaded with an unconditional model. We demonstrate SR3 on face and natural image super-resolution at high resolution and high magnification ratios (e.g., 64x64 -> 256x256 and 256x256 -> 1024x1024). SR3 achieves a human fool rate close to 50%, suggesting photo-realistic outputs.

## Acknowledgements

We thank Jimmy Ba, Adji Bousso Dieng, Chelsea Finn, Geoffrey Hinton, Natacha Mainville, Shingai Manjengwa, and Ali Punjani for providing their face images on which we demonstrate the face SR3 results. We thank Ben Poole, Samy Bengio and the Google Brain team for research discussions and technical assistance. We also thank authors of Menon et al. (2020) for generously providing us with baseline super-resolution samples for human evaluation.

## Appendix A: Task Specific Architectural Details

Table A.1 summarizes the primary architecture details for each super-resolution task. For a particular task, we use the same architecture for both SR3 and Regression models. To condition the diffusion model on the low resolution image, we first interpolate the low resolution image to the target high resolution, and then simply concatenate it with the input noisy high resolution image.

**Table A.1: Task specific architecture hyper-parameters for the U-Net model. Channel Dim is the dimension of the first U-Net layer, while the depth multipliers are the multipliers for subsequent resolutions.**

| Task | Channel Dim | Depth Multipliers | # ResNet Blocks | # Parameters |
|---|---|---|---|---|
| 16x16 -> 128x128 | 128 | {1, 2, 4, 8, 8} | 3 | 550M |
| 64x64 -> 256x256 | 128 | {1, 2, 4, 4, 8, 8} | 3 | 625M |
| 64x64 -> 512x512 | 64 | {1, 2, 4, 8, 8, 16, 16} | 3 | 625M |
| 256x256 -> 1024x1024 | 16 | {1, 2, 4, 8, 16, 32, 32, 32} | 2 | 150M |

## Appendix B: Justification of the Training Objective

### B.1 A Variational Bound Perspective

Following Ho et al. (2020), we justify the choice of the training objective for the probabilistic model from a variational lower bound perspective. If the forward diffusion process is viewed as a fixed approximate posterior to the inference process, one can derive the following variational lower bound on the marginal log-likelihood:

$$\mathbb{E}_{(\mathbf{x},\mathbf{y}_0)}\log p_\theta(\mathbf{y}_0 | \mathbf{x}) \geq \mathbb{E}_{\mathbf{x},\mathbf{y}_0}\mathbb{E}_{q(\mathbf{y}_{1:T}|\mathbf{y}_0)}\left[ \log p(\mathbf{y}_T) + \sum_{t \geq 1} \log \frac{p_{\theta} (\mathbf{y}_{t-1} | \mathbf{y}_t, \mathbf{x})}{q(\mathbf{y}_t|\mathbf{y}_{t-1})} \right]$$

Given the particular parameterization of the inference process outlined above, one can show (Ho et al., 2020) that the negative variational lower bound can be expressed as the following simplified loss, up to a constant weighting of each term for each time step:

$$\mathbb{E}_{\mathbf{x},\mathbf{y}_0,\boldsymbol{\epsilon}} \sum_{t=1}^T \frac{1}{T} \left\lVert \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_{\theta}(\mathbf{x}, \sqrt{\gamma_t} \mathbf{y}_0 + \sqrt{1 - \gamma_t}\boldsymbol{\epsilon}, \gamma_t) \right\rVert^{2}_{2}$$

where $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. Note that this objective function corresponds to $L_2$ norm in the training objective, and a characterization of $p(\gamma)$ in terms of a uniform distribution over $\{\gamma_1, \ldots, \gamma_T\}$.

### B.2 A Denoising Score-Matching Perspective

Our approach is also linked to denoising score matching (Hyvarinen and Dayan, 2005; Vincent, 2011; Raphan and Simoncelli, 2011; Saremi et al., 2018) for training unnormalized energy functions for density estimation. These methods learn a parametric score function to approximate the gradient of the empirical data log-density. To make sure that the gradient of the data log-density is well-defined, one often replaces each data point with a Gaussian distribution with a small variance. Song and Ermon (2020) advocate for the use of a multi-scale Gaussian mixture as the target density, where each data point is perturbed with different amounts of Gaussian noise, so that Langevin dynamics starting from pure noise can still yield reasonable samples.

One can view our approach as a variant of denoising score matching in which the target density is given by a mixture of $q(\tilde{\mathbf{y}} | \mathbf{y}_0, \gamma) = \mathcal{N}(\tilde{\mathbf{y}} \mid \sqrt{\gamma}\mathbf{y}_0, 1-\gamma)$ for different values of $\mathbf{y}_0$ and $\gamma$. Accordingly, the gradient of data log-density is given by

$$\frac{\mathrm{d} \log q(\tilde{\mathbf{y}} \mid \mathbf{y}_0, \gamma)}{\mathrm{d} \tilde{\mathbf{y}}} = -\frac{\tilde{\mathbf{y}} - \sqrt{\gamma}\mathbf{y}_0}{\sqrt{1-\gamma}} = -\boldsymbol{\epsilon}$$

which is used as the regression target of our model.

## Appendix C: Additional Experimental Results

Additional figures show more examples of SR3 on faces, natural images, and samples from unconditional generative models.

### Images with the Lowest and Highest Fool Rates

In interpreting the fool rate results, it is interesting to inspect those images that maximize the fool rates for a given technique, as well as those images that minimize the fool rate. This provides insight into the nature of the problems that models exhibit, as well as cases in which the model outputs are good enough to regularly fool people.

Images from PULSE for which the fool rate is low have obvious distortions, and the fool rates are lower than 10% for both tasks. For SR3, by comparison, the images with the lowest fool rates are still reasonably good, with much higher fool rates of 14% and 19% in Task-1, and 21% and 26% in Task-2.

The best fool rates for SR3 are 84% and 88%. The corresponding original images are somewhat noisy, and as a consequence, many subjects prefer the SR3 outputs.
