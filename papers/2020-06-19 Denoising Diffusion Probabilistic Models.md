# Denoising Diffusion Probabilistic Models

Jonathan Ho, Ajay Jain, Pieter Abbeel (UC Berkeley)

arXiv: [2006.11239](https://arxiv.org/abs/2006.11239)

[IMAGE: Generated samples on CelebA-HQ 256x256 (left) and unconditional CIFAR10 (right)]

## Abstract

We present high quality image synthesis results using diffusion probabilistic models, a class of latent variable models inspired by considerations from nonequilibrium thermodynamics. Our best results are obtained by training on a weighted variational bound designed according to a novel connection between diffusion probabilistic models and denoising score matching with Langevin dynamics, and our models naturally admit a progressive lossy decompression scheme that can be interpreted as a generalization of autoregressive decoding. On the unconditional CIFAR10 dataset, we obtain an Inception score of 9.46 and a state-of-the-art FID score of 3.17. On 256x256 LSUN, we obtain sample quality similar to ProgressiveGAN. Our implementation is available at https://github.com/hojonathanho/diffusion.

## 1. Introduction

Deep generative models of all kinds have recently exhibited high quality samples in a wide variety of data modalities. Generative adversarial networks (GANs), autoregressive models, flows, and variational autoencoders (VAEs) have synthesized striking image and audio samples [goodfellow2014generative; karras2018progressive; brock2018large; oord2016pixel; menick2018generating; kalchbrenner2017video; dinh2016density; kingma2018glow; prenger2019waveglow; oord2016wavenet; kalchbrenner2018efficient; kingma2013auto; razavi2019generating], and there have been remarkable advances in energy-based modeling and score matching that have produced images comparable to those of GANs [du2019implicit; song2019generative].

This paper presents progress in diffusion probabilistic models [sohl2015deep]. A diffusion probabilistic model (which we will call a "diffusion model" for brevity) is a parameterized Markov chain trained using variational inference to produce samples matching the data after finite time. Transitions of this chain are learned to reverse a diffusion process, which is a Markov chain that gradually adds noise to the data in the opposite direction of sampling until signal is destroyed. When the diffusion consists of small amounts of Gaussian noise, it is sufficient to set the sampling chain transitions to conditional Gaussians too, allowing for a particularly simple neural network parameterization.

Diffusion models are straightforward to define and efficient to train, but to the best of our knowledge, there has been no demonstration that they are capable of generating high quality samples. We show that diffusion models actually are capable of generating high quality samples, sometimes better than the published results on other types of generative models (Section 4). In addition, we show that a certain parameterization of diffusion models reveals an equivalence with denoising score matching over multiple noise levels during training and with annealed Langevin dynamics during sampling (Section 3.2) [song2019generative; vincent2011connection]. We obtained our best sample quality results using this parameterization (Section 4.2), so we consider this equivalence to be one of our primary contributions.

Despite their sample quality, our models do not have competitive log likelihoods compared to other likelihood-based models (our models do, however, have log likelihoods better than the large estimates annealed importance sampling has been reported to produce for energy based models and score matching [du2019implicit; song2019generative]). We find that the majority of our models' lossless codelengths are consumed to describe imperceptible image details (Section 4.3). We present a more refined analysis of this phenomenon in the language of lossy compression, and we show that the sampling procedure of diffusion models is a type of progressive decoding that resembles autoregressive decoding along a bit ordering that vastly generalizes what is normally possible with autoregressive models.

## 2. Background

Diffusion models [sohl2015deep] are latent variable models of the form ```latex $p_\theta(\mathbf{x}_0) \coloneqq\int p_\theta(\mathbf{x}_{0:T}) \,d\mathbf{x}_{1:T}$ ```, where ```latex $\mathbf{x}_1, \dotsc, \mathbf{x}_T$ ``` are latents of the same dimensionality as the data ```latex $\mathbf{x}_0 \sim q(\mathbf{x}_0)$ ```. The joint distribution ```latex $p_\theta(\mathbf{x}_{0:T})$ ``` is called the *reverse process*, and it is defined as a Markov chain with learned Gaussian transitions starting at ```latex $p(\mathbf{x}_T)=\mathcal{N}(\mathbf{x}_T; \mathbf{0}, \mathbf{I})$ ```:

```latex
$$p_\theta(\mathbf{x}_{0:T}) \coloneqq p(\mathbf{x}_T)\prod_{t=1}^T p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t), \qquad p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) \coloneqq\mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))$$
```

What distinguishes diffusion models from other types of latent variable models is that the approximate posterior ```latex $q(\mathbf{x}_{1:T}|\mathbf{x}_0)$ ```, called the *forward process* or *diffusion process*, is fixed to a Markov chain that gradually adds Gaussian noise to the data according to a variance schedule ```latex $\beta_1, \dotsc, \beta_T$ ```:

```latex
$$q(\mathbf{x}_{1:T} | \mathbf{x}_0) \coloneqq\prod_{t=1}^T q(\mathbf{x}_t | \mathbf{x}_{t-1} ), \qquad q(\mathbf{x}_t|\mathbf{x}_{t-1}) \coloneqq\mathcal{N}(\mathbf{x}_t;\sqrt{1-\beta_t}\mathbf{x}_{t-1},\beta_t \mathbf{I})$$
```

[IMAGE: The directed graphical model considered in this work]

Training is performed by optimizing the usual variational bound on negative log likelihood:

```latex
$$\mathbb{E}\left[-\log p_\theta(\mathbf{x}_0)\right] \leq \mathbb{E}_{q}\left[ - \log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} | \mathbf{x}_0)}\right] = \mathbb{E}_q\bigg[ -\log p(\mathbf{x}_T) - \sum_{t \geq 1} \log \frac{p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)}{q(\mathbf{x}_t|\mathbf{x}_{t-1})} \bigg] \eqqcolon L$$
```

The forward process variances ```latex $\beta_t$ ``` can be learned by reparameterization [kingma2013auto] or held constant as hyperparameters, and expressiveness of the reverse process is ensured in part by the choice of Gaussian conditionals in ```latex $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ ```, because both processes have the same functional form when ```latex $\beta_t$ ``` are small [sohl2015deep]. A notable property of the forward process is that it admits sampling ```latex $\mathbf{x}_t$ ``` at an arbitrary timestep ```latex $t$ ``` in closed form: using the notation ```latex $\alpha_t \coloneqq 1-\beta_t$ ``` and ```latex $\bar\alpha_t \coloneqq\prod_{s=1}^t \alpha_s$ ```, we have

```latex
$$q(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar\alpha_t}\mathbf{x}_0, (1-\bar\alpha_t)\mathbf{I})$$
```

Efficient training is therefore possible by optimizing random terms of ```latex $L$ ``` with stochastic gradient descent. Further improvements come from variance reduction by rewriting ```latex $L$ ``` as:

```latex
$$\mathbb{E}_q \bigg[ \underbrace{D_{\mathrm{KL}}(q(\mathbf{x}_T|\mathbf{x}_0) \| p(\mathbf{x}_T))}_{L_T} + \sum_{t > 1} \underbrace{D_{\mathrm{KL}}(q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0) \| p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t))}_{L_{t-1}} \underbrace{-\log p_\theta(\mathbf{x}_0|\mathbf{x}_1)}_{L_0} \bigg]$$
```

(See Appendix A for details. The labels on the terms are used in Section 3.) This equation uses KL divergence to directly compare ```latex $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ ``` against forward process posteriors, which are tractable when conditioned on ```latex $\mathbf{x}_0$ ```:

```latex
$$q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde\beta_t \mathbf{I})$$
```

where

```latex
$$\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) \coloneqq\frac{\sqrt{\bar\alpha_{t-1}}\beta_t }{1-\bar\alpha_t}\mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1- \bar\alpha_{t-1})}{1-\bar\alpha_t} \mathbf{x}_t \quad \text{and} \quad \tilde\beta_t \coloneqq\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$$
```

Consequently, all KL divergences are comparisons between Gaussians, so they can be calculated in a Rao-Blackwellized fashion with closed form expressions instead of high variance Monte Carlo estimates.

## 3. Diffusion models and denoising autoencoders

Diffusion models might appear to be a restricted class of latent variable models, but they allow a large number of degrees of freedom in implementation. One must choose the variances ```latex $\beta_t$ ``` of the forward process and the model architecture and Gaussian distribution parameterization of the reverse process. To guide our choices, we establish a new explicit connection between diffusion models and denoising score matching (Section 3.2) that leads to a simplified, weighted variational bound objective for diffusion models (Section 3.4). Ultimately, our model design is justified by simplicity and empirical results (Section 4). Our discussion is categorized by the terms of the variational bound.

### 3.1. Forward process and L_T

We ignore the fact that the forward process variances ```latex $\beta_t$ ``` are learnable by reparameterization and instead fix them to constants (see Section 4 for details). Thus, in our implementation, the approximate posterior ```latex $q$ ``` has no learnable parameters, so ```latex $L_T$ ``` is a constant during training and can be ignored.

### 3.2. Reverse process and L_{1:T-1}

Now we discuss our choices in ```latex $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))$ ``` for ```latex $1 < t \leq T$ ```. First, we set ```latex $\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) = \sigma_t^2 \mathbf{I}$ ``` to untrained time dependent constants. Experimentally, both ```latex $\sigma_t^2 = \beta_t$ ``` and ```latex $\sigma_t^2 = \tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$ ``` had similar results. The first choice is optimal for ```latex $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ ```, and the second is optimal for ```latex $\mathbf{x}_0$ ``` deterministically set to one point. These are the two extreme choices corresponding to upper and lower bounds on reverse process entropy for data with coordinatewise unit variance [sohl2015deep].

Second, to represent the mean ```latex $\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)$ ```, we propose a specific parameterization motivated by the following analysis of ```latex $L_t$ ```. With ```latex $p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \sigma_t^2\mathbf{I})$ ```, we can write:

```latex
$$L_{t-1} = \mathbb{E}_{q}\left[ \frac{1}{2\sigma_t^2} \|\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t,\mathbf{x}_0) - \boldsymbol{\mu}_\theta(\mathbf{x}_t, t)\|^2 \right] + C$$
```

where ```latex $C$ ``` is a constant that does not depend on ```latex $\theta$ ```. So, we see that the most straightforward parameterization of ```latex $\boldsymbol{\mu}_\theta$ ``` is a model that predicts ```latex $\tilde{\boldsymbol{\mu}}_t$ ```, the forward process posterior mean. However, we can expand further by reparameterizing as ```latex $\mathbf{x}_t(\mathbf{x}_0, \boldsymbol{\epsilon}) = \sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol{\epsilon}$ ``` for ```latex $\boldsymbol{\epsilon}\sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ ``` and applying the forward process posterior formula:

```latex
$$L_{t-1} - C = \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}}\left[ \frac{1}{2\sigma_t^2} \left\| \frac{1}{\sqrt{\alpha_t}}\left( \mathbf{x}_t(\mathbf{x}_0,\boldsymbol{\epsilon}) - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\boldsymbol{\epsilon}\right) - \boldsymbol{\mu}_\theta(\mathbf{x}_t(\mathbf{x}_0,\boldsymbol{\epsilon}), t) \right\|^2 \right]$$
```

This reveals that ```latex $\boldsymbol{\mu}_\theta$ ``` must predict ```latex $\frac{1}{\sqrt{\alpha_t}}\left( \mathbf{x}_t  - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\boldsymbol{\epsilon}\right)$ ``` given ```latex $\mathbf{x}_t$ ```. Since ```latex $\mathbf{x}_t$ ``` is available as input to the model, we may choose the parameterization

```latex
$$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right)$$
```

where ```latex $\boldsymbol{\epsilon}_\theta$ ``` is a function approximator intended to predict ```latex $\boldsymbol{\epsilon}$ ``` from ```latex $\mathbf{x}_t$ ```. To sample ```latex $\mathbf{x}_{t-1} \sim p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ ``` is to compute ```latex $\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right) + \sigma_t \mathbf{z}$ ```, where ```latex $\mathbf{z}\sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ ```. The complete sampling procedure (Algorithm 2) resembles Langevin dynamics with ```latex $\boldsymbol{\epsilon}_\theta$ ``` as a learned gradient of the data density. Furthermore, with this parameterization, the loss simplifies to:

```latex
$$\mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}}\left[ \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1-\bar\alpha_t)}  \left\| \boldsymbol{\epsilon}- \boldsymbol{\epsilon}_\theta(\sqrt{\bar\alpha_t} \mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol{\epsilon}, t) \right\|^2\right]$$
```

which resembles denoising score matching over multiple noise scales indexed by ```latex $t$ ``` [song2019generative]. As this expression is equal to (one term of) the variational bound for the Langevin-like reverse process, we see that optimizing an objective resembling denoising score matching is equivalent to using variational inference to fit the finite-time marginal of a sampling chain resembling Langevin dynamics.

To summarize, we can train the reverse process mean function approximator ```latex $\boldsymbol{\mu}_\theta$ ``` to predict ```latex $\tilde{\boldsymbol{\mu}}_t$ ```, or by modifying its parameterization, we can train it to predict ```latex $\boldsymbol{\epsilon}$ ```. (There is also the possibility of predicting ```latex $\mathbf{x}_0$ ```, but we found this to lead to worse sample quality early in our experiments.) We have shown that the ```latex $\boldsymbol{\epsilon}$ ```-prediction parameterization both resembles Langevin dynamics and simplifies the diffusion model's variational bound to an objective that resembles denoising score matching. Nonetheless, it is just another parameterization of ```latex $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ ```, so we verify its effectiveness in Section 4 in an ablation where we compare predicting ```latex $\boldsymbol{\epsilon}$ ``` against predicting ```latex $\tilde{\boldsymbol{\mu}}_t$ ```.

**Algorithm 1: Training**

1. Repeat:
   - ```latex $\mathbf{x}_0 \sim q(\mathbf{x}_0)$ ```
   - ```latex $t \sim \mathrm{Uniform}(\{1, \dotsc, T\})$ ```
   - ```latex $\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ ```
   - Take gradient descent step on ```latex $\nabla_\theta \left\| \boldsymbol{\epsilon}- \boldsymbol{\epsilon}_\theta(\sqrt{\bar\alpha_t} \mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol{\epsilon}, t) \right\|^2$ ```
2. Until converged

**Algorithm 2: Sampling**

1. ```latex $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ ```
2. For ```latex $t=T, \dotsc, 1$ ```:
   - ```latex $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ ``` if ```latex $t > 1$ ```, else ```latex $\mathbf{z} = \mathbf{0}$ ```
   - ```latex $\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left( \mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right) + \sigma_t \mathbf{z}$ ```
3. Return ```latex $\mathbf{x}_0$ ```

### 3.3. Data scaling, reverse process decoder, and L_0

We assume that image data consists of integers in ```latex $\{ 0, 1, \dotsc, 255\}$ ``` scaled linearly to ```latex $[-1, 1]$ ```. This ensures that the neural network reverse process operates on consistently scaled inputs starting from the standard normal prior ```latex $p(\mathbf{x}_T)$ ```. To obtain discrete log likelihoods, we set the last term of the reverse process to an independent discrete decoder derived from the Gaussian ```latex $\mathcal{N}(\mathbf{x}_0 ; \boldsymbol{\mu}_\theta(\mathbf{x}_1, 1), \sigma_1^2 \mathbf{I})$ ```:

```latex
$$p_\theta(\mathbf{x}_0 | \mathbf{x}_1) = \prod_{i=1}^D \int_{\delta_{-}(x_0^i)}^{\delta_{+}(x_0^i)} \mathcal{N}(x; \mu_\theta^i(\mathbf{x}_1, 1), \sigma_1^2) \, dx$$
```

```latex
$$\delta_{+}(x) = \begin{cases} \infty & \text{if}\ x=1 \\ x+\frac{1}{255} & \text{if}\ x < 1 \end{cases} \qquad \delta_{-}(x) = \begin{cases} -\infty & \text{if}\ x=-1 \\ x-\frac{1}{255} & \text{if}\ x > -1 \end{cases}$$
```

where ```latex $D$ ``` is the data dimensionality and the ```latex $i$ ``` superscript indicates extraction of one coordinate. (It would be straightforward to instead incorporate a more powerful decoder like a conditional autoregressive model, but we leave that to future work.) Similar to the discretized continuous distributions used in VAE decoders and autoregressive models [kingma2016improved; salimans2017pixelcnn++], our choice here ensures that the variational bound is a lossless codelength of discrete data, without need of adding noise to the data or incorporating the Jacobian of the scaling operation into the log likelihood. At the end of sampling, we display ```latex $\boldsymbol{\mu}_\theta(\mathbf{x}_1,1)$ ``` noiselessly.

### 3.4. Simplified training objective

With the reverse process and decoder defined above, the variational bound is clearly differentiable with respect to ```latex $\theta$ ``` and is ready to be employed for training. However, we found it beneficial to sample quality (and simpler to implement) to train on the following variant of the variational bound:

```latex
$$L_\mathrm{simple}(\theta) \coloneqq\mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[ \left\| \boldsymbol{\epsilon}- \boldsymbol{\epsilon}_\theta(\sqrt{\bar\alpha_t} \mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol{\epsilon}, t) \right\|^2\right]$$
```

where ```latex $t$ ``` is uniform between 1 and ```latex $T$ ```. The ```latex $t=1$ ``` case corresponds to ```latex $L_0$ ``` with the integral in the discrete decoder definition approximated by the Gaussian probability density function times the bin width, ignoring ```latex $\sigma_1^2$ ``` and edge effects. The ```latex $t > 1$ ``` cases correspond to an unweighted version of the loss, analogous to the loss weighting used by the NCSN denoising score matching model [song2019generative]. (```latex $L_T$ ``` does not appear because the forward process variances ```latex $\beta_t$ ``` are fixed.) Algorithm 1 displays the complete training procedure with this simplified objective.

Since our simplified objective discards the weighting, it is a weighted variational bound that emphasizes different aspects of reconstruction compared to the standard variational bound [gregor2016towards; higgins2017beta]. In particular, our diffusion process setup in Section 4 causes the simplified objective to down-weight loss terms corresponding to small ```latex $t$ ```. These terms train the network to denoise data with very small amounts of noise, so it is beneficial to down-weight them so that the network can focus on more difficult denoising tasks at larger ```latex $t$ ``` terms. We will see in our experiments that this reweighting leads to better sample quality.

## 4. Experiments

We set ```latex $T=1000$ ``` for all experiments so that the number of neural network evaluations needed during sampling matches previous work [sohl2015deep; song2019generative]. We set the forward process variances to constants increasing linearly from ```latex $\beta_1 = 10^{-4}$ ``` to ```latex $\beta_T = 0.02$ ```. These constants were chosen to be small relative to data scaled to ```latex $[-1, 1]$ ```, ensuring that reverse and forward processes have approximately the same functional form while keeping the signal-to-noise ratio at ```latex $\mathbf{x}_T$ ``` as small as possible (```latex $L_T = D_{\mathrm{KL}}(q(\mathbf{x}_T|\mathbf{x}_0) \| \mathcal{N}(\mathbf{0}, \mathbf{I})) \approx 10^{-5}$ ``` bits per dimension in our experiments).

To represent the reverse process, we use a U-Net backbone similar to an unmasked PixelCNN++ [salimans2017pixelcnn++; ronneberger2015unet] with group normalization throughout [wu2018group]. Parameters are shared across time, which is specified to the network using the Transformer sinusoidal position embedding [vaswani2017attention]. We use self-attention at the ```latex $16 \times 16$ ``` feature map resolution [wang2018non; vaswani2017attention]. Details are in Appendix B.

### 4.1. Sample quality

**Table 1: CIFAR10 results. NLL measured in bits/dim.**

| Model | IS | FID | NLL Test (Train) |
|---|---|---|---|
| **Conditional** | | | |
| EBM [du2019implicit] | 8.30 | 37.9 | |
| JEM [grathwohl2020your] | 8.76 | 38.4 | |
| BigGAN [brock2018large] | 9.22 | 14.73 | |
| StyleGAN2 + ADA (v1) [karras2020training] | **10.06** | **2.67** | |
| **Unconditional** | | | |
| Diffusion (original) [sohl2015deep] | | | <= 5.40 |
| Gated PixelCNN [oord2016conditional] | 4.60 | 65.93 | 3.03 (2.90) |
| Sparse Transformer [child2019generating] | | | **2.80** |
| PixelIQN [ostrovski2018autoregressive] | 5.29 | 49.46 | |
| EBM [du2019implicit] | 6.78 | 38.2 | |
| NCSNv2 [song2020improved] | | 31.75 | |
| NCSN [song2019generative] | 8.87 +/- 0.12 | 25.32 | |
| SNGAN [miyato2018spectral] | 8.22 +/- 0.05 | 21.7 | |
| SNGAN-DDLS [che2020your] | 9.09 +/- 0.10 | 15.42 | |
| StyleGAN2 + ADA (v1) [karras2020training] | **9.74** +/- 0.05 | 3.26 | |
| Ours (L, fixed isotropic Sigma) | 7.67 +/- 0.13 | 13.51 | <= 3.70 (3.69) |
| **Ours (L_simple)** | 9.46 +/- 0.11 | **3.17** | <= 3.75 (3.72) |

**Table 2: Unconditional CIFAR10 reverse process parameterization and training objective ablation.** Blank entries were unstable to train and generated poor samples with out-of-range scores.

| Objective | IS | FID |
|---|---|---|
| **mu-tilde prediction (baseline)** | | |
| L, learned diagonal Sigma | 7.28 +/- 0.10 | 23.69 |
| L, fixed isotropic Sigma | 8.06 +/- 0.09 | 13.22 |
| \|\|mu-tilde - mu-tilde_theta\|\|^2 | -- | -- |
| **epsilon prediction (ours)** | | |
| L, learned diagonal Sigma | -- | -- |
| L, fixed isotropic Sigma | 7.67 +/- 0.13 | 13.51 |
| \|\|epsilon-tilde - epsilon_theta\|\|^2 (L_simple) | **9.46 +/- 0.11** | **3.17** |

Table 1 shows Inception scores, FID scores, and negative log likelihoods (lossless codelengths) on CIFAR10. With our FID score of 3.17, our unconditional model achieves better sample quality than most models in the literature, including class conditional models. Our FID score is computed with respect to the training set, as is standard practice; when we compute it with respect to the test set, the score is 5.24, which is still better than many of the training set FID scores in the literature.

We find that training our models on the true variational bound yields better codelengths than training on the simplified objective, as expected, but the latter yields the best sample quality.

[IMAGE: LSUN Church samples. FID=7.89]

[IMAGE: LSUN Bedroom samples. FID=4.90]

### 4.2. Reverse process parameterization and training objective ablation

In Table 2, we show the sample quality effects of reverse process parameterizations and training objectives (Section 3.2). We find that the baseline option of predicting ```latex $\tilde{\boldsymbol{\mu}}$ ``` works well only when trained on the true variational bound instead of unweighted mean squared error, a simplified objective akin to ```latex $L_\mathrm{simple}$ ```. We also see that learning reverse process variances (by incorporating a parameterized diagonal ```latex $\boldsymbol{\Sigma}_\theta(\mathbf{x}_t)$ ``` into the variational bound) leads to unstable training and poorer sample quality compared to fixed variances. Predicting ```latex $\boldsymbol{\epsilon}$ ```, as we proposed, performs approximately as well as predicting ```latex $\tilde{\boldsymbol{\mu}}$ ``` when trained on the variational bound with fixed variances, but much better when trained with our simplified objective.

### 4.3. Progressive coding

Table 1 also shows the codelengths of our CIFAR10 models. The gap between train and test is at most 0.03 bits per dimension, which is comparable to the gaps reported with other likelihood-based models and indicates that our diffusion model is not overfitting. Still, while our lossless codelengths are better than the large estimates reported for energy based models and score matching using annealed importance sampling [du2019implicit], they are not competitive with other types of likelihood-based generative models [child2019generating].

Since our samples are nonetheless of high quality, we conclude that diffusion models have an inductive bias that makes them excellent lossy compressors. Treating the variational bound terms ```latex $L_1 + \cdots + L_T$ ``` as rate and ```latex $L_0$ ``` as distortion, our CIFAR10 model with the highest quality samples has a rate of **1.78** bits/dim and a distortion of **1.97** bits/dim, which amounts to a root mean squared error of 0.95 on a scale from 0 to 255. More than half of the lossless codelength describes imperceptible distortions.

#### Progressive lossy compression

We can probe further into the rate-distortion behavior of our model by introducing a progressive lossy code that mirrors the form of the variational bound: see Algorithms 3-4, which assume access to a procedure, such as minimal random coding [harsha2007communication; havasi2018minimal], that can transmit a sample ```latex $\mathbf{x}\sim q(\mathbf{x})$ ``` using approximately ```latex $D_{\mathrm{KL}}(q(\mathbf{x}) \| p(\mathbf{x}))$ ``` bits on average for any distributions ```latex $p$ ``` and ```latex $q$ ```, for which only ```latex $p$ ``` is available to the receiver beforehand.

**Algorithm 3: Sending x_0**

1. Send ```latex $\mathbf{x}_T \sim q(\mathbf{x}_T|\mathbf{x}_0)$ ``` using ```latex $p(\mathbf{x}_T)$ ```
2. For ```latex $t=T-1, \dotsc, 2, 1$ ```:
   - Send ```latex $\mathbf{x}_t \sim q(\mathbf{x}_t|\mathbf{x}_{t+1}, \mathbf{x}_0)$ ``` using ```latex $p_\theta(\mathbf{x}_t | \mathbf{x}_{t+1})$ ```
3. Send ```latex $\mathbf{x}_0$ ``` using ```latex $p_\theta(\mathbf{x}_0|\mathbf{x}_1)$ ```

**Algorithm 4: Receiving**

1. Receive ```latex $\mathbf{x}_T$ ``` using ```latex $p(\mathbf{x}_T)$ ```
2. For ```latex $t=T-1, \dotsc, 1, 0$ ```:
   - Receive ```latex $\mathbf{x}_t$ ``` using ```latex $p_\theta(\mathbf{x}_t | \mathbf{x}_{t+1})$ ```
3. Return ```latex $\mathbf{x}_0$ ```

When applied to ```latex $\mathbf{x}_0 \sim q(\mathbf{x}_0)$ ```, Algorithms 3-4 transmit ```latex $\mathbf{x}_T, \dotsc, \mathbf{x}_0$ ``` in sequence using a total expected codelength equal to the variational bound. The receiver, at any time ```latex $t$ ```, has the partial information ```latex $\mathbf{x}_t$ ``` fully available and can progressively estimate:

```latex
$$\mathbf{x}_0 \approx \hat\mathbf{x}_0 = \left( \mathbf{x}_t - \sqrt{1-\bar\alpha_t}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t) \right) / \sqrt{\bar\alpha_t}$$
```

(A stochastic reconstruction ```latex $\mathbf{x}_0 \sim p_\theta(\mathbf{x}_0|\mathbf{x}_t)$ ``` is also valid, but we do not consider it here because it makes distortion more difficult to evaluate.)

[IMAGE: Unconditional CIFAR10 test set rate-distortion vs. time. Distortion is measured in root mean squared error on a [0,255] scale.]

#### Progressive generation

We also run a progressive unconditional generation process given by progressive decompression from random bits. In other words, we predict the result of the reverse process, ```latex $\hat\mathbf{x}_0$ ```, while sampling from the reverse process using Algorithm 2. Large scale image features appear first and details appear last. Stochastic predictions ```latex $\mathbf{x}_0 \sim p_\theta(\mathbf{x}_0|\mathbf{x}_t)$ ``` with ```latex $\mathbf{x}_t$ ``` frozen for various ```latex $t$ ``` show that when ```latex $t$ ``` is small, all but fine details are preserved, and when ```latex $t$ ``` is large, only large scale features are preserved. Perhaps these are hints of conceptual compression [gregor2016towards].

[IMAGE: Unconditional CIFAR10 progressive generation (x-hat_0 over time, from left to right)]

[IMAGE: When conditioned on the same latent, CelebA-HQ 256x256 samples share high-level attributes. Bottom-right quadrants are x_t, and other quadrants are samples from p_theta(x_0 | x_t).]

#### Connection to autoregressive decoding

Note that the variational bound can be rewritten as:

```latex
$$L = D_{\mathrm{KL}}(q(\mathbf{x}_T) \| p(\mathbf{x}_T)) + \mathbb{E}_{q}\Bigg[ \sum_{t \geq 1} D_{\mathrm{KL}}(q(\mathbf{x}_{t-1}|\mathbf{x}_t) \| p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)) \Bigg] + H(\mathbf{x}_0)$$
```

Now consider setting the diffusion process length ```latex $T$ ``` to the dimensionality of the data, defining the forward process so that ```latex $q(\mathbf{x}_t|\mathbf{x}_0)$ ``` places all probability mass on ```latex $\mathbf{x}_0$ ``` with the first ```latex $t$ ``` coordinates masked out (i.e. ```latex $q(\mathbf{x}_t|\mathbf{x}_{t-1})$ ``` masks out the ```latex $t$ ```th coordinate), setting ```latex $p(\mathbf{x}_T)$ ``` to place all mass on a blank image, and, for the sake of argument, taking ```latex $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ ``` to be a fully expressive conditional distribution. With these choices, ```latex $D_{\mathrm{KL}}(q(\mathbf{x}_T) \| p(\mathbf{x}_T))=0$ ```, and minimizing ```latex $D_{\mathrm{KL}}(q(\mathbf{x}_{t-1}|\mathbf{x}_t) \| p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t))$ ``` trains ```latex $p_\theta$ ``` to copy coordinates ```latex $t+1, \dotsc, T$ ``` unchanged and to predict the ```latex $t$ ```th coordinate given ```latex $t+1, \dotsc, T$ ```. Thus, training ```latex $p_\theta$ ``` with this particular diffusion is training an autoregressive model.

We can therefore interpret the Gaussian diffusion model as a kind of autoregressive model with a generalized bit ordering that cannot be expressed by reordering data coordinates. Prior work has shown that such reorderings introduce inductive biases that have an impact on sample quality [menick2018generating], so we speculate that the Gaussian diffusion serves a similar purpose, perhaps to greater effect since Gaussian noise might be more natural to add to images compared to masking noise. Moreover, the Gaussian diffusion length is not restricted to equal the data dimension; for instance, we use ```latex $T=1000$ ```, which is less than the dimension of the ```latex $32\times 32 \times 3$ ``` or ```latex $256 \times 256 \times 3$ ``` images in our experiments. Gaussian diffusions can be made shorter for fast sampling or longer for model expressiveness.

### 4.4. Interpolation

We can interpolate source images ```latex $\mathbf{x}_0, \mathbf{x}'_0 \sim q(\mathbf{x}_0)$ ``` in latent space using ```latex $q$ ``` as a stochastic encoder, ```latex $\mathbf{x}_t, \mathbf{x}'_t \sim q(\mathbf{x}_t | \mathbf{x}_0)$ ```, then decoding the linearly interpolated latent ```latex $\bar{\mathbf{x}}_t = (1-\lambda) \mathbf{x}_0 + \lambda \mathbf{x}'_0$ ``` into image space by the reverse process, ```latex $\bar{\mathbf{x}}_0 \sim p(\mathbf{x}_0 | \bar{\mathbf{x}}_t)$ ```. In effect, we use the reverse process to remove artifacts from linearly interpolating corrupted versions of the source images. The reverse process produces high-quality reconstructions, and plausible interpolations that smoothly vary attributes such as pose, skin tone, hairstyle, expression and background, but not eyewear. Larger ```latex $t$ ``` results in coarser and more varied interpolations, with novel samples at ```latex $t=1000$ ```.

[IMAGE: Interpolations of CelebA-HQ 256x256 images with 500 timesteps of diffusion]

## 5. Related Work

While diffusion models might resemble flows [dinh2014nice; rezende2015variational; dinh2016density; kingma2018glow; chen2018neural; grathwohl2019ffjord; ho2019flow++] and VAEs [kingma2013auto; rezende2014stochastic; maaloe2019biva], diffusion models are designed so that ```latex $q$ ``` has no parameters and the top-level latent ```latex $\mathbf{x}_T$ ``` has nearly zero mutual information with the data ```latex $\mathbf{x}_0$ ```. Our ```latex $\boldsymbol{\epsilon}$ ```-prediction reverse process parameterization establishes a connection between diffusion models and denoising score matching over multiple noise levels with annealed Langevin dynamics for sampling [song2019generative; song2020improved]. Diffusion models, however, admit straightforward log likelihood evaluation, and the training procedure explicitly trains the Langevin dynamics sampler using variational inference. The connection also has the reverse implication that a certain weighted form of denoising score matching is the same as variational inference to train a Langevin-like sampler. Other methods for learning transition operators of Markov chains include infusion training [bordes2016learning], variational walkback [goyal2017variational], generative stochastic networks [alain2016gsns], and others [salimans2015markov; song2017nice; levy2018generalizing; nijkamp2019learning; lawson2019energy; wu2020stochastic].

By the known connection between score matching and energy-based modeling, our work could have implications for other recent work on energy-based models [xie2016theory; xie2017synthesizing; xie2018learning; gao2018learning; xie2019learning; gao2020flow; du2019implicit; nijkamp2019anatomy; grathwohl2020your; deng2020residual]. Our rate-distortion curves are computed over time in one evaluation of the variational bound, reminiscent of how rate-distortion curves can be computed over distortion penalties in one run of annealed importance sampling [huang2020evaluating]. Our progressive decoding argument can be seen in convolutional DRAW and related models [gregor2016towards; nichol2020vq] and may also lead to more general designs for subscale orderings or sampling strategies for autoregressive models [menick2018generating; wiggers2020predictive].

## 6. Conclusion

We have presented high quality image samples using diffusion models, and we have found connections among diffusion models and variational inference for training Markov chains, denoising score matching and annealed Langevin dynamics (and energy-based models by extension), autoregressive models, and progressive lossy compression. Since diffusion models seem to have excellent inductive biases for image data, we look forward to investigating their utility in other data modalities and as components in other types of generative models and machine learning systems.

## Broader Impact

Our work on diffusion models takes on a similar scope as existing work on other types of deep generative models, such as efforts to improve the sample quality of GANs, flows, autoregressive models, and so forth. Our paper represents progress in making diffusion models a generally useful tool in this family of techniques, so it may serve to amplify any impacts that generative models have had (and will have) on the broader world.

Unfortunately, there are numerous well-known malicious uses of generative models. Sample generation techniques can be employed to produce fake images and videos of high profile figures for political purposes. While fake images were manually created long before software tools were available, generative models such as ours make the process easier. Fortunately, CNN-generated images currently have subtle flaws that allow detection [wang2019cnngenerated], but improvements in generative models may make this more difficult. Generative models also reflect the biases in the datasets on which they are trained. As many large datasets are collected from the internet by automated systems, it can be difficult to remove these biases, especially when the images are unlabeled. If samples from generative models trained on these datasets proliferate throughout the internet, then these biases will only be reinforced further.

On the other hand, diffusion models may be useful for data compression, which, as data becomes higher resolution and as global internet traffic increases, might be crucial to ensure accessibility of the internet to wide audiences. Our work might contribute to representation learning on unlabeled raw data for a large range of downstream tasks, from image classification to reinforcement learning, and diffusion models might also become viable for creative uses in art, photography, and music.

## Appendix A: Extended derivations

Below is a derivation of the reduced variance variational bound for diffusion models. This material is from [sohl2015deep]; we include it here only for completeness.

```latex
$$\begin{aligned}
L &= \mathbb{E}_{q}\left[ - \log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} | \mathbf{x}_0)}\right] \\
  &= \mathbb{E}_{q}\left[ -\log p(\mathbf{x}_T) - \sum_{t \geq 1} \log \frac{p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)}{q(\mathbf{x}_t|\mathbf{x}_{t-1})} \right] \\
  &= \mathbb{E}_{q}\left[ -\log p(\mathbf{x}_T) - \sum_{t > 1} \log \frac{p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)}{q(\mathbf{x}_t|\mathbf{x}_{t-1})} - \log\frac{p_\theta(\mathbf{x}_0|\mathbf{x}_1)}{q(\mathbf{x}_1|\mathbf{x}_0)} \right] \\
  &= \mathbb{E}_{q}\left[ -\log p(\mathbf{x}_T) - \sum_{t > 1} \log \frac{p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)}{q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)}\cdot\frac{q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_t|\mathbf{x}_0)} - \log\frac{p_\theta(\mathbf{x}_0|\mathbf{x}_1)}{q(\mathbf{x}_1|\mathbf{x}_0)} \right] \\
  &= \mathbb{E}_{q}\left[ -\log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T|\mathbf{x}_0)} - \sum_{t > 1} \log \frac{p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)}{q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)} - \log p_\theta(\mathbf{x}_0|\mathbf{x}_1) \right] \\
  &= \mathbb{E}_{q}\left[ D_{\mathrm{KL}}(q(\mathbf{x}_T|\mathbf{x}_0) \| p(\mathbf{x}_T)) + \sum_{t > 1} D_{\mathrm{KL}}(q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0) \| p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)) - \log p_\theta(\mathbf{x}_0|\mathbf{x}_1) \right]
\end{aligned}$$
```

The following is an alternate version of ```latex $L$ ```. It is not tractable to estimate, but it is useful for the discussion on progressive coding:

```latex
$$\begin{aligned}
L &= \mathbb{E}_{q}\left[ -\log p(\mathbf{x}_T) - \sum_{t \geq 1} \log \frac{p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)}{q(\mathbf{x}_t|\mathbf{x}_{t-1})} \right] \\
  &= \mathbb{E}_{q}\left[ -\log p(\mathbf{x}_T) - \sum_{t \geq 1} \log \frac{p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)}{q(\mathbf{x}_{t-1}|\mathbf{x}_t)} \cdot \frac{q(\mathbf{x}_{t-1})}{q(\mathbf{x}_t)} \right] \\
  &= \mathbb{E}_{q}\left[ -\log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T)} - \sum_{t \geq 1} \log \frac{p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)}{q(\mathbf{x}_{t-1}|\mathbf{x}_t)} -\log q(\mathbf{x}_0) \right] \\
  &= D_{\mathrm{KL}}(q(\mathbf{x}_T) \| p(\mathbf{x}_T)) + \mathbb{E}_{q}\left[ \sum_{t \geq 1} D_{\mathrm{KL}}(q(\mathbf{x}_{t-1}|\mathbf{x}_t) \| p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)) \right] + H(\mathbf{x}_0)
\end{aligned}$$
```

## Appendix B: Experimental details

Our neural network architecture follows the backbone of PixelCNN++ [salimans2017pixelcnn++], which is a U-Net [ronneberger2015unet] based on a Wide ResNet [zagoruyko2016wide]. We replaced weight normalization [salimans2016weight] with group normalization [wu2018group] to make the implementation simpler. Our ```latex $32 \times 32$ ``` models use four feature map resolutions (```latex $32 \times 32$ ``` to ```latex $4 \times 4$ ```), and our ```latex $256 \times 256$ ``` models use six. All models have two convolutional residual blocks per resolution level and self-attention blocks at the ```latex $16 \times 16$ ``` resolution between the convolutional blocks [chen2018pixelsnail]. Diffusion time ```latex $t$ ``` is specified by adding the Transformer sinusoidal position embedding [vaswani2017attention] into each residual block. Our CIFAR10 model has 35.7 million parameters, and our LSUN and CelebA-HQ models have 114 million parameters. We also trained a larger variant of the LSUN Bedroom model with approximately 256 million parameters by increasing filter count.

We used TPU v3-8 (similar to 8 V100 GPUs) for all experiments. Our CIFAR model trains at 21 steps per second at batch size 128 (10.6 hours to train to completion at 800k steps), and sampling a batch of 256 images takes 17 seconds. Our CelebA-HQ/LSUN (256^2) models train at 2.2 steps per second at batch size 64, and sampling a batch of 128 images takes 300 seconds. We trained on CelebA-HQ for 0.5M steps, LSUN Bedroom for 2.4M steps, LSUN Cat for 1.8M steps, and LSUN Church for 1.2M steps. The larger LSUN Bedroom model was trained for 1.15M steps.

Apart from an initial choice of hyperparameters early on to make network size fit within memory constraints, we performed the majority of our hyperparameter search to optimize for CIFAR10 sample quality, then transferred the resulting settings over to the other datasets:

- We chose the ```latex $\beta_t$ ``` schedule from a set of constant, linear, and quadratic schedules, all constrained so that ```latex $L_T \approx 0$ ```. We set ```latex $T=1000$ ``` without a sweep, and we chose a linear schedule from ```latex $\beta_1=10^{-4}$ ``` to ```latex $\beta_T=0.02$ ```.
- We set the dropout rate on CIFAR10 to 0.1 by sweeping over the values {0.1, 0.2, 0.3, 0.4}. Without dropout on CIFAR10, we obtained poorer samples reminiscent of the overfitting artifacts in an unregularized PixelCNN++ [salimans2017pixelcnn++]. We set dropout rate on the other datasets to zero without sweeping.
- We used random horizontal flips during training for CIFAR10; we tried training both with and without flips, and found flips to improve sample quality slightly. We also used random horizontal flips for all other datasets except LSUN Bedroom.
- We tried Adam [kingma2014adam] and RMSProp early on in our experimentation process and chose the former. We left the hyperparameters to their standard values. We set the learning rate to ```latex $2 \times 10^{-4}$ ``` without any sweeping, and we lowered it to ```latex $2 \times 10^{-5}$ ``` for the ```latex $256\times256$ ``` images, which seemed unstable to train with the larger learning rate.
- We set the batch size to 128 for CIFAR10 and 64 for larger images. We did not sweep over these values.
- We used EMA on model parameters with a decay factor of 0.9999. We did not sweep over this value.

Final experiments were trained once and evaluated throughout training for sample quality. Sample quality scores and log likelihood are reported on the minimum FID value over the course of training. On CIFAR10, we calculated Inception and FID scores on 50000 samples using the original code from the OpenAI [salimans2016improved] and TTUR [heusel2017gans] repositories, respectively. On LSUN, we calculated FID scores on 50000 samples using code from the StyleGAN2 [karras2019analyzing] repository. CIFAR10 and CelebA-HQ were loaded as provided by TensorFlow Datasets (https://www.tensorflow.org/datasets), and LSUN was prepared using code from StyleGAN. Dataset splits (or lack thereof) are standard from the papers that introduced their usage in a generative modeling context. All details can be found in the source code release.

## Appendix C: Discussion on related work

Our model architecture, forward process definition, and prior differ from NCSN [song2019generative; song2020improved] in subtle but important ways that improve sample quality, and, notably, we directly train our sampler as a latent variable model rather than adding it after training post-hoc. In greater detail:

1. We use a U-Net with self-attention; NCSN uses a RefineNet with dilated convolutions. We condition all layers on ```latex $t$ ``` by adding in the Transformer sinusoidal position embedding, rather than only in normalization layers (NCSNv1) or only at the output (v2).

2. Diffusion models scale down the data with each forward process step (by a ```latex $\sqrt{1-\beta_t}$ ``` factor) so that variance does not grow when adding noise, thus providing consistently scaled inputs to the neural net reverse process. NCSN omits this scaling factor.

3. Unlike NCSN, our forward process destroys signal (```latex $D_{\mathrm{KL}}(q(\mathbf{x}_T|\mathbf{x}_0) \| \mathcal{N}(\mathbf{0},\mathbf{I})) \approx 0$ ```), ensuring a close match between the prior and aggregate posterior of ```latex $\mathbf{x}_T$ ```. Also unlike NCSN, our ```latex $\beta_t$ ``` are very small, which ensures that the forward process is reversible by a Markov chain with conditional Gaussians. Both of these factors prevent distribution shift when sampling.

4. Our Langevin-like sampler has coefficients (learning rate, noise scale, etc.) derived rigorously from ```latex $\beta_t$ ``` in the forward process. Thus, our training procedure directly trains our sampler to match the data distribution after ```latex $T$ ``` steps: it trains the sampler as a latent variable model using variational inference. In contrast, NCSN's sampler coefficients are set by hand post-hoc, and their training procedure is not guaranteed to directly optimize a quality metric of their sampler.

## Appendix D: Additional information

**Table 3: FID scores for LSUN 256x256 datasets**

| Model | LSUN Bedroom | LSUN Church | LSUN Cat |
|---|---|---|---|
| ProgressiveGAN [karras2018progressive] | 8.34 | 6.42 | 37.52 |
| StyleGAN [karras2019style] | **2.65** | 4.21* | 8.53* |
| StyleGAN2 [karras2019analyzing] | - | **3.86** | **6.93** |
| Ours (L_simple) | 6.36 | 7.89 | 19.75 |
| Ours (L_simple, large) | 4.90 | - | - |

#### Progressive compression

Our lossy compression argument in Section 4.3 is only a proof of concept, because Algorithms 3-4 depend on a procedure such as minimal random coding [havasi2018minimal], which is not tractable for high dimensional data. These algorithms serve as a compression interpretation of the variational bound of [sohl2015deep], not yet as a practical compression system.

**Table 4: Unconditional CIFAR10 test set rate-distortion values**

| Reverse process time (T-t+1) | Rate (bits/dim) | Distortion (RMSE [0, 255]) |
|---|---|---|
| 1000 | 1.77581 | 0.95136 |
| 900 | 0.11994 | 12.02277 |
| 800 | 0.05415 | 18.47482 |
| 700 | 0.02866 | 24.43656 |
| 600 | 0.01507 | 30.80948 |
| 500 | 0.00716 | 38.03236 |
| 400 | 0.00282 | 46.12765 |
| 300 | 0.00081 | 54.18826 |
| 200 | 0.00013 | 60.97170 |
| 100 | 0.00000 | 67.60125 |

## Appendix E: Samples

#### Latent structure and reverse process stochasticity

During sampling, both the prior ```latex $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ ``` and Langevin dynamics are stochastic. To understand the significance of the second source of noise, we sampled multiple images conditioned on the same intermediate latent for the CelebA ```latex $256 \times 256$ ``` dataset. Multiple draws from the reverse process ```latex $\mathbf{x}_0 \sim p_\theta(\mathbf{x}_0 | \mathbf{x}_t)$ ``` that share the latent ```latex $\mathbf{x}_t$ ``` for ```latex $t \in \{1000, 750, 500, 250\}$ ``` show that when the chain is split after the prior draw at ```latex $\mathbf{x}_{T=1000}$ ```, the samples differ significantly. However, when the chain is split after more steps, samples share high-level attributes like gender, hair color, eyewear, saturation, pose and facial expression. This indicates that intermediate latents like ```latex $\mathbf{x}_{750}$ ``` encode these attributes, despite their imperceptibility.

#### Coarse-to-fine interpolation

Interpolations between a pair of source CelebA ```latex $256\times256$ ``` images show that increasing the number of diffusion steps destroys more structure in the source images, which the model completes during the reverse process. This allows interpolation at both fine granularities and coarse granularities. In the limiting case of 0 diffusion steps, the interpolation mixes source images in pixel space. On the other hand, after 1000 diffusion steps, source information is lost and interpolations are novel samples.
