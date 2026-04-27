# Abstract

Classifier guidance is a recently introduced method to trade off mode coverage and sample fidelity in conditional diffusion models post training, in the same spirit as low temperature sampling or truncation in other types of generative models. Classifier guidance combines the score estimate of a diffusion model with the gradient of an image classifier and thereby requires training an image classifier separate from the diffusion model. It also raises the question of whether guidance can be performed without a classifier. We show that guidance can be indeed performed by a pure generative model without such a classifier: in what we call classifier-free guidance, we jointly train a conditional and an unconditional diffusion model, and we combine the resulting conditional and unconditional score estimates to attain a trade-off between sample quality and diversity similar to that obtained using classifier guidance.

A short version of this paper appeared in the NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications.

# Introduction

[IMAGE: Classifier-free guidance on the malamute class for a 64x64 ImageNet diffusion model. Left to right: increasing amounts of classifier-free guidance, starting from non-guided samples on the left.]

[IMAGE: The effect of guidance on a mixture of three Gaussians, each mixture component representing data conditioned on a class. The leftmost plot is the non-guided marginal density. Left to right are densities of mixtures of normalized guided conditionals with increasing guidance strength.]

Diffusion models have recently emerged as an expressive and flexible family of generative models, delivering competitive sample quality and likelihood scores on image and audio synthesis tasks [sohl2015deep; song2019generative; ho2020denoising; song2020score; kingma2021variational; song2021maximum]. These models have delivered audio synthesis performance rivaling the quality of autoregressive models with substantially fewer inference steps [chen2020wavegrad; kong2020diffwave], and they have delivered ImageNet generation results outperforming BigGAN-deep [brock2018large] and VQ-VAE-2 [razavi2019generating] in terms of FID score and classification accuracy score [ho2021cascaded; dhariwal2021diffusion].

[dhariwal2021diffusion] proposed *classifier guidance*, a technique to boost the sample quality of a diffusion model using an extra trained classifier. Prior to classifier guidance, it was not known how to generate "low temperature" samples from a diffusion model similar to those produced by truncated BigGAN [brock2018large] or low temperature Glow [kingma2018glow]: naive attempts, such as scaling the model score vectors or decreasing the amount of Gaussian noise added during diffusion sampling, are ineffective [dhariwal2021diffusion]. Classifier guidance instead mixes a diffusion model's score estimate with the input gradient of the log probability of a classifier. By varying the strength of the classifier gradient, [dhariwal2021diffusion] can trade off Inception score [salimans2016improved] and FID score [heusel2017gans] (or precision and recall) in a manner similar to varying the truncation parameter of BigGAN.

We are interested in whether classifier guidance can be performed without a classifier. Classifier guidance complicates the diffusion model training pipeline because it requires training an extra classifier, and this classifier must be trained on noisy data so it is generally not possible to plug in a pre-trained classifier. Furthermore, because classifier guidance mixes a score estimate with a classifier gradient during sampling, classifier-guided diffusion sampling can be interpreted as attempting to confuse an image classifier with a gradient-based adversarial attack. This raises the question of whether classifier guidance is successful at boosting classifier-based metrics such as FID and Inception score (IS) simply because it is adversarial against such classifiers. Stepping in direction of classifier gradients also bears some resemblance to GAN training, particularly with nonparameteric generators; this also raises the question of whether classifier-guided diffusion models perform well on classifier-based metrics because they are beginning to resemble GANs, which are already known to perform well on such metrics.

To resolve these questions, we present *classifier-free guidance*, our guidance method which avoids any classifier entirely. Rather than sampling in the direction of the gradient of an image classifier, classifier-free guidance instead mixes the score estimates of a conditional diffusion model and a jointly trained unconditional diffusion model. By sweeping over the mixing weight, we attain a FID/IS tradeoff similar to that attained by classifier guidance. Our classifier-free guidance results demonstrate that pure generative diffusion models are capable of synthesizing extremely high fidelity samples possible with other types of generative models.

# Background

We train diffusion models in continuous time [song2020score; chen2020wavegrad; kingma2021variational]: letting ```latex $\mathbf{x}\sim p(\mathbf{x})$ ``` and ```latex $\mathbf{z}= \{\mathbf{z}_\lambda \,|\, \lambda \in [\lambda_{\mathrm{min}}, \lambda_{\mathrm{max}}]\}$ ``` for hyperparameters ```latex $\lambda_{\mathrm{min}} < \lambda_{\mathrm{max}} \in \mathbb{R}$ ```, the forward process ```latex $q(\mathbf{z}|\mathbf{x})$ ``` is the variance-preserving Markov process [sohl2015deep]:

```latex
$$\begin{aligned}
q(\mathbf{z}_\lambda|\mathbf{x}) &= \mathcal{N}(\alpha_\lambda \mathbf{x}, \sigma_\lambda^2 \mathbf{I}), \ \text{where}\ \alpha_\lambda^2 = 1/(1+e^{-\lambda}),\ \sigma_\lambda^2 = 1-\alpha_\lambda^2 \\
q(\mathbf{z}_{\lambda} | \mathbf{z}_{\lambda'}) &= \mathcal{N}((\alpha_{\lambda}/\alpha_{\lambda'})\mathbf{z}_{\lambda'}, \sigma_{\lambda|\lambda'}^2\mathbf{I}), \ \text{where}\ \lambda < \lambda',\
\sigma^2_{\lambda|\lambda'} = (1-e^{\lambda-\lambda'})\sigma_\lambda^2
\end{aligned}$$
```

We will use the notation ```latex $p(\mathbf{z})$ ``` (or ```latex $p(\mathbf{z}_\lambda)$ ```) to denote the marginal of ```latex $\mathbf{z}$ ``` (or ```latex $\mathbf{z}_\lambda$ ```) when ```latex $\mathbf{x}\sim p(\mathbf{x})$ ``` and ```latex $\mathbf{z}\sim q(\mathbf{z}|\mathbf{x})$ ```. Note that ```latex $\lambda = \log \alpha_\lambda^2/\sigma_\lambda^2$ ```, so ```latex $\lambda$ ``` can be interpreted as the log signal-to-noise ratio of ```latex $\mathbf{z}_\lambda$ ```, and the forward process runs in the direction of decreasing ```latex $\lambda$ ```.

Conditioned on ```latex $\mathbf{x}$ ```, the forward process can be described in reverse by the transitions ```latex $q(\mathbf{z}_{\lambda'}|\mathbf{z}_\lambda,\mathbf{x}) = \mathcal{N}(\tilde{\boldsymbol{\mu}}_{\lambda'|\lambda}(\mathbf{z}_\lambda,\mathbf{x}), \tilde\sigma^2_{\lambda'|\lambda}\mathbf{I})$ ```, where

```latex
$$\begin{aligned}
\tilde{\boldsymbol{\mu}}_{\lambda'|\lambda}(\mathbf{z}_\lambda,\mathbf{x}) = e^{\lambda-\lambda'}(\alpha_{\lambda'}/\alpha_{\lambda})\mathbf{z}_\lambda + (1-e^{\lambda - \lambda'})\alpha_{\lambda'}\mathbf{x},
\quad \tilde\sigma^2_{\lambda'|\lambda} = (1-e^{\lambda-\lambda'})\sigma_{\lambda'}^2
\end{aligned}$$
```

The reverse process generative model starts from ```latex $p_\theta(\mathbf{z}_{\lambda_{\mathrm{min}}}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ ```. We specify the transitions:

```latex
$$\begin{aligned}
p_\theta(\mathbf{z}_{\lambda'}|\mathbf{z}_{\lambda}) = \mathcal{N}(\tilde{\boldsymbol{\mu}}_{\lambda'|\lambda}(\mathbf{z}_\lambda,\mathbf{x}_\theta(\mathbf{z}_\lambda)),  (\tilde\sigma^2_{\lambda'|\lambda})^{1-v} (\sigma^2_{\lambda|\lambda'})^v)
\end{aligned}$$
```

During sampling, we apply this transition along an increasing sequence ```latex $\lambda_{\mathrm{min}} = \lambda_1 < \cdots < \lambda_T = \lambda_{\mathrm{max}}$ ``` for ```latex $T$ ``` timesteps; in other words, we follow the discrete time ancestral sampler of [sohl2015deep; ho2020denoising]. If the model ```latex $\mathbf{x}_\theta$ ``` is correct, then as ```latex $T\rightarrow\infty$ ```, we obtain samples from an SDE whose sample paths are distributed as ```latex $p(\mathbf{z})$ ``` [song2020score], and we use ```latex $p_\theta(\mathbf{z})$ ``` to denote the continuous time model distribution. The variance is a log-space interpolation of ```latex $\tilde\sigma^2_{\lambda'|\lambda}$ ``` and ```latex $\sigma^2_{\lambda|\lambda'}$ ``` as suggested by [nichol2021improved]; we found it effective to use a constant hyperparameter ```latex $v$ ``` rather than learned ```latex $\mathbf{z}_\lambda$ ```-dependent ```latex $v$ ```. Note that the variances simplify to ```latex $\tilde\sigma^2_{\lambda'|\lambda}$ ``` as ```latex $\lambda'\rightarrow\lambda$ ```, so ```latex $v$ ``` has an effect only when sampling with non-infinitesimal timesteps as done in practice.

The reverse process mean comes from an estimate ```latex $\mathbf{x}_\theta(\mathbf{z}_\lambda) \approx \mathbf{x}$ ``` plugged into ```latex $q(\mathbf{z}_{\lambda'}|\mathbf{z}_\lambda,\mathbf{x})$ ``` [ho2020denoising; kingma2021variational] (```latex $\mathbf{x}_\theta$ ``` also receives ```latex $\lambda$ ``` as input, but we suppress this to keep our notation clean). We parameterize ```latex $\mathbf{x}_\theta$ ``` in terms of ```latex ${\boldsymbol{\epsilon}}$ ```-prediction [ho2020denoising]: ```latex $\mathbf{x}_\theta(\mathbf{z}_\lambda) = (\mathbf{z}_\lambda - \sigma_\lambda {\boldsymbol{\epsilon}}_\theta(\mathbf{z}_\lambda)) / \alpha_\lambda$ ```, and we train on the objective

```latex
$$\begin{aligned}
    \mathbb{E}_{{\boldsymbol{\epsilon}},\lambda}\!\left[\|{\boldsymbol{\epsilon}}_\theta(\mathbf{z}_\lambda) - {\boldsymbol{\epsilon}}\|^2_2\right]
\end{aligned}$$
```

where ```latex ${\boldsymbol{\epsilon}}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ ```, ```latex $\mathbf{z}_\lambda = \alpha_\lambda\mathbf{x}+ \sigma_\lambda{\boldsymbol{\epsilon}}$ ```, and ```latex $\lambda$ ``` is drawn from a distribution ```latex $p(\lambda)$ ``` over ```latex $[\lambda_{\mathrm{min}}, \lambda_{\mathrm{max}}]$ ```. This objective is denoising score matching [vincent2011connection; hyvarinen2005estimation] over multiple noise scales [song2019generative], and when ```latex $p(\lambda)$ ``` is uniform, the objective is proportional to the variational lower bound on the marginal log likelihood of the latent variable model ```latex $\int p_\theta(\mathbf{x}|\mathbf{z}) p_\theta(\mathbf{z}) d\mathbf{z}$ ```, ignoring the term for the unspecified decoder ```latex $p_\theta(\mathbf{x}|\mathbf{z})$ ``` and for the prior at ```latex $\mathbf{z}_{\lambda_\mathrm{min}}$ ``` [kingma2021variational].

If ```latex $p(\lambda)$ ``` is not uniform, the objective can be interpreted as weighted variational lower bound whose weighting can be tuned for sample quality [ho2020denoising; kingma2021variational]. We use a ```latex $p(\lambda)$ ``` inspired by the discrete time cosine noise schedule of [nichol2021improved]: we sample ```latex $\lambda$ ``` via ```latex $\lambda = -2\log\tan(au+b)$ ``` for uniformly distributed ```latex $u \in [0,1]$ ```, where ```latex $b = \arctan(e^{-\lambda_{\mathrm{max}}/2})$ ``` and ```latex $a = \arctan(e^{-\lambda_{\mathrm{min}}/2}) - b$ ```. This represents a hyperbolic secant distribution modified to be supported on a bounded interval. For finite timestep generation, we use ```latex $\lambda$ ``` values corresponding to uniformly spaced ```latex $u \in [0, 1]$ ```, and the final generated sample is ```latex $\mathbf{x}_\theta(\mathbf{z}_{\lambda_\mathrm{max}})$ ```.

Because the loss for ```latex ${\boldsymbol{\epsilon}}_\theta(\mathbf{z}_\lambda)$ ``` is denoising score matching for all ```latex $\lambda$ ```, the score ```latex ${\boldsymbol{\epsilon}}_\theta(\mathbf{z}_\lambda)$ ``` learned by our model estimates the gradient of the log-density of the distribution of our noisy data ```latex $\mathbf{z}_\lambda$ ```, that is ```latex ${\boldsymbol{\epsilon}}_\theta(\mathbf{z}_\lambda) \approx -\sigma_\lambda \nabla_{\mathbf{z}_\lambda}\log p(\mathbf{z}_\lambda)$ ```; note, however, that because we use unconstrained neural networks to define ```latex ${\boldsymbol{\epsilon}}_\theta$ ```, there need not exist any scalar potential whose gradient is ```latex ${\boldsymbol{\epsilon}}_\theta$ ```. Sampling from the learned diffusion model resembles using Langevin diffusion to sample from a sequence of distributions ```latex $p(\mathbf{z}_\lambda)$ ``` that converges to the conditional distribution ```latex $p(\mathbf{x})$ ``` of the original data ```latex $\mathbf{x}$ ```.

In the case of conditional generative modeling, the data ```latex $\mathbf{x}$ ``` is drawn jointly with conditioning information ```latex $\mathbf{c}$ ```, i.e. a class label for class-conditional image generation. The only modification to the model is that the reverse process function approximator receives ```latex $\mathbf{c}$ ``` as input, as in ```latex ${\boldsymbol{\epsilon}}_\theta(\mathbf{z}_\lambda, \mathbf{c})$ ```.

# Guidance

An interesting property of certain generative models, such as GANs and flow-based models, is the ability to perform truncated or low temperature sampling by decreasing the variance or range of noise inputs to the generative model at sampling time. The intended effect is to decrease the diversity of the samples while increasing the quality of each individual sample. Truncation in BigGAN [brock2018large], for example, yields a tradeoff curve between FID score and Inception score for low and high amounts of truncation, respectively. Low temperature sampling in Glow [kingma2018glow] has a similar effect.

Unfortunately, straightforward attempts of implementing truncation or low temperature sampling in diffusion models are ineffective. For example, scaling model scores or decreasing the variance of Gaussian noise in the reverse process cause the diffusion model to generate blurry, low quality samples [dhariwal2021diffusion].

## Classifier guidance

To obtain a truncation-like effect in diffusion models, [dhariwal2021diffusion] introduce *classifier guidance*, where the diffusion score ```latex ${\boldsymbol{\epsilon}}_\theta(\mathbf{z}_\lambda, \mathbf{c}) \approx -\sigma_\lambda \nabla_{\mathbf{z}_\lambda}\log p(\mathbf{z}_\lambda |  \mathbf{c})$ ``` is modified to include the gradient of the log likelihood of an auxiliary classifier model ```latex $p_{\theta}(\mathbf{c}| \mathbf{z}_\lambda)$ ``` as follows:

```latex
$$\tilde{{\boldsymbol{\epsilon}}}_\theta(\mathbf{z}_\lambda, \mathbf{c}) = {\boldsymbol{\epsilon}}_\theta(\mathbf{z}_\lambda, \mathbf{c}) - w\sigma_{\lambda}\nabla_{\mathbf{z}_\lambda}\log p_{\theta}(\mathbf{c}| \mathbf{z}_\lambda) \approx -\sigma_{\lambda}\nabla_{\mathbf{z}_\lambda}[\log p(\mathbf{z}_\lambda |  \mathbf{c}) + w \log p_{\theta}(\mathbf{c}| \mathbf{z}_\lambda) ],$$
```

where ```latex $w$ ``` is a parameter that controls the strength of the classifier guidance. This modified score ```latex $\tilde{{\boldsymbol{\epsilon}}}_\theta(\mathbf{z}_\lambda, \mathbf{c})$ ``` is then used in place of ```latex ${\boldsymbol{\epsilon}}_\theta(\mathbf{z}_\lambda, \mathbf{c})$ ``` when sampling from the diffusion model, resulting in approximate samples from the distribution

```latex
$$\tilde{p}_{\theta}(\mathbf{z}_\lambda | \mathbf{c}) \propto p_{\theta}(\mathbf{z}_\lambda | \mathbf{c})p_{\theta}(\mathbf{c}| \mathbf{z}_\lambda)^{w}.$$
```

The effect is that of up-weighting the probability of data for which the classifier ```latex $p_{\theta}(\mathbf{c}| \mathbf{z}_\lambda)$ ``` assigns high likelihood to the correct label: data that can be classified well scores high on the Inception score of perceptual quality [salimans2016improved], which rewards generative models for this by design. [dhariwal2021diffusion] therefore find that by setting ```latex $w > 0$ ``` they can improve the Inception score of their diffusion model, at the expense of decreased diversity in their samples.

Figure 2 illustrates the effect of numerically solved guidance ```latex $\tilde{p}_{\theta}(\mathbf{z}_\lambda | \mathbf{c}) \propto p_{\theta}(\mathbf{z}_\lambda | \mathbf{c})p_{\theta}(\mathbf{c}| \mathbf{z}_\lambda)^{w}$ ``` on a toy 2D example of three classes, in which the conditional distribution for each class is an isotropic Gaussian. The form of each conditional upon applying guidance is markedly non-Gaussian. As guidance strength is increased, each conditional places probability mass farther away from other classes and towards directions of high confidence given by logistic regression, and most of the mass becomes concentrated in smaller regions. This behavior can be seen as a simplistic manifestation of the Inception score boost and sample diversity decrease that occur when classifier guidance strength is increased in an ImageNet model.

Applying classifier guidance with weight ```latex $w+1$ ``` to an unconditional model would theoretically lead to the same result as applying classifier guidance with weight ```latex $w$ ``` to a conditional model, because ```latex $p_{\theta}(\mathbf{z}_\lambda | \mathbf{c})p_{\theta}(\mathbf{c}| \mathbf{z}_\lambda)^{w} \propto p_{\theta}(\mathbf{z}_\lambda)p_{\theta}(\mathbf{c}| \mathbf{z}_\lambda)^{w+1}$ ```; or in terms of scores,

```latex
$$\begin{aligned}
{\boldsymbol{\epsilon}}_\theta(\mathbf{z}_\lambda) - (w+1)\sigma_{\lambda}\nabla_{\mathbf{z}_\lambda}\log p_{\theta}(\mathbf{c}| \mathbf{z}_\lambda) &\approx -\sigma_{\lambda}\nabla_{\mathbf{z}_\lambda}[\log p(\mathbf{z}_\lambda) + (w+1) \log p_{\theta}(\mathbf{c}| \mathbf{z}_\lambda) ] \\
&= -\sigma_{\lambda}\nabla_{\mathbf{z}_\lambda}[\log p(\mathbf{z}_\lambda|\mathbf{c}) + w\log p_{\theta}(\mathbf{c}| \mathbf{z}_\lambda) ],
\end{aligned}$$
```

but interestingly, [dhariwal2021diffusion] obtain their best results when applying classifier guidance to an already class-conditional model, as opposed to applying guidance to an unconditional model. For this reason, we will stay in the setup of guiding an already conditional model.

## Classifier-free guidance

While classifier guidance successfully trades off IS and FID as expected from truncation or low temperature sampling, it is nonetheless reliant on gradients from an image classifier and we seek to eliminate the classifier for the reasons stated in Section 1. Here, we describe *classifier-free guidance*, which achieves the same effect without such gradients. Classifier-free guidance is an alternative method of modifying ```latex ${\boldsymbol{\epsilon}}_\theta(\mathbf{z}_\lambda, \mathbf{c})$ ``` to have the same effect as classifier guidance, but without a classifier. Algorithms 1 and 2 describe training and sampling with classifier-free guidance in detail.

**Algorithm 1: Training**

- Input: ```latex $p_\mathrm{uncond}$ ```: probability of unconditional training
- Sample ```latex $(\mathbf{x},\mathbf{c}) \sim p(\mathbf{x},\mathbf{c})$ ```
- Set ```latex $\mathbf{c}\gets \varnothing$ ``` with probability ```latex $p_\mathrm{uncond}$ ```
- Sample ```latex $\lambda \sim p(\lambda)$ ```
- Sample ```latex ${\boldsymbol{\epsilon}}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ ```
- Compute ```latex $\mathbf{z}_\lambda = \alpha_\lambda\mathbf{x}+ \sigma_\lambda {\boldsymbol{\epsilon}}$ ```
- Take gradient step on ```latex $\nabla_\theta \left\| {\boldsymbol{\epsilon}}_\theta(\mathbf{z}_\lambda,\mathbf{c}) - {\boldsymbol{\epsilon}}\right\|^2$ ```

**Algorithm 2: Sampling**

- Input: ```latex $w$ ```: guidance strength; ```latex $\mathbf{c}$ ```: conditioning information; ```latex $\lambda_1, \dotsc, \lambda_T$ ```: increasing log SNR sequence with ```latex $\lambda_1=\lambda_{\mathrm{min}}$ ```, ```latex $\lambda_T=\lambda_{\mathrm{max}}$ ```
- Sample ```latex $\mathbf{z}_{1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ ```
- For ```latex $t = 1, \dotsc, T$ ```:
  - Form the classifier-free guided score at log SNR ```latex $\lambda_t$ ```: ```latex $\tilde{{\boldsymbol{\epsilon}}}_t = (1+w){\boldsymbol{\epsilon}}_\theta(\mathbf{z}_{t}, \mathbf{c}) - w{\boldsymbol{\epsilon}}_{\theta}(\mathbf{z}_{t})$ ```
  - Sampling step: ```latex $\tilde\mathbf{x}_t = (\mathbf{z}_{t}-\sigma_{\lambda_t}\tilde{\boldsymbol{\epsilon}}_t)/\alpha_{\lambda_t}$ ```
  - ```latex $\mathbf{z}_{t+1} \sim \mathcal{N}(\tilde{\boldsymbol{\mu}}_{\lambda_{t+1}|\lambda_t}(\mathbf{z}_{t},\tilde\mathbf{x}_t),  (\tilde\sigma^2_{\lambda_{t+1}|\lambda_t})^{1-v} (\sigma^2_{\lambda_t|\lambda_{t+1}})^v)$ ``` if ```latex $t<T$ ``` else ```latex $\mathbf{z}_{t+1}=\tilde\mathbf{x}_t$ ```
- Return ```latex $\mathbf{z}_{T+1}$ ```

Instead of training a separate classifier model, we choose to train an unconditional denoising diffusion model ```latex $p_{\theta}(\mathbf{z})$ ``` parameterized through a score estimator ```latex ${\boldsymbol{\epsilon}}_{\theta}(\mathbf{z}_\lambda)$ ``` together with the conditional model ```latex $p_{\theta}(\mathbf{z}| \mathbf{c})$ ``` parameterized through ```latex ${\boldsymbol{\epsilon}}_{\theta}(\mathbf{z}_\lambda, \mathbf{c})$ ```. We use a single neural network to parameterize both models, where for the unconditional model we can simply input a null token ```latex $\varnothing$ ``` for the class identifier ```latex $\mathbf{c}$ ``` when predicting the score, i.e. ```latex ${\boldsymbol{\epsilon}}_{\theta}(\mathbf{z}_\lambda) = {\boldsymbol{\epsilon}}_{\theta}(\mathbf{z}_\lambda, \mathbf{c}= \varnothing)$ ```. We jointly train the unconditional and conditional models simply by randomly setting ```latex $\mathbf{c}$ ``` to the unconditional class identifier ```latex $\varnothing$ ``` with some probability ```latex $p_\mathrm{uncond}$ ```, set as a hyperparameter. (It would certainly be possible to train separate models instead of jointly training them together, but we choose joint training because it is extremely simple to implement, does not complicate the training pipeline, and does not increase the total number of parameters.) We then perform sampling using the following linear combination of the conditional and unconditional score estimates:

```latex
$$\begin{aligned}
    \tilde{{\boldsymbol{\epsilon}}}_\theta(\mathbf{z}_\lambda, \mathbf{c}) = (1+w){\boldsymbol{\epsilon}}_\theta(\mathbf{z}_\lambda, \mathbf{c}) - w{\boldsymbol{\epsilon}}_{\theta}(\mathbf{z}_\lambda)
\end{aligned}$$
```

This equation has no classifier gradient present, so taking a step in the ```latex $\tilde{\boldsymbol{\epsilon}}_\theta$ ``` direction cannot be interpreted as a gradient-based adversarial attack on an image classifier. Furthermore, ```latex $\tilde{\boldsymbol{\epsilon}}_\theta$ ``` is constructed from score estimates that are non-conservative vector fields due to the use of unconstrained neural networks, so there in general cannot exist a scalar potential such as a classifier log likelihood for which ```latex $\tilde{\boldsymbol{\epsilon}}_\theta$ ``` is the classifier-guided score.

Despite the fact that there in general may not exist a classifier for which the above equation is the classifier-guided score, it is in fact inspired by the gradient of an implicit classifier ```latex $p^{i}(\mathbf{c}| \mathbf{z}_\lambda) \propto p(\mathbf{z}_\lambda | \mathbf{c})/p(\mathbf{z}_\lambda)$ ```. If we had access to exact scores ```latex ${\boldsymbol{\epsilon}}^*(\mathbf{z}_\lambda, \mathbf{c})$ ``` and ```latex ${\boldsymbol{\epsilon}}^*(\mathbf{z}_\lambda)$ ``` (of ```latex $p(\mathbf{z}_\lambda|\mathbf{c})$ ``` and ```latex $p(\mathbf{z}_\lambda)$ ```, respectively), then the gradient of this implicit classifier would be ```latex $\nabla_{\mathbf{z}_\lambda} \log p^{i}(\mathbf{c}| \mathbf{z}_\lambda) = -\frac{1}{\sigma_{\lambda}}[{\boldsymbol{\epsilon}}^*(\mathbf{z}_\lambda, \mathbf{c}) - {\boldsymbol{\epsilon}}^*(\mathbf{z}_\lambda)]$ ```, and classifier guidance with this implicit classifier would modify the score estimate into ```latex $\tilde{{\boldsymbol{\epsilon}}}^*(\mathbf{z}_\lambda, \mathbf{c}) = (1+w){\boldsymbol{\epsilon}}^*(\mathbf{z}_\lambda, \mathbf{c}) - w{\boldsymbol{\epsilon}}^*(\mathbf{z}_\lambda)$ ```. Note the resemblance to the classifier-free guidance equation, but also note that ```latex $\tilde{{\boldsymbol{\epsilon}}}^*(\mathbf{z}_\lambda,\mathbf{c})$ ``` differs fundamentally from ```latex $\tilde{{\boldsymbol{\epsilon}}}_\theta(\mathbf{z}_\lambda,\mathbf{c})$ ```. The former is constructed from the scaled classifier gradient ```latex ${\boldsymbol{\epsilon}}^*(\mathbf{z}_\lambda, \mathbf{c}) - {\boldsymbol{\epsilon}}^*(\mathbf{z}_\lambda)$ ```; the latter is constructed from the estimate ```latex ${\boldsymbol{\epsilon}}_\theta(\mathbf{z}_\lambda, \mathbf{c}) - {\boldsymbol{\epsilon}}_\theta(\mathbf{z}_\lambda)$ ```, and this expression is not in general the (scaled) gradient of any classifier, again because the score estimates are the outputs of unconstrained neural networks.

It is not obvious a priori that inverting a generative model using Bayes' rule yields a good classifier that provides a useful guidance signal. For example, [grandvalet2004semi] find that discriminative models generally outperform implicit classifiers derived from generative models, even in artificial cases where the specification of those generative models exactly matches the data distribution. In cases such as ours, where we expect the model to be misspecified, classifiers derived by Bayes' rule can be inconsistent [grunwald2007suboptimal] and we lose all guarantees on their performance. Nevertheless, in Section 4, we show empirically that classifier-free guidance is able to trade off FID and IS in the same way as classifier guidance. In Section 5 we discuss the implications of classifier-free guidance in relation to classifier guidance.

# Experiments

[IMAGE: Classifier-free guidance on 128x128 ImageNet. Left: non-guided samples, right: classifier-free guided samples with w=3.0. Interestingly, strongly guided samples such as these display saturated colors.]

We train diffusion models with classifier-free guidance on area-downsampled class-conditional ImageNet [russakovsky2015imagenet], the standard setting for studying tradeoffs between FID and Inception scores starting from the BigGAN paper [brock2018large].

The purpose of our experiments is to serve as a proof of concept to demonstrate that classifier-free guidance is able to attain a FID/IS tradeoff similar to classifier guidance and to understand the behavior of classifier-free guidance, not necessarily to push sample quality metrics to state of the art on these benchmarks. For this purpose, we use the same model architectures and hyperparameters as the guided diffusion models of [dhariwal2021diffusion] (apart from continuous time training as specified in Section 2); those hyperparameter settings were tuned for classifier guidance and hence may be suboptimal for classifier-free guidance. Furthermore, since we amortize the conditional and unconditional models into the same architecture without an extra classifier, we in fact are using less model capacity than previous work. Nevertheless, our classifier-free guided models still produce competitive sample quality metrics and sometimes outperform prior work, as can be seen in the following sections.

## Varying the classifier-free guidance strength

Here we experimentally verify the main claim of this paper: that classifier-free guidance is able to trade off IS and FID in a manner like classifier guidance or GAN truncation. We apply our proposed classifier-free guidance to 64x64 and 128x128 class-conditional ImageNet generation. In Table 1 and Figure 4, we show sample quality effects of sweeping over the guidance strength ```latex $w$ ``` on our 64x64 ImageNet models; Table 2 and Figure 5 show the same for our 128x128 models. We consider ```latex $w \in \{0, 0.1, 0.2, \ldots, 4\}$ ``` and calculate FID and Inception Scores with 50000 samples for each value following the procedures of [heusel2017gans] and [salimans2016improved]. All models used log SNR endpoints ```latex $\lambda_\mathrm{min}=-20$ ``` and ```latex $\lambda_\mathrm{max}=20$ ```. The 64x64 models used sampler noise interpolation coefficient ```latex $v=0.3$ ``` and were trained for 400 thousand steps; the 128x128 models used ```latex $v=0.2$ ``` and were trained for 2.7 million steps.

We obtain the best FID results with a small amount of guidance (```latex $w = 0.1$ ``` or ```latex $w=0.3$ ```, depending on the dataset) and the best IS result with strong guidance (```latex $w \geq 4$ ```). Between these two extremes we see a clear trade-off between these two metrics of perceptual quality, with FID monotonically decreasing and IS monotonically increasing with ```latex $w$ ```. Our results compare favorably to [dhariwal2021diffusion] and [ho2021cascaded], and in fact our 128x128 results are the state of the art in the literature. At ```latex $w=0.3$ ```, our model's FID score on 128x128 ImageNet outperforms the classifier-guided ADM-G, and at ```latex $w=4.0$ ```, our model outperforms BigGAN-deep at both FID and IS when BigGAN-deep is evaluated its best-IS truncation level.

**Table 1: ImageNet 64x64 results (w=0.0 refers to non-guided models).**

| Model | FID (down) | IS (up) |
|---|---|---|
| ADM [dhariwal2021diffusion] | 2.07 | - |
| CDM [ho2021cascaded] | **1.48** | 67.95 |
| **Ours** | p_uncond=0.1/0.2/0.5 | |
| w=0.0 | 1.8 / 1.8 / 2.21 | 53.71 / 52.9 / 47.61 |
| w=0.1 | 1.55 / 1.62 / 1.91 | 66.11 / 64.58 / 56.1 |
| w=0.2 | 2.04 / 2.1 / 2.08 | 78.91 / 76.99 / 65.6 |
| w=0.3 | 3.03 / 2.93 / 2.65 | 92.8 / 88.64 / 74.92 |
| w=0.4 | 4.3 / 4 / 3.44 | 106.2 / 101.11 / 84.27 |
| w=0.5 | 5.74 / 5.19 / 4.34 | 119.3 / 112.15 / 92.95 |
| w=0.6 | 7.19 / 6.48 / 5.27 | 131.1 / 122.13 / 102 |
| w=0.7 | 8.62 / 7.73 / 6.23 | 141.8 / 131.6 / 109.8 |
| w=0.8 | 10.08 / 8.9 / 7.25 | 151.6 / 140.82 / 116.9 |
| w=0.9 | 11.41 / 10.09 / 8.21 | 161 / 150.26 / 124.6 |
| w=1.0 | 12.6 / 11.21 / 9.13 | 170.1 / 158.29 / 131.1 |
| w=2.0 | 21.03 / 18.79 / 16.16 | 225.5 / 212.98 / 183 |
| w=3.0 | 24.83 / 22.36 / 19.75 | 250.4 / 237.65 / 208.9 |
| w=4.0 | 26.22 / 23.84 / 21.48 | **260.2** / 248.97 / 225.1 |

[IMAGE: IS/FID curves over guidance strengths for ImageNet 64x64 models. Each curve represents a model with unconditional training probability p_uncond.]

## Varying the unconditional training probability

The main hyperparameter of classifier-free guidance at training time is ```latex $p_\mathrm{uncond}$ ```, the probability of training on unconditional generation during joint training of the conditional and unconditional diffusion models. Here, we study the effect of training models on varying ```latex $p_\mathrm{uncond}$ ``` on 64x64 ImageNet.

Table 1 and Figure 4 show the effects of ```latex $p_\mathrm{uncond}$ ``` on sample quality. We trained models with ```latex $p_\mathrm{uncond}\in\{0.1,0.2,0.5\}$ ```, all for 400 thousand training steps, and evaluated sample quality across various guidance strengths. We find ```latex $p_\mathrm{uncond}=0.5$ ``` consistently performs worse than ```latex $p_\mathrm{uncond}\in\{0.1,0.2\}$ ``` across the entire IS/FID frontier; ```latex $p_\mathrm{uncond}\in\{0.1,0.2\}$ ``` perform about equally as well as each other.

Based on these findings, we conclude that only a relatively small portion of the model capacity of the diffusion model needs to be dedicated to the unconditional generation task in order to produce classifier-free guided scores effective for sample quality. Interestingly, for classifier guidance, [dhariwal2021diffusion] report that relatively small classifiers with little capacity are sufficient for effective classifier guided sampling, mirroring this phenomenon that we found with classifier-free guided models.

## Varying the number of sampling steps

Since the number of sampling steps ```latex $T$ ``` is known to have a major impact on the sample quality of a diffusion model, here we study the effect of varying ```latex $T$ ``` on our 128x128 ImageNet model. Table 2 and Figure 5 show the effect of varying ```latex $T\in\{128,256,1024\}$ ``` over a range of guidance strengths. As expected, sample quality improves when ```latex $T$ ``` is increased, and for this model ```latex $T=256$ ``` attains a good balance between sample quality and sampling speed.

Note that ```latex $T=256$ ``` is approximately the same number of sampling steps used by ADM-G [dhariwal2021diffusion], which is outperformed by our model. However, it is important to note that each sampling step for our method requires evaluating the denoising model twice, once for the conditional ```latex ${\boldsymbol{\epsilon}}_\theta(\mathbf{z}_\lambda,\mathbf{c})$ ``` and once for the unconditional ```latex ${\boldsymbol{\epsilon}}_\theta(\mathbf{z}_\lambda)$ ```. Because we used the same model architecture as ADM-G, the fair comparison in terms of sampling speed would be our ```latex $T=128$ ``` setting, which underperforms compared to ADM-G in terms of FID score.

**Table 2: ImageNet 128x128 results (w=0.0 refers to non-guided models).**

| Model | FID (down) | IS (up) |
|---|---|---|
| BigGAN-deep, max IS [brock2018large] | 25 | 253 |
| BigGAN-deep [brock2018large] | 5.7 | 124.5 |
| CDM [ho2021cascaded] | 3.52 | 128.8 |
| LOGAN [wu2019logan] | 3.36 | 148.2 |
| ADM-G [dhariwal2021diffusion] | 2.97 | - |
| **Ours** | T=128/256/1024 | |
| w=0.0 | 8.11 / 7.27 / 7.22 | 81.46 / 82.45 / 81.54 |
| w=0.1 | 5.31 / 4.53 / 4.5 | 105.01 / 106.12 / 104.67 |
| w=0.2 | 3.7 / 3.03 / 3 | 130.79 / 132.54 / 130.09 |
| w=0.3 | 3.04 / **2.43** / **2.43** | 156.09 / 158.47 / 156 |
| w=0.4 | 3.02 / 2.49 / 2.48 | 183.01 / 183.41 / 180.88 |
| w=0.5 | 3.43 / 2.98 / 2.96 | 206.94 / 207.98 / 204.31 |
| w=0.6 | 4.09 / 3.76 / 3.73 | 227.72 / 228.83 / 226.76 |
| w=0.7 | 4.96 / 4.67 / 4.69 | 247.92 / 249.25 / 247.89 |
| w=0.8 | 5.93 / 5.74 / 5.71 | 265.54 / 267.99 / 265.52 |
| w=0.9 | 6.89 / 6.8 / 6.81 | 280.19 / 283.41 / 281.14 |
| w=1.0 | 7.88 / 7.86 / 7.8 | 295.29 / 297.98 / 294.56 |
| w=2.0 | 15.9 / 15.93 / 15.75 | 378.56 / 377.37 / 373.18 |
| w=3.0 | 19.77 / 19.77 / 19.56 | 409.16 / 407.44 / 405.68 |
| w=4.0 | 21.55 / 21.53 / 21.45 | **422.29** / 421.03 / 419.06 |

[IMAGE: IS/FID curves over guidance strengths for ImageNet 128x128 models. Each curve represents sampling with a different number of timesteps T.]

# Discussion

The most practical advantage of our classifier-free guidance method is its extreme simplicity: it is only a one-line change of code during training---to randomly drop out the conditioning---and during sampling---to mix the conditional and unconditional score estimates. Classifier guidance, by contrast, complicates the training pipeline since it requires training an extra classifier. This classifier must be trained on noisy ```latex $\mathbf{z}_\lambda$ ```, so it is not possible to plug in a standard pre-trained classifier.

Since classifier-free guidance is able to trade off IS and FID like classifier guidance without needing an extra trained classifier, we have demonstrated that guidance can be performed with a pure generative model. Furthermore, our diffusion models are parameterized by unconstrained neural networks and therefore their score estimates do not necessarily form conservative vector fields, unlike classifier gradients [salimans2021should]. Therefore, our classifier-free guided sampler follows step directions that do not resemble classifier gradients at all and thus cannot be interpreted as a gradient-based adversarial attack on a classifier, and hence our results show that boosting the classifier-based IS and FID metrics can be accomplished with pure generative models with a sampling procedure that is not adversarial against image classifiers using classifier gradients.

We also have arrived at an intuitive explanation for how guidance works: it decreases the unconditional likelihood of the sample while increasing the conditional likelihood. Classifier-free guidance accomplishes this by decreasing the unconditional likelihood with a *negative* score term, which to our knowledge has not yet been explored and may find uses in other applications.

Classifier-free guidance as presented here relies on training an unconditional model, but in some cases this can be avoided. If the class distribution is known and there are only a few classes, we can use the fact that ```latex $\sum_\mathbf{c}p(\mathbf{x}|\mathbf{c}) p(\mathbf{c}) = p(\mathbf{x})$ ``` to obtain an unconditional score from conditional scores without explicitly training for the unconditional score. Of course, this would require as many forward passes as there are possible values of ```latex $\mathbf{c}$ ``` and would be inefficient for high dimensional conditioning.

A potential disadvantage of classifier-free guidance is sampling speed. Generally, classifiers can be smaller and faster than generative models, so classifier guided sampling may be faster than classifier-free guidance because the latter needs to run two forward passes of the diffusion model, one for conditional score and another for the unconditional score. The necessity to run multiple passes of the diffusion model might be mitigated by changing the architecture to inject conditioning late in the network, but we leave this exploration for future work.

Finally, any guidance method that increases sample fidelity at the expense of diversity must face the question of whether decreased diversity is acceptable. There may be negative impacts in deployed models, since sample diversity is important to maintain in applications where certain parts of the data are underrepresented in the context of the rest of the data. It would be an interesting avenue of future work to try to boost sample quality while maintaining sample diversity.

# Conclusion

We have presented classifier-free guidance, a method to increase sample quality while decreasing sample diversity in diffusion models. Classifier-free guidance can be thought of as classifier guidance without a classifier, and our results showing the effectiveness of classifier-free guidance confirm that pure generative diffusion models are capable of maximizing classifier-based sample quality metrics while entirely avoiding classifier gradients. We look forward to further explorations of classifier-free guidance in a wider variety of settings and data modalities.

# Samples

[IMAGE: Classifier-free guidance on ImageNet 64x64. Subfigures show: (a) Non-guided conditional sampling: FID=1.80, IS=53.71; (b) Classifier-free guidance with w=1.0: FID=12.6, IS=170.1; (c) Classifier-free guidance with w=3.0: FID=24.83, IS=250.4. Left: random classes. Right: single class (malamute). The same random seed was used for sampling in each subfigure.]

[IMAGE: Classifier-free guidance on ImageNet 128x128. Subfigures show: (a) Non-guided conditional sampling: FID=7.27, IS=82.45; (b) Classifier-free guidance with w=1.0: FID=7.86, IS=297.98; (c) Classifier-free guidance with w=4.0: FID=21.53, IS=421.03. Left: random classes. Right: single class (malamute). The same random seed was used for sampling in each subfigure.]

[IMAGE: More examples of classifier-free guidance on 128x128 ImageNet. Left: non-guided samples, right: classifier-free guided samples with w=3.0.]
