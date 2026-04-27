# Abstract

We introduce a new paradigm for generative modeling built on Continuous Normalizing Flows (CNFs), allowing us to train CNFs at unprecedented scale. Specifically, we present the notion of Flow Matching (FM), a simulation-free approach for training CNFs based on regressing vector fields of fixed conditional probability paths. Flow Matching is compatible with a general family of Gaussian probability paths for transforming between noise and data samples---which subsumes existing diffusion paths as specific instances. Interestingly, we find that employing FM with diffusion paths results in a more robust and stable alternative for training diffusion models. Furthermore, Flow Matching opens the door to training CNFs with other, non-diffusion probability paths. An instance of particular interest is using Optimal Transport (OT) displacement interpolation to define the conditional probability paths. These paths are more efficient than diffusion paths, provide faster training and sampling, and result in better generalization. Training CNFs using Flow Matching on ImageNet leads to consistently better performance than alternative diffusion-based methods in terms of both likelihood and sample quality, and allows fast and reliable sample generation using off-the-shelf numerical ODE solvers.

# Introduction

Deep generative models are a class of deep learning algorithms aimed at estimating and sampling from an unknown data distribution. The recent influx of amazing advances in generative modeling, *e.g.*, for image generation [ramesh2022hierarchical; rombach2022high], is mostly facilitated by the scalable and relatively stable training of diffusion-based models [ho2020denoising; song2020score]. However, the restriction to simple diffusion processes leads to a rather confined space of sampling probability paths, resulting in very long training times and the need to adopt specialized methods (*e.g.*, [song2020denoising; zhang2022fast]) for efficient sampling.

In this work we consider the general and deterministic framework of Continuous Normalizing Flows (CNFs; [chen2018neural]). CNFs are capable of modeling arbitrary probability path

[IMAGE: Curated ImageNet 128x128 samples (figures/imagenet128/imagenet128_curated_.png)]

and are in particular known to encompass the probability paths modeled by diffusion processes [song2021maximum]. However, aside from diffusion that can be trained efficiently via, *e.g.*, denoising score matching [vincent2011connection], no scalable CNF training algorithms are known. Indeed, maximum likelihood training (*e.g.*, [ffjord2018]) require expensive numerical ODE simulations, while existing simulation-free methods either involve intractable integrals [rozen2021moser] or biased gradients [ben2022matching].

The goal of this work is to propose Flow Matching (FM), an efficient simulation-free approach to training CNF models, allowing the adoption of general probability paths to supervise CNF training. Importantly, FM breaks the barriers for scalable CNF training beyond diffusion, and sidesteps the need to reason about diffusion processes to directly work with probability paths.

In particular, we propose the Flow Matching objective (Section 3), a simple and intuitive training objective to regress onto a target vector field that generates a desired probability path. We first show that we can construct such target vector fields through per-example (*i.e.*, conditional) formulations. Then, inspired by denoising score matching, we show that a per-example training objective, termed Conditional Flow Matching (CFM), provides equivalent gradients and does not require explicit knowledge of the intractable target vector field. Furthermore, we discuss a general family of per-example probability paths (Section 4) that can be used for Flow Matching, which subsumes existing diffusion paths as special instances. Even on diffusion paths, we find that using FM provides more robust and stable training, and achieves superior performance compared to score matching. Furthermore, this family of probability paths also includes a particularly interesting case: the vector field that corresponds to an Optimal Transport (OT) displacement interpolant [mccann1997convexity]. We find that conditional OT paths are simpler than diffusion paths, forming straight line trajectories whereas diffusion paths result in curved paths. These properties seem to empirically translate to faster training, faster generation, and better performance.

We empirically validate Flow Matching and the construction via Optimal Transport paths on ImageNet, a large and highly diverse image dataset. We find that we can easily train models to achieve favorable performance in both likelihood estimation and sample quality amongst competing diffusion-based methods. Furthermore, we find that our models produce better trade-offs between computational cost and sample quality compared to prior methods.

# Preliminaries: Continuous Normalizing Flows

Let ```latex $\mathbb R^d$ ``` denote the data space with data points ```latex $x=(x^1,\ldots,x^d) \in \mathbb R^d$ ```. Two important objects we use in this paper are: the *probability density path* ```latex $p:[0,1]\times \mathbb R^d \rightarrow\mathbb R_{>0}$ ```, which is a time dependent probability density function, *i.e.*, ```latex $\int p_t(x)dx = 1$ ```, and a *time-dependent vector field*, ```latex $v:[0,1]\times \mathbb R^d \rightarrow\mathbb R^d$ ```. A vector field ```latex $v_t$ ``` can be used to construct a time-dependent diffeomorphic map, called a *flow*, ```latex $\phi:[0,1]\times \mathbb R^d \rightarrow\mathbb R^d$ ```, defined via the ordinary differential equation (ODE):

```latex
$$\frac{d}{dt}\phi_t(x) = v_t(\phi_t(x)), \quad \phi_0(x) = x$$
```

Previously, [chen2018neural] suggested modeling the vector field ```latex $v_t$ ``` with a neural network, ```latex $v_t(x;\theta)$ ```, where ```latex $\theta\in \mathbb R^p$ ``` are its learnable parameters, which in turn leads to a deep parametric model of the flow ```latex $\phi_t$ ```, called a *Continuous Normalizing Flow* (CNF). A CNF is used to reshape a simple prior density ```latex $p_0$ ``` (*e.g.*, pure noise) to a more complicated one, ```latex $p_1$ ```, via the push-forward equation

```latex
$$p_t = [\phi_t]_* p_0$$
```

where the push-forward (or change of variables) operator ```latex $*$ ``` is defined by

```latex
$$[\phi_t]_* p_0(x) = p_0(\phi_t^{-1}(x))\det \left [ \frac{\partial \phi_t^{-1}}{\partial x}(x)\right ].$$
```

A vector field ```latex $v_t$ ``` is said to *generate* a probability density path ```latex $p_t$ ``` if its flow ```latex $\phi_t$ ``` satisfies the push-forward equation. One practical way to test if a vector field generates a probability path is using the continuity equation, which is a key component in our proofs, see Appendix 10.

# Flow Matching

Let ```latex $x_1$ ``` denote a random variable distributed according to some unknown data distribution ```latex $q(x_1)$ ```. We assume we only have access to data samples from ```latex $q(x_1)$ ``` but have no access to the density function itself. Furthermore, we let ```latex $p_t$ ``` be a probability path such that ```latex $p_0=p$ ``` is a simple distribution, *e.g.*, the standard normal distribution ```latex $p(x) = {\mathcal{N}}(x | 0, I)$ ```, and let ```latex $p_1$ ``` be approximately equal in distribution to ```latex $q$ ```. We will later discuss how to construct such a path. The Flow Matching objective is then designed to match this target probability path, which will allow us to flow from ```latex $p_0$ ``` to ```latex $p_1$ ```.

Given a target probability density path ```latex $p_t(x)$ ``` and a corresponding vector field ```latex $u_t(x)$ ```, which generates ```latex $p_t(x)$ ```, we define the Flow Matching (FM) objective as

```latex
$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t,p_t(x)}\| v_t(x)-u_t(x) \|^2,$$
```

where ```latex $\theta$ ``` denotes the learnable parameters of the CNF vector field ```latex $v_t$ ```, ```latex $t\sim {\mathcal{U}}[0,1]$ ``` (uniform distribution), and ```latex $x\sim p_t(x)$ ```. Simply put, the FM loss regresses the vector field ```latex $u_t$ ``` with a neural network ```latex $v_t$ ```. Upon reaching zero loss, the learned CNF model will generate ```latex $p_t(x)$ ```.

Flow Matching is a simple and attractive objective, but naively on its own, it is intractable to use in practice since we have no prior knowledge for what an appropriate ```latex $p_t$ ``` and ```latex $u_t$ ``` are. There are many choices of probability paths that can satisfy ```latex $p_1(x) \approx q(x)$ ```, and more importantly, we generally don't have access to a closed form ```latex $u_t$ ``` that generates the desired ```latex $p_t$ ```. In this section, we show that we can construct both ```latex $p_t$ ``` and ```latex $u_t$ ``` using probability paths and vector fields that are only defined *per sample*, and an appropriate method of aggregation provides the desired ```latex $p_t$ ``` and ```latex $u_t$ ```. Furthermore, this construction allows us to create a much more tractable objective for Flow Matching.

## Constructing p_t, u_t from conditional probability paths and vector fields

A simple way to construct a target probability path is via a mixture of simpler probability paths: Given a particular data sample ```latex $x_1$ ``` we denote by ```latex $p_t(x\vert x_1)$ ``` a *conditional probability path* such that it satisfies ```latex $p_0(x|x_1) = p(x)$ ``` at time ```latex $t=0$ ```, and we design ```latex $p_1(x|x_1)$ ``` at ```latex $t=1$ ``` to be a distribution concentrated around ```latex $x=x_1$ ```, *e.g.*, ```latex $p_1(x|x_1)={\mathcal{N}}(x|x_1,\sigma^2 I)$ ```, a normal distribution with ```latex $x_1$ ``` mean and a sufficiently small standard deviation ```latex $\sigma > 0$ ```. Marginalizing the conditional probability paths over ```latex $q(x_1)$ ``` give rise to *the marginal probability path*

```latex
$$p_t(x)=\int p_t(x|x_1)q(x_1)dx_1,$$
```

where in particular at time ```latex $t=1$ ```, the marginal probability ```latex $p_1$ ``` is a mixture distribution that closely approximates the data distribution ```latex $q$ ```,

```latex
$$p_1(x)=\int p_1(x|x_1)q(x_1)dx_1\approx q(x).$$
```

Interestingly, we can also define a *marginal vector field*, by "marginalizing" over the conditional vector fields in the following sense (we assume ```latex $p_t(x)>0$ ``` for all ```latex $t$ ``` and ```latex $x$ ```):

```latex
$$u_t(x) = \int u_t(x\vert x_1) \frac{p_t(x\vert x_1)q(x_1)}{p_t(x)}dx_1,$$
```

where ```latex $u_t(\cdot|x_1):\mathbb R^d\rightarrow\mathbb R^d$ ``` is a conditional vector field that generates ```latex $p_t(\cdot\vert x_1)$ ```. It may not seem apparent, but this way of aggregating the conditional vector fields actually results in the correct vector field for modeling the marginal probability path.

Our first key observation is this:

> *The marginal vector field generates the marginal probability path.*

This provides a surprising connection between the conditional VFs (those that generate conditional probability paths) and the marginal VF (those that generate the marginal probability path). This connection allows us to break down the unknown and intractable marginal VF into simpler conditional VFs, which are much simpler to define as these only depend on a single data sample. We formalize this in the following theorem.

**Theorem (Marginal VF).** Given vector fields ```latex $u_t(x|x_1)$ ``` that generate conditional probability paths ```latex $p_t(x|x_1)$ ```, for any distribution ```latex $q(x_1)$ ```, the marginal vector field ```latex $u_t$ ``` generates the marginal probability path ```latex $p_t$ ```, *i.e.*, ```latex $u_t$ ``` and ```latex $p_t$ ``` satisfy the continuity equation.

Theorem (Marginal VF) can also be derived from the Diffusion Mixture Representation Theorem in [peluchetti2021non] that provides a formula for the marginal drift and diffusion coefficients in diffusion SDEs.

## Conditional Flow Matching

Unfortunately, due to the intractable integrals in the definitions of the marginal probability path and VF, it is still intractable to compute ```latex $u_t$ ```, and consequently, intractable to naively compute an unbiased estimator of the original Flow Matching objective. Instead, we propose a simpler objective, which surprisingly will result in the same optima as the original objective. Specifically, we consider the *Conditional Flow Matching* (CFM) objective,

```latex
$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t,q(x_1),p_t(x\vert x_1)} \big\| v_t(x) - u_t(x\vert x_1) \big\|^2,$$
```

where ```latex $t\sim {\mathcal{U}}[0,1]$ ```, ```latex $x_1\sim q(x_1)$ ```, and now ```latex $x\sim p_t(x|x_1)$ ```. Unlike the FM objective, the CFM objective allows us to easily sample unbiased estimates as long as we can efficiently sample from ```latex $p_t(x|x_1)$ ``` and compute ```latex $u_t(x|x_1)$ ```, both of which can be easily done as they are defined on a per-sample basis.

Our second key observation is therefore:

> *The FM and CFM objectives have identical gradients w.r.t. ```latex $\theta$ ```.*

That is, optimizing the CFM objective is equivalent (in expectation) to optimizing the FM objective. Consequently, this allows us to train a CNF to generate the marginal probability path ```latex $p_t$ ```---which in particular, approximates the unknown data distribution ```latex $q$ ``` at ```latex $t=1$ ```--- without ever needing access to either the marginal probability path or the marginal vector field. We simply need to design suitable *conditional* probability paths and vector fields. We formalize this property in the following theorem.

**Theorem (CFM).** Assuming that ```latex $p_t(x)>0$ ``` for all ```latex $x\in\mathbb R^d$ ``` and ```latex $t\in [0,1]$ ```, then, up to a constant independent of ```latex $\theta$ ```, ```latex ${\mathcal{L}}_{\text{CFM}}$ ``` and ```latex ${\mathcal{L}}_{\text{FM}}$ ``` are equal. Hence, ```latex $\nabla_\theta \mathcal{L}_{\text{FM}}(\theta) = \nabla_\theta \mathcal{L}_{\text{CFM}}(\theta)$ ```.

# Conditional Probability Paths and Vector Fields

The Conditional Flow Matching objective works with any choice of conditional probability path and conditional vector fields. In this section, we discuss the construction of ```latex $p_t(x \vert x_1)$ ``` and ```latex $u_t(x \vert x_1)$ ``` for a general family of Gaussian conditional probability paths. Namely, we consider conditional probability paths of the form

```latex
$$p_t(x\vert x_1) = {\mathcal{N}}(x\, \vert\, \mu_t(x_1), \sigma_t(x_1)^2 I ),$$
```

where ```latex $\mu:[0,1]\times \mathbb R^d\rightarrow\mathbb R^d$ ``` is the time-dependent mean of the Gaussian distribution, while ```latex $\sigma:[0,1]\times\mathbb R\rightarrow\mathbb R_{>0}$ ``` describes a time-dependent scalar standard deviation (std). We set ```latex $\mu_0(x_1) = 0$ ``` and ```latex $\sigma_0(x_1) = 1$ ```, so that all conditional probability paths converge to the same standard Gaussian noise distribution at ```latex $t=0$ ```, ```latex $p(x)={\mathcal{N}}(x|0,I)$ ```. We then set ```latex $\mu_1(x_1)=x_1$ ``` and ```latex $\sigma_1(x_1)=\sigma_\text{min}$ ```, which is set sufficiently small so that ```latex $p_1(x \vert x_1)$ ``` is a concentrated Gaussian distribution centered at ```latex $x_1$ ```.

There is an infinite number of vector fields that generate any particular probability path (*e.g.*, by adding a divergence free component to the continuity equation), but the vast majority of these is due to the presence of components that leave the underlying distribution invariant---for instance, rotational components when the distribution is rotation-invariant---leading to unnecessary extra compute. We decide to use the simplest vector field corresponding to a canonical transformation for Gaussian distributions. Specifically, consider the flow (conditioned on ```latex $x_1$ ```)

```latex
$$\psi_t(x) = \sigma_t(x_1)x + \mu_t(x_1).$$
```

When ```latex $x$ ``` is distributed as a standard Gaussian, ```latex $\psi_t(x)$ ``` is the affine transformation that maps to a normally-distributed random variable with mean ```latex $\mu_t(x_1)$ ``` and std ```latex $\sigma_t(x_1)$ ```. That is to say, ```latex $\psi_t$ ``` pushes the noise distribution ```latex $p_0(x \vert x_1)=p(x)$ ``` to ```latex $p_t(x\vert x_1)$ ```, *i.e.*,

```latex
$$\left [\psi_t\right ]_*p(x) = p_t(x\vert x_1).$$
```

This flow then provides a vector field that generates the conditional probability path:

```latex
$$\frac{d}{dt}\psi_t(x) = u_t(\psi_t(x)\vert x_1).$$
```

Reparameterizing ```latex $p_t(x | x_1)$ ``` in terms of just ```latex $x_0$ ``` and plugging into the CFM loss we get

```latex
$${\mathcal{L}}_{\text{CFM}}(\theta) =\mathbb{E}_{t, q(x_1), p(x_0)} \Big\|v_t(\psi_t(x_0)) - \frac{d}{dt}\psi_t(x_0)\Big\|^2.$$
```

Since ```latex $\psi_t$ ``` is a simple (invertible) affine map we can solve for ```latex $u_t$ ``` in a closed form. Let ```latex $f'$ ``` denote the derivative with respect to time, *i.e.*, ```latex $f' = \frac{d}{dt}f$ ```, for a time-dependent function ```latex $f$ ```.

**Theorem (Conditional VF).** Let ```latex $p_t(x\vert x_1)$ ``` be a Gaussian probability path and ```latex $\psi_t$ ``` its corresponding flow map. Then, the unique vector field that defines ```latex $\psi_t$ ``` has the form:

```latex
$$u_t(x\vert x_1) = \frac{\sigma'_t(x_1)}{\sigma_t(x_1)}\left (x-\mu_t(x_1)\right ) + \mu'_t(x_1).$$
```

Consequently, ```latex $u_t(x\vert x_1)$ ``` generates the Gaussian path ```latex $p_t(x\vert x_1)$ ```.

## Special instances of Gaussian conditional probability paths

Our formulation is fully general for arbitrary functions ```latex $\mu_t(x_1)$ ``` and ```latex $\sigma_t(x_1)$ ```, and we can set them to any differentiable function satisfying the desired boundary conditions. We first discuss the special cases that recover probability paths corresponding to previously-used diffusion processes. Since we directly work with probability paths, we can simply depart from reasoning about diffusion processes altogether. Therefore, in the second example below, we directly formulate a probability path based on the Wasserstein-2 optimal transport solution as an interesting instance.

#### Example I: Diffusion conditional VFs.

Diffusion models start with data points and gradually add noise until it approximates pure noise. These can be formulated as stochastic processes, which have strict requirements in order to obtain closed form representation at arbitrary times ```latex $t$ ```, resulting in Gaussian conditional probability paths ```latex $p_t(x\vert x_1)$ ``` with specific choices of mean ```latex $\mu_t(x_1)$ ``` and std ```latex $\sigma_t(x_1)$ ``` [sohl2015deep; ho2020denoising; song2020score]. For example, the reversed (noise to data) Variance Exploding (VE) path has the form

```latex
$$p_t(x) = {\mathcal{N}}(x | x_1 , \sigma_{1-t}^2 I ),$$
```

where ```latex $\sigma_t$ ``` is an increasing function, ```latex $\sigma_0=0$ ```, and ```latex $\sigma_1 \gg 1$ ```. This provides the choices of ```latex $\mu_t(x_1)= x_1$ ``` and ```latex $\sigma_t(x_1)= \sigma_{1-t}$ ```. Plugging these into the conditional VF formula of the Conditional VF Theorem we get

```latex
$$u_t(x|x_1) = -\frac{\sigma'_{1-t}}{\sigma_{1-t}}(x-x_1).$$
```

The reversed (noise to data) Variance Preserving (VP) diffusion path has the form

```latex
$$p_t(x\vert x_1) = {\mathcal{N}}(x\, \vert \, \alpha_{1-t} x_1 , \left (1-\alpha_{1-t}^2\right )I), \text{where } \alpha_t = e^{-\frac{1}{2}T(t)}, T(t)=\int_0^{t} \beta(s)ds,$$
```

and ```latex $\beta$ ``` is the noise scale function. This provides the choices of ```latex $\mu_t(x_1)=\alpha_{1-t} x_1$ ``` and ```latex $\sigma_t(x_1)=\sqrt{1-\alpha_{1-t}^2}$ ```. Plugging these into the conditional VF formula we get

```latex
$$u_t(x\vert x_1) = \frac{\alpha'_{1-t}}{1-\alpha^2_{1-t}} \left (\alpha_{1-t}x-x_1\right ) = -\frac{T'(1-t)}{2}\left [\frac{e^{-T(1-t)}x - e^{-\frac{1}{2}T(1-t)}x_1}{1-e^{-T(1-t)}}\right ].$$
```

Our construction of the conditional VF ```latex $u_t(x|x_1)$ ``` does in fact coincide with the vector field previously used in the deterministic probability flow ([song2020score], equation 13) when restricted to these conditional diffusion processes; see details in Appendix 12. Nevertheless, combining the diffusion conditional VF with the Flow Matching objective offers an attractive training alternative---which we find to be more stable and robust in our experiments---to existing score matching approaches.

Another important observation is that, as these probability paths were previously derived as solutions of diffusion processes, they do not actually reach a true noise distribution in finite time. In practice, ```latex $p_0(x)$ ``` is simply approximated by a suitable Gaussian distribution for sampling and likelihood evaluation. Instead, our construction provides full control over the probability path, and we can just directly set ```latex $\mu_t$ ``` and ```latex $\sigma_t$ ```, as we will do next.

[IMAGE: Figure 1 - Compared to the diffusion path's conditional score function, the OT path's conditional vector field has constant direction in time and is arguably simpler to fit with a parametric model. Blue denotes larger magnitude, red denotes smaller magnitude.]

#### Example II: Optimal Transport conditional VFs.

An arguably more natural choice for conditional probability paths is to define the mean and the std to simply change linearly in time, *i.e.*,

```latex
$$\mu_t(x) = tx_1, \text{ and } \sigma_t(x) = 1-(1-\sigma_{\text{min}})t.$$
```

According to the Conditional VF Theorem this path is generated by the VF

```latex
$$u_t(x|x_1)= \frac{x_1-(1-\sigma_{\text{min}})x}{1-(1-\sigma_{\text{min}})t},$$
```

which, in contrast to the diffusion conditional VF, is defined for all ```latex $t\in[0,1]$ ```. The conditional flow that corresponds to ```latex $u_t(x|x_1)$ ``` is

```latex
$$\psi_t(x) = (1-(1-\sigma_{\text{min}})t)x + tx_1,$$
```

and in this case, the CFM loss takes the form:

```latex
$${\mathcal{L}}_{\text{CFM}}(\theta) =\mathbb{E}_{t, q(x_1), p(x_0)} \Big\|v_t(\psi_t(x_0)) - \Big (x_1 - (1-\sigma_{\min})x_0\Big) \Big\|^2.$$
```

Allowing the mean and std to change linearly not only leads to simple and intuitive paths, but it is actually also optimal in the following sense. The conditional flow ```latex $\psi_t(x)$ ``` is in fact the Optimal Transport (OT) *displacement map* between the two Gaussians ```latex $p_0(x|x_1)$ ``` and ```latex $p_1(x|x_1)$ ```. The OT *interpolant*, which is a probability path, is defined to be (see Definition 1.1 in [mccann1997convexity]):

```latex
$$p_t = [(1-t)\mathrm{id} + t\psi]_\star p_0$$
```

where ```latex $\psi:\mathbb R^d\rightarrow\mathbb R^d$ ``` is the OT map pushing ```latex $p_0$ ``` to ```latex $p_1$ ```, ```latex $\mathrm{id}$ ``` denotes the identity map, *i.e.*, ```latex $\mathrm{id}(x)=x$ ```, and ```latex $(1-t)\mathrm{id} + t\psi$ ``` is called the OT displacement map. Example 1.7 in [mccann1997convexity] shows, that in our case of two Gaussians where the first is a standard one, the OT displacement map takes the above form.

[IMAGE: Figure - Diffusion vs OT sampling trajectories. Diffusion trajectories can overshoot, while OT paths stay straight.]

Intuitively, particles under the OT displacement map always move in straight line trajectories and with constant speed. Sampling trajectory from diffusion paths can "overshoot" the final sample, resulting in unnecessary backtracking, whilst the OT paths are guaranteed to stay straight.

Figure 1 compares the diffusion conditional score function (the regression target in a typical diffusion methods), *i.e.*, ```latex $\nabla \log p_t(x|x_1)$ ```, with the OT conditional VF. The start (```latex $p_0$ ```) and end (```latex $p_1$ ```) Gaussians are identical in both examples. An interesting observation is that the OT VF has a constant direction in time, which arguably leads to a simpler regression task. This property can also be verified directly from the OT VF formula as the VF can be written in the form ```latex $u_t(x|x_1)=g(t)h(x|x_1)$ ```. Lastly, we note that although the conditional flow is optimal, this by no means imply that the marginal VF is an optimal transport solution. Nevertheless, we expect the marginal vector field to remain relatively simple.

# Related Work

Continuous Normalizing Flows were introduced in [chen2018neural] as a continuous-time version of Normalizing Flows (see *e.g.*, [kobyzev2020normalizing; papamakarios2021normalizing] for an overview). Originally, CNFs are trained with the maximum likelihood objective, but this involves expensive ODE simulations for the forward and backward propagation, resulting in high time complexity due to the sequential nature of ODE simulations. Although some works demonstrated the capability of CNF generative models for image synthesis [ffjord2018], scaling up to very high dimensional images is inherently difficult. A number of works attempted to regularize the ODE to be easier to solve, *e.g.*, using augmentation [dupont2019aug], adding regularization terms [yang2019potential; finlay2020how; onken2021ot-flow; tong2020trajectorynet; kelly2020learning], or stochastically sampling the integration interval [du2022toflow]. These works merely aim to regularize the ODE but do not change the fundamental training algorithm.

In order to speed up CNF training, some works have developed simulation-free CNF training frameworks by explicitly designing the target probability path and the dynamics. For instance, [rozen2021moser] consider a linear interpolation between the prior and the target density but involves integrals that were difficult to estimate in high dimensions, while [ben2022matching] consider general probability paths similar to this work but suffers from biased gradients in the stochastic minibatch regime. In contrast, the Flow Matching framework allows simulation-free training with unbiased gradients and readily scales to very high dimensions.

Another approach to simulation-free training relies on the construction of a diffusion process to indirectly define the target probability path [sohl2015deep; ho2020denoising; song2019score]. [song2020score] shows that diffusion models are trained using denoising score matching [vincent2011connection], a conditional objective that provides unbiased gradients with respect to the score matching objective. Conditional Flow Matching draws inspiration from this result, but generalizes to matching vector fields directly. Due to the ease of scalability, diffusion models have received increased attention, producing a variety of improvements such as loss-rescaling [song2021maximum], adding classifier guidance along with architectural improvements [dhariwal2021diffusion], and learning the noise schedule [nichol2021improved; kingma2021vdm]. However, [nichol2021improved] and [kingma2021vdm] only consider a restricted setting of Gaussian conditional paths defined by simple diffusion processes with a single parameter---in particular, it does not include our conditional OT path. In another line of works, [DeBortoli2021schscore; wang2021schbridges; peluchetti2021non] proposed finite time diffusion constructions via diffusion bridges theory resolving the approximation error incurred by infinite time denoising constructions. While existing works make use of a connection between diffusion processes and continuous normalizing flows with the same probability path [maoutsa2020interacting; song2020score; song2021maximum], our work allows us to generalize beyond the class of probability paths modeled by simple diffusion. With our work, it is possible to completely sidestep the diffusion process construction and reason directly with probability paths, while still retaining efficient training and log-likelihood evaluations. Lastly, concurrently to our work [liu2022flow; albergo2022building] arrived at similar conditional objectives for simulation-free training of CNFs, while [neklyudov2023action] derived an implicit objective when ```latex $u_t$ ``` is assumed to be a gradient field.

[IMAGE: Figure 2 - (left) Trajectories of CNFs trained with different objectives on 2D checkerboard data. The OT path introduces the checkerboard pattern much earlier, while FM results in more stable training. (right) FM with OT results in more efficient sampling, solved using the midpoint scheme.]

# Experiments

We explore the empirical benefits of using Flow Matching on the image datasets of CIFAR-10 [krizhevsky2009learning] and ImageNet at resolutions 32, 64, and 128 [chrabaszcz2017downsampled; deng2009-imagenet]. We also ablate the choice of diffusion path in Flow Matching, particularly between the standard variance preserving diffusion path and the optimal transport path. We discuss how sample generation is improved by directly parameterizing the generating vector field and using the Flow Matching objective. Lastly we show Flow Matching can also be used in the conditional generation setting. Unless otherwise specified, we evaluate likelihood and samples from the model using `dopri5` [dormand1980family] at absolute and relative tolerances of 1e-5. Generated samples can be found in the Appendix, and all implementation details are in Appendix 13.

## Density Modeling and Sample Quality on ImageNet

We start by comparing the same model architecture, *i.e.*, the U-Net architecture from [dhariwal2021diffusion] with minimal changes, trained on CIFAR-10, and ImageNet 32/64 with different popular diffusion-based losses: DDPM from [ho2020denoising], Score Matching (SM) [song2020score], and Score Flow (SF) [song2021maximum]; see Appendix 13.1 for exact details. Table (left) summarizes our results alongside these baselines reporting negative log-likelihood (NLL) in units of bits per dimension (BPD), sample quality as measured by the Frechet Inception Distance (FID; [heusel2017gans]), and averaged number of function evaluations (NFE) required for the adaptive solver to reach its a prespecified numerical tolerance, averaged over 50k samples. All models are trained using the same architecture, hyperparameter values and number of training iterations, where baselines are allowed more iterations for better convergence. Note that these are *unconditional* models. On both CIFAR-10 and ImageNet, FM-OT consistently obtains best results across all our quantitative measures compared to competing methods. We are noticing a higher that usual FID performance in CIFAR-10 compared to previous works [ho2020denoising; song2020score; song2021maximum] that can possibly be explained by the fact that our used architecture was not optimized for CIFAR-10.

Secondly, Table (right) compares a model trained using Flow Matching with the OT path on ImageNet at resolution 128x128. Our FID is state-of-the-art with the exception of IC-GAN [casanova2021instance] which uses conditioning with a self-supervised ResNet50 model, and therefore is left out of this table.

[IMAGE: Figure - FID vs epoch for ImageNet 64x64 training curves]

**Faster training.** While existing works train diffusion models with a very high number of iterations (*e.g.*, 1.3m and 10m iterations are reported by Score Flow and VDM, respectively), we find that Flow Matching generally converges much faster. FM-OT is able to lower the FID faster and to a greater extent than the alternatives. For ImageNet-128 [dhariwal2021diffusion] train for 4.36m iterations with batch size 256, while FM (with 25% larger model) used 500k iterations with batch size 1.5k, *i.e.*, 33% less image throughput. Furthermore, the cost of sampling from a model can drastically change during training for score matching, whereas the sampling cost stays constant when training with Flow Matching.

## Sampling Efficiency

[IMAGE: Figure 3 - Sample paths from the same initial noise with models trained on ImageNet 64x64. The OT path reduces noise roughly linearly, while diffusion paths visibly remove noise only towards the end of the path.]

For sampling, we first draw a random noise sample ```latex $x_0 \sim {\mathcal{N}}(0,I)$ ``` then compute ```latex $\phi_1(x_0)$ ``` by solving the ODE with the trained VF, ```latex $v_t$ ```, on the interval ```latex $t\in [0, 1]$ ``` using an ODE solver. While diffusion models can also be sampled through an SDE formulation, this can be highly inefficient and many methods that propose fast samplers (*e.g.*, [song2020denoising; zhang2022fast]) directly make use of the ODE perspective (see Appendix 12). In part, this is due to ODE solvers being much more efficient---yielding lower error at similar computational costs [kloeden2012numerical]---and the multitude of available ODE solver schemes. When compared to our ablation models, we find that models trained using Flow Matching with the OT path always result in the most efficient sampler, regardless of ODE solver, as demonstrated next.

**Sample paths.** We first qualitatively visualize the difference in sampling paths between diffusion and OT. Figure 3 shows samples from ImageNet-64 models using identical random seeds, where we find that the OT path model starts generating images sooner than the diffusion path models, where noise dominates the image until the very last time point. We additionally depict the probability density paths in 2D generation of a checkerboard pattern, Figure 2 (left), noticing a similar trend.

**Low-cost samples.** We next switch to fixed-step solvers and compare low (<=100) NFE samples computed with the ImageNet-32 models. In Figure 4 (left), we compare the per-pixel MSE of low NFE solutions compared with 1000 NFE solutions (we use 256 random noise seeds), and notice that the FM with OT model produces the best numerical error, in terms of computational cost, requiring roughly only 60% of the NFEs to reach the same error threshold as diffusion models. Secondly, Figure 4 (right) shows how FID changes as a result of the computational cost, where we find FM with OT is able to achieve decent FID even at very low NFE values, producing better trade-off between sample quality and cost compared to ablated models. Figure 2 (right) shows low-cost sampling effects for the 2D checkerboard experiment.

[IMAGE: Figure 4 - Flow Matching, especially when using OT paths, allows fewer evaluations for sampling while retaining similar numerical error (left) and sample quality (right). Results on ImageNet 32x32.]

## Conditional sampling from low-resolution images

Lastly, we experimented with Flow Matching for conditional image generation. In particular, upsampling images from 64x64 to 256x256. We follow the evaluation procedure in [saharia2022image] and compute the FID of the upsampled validation images; baselines include reference (FID of original validation set), and regression. FM-OT achieves similar PSNR and SSIM values to [saharia2022image] while considerably improving on FID and IS, which as argued by [saharia2022image] is a better indication of generation quality.

# Conclusion

We introduced Flow Matching, a new simulation-free framework for training Continuous Normalizing Flow models, relying on conditional constructions to effortlessly scale to very high dimensions. Furthermore, the FM framework provides an alternative view on diffusion models, and suggests forsaking the stochastic/diffusion construction in favor of more directly specifying the probability path, allowing us to, *e.g.*, construct paths that allow faster sampling and/or improve generation. We experimentally showed the ease of training and sampling when using the Flow Matching framework, and in the future, we expect FM to open the door to allowing a multitude of probability paths (*e.g.*, non-isotropic Gaussians or more general kernels altogether).

# Social responsibility

Along side its many positive applications, image generation can also be used for harmful proposes. Using content-controlled training sets and image validation/classification can help reduce these uses. Furthermore, the energy demand for training large deep learning models is increasing at a rapid pace [openaiandcompute; thompson2020computational], focusing on methods that are able to train using less gradient updates / image throughput can lead to significant time and energy savings.

# Appendix

## Theorem Proofs

**Proof of Theorem (Marginal VF).** To verify this, we check that ```latex $p_t$ ``` and ```latex $u_t$ ``` satisfy the continuity equation:

```latex
$$\frac{d}{dt}p_t(x) = \int \Big( \frac{d}{dt} p_t(x\vert x_1)  \Big) q(x_1) dx_1 = -\int \mathrm{div}\Big(   u_t(x\vert x_1)p_t(x\vert x_1)  \Big) q(x_1) dx_1 = -\mathrm{div}\Big(\int    u_t(x\vert x_1)p_t(x\vert x_1)   q(x_1) dx_1\Big) = -\mathrm{div} \Big( u_t(x) p_t(x) \Big ),$$
```

where in the second equality we used the fact that ```latex $u_t(\cdot\vert x_1)$ ``` generates ```latex $p_t(\cdot\vert x_1)$ ```, in the last equality we used the marginal VF definition. Furthermore, the first and third equalities are justified by assuming the integrands satisfy the regularity conditions of the Leibniz Rule (for exchanging integration and differentiation).

**Proof of Theorem (CFM).** To ensure existence of all integrals and to allow the changing of integration order (by Fubini's Theorem) we assume that ```latex $q(x)$ ``` and ```latex $p_t(x|x_1)$ ``` are decreasing to zero at a sufficient speed as ```latex $\left\Vert x\right\Vert\rightarrow\infty$ ```, and that ```latex $u_t,v_t,\nabla_\theta v_t$ ``` are bounded.

First, using the standard bilinearity of the 2-norm we have that

```latex
$$\|v_t(x)-u_t(x)\|^2 = \left\Vert v_t(x)\right\Vert^2 - 2\left \langle v_t(x),u_t(x) \right \rangle  + \left\Vert u_t(x)\right\Vert^2$$
$$\|v_t(x)-u_t(x\vert x_1)\|^2  = \left\Vert v_t(x)\right\Vert^2 - 2\left \langle v_t(x),u_t(x\vert x_1) \right \rangle  + \left\Vert u_t(x\vert x_1)\right\Vert^2$$
```

Next, note that

```latex
$$\mathbb{E}_{p_t(x)} \|v_t(x)\|^2 = \int \|v_t(x)\|^2 p_t(x) dx = \int \|v_t(x)\|^2 p_t(x\vert x_1)q(x_1) dx_1 dx = \mathbb{E}_{q(x_1),p_t(x\vert x_1)}\|v_t(x)\|^2,$$
```

where in the second equality we use the marginal probability path definition, and in the third equality we change the order of integration. Next,

```latex
$$\mathbb{E}_{p_t(x)}\left \langle v_t(x),u_t(x) \right \rangle = \int\left \langle v_t(x), u_t(x\vert x_1)  \right \rangle p_t(x\vert x_1)q(x_1)dx_1dx = \mathbb{E}_{q(x_1),p_t(x\vert x_1)} \left \langle v_t(x), u_t(x\vert x_1)  \right \rangle,$$
```

where in the last equality we change again the order of integration.

**Proof of Theorem (Conditional VF).** For notational simplicity let ```latex $w_t(x)=u_t(x\vert x_1)$ ```. Since ```latex $\psi_t$ ``` is invertible (as ```latex $\sigma_t(x_1)>0$ ```) we let ```latex $x=\psi^{-1}(y)$ ``` and get

```latex
$$\psi'_t(\psi^{-1}(y)) = w_t(y).$$
```

Now, inverting ```latex $\psi_t(x)$ ``` provides ```latex $\psi_t^{-1}(y)=\frac{y-\mu_t(x_1)}{\sigma_t(x_1)}$ ```. Differentiating ```latex $\psi_t$ ``` with respect to ```latex $t$ ``` gives ```latex $\psi_t'(x)=\sigma_t'(x_1)x + \mu_t'(x_1)$ ```. Plugging these together we get

```latex
$$w_t(y) = \frac{\sigma'_t(x_1)}{\sigma_t(x_1)}\left (y-\mu_t(x_1)\right ) + \mu_t'(x_1)$$
```

as required.

## The continuity equation

One method of testing if a vector field ```latex $v_t$ ``` generates a probability path ```latex $p_t$ ``` is the continuity equation [villani2009optimal]. It is a Partial Differential Equation (PDE) providing a necessary and sufficient condition to ensuring that a vector field ```latex $v_t$ ``` generates ```latex $p_t$ ```,

```latex
$$\frac{d}{dt}p_t(x) + \mathrm{div}(p_t(x) v_t(x)) = 0,$$
```

where the divergence operator, ```latex $\mathrm{div}$ ```, is defined with respect to the spatial variable ```latex $x=(x^1,\ldots,x^d)$ ```, *i.e.*, ```latex $\mathrm{div}=\sum_{i=1}^d \frac{\partial }{\partial x^i}$ ```.

## Computing probabilities of the CNF model

We are given an arbitrary data point ```latex $x_1\in \mathbb R^d$ ``` and need to compute the model probability at that point, *i.e.*, ```latex $p_1(x_1)$ ```.

#### ODE for computing p_1(x_1)

The continuity equation with the flow ODE lead to the instantaneous change of variable [chen2018neural; ben2022matching]:

```latex
$$\frac{d}{dt}\log p_t(\phi_t(x)) + \mathrm{div}(v_t(\phi_t(x))=0.$$
```

Integrating ```latex $t\in [0,1]$ ``` gives:

```latex
$$\log p_1(\phi_1(x)) - \log p_0(\phi_0(x)) = -\int_0^1 \mathrm{div}(v_t(\phi_t(x))) dt$$
```

Therefore, the log probability can be computed together with the flow trajectory by solving the ODE system. Given initial conditions, the solution is uniquely defined. To compute ```latex $p_1(x_1)$ ``` we solve the reverse ODE and obtain:

```latex
$$\log p_1(x_1) = \log p_0(x_0) - f(0).$$
```

#### Unbiased estimator to p_1(x_1)

Solving the system requires computation of ```latex $\mathrm{div}$ ``` of VFs in ```latex $\mathbb R^d$ ``` which is costly. [ffjord2018] suggest replacing the divergence by the (unbiased) Hutchinson trace estimator, where ```latex $z\in \mathbb R^d$ ``` is a sample from a random variable such that ```latex $\mathbb{E}zz^T=I$ ```. This leads to an unbiased estimator for ```latex $\log p_1(x_1)$ ```.

#### Transformed data

Often, before training our generative model we transform the data, *e.g.*, we scale and/or translate the data. For images ```latex $d=H\times W\times 3$ ``` and we consider a transform that maps each pixel value from ```latex $[-1,1]$ ``` to ```latex $[0,256]$ ```.

#### Bits-Per-Dimension (BPD) computation

BPD is defined by

```latex
$$\mathrm{BPD} = \mathbb{E}_{x_1}\left [ -\frac{\log_2 p_1(x_1)}{d}\right ] = \mathbb{E}_{x_1}\left [ -\frac{\log p_1(x_1)}{d\log 2}\right ]$$
```

Averaging the unbiased estimator on a large test set ```latex $x_1$ ``` provides a good approximation to the test set BPD.

## Diffusion conditional vector fields

We derive the vector field governing the Probability Flow ODE (equation 13 in [song2020score]) for the VE and VP diffusion paths and note that it coincides with the conditional vector fields we derive using the Conditional VF Theorem.

**Lemma (Time Reversal).** Consider a flow defined by a vector field ```latex $u_t(x)$ ``` generating probability density path ```latex $p_t(x)$ ```. Then, the vector field ```latex $\tilde{u}_t(x) = -u_{1-t}(x)$ ``` generates the path ```latex $\tilde{p}_t(x) = p_{1-t}(x)$ ``` when initiated from ```latex $\tilde{p}_0(x) = p_1(x)$ ```.

#### Conditional VFs for Fokker-Planck probability paths

Consider a Stochastic Differential Equation (SDE) of the standard form ```latex $dy = f_t dt + g_t dw$ ``` with drift ```latex $f_t$ ```, diffusion coefficient ```latex $g_t$ ```, and ```latex $dw$ ``` is the Wiener process. The probability density is characterized by the Fokker-Planck equation. Rewriting in the form of the continuity equation gives the vector field

```latex
$$w_t = f_t - \frac{g_t^2}{2}\nabla \log p_t$$
```

which generates ```latex $p_t$ ```.

#### Variance Exploding (VE) path

The SDE for the VE path is ```latex $dy = \sqrt{\frac{d}{dt}\sigma_t^2}dw$ ```, moving from data at ```latex $t=0$ ``` to noise at ```latex $t=1$ ``` with the conditional probability path ```latex $p_t(y|y_0) = {\mathcal{N}}(y | y_0, \sigma^2_t I)$ ```. The conditional VF is ```latex $w_t(y|y_0) = \frac{ \sigma_t'}{\sigma_t}(y-y_0)$ ```. Using the Time Reversal Lemma we get the reversed conditional VF, which coincides with the one derived via the Conditional VF Theorem.

#### Variance Preserving (VP) path

The SDE for the VP path is ```latex $dy = -\frac{T'(t)}{2}y + \sqrt{T'(t)}dw$ ```, where ```latex $T(t)=\int_0^t \beta(s)ds$ ```. The conditional probability path is ```latex $p_t(y \vert  y_0) = {\mathcal{N}}(y \vert e^{-\frac{1}{2}T(t)}y_0, (1-e^{-T(t)})I)$ ```. Using the Fokker-Planck VF formula and the Time Reversal Lemma, the reversed conditional VF coincides with the one derived via the Conditional VF Theorem.

## Implementation details

For the 2D example we used an MLP with 5-layers of 512 neurons each, while for images we used the UNet architecture from [dhariwal2021diffusion]. For images, we center crop images and resize to the appropriate dimension, whereas for the 32x32 and 64x64 resolutions we use the same pre-processing as [chrabaszcz2017downsampled]. The three methods (FM-OT, FM-Diffusion, and SM-Diffusion) are always trained on the same architecture, same hyper-parameters, and for the same number of epochs.

### Diffusion baselines

**Losses.** We consider three options as diffusion baselines that correspond to the most popular diffusion loss parametrizations [song2019score; song2021maximum; ho2020denoising; kingma2021vdm].

Score Matching loss is:

```latex
$${\mathcal{L}}_{\text{SM}}(\theta) = \mathbb{E}_{t,q(x_1),p_t(x|x_1)} \lambda(t) \left\Vert s_t(x) - \nabla \log p_t(x|x_1)\right\Vert^2 = \mathbb{E}_{t,q(x_1),p_t(x|x_1)} \lambda(t) \left\Vert s_t(x) - \frac{x-\mu_t(x_1)}{\sigma_t^2(x_1)}\right\Vert^2.$$
```

Taking ```latex $\lambda(t) = \sigma_t^{2}(x_1)$ ``` corresponds to the original Score Matching (SM) loss from [song2019score], while ```latex $\lambda(t)=\beta(1-t)$ ``` corresponds to the Score Flow (SF) loss [song2021maximum]; ```latex $s_t$ ``` is the learnable score function.

DDPM (Noise Matching) loss from [ho2020denoising] is:

```latex
$${\mathcal{L}}_{\text{NM}}(\theta) = \mathbb{E}_{t,q(x_1),p_0(x_0)} \Big \|  {\epsilon_t(\sigma_t(x_1) x_0 + \mu_t(x_1)) - x_0 }\Big \|^2$$
```

where ```latex $p_0(x)={\mathcal{N}}(x|0,I)$ ``` and ```latex $\epsilon_t$ ``` is the learnable noise function.

**Diffusion path.** For the diffusion path we use the standard VP diffusion, namely,

```latex
$$\mu_t(x_1) = \alpha_{1-t}x_1, \quad \sigma_t(x_1) = \sqrt{1-\alpha_{1-t}^2},\quad  \text{where } \alpha_t = e^{-\frac{1}{2}T(t)},\quad  T(t)=\int_0^{t} \beta(s)ds,$$
```

with ```latex $\beta(s) = \beta_{\min} + s(\beta_{\max}-\beta_{\min})$ ``` and ```latex $\beta_{\min}=0.1$ ```, ```latex $\beta_{\max}=20$ ```.

**Sampling.** Score matching samples are produced by solving the ODE with the vector field:

```latex
$$u_t(x) = -\frac{T'(1-t)}{2}\left [s_t(x) - x\right ].$$
```

DDPM samples are computed similarly after setting ```latex $s_t(x) = \epsilon_t(x)/\sigma_t$ ```.

### Training & evaluation details

We use full 32 bit-precision for training CIFAR10 and ImageNet-32 and 16-bit mixed precision for training ImageNet-64/128/256. All models are trained using the Adam optimizer with ```latex $\beta_1 = 0.9$ ```, ```latex $\beta_2=0.999$ ```, weight decay = 0.0, and ```latex $\epsilon = 1e{-8}$ ```. All methods (FM-OT, FM-Diffusion, SM-Diffusion) use identical architectures, same parameters, and same number of epochs. We use either a constant learning rate schedule or a polynomial decay schedule. The polynomial decay learning rate schedule includes a warm-up phase for a specified number of training steps.

When reporting negative log-likelihood, we dequantize using the standard uniform dequantization. We report an importance-weighted estimate using the `torchdiffeq` [torchdiffeq] library with `dopri5` solver at `atol=rtol=1e-5`.

When computing FID/Inception scores for CIFAR10, ImageNet-32/64 we use the TensorFlow GAN library. To remain comparable to [dhariwal2021diffusion] for ImageNet-128 we use their evaluation script from their publicly available code repository.
