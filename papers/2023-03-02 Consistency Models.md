# Abstract

Diffusion models have significantly advanced the fields of image, audio, and video generation, but they depend on an iterative sampling process that causes slow generation. To overcome this limitation, we propose *consistency models*, a new family of models that generate high quality samples by directly mapping noise to data. They support fast one-step generation by design, while still allowing multistep sampling to trade compute for sample quality. They also support zero-shot data editing, such as image inpainting, colorization, and super-resolution, without requiring explicit training on these tasks. Consistency models can be trained either by distilling pre-trained diffusion models, or as standalone generative models altogether. Through extensive experiments, we demonstrate that they outperform existing distillation techniques for diffusion models in one- and few-step sampling, achieving the new state-of-the-art FID of 3.55 on CIFAR-10 and 6.20 on ImageNet ```latex $64\times 64$ ``` for one-step generation. When trained in isolation, consistency models become a new family of generative models that can outperform existing one-step, non-adversarial generative models on standard benchmarks such as CIFAR-10, ImageNet ```latex $64\times 64$ ``` and LSUN ```latex $256\times 256$ ```.

# Introduction

[IMAGE: Given a Probability Flow (PF) ODE that smoothly converts data to noise, we learn to map any point (e.g., x_t, x_t', and x_T) on the ODE trajectory to its origin (e.g., x_0) for generative modeling. Models of these mappings are called consistency models, as their outputs are trained to be consistent for points on the same trajectory.]

Diffusion models [sohl2015deep; song2019generative; song2020improved; ho2020denoising; song2021scorebased], also known as score-based generative models, have achieved unprecedented success across multiple fields, including image generation [dhariwal2021diffusion; nichol2021glide; ramesh2022hierarchical; saharia2022photorealistic; rombach2022high], audio synthesis [kong2020diffwave; chen2021wavegrad; popov2021grad], and video generation [ho2022video; ho2022imagen]. A key feature of diffusion models is the iterative sampling process which progressively removes noise from random initial vectors. This iterative process provides a flexible trade-off of compute and sample quality, as using extra compute for more iterations usually yields samples of better quality. It is also the crux of many zero-shot data editing capabilities of diffusion models, enabling them to solve challenging inverse problems ranging from image inpainting, colorization, stroke-guided image editing, to Computed Tomography and Magnetic Resonance Imaging [song2019generative; song2021scorebased; song2021medical; song2023pseudoinverseguided; kawar2021snips; kawar2022denoising; chung2023diffusion; meng2021sdedit]. However, compared to single-step generative models like GANs [goodfellow2014generative], VAEs [kingma2013auto; rezende2014stochastic], or normalizing flows [dinh2014nice; dinh2016density; kingma2018glow], the iterative generation procedure of diffusion models typically requires 10--2000 times more compute for sample generation [song2020improved; ho2020denoising; song2021scorebased; zhang2022fast; lu2022dpm], causing slow inference and limited real-time applications.

Our objective is to create generative models that facilitate efficient, single-step generation without sacrificing important advantages of iterative sampling, such as trading compute for sample quality when necessary, as well as performing zero-shot data editing tasks. We build on top of the probability flow (PF) ordinary differential equation (ODE) in continuous-time diffusion models [song2021scorebased], whose trajectories smoothly transition the data distribution into a tractable noise distribution. We propose to learn a model that maps any point at any time step to the trajectory's starting point. A notable property of our model is self-consistency: *points on the same trajectory map to the same initial point*. We therefore refer to such models as **consistency models**. Consistency models allow us to generate data samples (initial points of ODE trajectories, e.g., ```latex ${\mathbf{x}}_0$ ```) by converting random noise vectors (endpoints of ODE trajectories, e.g., ```latex ${\mathbf{x}}_T$ ```) with only one network evaluation. Importantly, by chaining the outputs of consistency models at multiple time steps, we can improve sample quality and perform zero-shot data editing at the cost of more compute, similar to what iterative sampling enables for diffusion models.

To train a consistency model, we offer two methods based on enforcing the self-consistency property. The first method relies on using numerical ODE solvers and a pre-trained diffusion model to generate pairs of adjacent points on a PF ODE trajectory. By minimizing the difference between model outputs for these pairs, we can effectively distill a diffusion model into a consistency model, which allows generating high-quality samples with one network evaluation. By contrast, our second method eliminates the need for a pre-trained diffusion model altogether, allowing us to train a consistency model in isolation. This approach situates consistency models as an independent family of generative models. Importantly, neither approach necessitates adversarial training, and they both place minor constraints on the architecture, allowing the use of flexible neural networks for parameterizing consistency models.

We demonstrate the efficacy of consistency models on several image datasets, including CIFAR-10 [krizhevsky2009learning], ImageNet ```latex $64\times 64$ ``` [deng2009imagenet], and LSUN ```latex $256\times 256$ ``` [yu2015lsun]. Empirically, we observe that as a distillation approach, consistency models outperform existing diffusion distillation methods like progressive distillation [salimans2022progressive] across a variety of datasets in few-step generation: On CIFAR-10, consistency models reach new state-of-the-art FIDs of 3.55 and 2.93 for one-step and two-step generation; on ImageNet ```latex $64\times 64$ ```, it achieves record-breaking FIDs of 6.20 and 4.70 with one and two network evaluations respectively. When trained as standalone generative models, consistency models can match or surpass the quality of one-step samples from progressive distillation, despite having no access to pre-trained diffusion models. They are also able to outperform many GANs, and existing non-adversarial, single-step generative models across multiple datasets. Furthermore, we show that consistency models can be used to perform a wide range of zero-shot data editing tasks, including image denoising, interpolation, inpainting, colorization, super-resolution, and stroke-guided image editing (SDEdit, meng2021sdedit).

# Diffusion Models

Consistency models are heavily inspired by the theory of continuous-time diffusion models [song2021scorebased; karras2022edm]. Diffusion models generate data by progressively perturbing data to noise via Gaussian perturbations, then creating samples from noise via sequential denoising steps. Let ```latex $p_\text{data}({\mathbf{x}})$ ``` denote the data distribution. Diffusion models start by diffusing ```latex $p_\text{data}({\mathbf{x}})$ ``` with a stochastic differential equation (SDE) [song2021scorebased]:

```latex
$$\mathop{}\!\mathrm{d}{\mathbf{x}}_t = \bm{\mu}({\mathbf{x}}_t, t) \mathop{}\!\mathrm{d}t + \sigma(t)\mathop{}\!\mathrm{d}{\mathbf{w}}_t$$
```

where ```latex $t\in[0, T]$ ```, ```latex $T>0$ ``` is a fixed constant, ```latex $\bm{\mu}(\cdot, \cdot)$ ``` and ```latex $\sigma(\cdot)$ ``` are the drift and diffusion coefficients respectively, and ```latex $\{{\mathbf{w}}_t\}_{t\in[0,T]}$ ``` denotes the standard Brownian motion. We denote the distribution of ```latex ${\mathbf{x}}_t$ ``` as ```latex $p_t({\mathbf{x}})$ ``` and as a result ```latex $p_0({\mathbf{x}}) \equiv p_\text{data}({\mathbf{x}})$ ```. A remarkable property of this SDE is the existence of an ordinary differential equation (ODE), dubbed the *Probability Flow (PF) ODE* by song2021scorebased, whose solution trajectories sampled at ```latex $t$ ``` are distributed according to ```latex $p_t({\mathbf{x}})$ ```:

```latex
$$\mathop{}\!\mathrm{d}{\mathbf{x}}_t = \left[\bm{\mu}({\mathbf{x}}_t, t) - \frac{1}{2} \sigma(t)^2 \nabla \log p_t({\mathbf{x}}_t)\right] \mathop{}\!\mathrm{d}t$$
```

Here ```latex $\nabla \log p_t({\mathbf{x}})$ ``` is the *score function* of ```latex $p_t({\mathbf{x}})$ ```; hence diffusion models are also known as *score-based generative models* [song2019generative; song2020improved; song2021scorebased].

Typically, the SDE is designed such that ```latex $p_T({\mathbf{x}})$ ``` is close to a tractable Gaussian distribution ```latex $\pi({\mathbf{x}})$ ```. We hereafter adopt the settings in karras2022edm, where ```latex $\bm{\mu}({\mathbf{x}}, t) = \bm{0}$ ``` and ```latex $\sigma(t) = \sqrt{2t}$ ```. In this case, we have ```latex $p_t({\mathbf{x}}) = p_\text{data}({\mathbf{x}}) \otimes \mathcal{N}(\bm{0}, t^2 {\bm{I}})$ ```, where ```latex $\otimes$ ``` denotes the convolution operation, and ```latex $\pi({\mathbf{x}}) = \mathcal{N}(\bm{0}, T^2{\bm{I}})$ ```. For sampling, we first train a *score model* ```latex ${\bm{s}}_{\bm{\phi}}({\mathbf{x}}, t) \approx \nabla \log p_t({\mathbf{x}})$ ``` via *score matching* [hyvarinen2005estimation; vincent2011connection; song2019sliced; song2019generative; ho2020denoising], then plug it into the PF ODE to obtain an empirical estimate of the PF ODE, which takes the form of:

```latex
$$\frac{\mathop{}\!\mathrm{d}{\mathbf{x}}_t}{\mathop{}\!\mathrm{d}t} = -t {\bm{s}}_{\bm{\phi}}({\mathbf{x}}_t, t)$$
```

We call this the *empirical PF ODE*. Next, we sample ```latex $\hat{{\mathbf{x}}}_T \sim \pi = \mathcal{N}(\bm{0}, T^2 {\bm{I}})$ ``` to initialize the empirical PF ODE and solve it backwards in time with any numerical ODE solver, such as Euler [song2020denoising; song2021scorebased] and Heun solvers [karras2022edm], to obtain the solution trajectory ```latex $\{\hat{{\mathbf{x}}}_t\}_{t\in[0,T]}$ ```. The resulting ```latex $\hat{{\mathbf{x}}}_0$ ``` can then be viewed as an approximate sample from the data distribution ```latex $p_\text{data}({\mathbf{x}})$ ```. To avoid numerical instability, one typically stops the solver at ```latex $t=\epsilon$ ```, where ```latex $\epsilon$ ``` is a fixed small positive number, and accepts ```latex $\hat{{\mathbf{x}}}_{\epsilon}$ ``` as the approximate sample. Following karras2022edm, we rescale image pixel values to ```latex $[-1,1]$ ```, and set ```latex $T=80, \epsilon=0.002$ ```.

Diffusion models are bottlenecked by their slow sampling speed. Clearly, using ODE solvers for sampling requires iterative evaluations of the score model ```latex ${\bm{s}}_{\bm{\phi}}({\mathbf{x}}, t)$ ```, which is computationally costly. Existing methods for fast sampling include faster numerical ODE solvers [song2020denoising; zhang2022fast; lu2022dpm; dockhorn2022genie], and distillation techniques [luhman2021knowledge; salimans2022progressive; meng2022distillation; zheng2022fast]. However, ODE solvers still need more than 10 evaluation steps to generate competitive samples. Most distillation methods like luhman2021knowledge and zheng2022fast rely on collecting a large dataset of samples from the diffusion model prior to distillation, which itself is computationally expensive. To our best knowledge, the only distillation approach that does not suffer from this drawback is progressive distillation (PD, salimans2022progressive), with which we compare consistency models extensively in our experiments.

# Consistency Models

[IMAGE: Consistency models are trained to map points on any trajectory of the PF ODE to the trajectory's origin.]

We propose consistency models, a new type of models that support single-step generation at the core of its design, while still allowing iterative generation for trade-offs between sample quality and compute, and zero-shot data editing. Consistency models can be trained in either the distillation mode or the isolation mode. In the former case, consistency models distill the knowledge of pre-trained diffusion models into a single-step sampler, significantly improving other distillation approaches in sample quality, while allowing zero-shot image editing applications. In the latter case, consistency models are trained in isolation, with no dependence on pre-trained diffusion models. This makes them an independent new class of generative models.

Below we introduce the definition, parameterization, and sampling of consistency models, plus a brief discussion on their applications to zero-shot data editing.

**Definition**  Given a solution trajectory ```latex $\{ {\mathbf{x}}_t \}_{t\in[\epsilon,T]}$ ``` of the PF ODE, we define the *consistency function* as ```latex ${\bm{f}}: ({\mathbf{x}}_t, t) \mapsto {\mathbf{x}}_{\epsilon}$ ```. A consistency function has the property of *self-consistency*: its outputs are consistent for arbitrary pairs of ```latex $({\mathbf{x}}_t, t)$ ``` that belong to the same PF ODE trajectory, i.e., ```latex ${\bm{f}}({\mathbf{x}}_t, t) = {\bm{f}}({\mathbf{x}}_{t'}, t')$ ``` for all ```latex $t, t' \in [\epsilon,T]$ ```. The goal of a *consistency model*, symbolized as ```latex ${\bm{f}}_{\bm{\theta}}$ ```, is to estimate this consistency function ```latex ${\bm{f}}$ ``` from data by learning to enforce the self-consistency property. Note that a similar definition is used for neural flows [bilovs2021neural] in the context of neural ODEs [chen2018neural]. Compared to neural flows, however, we do not enforce consistency models to be invertible.

**Parameterization**  For any consistency function ```latex ${\bm{f}}(\cdot, \cdot)$ ```, we have ```latex ${\bm{f}}({\mathbf{x}}_\epsilon, \epsilon) = {\mathbf{x}}_\epsilon$ ```, i.e., ```latex ${\bm{f}}(\cdot, \epsilon)$ ``` is an identity function. We call this constraint the *boundary condition*. All consistency models have to meet this boundary condition, as it plays a crucial role in the successful training of consistency models. This boundary condition is also the most confining architectural constraint on consistency models. For consistency models based on deep neural networks, we discuss two ways to implement this boundary condition *almost for free*. Suppose we have a free-form deep neural network ```latex $F_{\bm{\theta}}({\mathbf{x}}, t)$ ``` whose output has the same dimensionality as ```latex ${\mathbf{x}}$ ```. The first way is to simply parameterize the consistency model as:

```latex
$${\bm{f}}_{\bm{\theta}}({\mathbf{x}}, t) = \begin{cases}
    {\mathbf{x}}&\quad t = \epsilon\\
    F_{\bm{\theta}}({\mathbf{x}}, t) &\quad t \in (\epsilon, T]
\end{cases}$$
```

The second method is to parameterize the consistency model using skip connections:

```latex
$${\bm{f}}_{\bm{\theta}}({\mathbf{x}}, t) = c_\text{skip}(t) {\mathbf{x}}+ c_\text{out}(t) F_{\bm{\theta}}({\mathbf{x}}, t)$$
```

where ```latex $c_\text{skip}(t)$ ``` and ```latex $c_\text{out}(t)$ ``` are differentiable functions such that ```latex $c_\text{skip}(\epsilon) = 1$ ```, and ```latex $c_\text{out}(\epsilon) = 0$ ```. This way, the consistency model is differentiable at ```latex $t = \epsilon$ ``` if ```latex $F_{\bm{\theta}}({\mathbf{x}}, t), c_\text{skip}(t), c_\text{out}(t)$ ``` are all differentiable, which is critical for training continuous-time consistency models. The second parameterization bears strong resemblance to many successful diffusion models [karras2022edm; balaji2022eDiff-I], making it easier to borrow powerful diffusion model architectures for constructing consistency models. We therefore follow the second parameterization in all experiments.

**Sampling**  With a well-trained consistency model ```latex ${\bm{f}}_{\bm{\theta}}(\cdot, \cdot)$ ```, we can generate samples by sampling from the initial distribution ```latex $\hat{{\mathbf{x}}}_T \sim \mathcal{N}(\bm{0}, T^2 {\bm{I}})$ ``` and then evaluating the consistency model for ```latex $\hat{{\mathbf{x}}}_\epsilon = {\bm{f}}_{\bm{\theta}}(\hat{{\mathbf{x}}}_T, T)$ ```. This involves only one forward pass through the consistency model and therefore *generates samples in a single step*. Importantly, one can also evaluate the consistency model multiple times by alternating denoising and noise injection steps for improved sample quality. This *multistep* sampling procedure provides the flexibility to trade compute for sample quality. It also has important applications in zero-shot data editing. In practice, we find time points ```latex $\{\tau_1, \tau_2, \cdots, \tau_{N-1}\}$ ``` with a greedy algorithm, where the time points are pinpointed one at a time using ternary search to optimize the FID of samples. This assumes that given prior time points, the FID is a unimodal function of the next time point. We find this assumption to hold empirically in our experiments, and leave the exploration of better strategies as future work.

**Multistep Sampling Algorithm:** Given consistency model ```latex ${\bm{f}}_{\bm{\theta}}(\cdot, \cdot)$ ```, sequence of time points ```latex $\tau_1 > \tau_2 > \cdots > \tau_{N-1}$ ```, initial noise ```latex $\hat{{\mathbf{x}}}_T$ ```: compute ```latex ${\mathbf{x}}\gets {\bm{f}}_{\bm{\theta}}(\hat{{\mathbf{x}}}_T, T)$ ```, then for each ```latex $n$ ```: sample ```latex ${\mathbf{z}}\sim \mathcal{N}(\bm{0}, {\bm{I}})$ ```, compute ```latex $\hat{{\mathbf{x}}}_{\tau_n} \gets {\mathbf{x}}+ \sqrt{\tau_n^2 - \epsilon^2} {\mathbf{z}}$ ```, and update ```latex ${\mathbf{x}}\gets {\bm{f}}_{\bm{\theta}}(\hat{{\mathbf{x}}}_{\tau_n}, \tau_n)$ ```. Return ```latex ${\mathbf{x}}$ ```.

**Zero-Shot Data Editing**  Similar to diffusion models, consistency models enable various data editing and manipulation applications in zero shot; they do not require explicit training to perform these tasks. For example, consistency models define a one-to-one mapping from a Gaussian noise vector to a data sample. Similar to latent variable models like GANs, VAEs, and normalizing flows, consistency models can easily interpolate between samples by traversing the latent space. As consistency models are trained to recover ```latex ${\mathbf{x}}_\epsilon$ ``` from any noisy input ```latex ${\mathbf{x}}_t$ ``` where ```latex $t \in [\epsilon, T]$ ```, they can perform denoising for various noise levels. Moreover, the multistep generation procedure is useful for solving certain inverse problems in zero shot by using an iterative replacement procedure similar to that of diffusion models [song2019generative; song2021scorebased; ho2022video]. This enables many applications in the context of image editing, including inpainting, colorization, super-resolution, and stroke-guided image editing as in SDEdit [meng2021sdedit].

# Training Consistency Models via Distillation

We present our first method for training consistency models based on distilling a pre-trained score model ```latex ${\bm{s}}_{\bm{\phi}}({\mathbf{x}}, t)$ ```. Our discussion revolves around the empirical PF ODE, obtained by plugging the score model ```latex ${\bm{s}}_{\bm{\phi}}({\mathbf{x}}, t)$ ``` into the PF ODE. Consider discretizing the time horizon ```latex $[\epsilon, T]$ ``` into ```latex $N-1$ ``` sub-intervals, with boundaries ```latex $t_1=\epsilon < t_2 < \cdots < t_{N}=T$ ```. In practice, we follow karras2022edm to determine the boundaries with the formula ```latex $t_i = (\epsilon^{1/\rho} + \nicefrac{i-1}{N-1} (T^{1/\rho} - \epsilon^{1/\rho}))^\rho$ ```, where ```latex $\rho=7$ ```. When ```latex $N$ ``` is sufficiently large, we can obtain an accurate estimate of ```latex ${\mathbf{x}}_{t_n}$ ``` from ```latex ${\mathbf{x}}_{t_{n+1}}$ ``` by running one discretization step of a numerical ODE solver. This estimate, which we denote as ```latex $\hat{{\mathbf{x}}}_{t_n}^{\bm{\phi}}$ ```, is defined by:

```latex
$$\hat{{\mathbf{x}}}_{t_n}^{\bm{\phi}}\coloneqq {\mathbf{x}}_{t_{n+1}} + (t_n - t_{n+1})\Phi({\mathbf{x}}_{t_{n+1}}, t_{n+1}; {\bm{\phi}})$$
```

where ```latex $\Phi(\cdots; {\bm{\phi}})$ ``` represents the update function of a one-step ODE solver applied to the empirical PF ODE. For example, when using the Euler solver, we have ```latex $\Phi({\mathbf{x}}, t; {\bm{\phi}}) = -t {\bm{s}}_{\bm{\phi}}({\mathbf{x}}, t)$ ``` which corresponds to the following update rule:

```latex
$$\hat{{\mathbf{x}}}_{t_n}^{\bm{\phi}}= {\mathbf{x}}_{t_{n+1}} - (t_n - t_{n+1}) t_{n+1} {\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}}, t_{n+1})$$
```

For simplicity, we only consider one-step ODE solvers in this work. It is straightforward to generalize our framework to multistep ODE solvers and we leave it as future work.

Due to the connection between the PF ODE and the SDE, one can sample along the distribution of ODE trajectories by first sampling ```latex ${\mathbf{x}}\sim p_\text{data}$ ```, then adding Gaussian noise to ```latex ${\mathbf{x}}$ ```. Specifically, given a data point ```latex ${\mathbf{x}}$ ```, we can generate a pair of adjacent data points ```latex $(\hat{{\mathbf{x}}}_{t_n}^{\bm{\phi}}, {\mathbf{x}}_{t_{n+1}})$ ``` on the PF ODE trajectory efficiently by sampling ```latex ${\mathbf{x}}$ ``` from the dataset, followed by sampling ```latex ${\mathbf{x}}_{t_{n+1}}$ ``` from the transition density of the SDE ```latex $\mathcal{N}({\mathbf{x}}, t_{n+1}^2 {\bm{I}})$ ```, and then computing ```latex $\hat{{\mathbf{x}}}_{t_n}^{\bm{\phi}}$ ``` using one discretization step of the numerical ODE solver. Afterwards, we train the consistency model by minimizing its output differences on the pair ```latex $(\hat{{\mathbf{x}}}_{t_n}^{\bm{\phi}}, {\mathbf{x}}_{t_{n+1}})$ ```. This motivates our following *consistency distillation* loss for training consistency models.

**Definition 1** (Consistency Distillation Loss). The consistency distillation loss is defined as:

```latex
$$\mathcal{L}_\text{CD}^N({\bm{\theta}}, {\bm{\theta}}^{-}; {\bm{\phi}}) \coloneqq \mathbb{E}[\lambda(t_n) d({\bm{f}}_{\bm{\theta}}({{\mathbf{x}}}_{t_{n+1}}, t_{n+1}), {\bm{f}}_{{\bm{\theta}}^-}(\hat{{\mathbf{x}}}_{t_n}^{\bm{\phi}}, t_n))]$$
```

where the expectation is taken with respect to ```latex ${\mathbf{x}}\sim p_\text{data}$ ```, ```latex $n \sim \mathcal{U}\llbracket 1,N-1 \rrbracket$ ```, and ```latex ${\mathbf{x}}_{t_{n+1}} \sim \mathcal{N}({\mathbf{x}}; t_{n+1}^2 {\bm{I}})$ ```. Here ```latex $\mathcal{U}\llbracket 1, N-1 \rrbracket$ ``` denotes the uniform distribution over ```latex $\{1,2,\cdots, N-1\}$ ```, ```latex $\lambda(\cdot) \in \mathbb{R}^+$ ``` is a positive weighting function, ```latex ${\bm{\theta}}^-$ ``` denotes a running average of the past values of ```latex ${\bm{\theta}}$ ``` during the course of optimization, and ```latex $d(\cdot, \cdot)$ ``` is a metric function that satisfies ```latex $\forall {\mathbf{x}}, {\mathbf{y}}: d({\mathbf{x}}, {\mathbf{y}}) \geq 0$ ``` and ```latex $d({\mathbf{x}}, {\mathbf{y}}) = 0$ ``` if and only if ```latex ${\mathbf{x}}= {\mathbf{y}}$ ```.

In our experiments, we consider the squared ```latex $\ell_2$ ``` distance ```latex $d({\mathbf{x}}, {\mathbf{y}}) = \|{\mathbf{x}}- {\mathbf{y}}\|^2_2$ ```, ```latex $\ell_1$ ``` distance ```latex $d({\mathbf{x}}, {\mathbf{y}}) = \|{\mathbf{x}}-{\mathbf{y}}\|_1$ ```, and the Learned Perceptual Image Patch Similarity (LPIPS, zhang2018perceptual). We find ```latex $\lambda(t_n) \equiv 1$ ``` performs well across all tasks and datasets. In practice, we minimize the objective by stochastic gradient descent on the model parameters ```latex ${\bm{\theta}}$ ```, while updating ```latex ${\bm{\theta}}^-$ ``` with exponential moving average (EMA). That is, given a decay rate ```latex $0 \leq \mu < 1$ ```, we perform the following update after each optimization step:

```latex
$${\bm{\theta}}^- \leftarrow \operatorname{stopgrad}(\mu {\bm{\theta}}^- + (1-\mu) {\bm{\theta}})$$
```

In alignment with the convention in deep reinforcement learning [mnih2013playing; mnih2015human; lillicrap2015continuous] and momentum based contrastive learning [grill2020bootstrap; he2020momentum], we refer to ```latex ${\bm{f}}_{{\bm{\theta}}^-}$ ``` as the "target network", and ```latex ${\bm{f}}_{\bm{\theta}}$ ``` as the "online network". We find that compared to simply setting ```latex ${\bm{\theta}}^- = {\bm{\theta}}$ ```, the EMA update and "stopgrad" operator can greatly stabilize the training process and improve the final performance of the consistency model.

**Consistency Distillation Algorithm:** Given dataset ```latex $\mathcal{D}$ ```, initial model parameter ```latex ${\bm{\theta}}$ ```, learning rate ```latex $\eta$ ```, ODE solver ```latex $\Phi(\cdot, \cdot; {\bm{\phi}})$ ```, ```latex $d(\cdot, \cdot)$ ```, ```latex $\lambda(\cdot)$ ```, and ```latex $\mu$ ```: Set ```latex ${\bm{\theta}}^- \gets {\bm{\theta}}$ ```. Repeat: Sample ```latex ${\mathbf{x}}\sim \mathcal{D}$ ``` and ```latex $n \sim \mathcal{U}\llbracket 1,N-1 \rrbracket$ ```. Sample ```latex ${\mathbf{x}}_{t_{n+1}} \sim \mathcal{N}({\mathbf{x}}; t_{n+1}^2 {\bm{I}})$ ```. Compute ```latex $\hat{{\mathbf{x}}}_{t_n}^{\bm{\phi}}\gets {\mathbf{x}}_{t_{n+1}} + (t_n - t_{n+1})\Phi({\mathbf{x}}_{t_{n+1}}, t_{n+1}; {\bm{\phi}})$ ```. Compute loss ```latex $\mathcal{L}({\bm{\theta}}, {\bm{\theta}}^{-}; {\bm{\phi}}) \gets \lambda(t_n) d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}}, t_{n+1}), {\bm{f}}_{{\bm{\theta}}^-}(\hat{{\mathbf{x}}}_{t_n}^{\bm{\phi}}, t_n))$ ```. Update ```latex ${\bm{\theta}}\gets {\bm{\theta}}- \eta \nabla_{\bm{\theta}}\mathcal{L}({\bm{\theta}}, {\bm{\theta}}^{-}; {\bm{\phi}})$ ```. Update ```latex ${\bm{\theta}}^- \gets \operatorname{stopgrad}(\mu {\bm{\theta}}^- + (1-\mu) {\bm{\theta}})$ ```.

Below we provide a theoretical justification for consistency distillation based on asymptotic analysis.

**Theorem 1.** Let ```latex $\Delta t \coloneqq \max_{n \in \llbracket 1, N-1\rrbracket}\{|t_{n+1} - t_{n}|\}$ ```, and ```latex ${\bm{f}}(\cdot,\cdot;{\bm{\phi}})$ ``` be the consistency function of the empirical PF ODE. Assume ```latex ${\bm{f}}_{\bm{\theta}}$ ``` satisfies the Lipschitz condition: there exists ```latex $L > 0$ ``` such that for all ```latex $t \in [\epsilon, T]$ ```, ```latex ${\mathbf{x}}$ ```, and ```latex ${\mathbf{y}}$ ```, we have ```latex $\left\lVert{\bm{f}}_{\bm{\theta}}({\mathbf{x}}, t) - {\bm{f}}_{\bm{\theta}}({\mathbf{y}}, t)\right\rVert_2 \leq L \left\lVert{\mathbf{x}}- {\mathbf{y}}\right\rVert_2$ ```. Assume further that for all ```latex $n \in \llbracket 1, N-1 \rrbracket$ ```, the ODE solver called at ```latex $t_{n+1}$ ``` has local error uniformly bounded by ```latex $O((t_{n+1} - t_n)^{p+1})$ ``` with ```latex $p\geq 1$ ```. Then, if ```latex $\mathcal{L}_\text{CD}^N({\bm{\theta}}, {\bm{\theta}}; {\bm{\phi}}) = 0$ ```, we have:

```latex
$$\sup_{n, {\mathbf{x}}}\|{\bm{f}}_{{\bm{\theta}}}({\mathbf{x}}, t_n) - {\bm{f}}({\mathbf{x}}, t_n; {\bm{\phi}})\|_2 = O((\Delta t)^p)$$
```

*Proof.* The proof is based on induction and parallels the classic proof of global error bounds for numerical ODE solvers [suli2003introduction]. Full proof in Appendix.

Since ```latex ${\bm{\theta}}^{-}$ ``` is a running average of the history of ```latex ${\bm{\theta}}$ ```, we have ```latex ${\bm{\theta}}^{-} = {\bm{\theta}}$ ``` when the optimization converges. That is, the target and online consistency models will eventually match each other. If the consistency model additionally achieves zero consistency distillation loss, then Theorem 1 implies that, under some regularity conditions, the estimated consistency model can become arbitrarily accurate, as long as the step size of the ODE solver is sufficiently small. Importantly, our boundary condition ```latex ${\bm{f}}_{\bm{\theta}}({\mathbf{x}}, \epsilon) \equiv {\mathbf{x}}$ ``` precludes the trivial solution ```latex ${\bm{f}}_{\bm{\theta}}({\mathbf{x}}, t) \equiv \bm{0}$ ``` from arising in consistency model training.

The consistency distillation loss ```latex $\mathcal{L}_\text{CD}^N({\bm{\theta}}, {\bm{\theta}}^{-}; {\bm{\phi}})$ ``` can be extended to hold for infinitely many time steps (```latex $N \to \infty$ ```) if ```latex ${\bm{\theta}}^{-} = {\bm{\theta}}$ ``` or ```latex ${\bm{\theta}}^{-} = \operatorname{stopgrad}({\bm{\theta}})$ ```. The resulting continuous-time loss functions do not require specifying ```latex $N$ ``` nor the time steps ```latex $\{t_1, t_2, \cdots, t_N\}$ ```. Nonetheless, they involve Jacobian-vector products and require forward-mode automatic differentiation for efficient implementation, which may not be well-supported in some deep learning frameworks.

# Training Consistency Models in Isolation

Consistency models can be trained without relying on any pre-trained diffusion models. This differs from existing diffusion distillation techniques, making consistency models a new independent family of generative models.

**Consistency Training Algorithm:** Given dataset ```latex $\mathcal{D}$ ```, initial model parameter ```latex ${\bm{\theta}}$ ```, learning rate ```latex $\eta$ ```, step schedule ```latex $N(\cdot)$ ```, EMA decay rate schedule ```latex $\mu(\cdot)$ ```, ```latex $d(\cdot, \cdot)$ ```, and ```latex $\lambda(\cdot)$ ```: Set ```latex ${\bm{\theta}}^- \gets {\bm{\theta}}$ ``` and ```latex $k \gets 0$ ```. Repeat: Sample ```latex ${\mathbf{x}}\sim \mathcal{D}$ ```, and ```latex $n \sim \mathcal{U}\llbracket 1,N(k)-1 \rrbracket$ ```. Sample ```latex ${\mathbf{z}}\sim \mathcal{N}(\bm{0}, {\bm{I}})$ ```. Compute loss ```latex $\mathcal{L}({\bm{\theta}}, {\bm{\theta}}^{-}) \gets \lambda(t_n) d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}+ t_{n+1} {\mathbf{z}}, t_{n+1}),{\bm{f}}_{{\bm{\theta}}^-}({\mathbf{x}}+ t_n {\mathbf{z}}, t_n))$ ```. Update ```latex ${\bm{\theta}}\gets {\bm{\theta}}- \eta \nabla_{\bm{\theta}}\mathcal{L}({\bm{\theta}}, {\bm{\theta}}^{-})$ ```. Update ```latex ${\bm{\theta}}^- \gets \operatorname{stopgrad}(\mu(k) {\bm{\theta}}^- + (1-\mu(k)) {\bm{\theta}})$ ```. Increment ```latex $k \gets k + 1$ ```.

Recall that in consistency distillation, we rely on a pre-trained score model ```latex ${\bm{s}}_{\bm{\phi}}({\mathbf{x}}, t)$ ``` to approximate the ground truth score function ```latex $\nabla \log p_t({\mathbf{x}})$ ```. It turns out that we can avoid this pre-trained score model altogether by leveraging the following unbiased estimator:

```latex
$$\nabla \log p_t({\mathbf{x}}_t) = -\mathbb{E}\left[ \frac{{\mathbf{x}}_t - {\mathbf{x}}}{t^2} \mathrel\bigg| {\mathbf{x}}_t \right]$$
```

where ```latex ${\mathbf{x}}\sim p_\text{data}$ ``` and ```latex ${\mathbf{x}}_t \sim \mathcal{N}({\mathbf{x}}; t^2 {\bm{I}})$ ```. That is, given ```latex ${\mathbf{x}}$ ``` and ```latex ${\mathbf{x}}_t$ ```, we can estimate ```latex $\nabla \log p_t({\mathbf{x}}_t)$ ``` with ```latex $-({\mathbf{x}}_t-{\mathbf{x}})/t^2$ ```.

This unbiased estimate suffices to replace the pre-trained diffusion model in consistency distillation when using the Euler method as the ODE solver in the limit of ```latex $N\to\infty$ ```, as justified by the following result.

**Theorem 2.** Let ```latex $\Delta t \coloneqq \max_{n \in \llbracket 1, N-1\rrbracket}\{|t_{n+1} - t_{n}|\}$ ```. Assume ```latex $d$ ``` and ```latex ${\bm{f}}_{{\bm{\theta}}^{-}}$ ``` are both twice continuously differentiable with bounded second derivatives, the weighting function ```latex $\lambda(\cdot)$ ``` is bounded, and ```latex $\mathbb{E}[\left\lVert\nabla \log p_{t_n}({\mathbf{x}}_{t_{n}})\right\rVert_2^2] < \infty$ ```. Assume further that we use the Euler ODE solver, and the pre-trained score model matches the ground truth, i.e., ```latex $\forall t\in[\epsilon, T]: {\bm{s}}_{{\bm{\phi}}}({\mathbf{x}}, t) \equiv \nabla \log p_t({\mathbf{x}})$ ```. Then:

```latex
$$\mathcal{L}_\text{CD}^N({\bm{\theta}}, {\bm{\theta}}^{-}; {\bm{\phi}}) = \mathcal{L}_\text{CT}^N({\bm{\theta}}, {\bm{\theta}}^{-}) + o(\Delta t)$$
```

where the consistency training objective ```latex $\mathcal{L}_\text{CT}^N({\bm{\theta}}, {\bm{\theta}}^{-})$ ``` is defined as:

```latex
$$\mathbb{E}[\lambda(t_n) d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}+ t_{n+1}{\mathbf{z}}, t_{n+1}), {\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+ t_n{\mathbf{z}}, t_n))]$$
```

where ```latex ${\mathbf{z}}\sim \mathcal{N}(\bf{0}, {\bm{I}})$ ```. Moreover, ```latex $\mathcal{L}_\text{CT}^N({\bm{\theta}}, {\bm{\theta}}^{-}) \geq O(\Delta t)$ ``` if ```latex $\inf_N \mathcal{L}_\text{CD}^N({\bm{\theta}}, {\bm{\theta}}^{-}; {\bm{\phi}}) > 0$ ```.

*Proof.* The proof is based on Taylor series expansion and properties of score functions. A complete proof is provided in the Appendix.

We refer to this as the *consistency training* (CT) loss. Crucially, ```latex $\mathcal{L}({\bm{\theta}}, {\bm{\theta}}^{-})$ ``` only depends on the online network ```latex ${\bm{f}}_{\bm{\theta}}$ ```, and the target network ```latex ${\bm{f}}_{{\bm{\theta}}^{-}}$ ```, while being completely agnostic to diffusion model parameters ```latex ${\bm{\phi}}$ ```.

For improved practical performance, we propose to progressively increase ```latex $N$ ``` during training according to a schedule function ```latex $N(\cdot)$ ```. The intuition is that the consistency training loss has less "variance" but more "bias" with respect to the underlying consistency distillation loss when ```latex $N$ ``` is small (i.e., ```latex $\Delta t$ ``` is large), which facilitates faster convergence at the beginning of training. On the contrary, it has more "variance" but less "bias" when ```latex $N$ ``` is large (i.e., ```latex $\Delta t$ ``` is small), which is desirable when closer to the end of training. For best performance, we also find that ```latex $\mu$ ``` should change along with ```latex $N$ ```, according to a schedule function ```latex $\mu(\cdot)$ ```.

Similar to consistency distillation, the consistency training loss ```latex $\mathcal{L}_\text{CT}^N ({\bm{\theta}}, {\bm{\theta}}^{-})$ ``` can be extended to hold in continuous time (i.e., ```latex $N \to \infty$ ```) if ```latex ${\bm{\theta}}^{-} = \operatorname{stopgrad}({\bm{\theta}})$ ```. This continuous-time loss function does not require schedule functions for ```latex $N$ ``` or ```latex $\mu$ ```, but requires forward-mode automatic differentiation for efficient implementation. Unlike the discrete-time CT loss, there is no undesirable "bias" associated with the continuous-time objective, as we effectively take ```latex $\Delta t \to 0$ ```.

# Experiments

We employ consistency distillation and consistency training to learn consistency models on real image datasets, including CIFAR-10 [krizhevsky2009learning], ImageNet ```latex $64\times 64$ ``` [deng2009imagenet], LSUN Bedroom ```latex $256\times 256$ ```, and LSUN Cat ```latex $256\times 256$ ``` [yu2015lsun]. Results are compared according to Frechet Inception Distance (FID, heusel2017gans, lower is better), Inception Score (IS, salimans2016improved, higher is better), Precision (Prec., kynkaanniemi2019improved, higher is better), and Recall (Rec., kynkaanniemi2019improved, higher is better).

[IMAGE: Various factors that affect consistency distillation (CD) and consistency training (CT) on CIFAR-10. The best configuration for CD is LPIPS, Heun ODE solver, and N = 18. Adaptive schedule functions for N and mu make CT converge significantly faster than fixing them to be constants.]

[IMAGE: Multistep image generation with consistency distillation (CD) on CIFAR-10, ImageNet 64x64, Bedroom 256x256, and Cat 256x256. CD outperforms progressive distillation (PD) across all datasets and sampling steps. The only exception is single-step generation on Bedroom 256x256.]

## Training Consistency Models

We perform a series of experiments on CIFAR-10 to understand the effect of various hyperparameters on the performance of consistency models trained by consistency distillation (CD) and consistency training (CT). We first focus on the effect of the metric function ```latex $d(\cdot, \cdot)$ ```, the ODE solver, and the number of discretization steps ```latex $N$ ``` in CD, then investigate the effect of the schedule functions ```latex $N(\cdot)$ ``` and ```latex $\mu(\cdot)$ ``` in CT.

To set up our experiments for CD, we consider the squared ```latex $\ell_2$ ``` distance, ```latex $\ell_1$ ``` distance, and the Learned Perceptual Image Patch Similarity (LPIPS) as the metric function. For the ODE solver, we compare Euler's forward method and Heun's second order method as detailed in karras2022edm. For the number of discretization steps ```latex $N$ ```, we compare ```latex $N \in \{9, 12, 18, 36, 50, 60, 80, 120\}$ ```. All consistency models trained by CD in our experiments are initialized with the corresponding pre-trained diffusion models, whereas models trained by CT are randomly initialized.

The optimal metric for CD is LPIPS, which outperforms both ```latex $\ell_1$ ``` and ```latex $\ell_2$ ``` by a large margin over all training iterations. This is expected as the outputs of consistency models are images on CIFAR-10, and LPIPS is specifically designed for measuring the similarity between natural images. Heun ODE solver and ```latex $N=18$ ``` are the best choices. Both are in line with the recommendation of karras2022edm despite the fact that we are training consistency models, not diffusion models. Moreover, with the same ```latex $N$ ```, Heun's second order solver uniformly outperforms Euler's first order solver. This corroborates with Theorem 1, which states that the optimal consistency models trained by higher order ODE solvers have smaller estimation errors with the same ```latex $N$ ```. The results also indicate that once ```latex $N$ ``` is sufficiently large, the performance of CD becomes insensitive to ```latex $N$ ```.

Due to the strong connection between CD and CT, we adopt LPIPS for our CT experiments throughout this paper. Unlike CD, there is no need for using Heun's second order solver in CT as the loss function does not rely on any particular numerical ODE solver. The convergence of CT is highly sensitive to ```latex $N$ ```---smaller ```latex $N$ ``` leads to faster convergence but worse samples, whereas larger ```latex $N$ ``` leads to slower convergence but better samples upon convergence. This matches our analysis and motivates our practical choice of progressively growing ```latex $N$ ``` and ```latex $\mu$ ``` for CT to balance the trade-off between convergence speed and sample quality. Adaptive schedules of ```latex $N$ ``` and ```latex $\mu$ ``` significantly improve the convergence speed and sample quality of CT.

### CIFAR-10 Results

| METHOD | NFE | FID | IS |
|--------|-----|-----|-----|
| **Diffusion + Samplers** | | | |
| DDIM [song2020denoising] | 50 | 4.67 | |
| DDIM [song2020denoising] | 20 | 6.84 | |
| DDIM [song2020denoising] | 10 | 8.23 | |
| DPM-solver-2 [lu2022dpm] | 10 | 5.94 | |
| DPM-solver-fast [lu2022dpm] | 10 | 4.70 | |
| 3-DEIS [zhang2022fast] | 10 | **4.17** | |
| **Diffusion + Distillation** | | | |
| Knowledge Distillation* [luhman2021knowledge] | 1 | 9.36 | |
| DFNO* [zheng2022fast] | 1 | 4.12 | |
| 1-Rectified Flow (+distill)* [liu2022flow] | 1 | 6.18 | 9.08 |
| 2-Rectified Flow (+distill)* [liu2022flow] | 1 | 4.85 | 9.01 |
| 3-Rectified Flow (+distill)* [liu2022flow] | 1 | 5.21 | 8.79 |
| PD [salimans2022progressive] | 1 | 8.34 | 8.69 |
| **CD** | 1 | **3.55** | **9.48** |
| PD [salimans2022progressive] | 2 | 5.58 | 9.05 |
| **CD** | 2 | **2.93** | **9.75** |
| **Direct Generation** | | | |
| BigGAN [brock2018large] | 1 | 14.7 | 9.22 |
| Diffusion GAN [xiao2022tackling] | 1 | 14.6 | 8.93 |
| AutoGAN [gong2019autogan] | 1 | 12.4 | 8.55 |
| E2GAN [tian2020off] | 1 | 11.3 | 8.51 |
| ViTGAN [lee2021vitgan] | 1 | 6.66 | 9.30 |
| TransGAN [jiang2021transgan] | 1 | 9.26 | 9.05 |
| StyleGAN2-ADA [karras2020analyzing] | 1 | 2.92 | **9.83** |
| StyleGAN-XL [sauer2022stylegan] | 1 | **1.85** | |
| Score SDE [song2021scorebased] | 2000 | 2.20 | **9.89** |
| DDPM [ho2020denoising] | 1000 | 3.17 | 9.46 |
| LSGM [vahdat2021score] | 147 | 2.10 | |
| PFGM [xu2022poisson] | 110 | 2.35 | 9.68 |
| EDM [karras2022edm] | 35 | **2.04** | 9.84 |
| 1-Rectified Flow [liu2022flow] | 1 | 378 | 1.13 |
| Glow [kingma2018glow] | 1 | 48.9 | 3.92 |
| Residual Flow [chen2019residual] | 1 | 46.4 | |
| GLFlow [xiao2019generative] | 1 | 44.6 | |
| DenseFlow [grcic2021densely] | 1 | 34.9 | |
| DC-VAE [parmar2021dual] | 1 | 17.9 | 8.20 |
| **CT** | 1 | **8.70** | **8.49** |
| **CT** | 2 | **5.83** | **8.85** |

### ImageNet 64x64, LSUN Bedroom and Cat 256x256 Results

| METHOD | NFE | FID | Prec. | Rec. |
|--------|-----|-----|-------|------|
| **ImageNet 64x64** | | | | |
| PD [salimans2022progressive] | 1 | 15.39 | 0.59 | 0.62 |
| DFNO [zheng2022fast] | 1 | 8.35 | | |
| **CD** | 1 | 6.20 | 0.68 | 0.63 |
| PD [salimans2022progressive] | 2 | 8.95 | 0.63 | **0.65** |
| **CD** | 2 | **4.70** | **0.69** | 0.64 |
| ADM [dhariwal2021diffusion] | 250 | **2.07** | 0.74 | 0.63 |
| EDM [karras2022edm] | 79 | 2.44 | 0.71 | **0.67** |
| BigGAN-deep [brock2018large] | 1 | 4.06 | **0.79** | 0.48 |
| **CT** | 1 | 13.0 | 0.71 | 0.47 |
| **CT** | 2 | 11.1 | 0.69 | 0.56 |
| **LSUN Bedroom 256x256** | | | | |
| PD [salimans2022progressive] | 1 | 16.92 | 0.47 | 0.27 |
| PD [salimans2022progressive] | 2 | 8.47 | 0.56 | **0.39** |
| **CD** | 1 | 7.80 | 0.66 | 0.34 |
| **CD** | 2 | **5.22** | **0.68** | **0.39** |
| DDPM [ho2020denoising] | 1000 | 4.89 | 0.60 | 0.45 |
| ADM [dhariwal2021diffusion] | 1000 | **1.90** | 0.66 | **0.51** |
| EDM [karras2022edm] | 79 | 3.57 | 0.66 | 0.45 |
| PGGAN [karras2018progressive] | 1 | 8.34 | | |
| PG-SWGAN [wu2019sliced] | 1 | 8.0 | | |
| TDPM (GAN) [zheng2023truncated] | 1 | 5.24 | | |
| StyleGAN2 [karras2020analyzing] | 1 | 2.35 | 0.59 | 0.48 |
| **CT** | 1 | 16.0 | 0.60 | 0.17 |
| **CT** | 2 | 7.85 | **0.68** | 0.33 |
| **LSUN Cat 256x256** | | | | |
| PD [salimans2022progressive] | 1 | 29.6 | 0.51 | 0.25 |
| PD [salimans2022progressive] | 2 | 15.5 | 0.59 | 0.36 |
| **CD** | 1 | 11.0 | 0.65 | 0.36 |
| **CD** | 2 | **8.84** | **0.66** | **0.40** |
| DDPM [ho2020denoising] | 1000 | 17.1 | 0.53 | 0.48 |
| ADM [dhariwal2021diffusion] | 1000 | **5.57** | 0.63 | **0.52** |
| EDM [karras2022edm] | 79 | 6.69 | **0.70** | 0.43 |
| PGGAN [karras2018progressive] | 1 | 37.5 | | |
| StyleGAN2 [karras2020analyzing] | 1 | 7.25 | 0.58 | 0.43 |
| **CT** | 1 | 20.7 | 0.56 | 0.23 |
| **CT** | 2 | 11.7 | 0.63 | 0.36 |

## Few-Step Image Generation

**Distillation**  In current literature, the most directly comparable approach to our consistency distillation (CD) is progressive distillation (PD, salimans2022progressive); both are thus far the only distillation approaches that *do not construct synthetic data before distillation*. In stark contrast, other distillation techniques, such as knowledge distillation [luhman2021knowledge] and DFNO [zheng2022fast], have to prepare a large synthetic dataset by generating numerous samples from the diffusion model with expensive numerical ODE/SDE solvers. We perform comprehensive comparison for PD and CD on CIFAR-10, ImageNet ```latex $64\times 64$ ```, and LSUN ```latex $256\times 256$ ```, with all results reported in the figures. All methods distill from an EDM [karras2022edm] model that we pre-trained in-house. We note that across all sampling iterations, *using the LPIPS metric uniformly improves PD compared to the squared ```latex $\ell_2$ ``` distance in the original paper of salimans2022progressive*. Both PD and CD improve as we take more sampling steps. We find that CD uniformly outperforms PD across all datasets, sampling steps, and metric functions considered, except for single-step generation on Bedroom ```latex $256\times 256$ ```, where CD with ```latex $\ell_2$ ``` slightly underperforms PD with ```latex $\ell_2$ ```. CD even outperforms distillation approaches that require synthetic dataset construction, such as Knowledge Distillation [luhman2021knowledge] and DFNO [zheng2022fast].

[IMAGE: Samples generated by EDM (top), CT + single-step generation (middle), and CT + 2-step generation (Bottom). All corresponding images are generated from the same initial noise.]

**Direct Generation**  We compare the sample quality of consistency training (CT) with other generative models using one-step and two-step generation. We also include PD and CD results for reference. We observe that CT outperforms existing single-step, non-adversarial generative models, i.e., VAEs and normalizing flows, by a significant margin on CIFAR-10. Moreover, *CT achieves comparable quality to one-step samples from PD without relying on distillation*. Importantly, *all samples obtained from the same initial noise vector share significant structural similarity*, even though CT and EDM models are trained independently from one another. This indicates that CT is less likely to suffer from mode collapse, as EDMs do not.

## Zero-Shot Image Editing

Similar to diffusion models, consistency models allow zero-shot image editing by modifying the multistep sampling process. We demonstrate this capability with a consistency model trained on the LSUN bedroom dataset using consistency distillation. Such a consistency model can colorize gray-scale bedroom images at test time, even though it has never been trained on colorization tasks. The same consistency model can generate high-resolution images from low-resolution inputs. It can also generate images based on stroke inputs created by humans, as in SDEdit for diffusion models [meng2021sdedit]. Again, this editing capability is zero-shot, as the model has not been trained on stroke inputs. We additionally demonstrate the zero-shot capability of consistency models on inpainting, interpolation, and denoising.

[IMAGE: Zero-shot image editing with a consistency model trained by consistency distillation on LSUN Bedroom 256x256, showing colorization, super-resolution, and stroke-guided image generation.]

# Conclusion

We have introduced consistency models, a type of generative models that are specifically designed to support one-step and few-step generation. We have empirically demonstrated that our consistency distillation method outshines the existing distillation techniques for diffusion models on multiple image benchmarks and small sampling iterations. Furthermore, as a standalone generative model, consistency models generate better samples than existing single-step generation models except for GANs. Similar to diffusion models, they also allow zero-shot image editing applications such as inpainting, colorization, super-resolution, denoising, interpolation, and stroke-guided image generation.

In addition, consistency models share striking similarities with techniques employed in other fields, including deep Q-learning [mnih2015human] and momentum-based contrastive learning [grill2020bootstrap; he2020momentum]. This offers exciting prospects for cross-pollination of ideas and methods among these diverse fields.

# Appendix: Proofs

## Notations

We use ```latex ${\bm{f}}_{{\bm{\theta}}}({\mathbf{x}}, t)$ ``` to denote a consistency model parameterized by ```latex ${\bm{\theta}}$ ```, and ```latex ${\bm{f}}({\mathbf{x}}, t; {\bm{\phi}})$ ``` the consistency function of the empirical PF ODE. Here ```latex ${\bm{\phi}}$ ``` symbolizes its dependency on the pre-trained score model ```latex ${\bm{s}}_{\bm{\phi}}({\mathbf{x}}, t)$ ```. For the consistency function of the PF ODE, we denote it as ```latex ${\bm{f}}({\mathbf{x}}, t)$ ```. Given a multi-variate function ```latex ${\bm{h}}({\mathbf{x}}, {\mathbf{y}})$ ```, we let ```latex $\partial_1 {\bm{h}}({\mathbf{x}}, {\mathbf{y}})$ ``` denote the Jacobian of ```latex ${\bm{h}}$ ``` over ```latex ${\mathbf{x}}$ ```, and analogously ```latex $\partial_2 {\bm{h}}({\mathbf{x}}, {\mathbf{y}})$ ``` denote the Jacobian of ```latex ${\bm{h}}$ ``` over ```latex ${\mathbf{y}}$ ```. Unless otherwise stated, ```latex ${\mathbf{x}}$ ``` is supposed to be a random variable sampled from the data distribution ```latex $p_\text{data}({\mathbf{x}})$ ```, ```latex $n$ ``` is sampled uniformly at random from ```latex $\llbracket 1, N-1 \rrbracket$ ```, and ```latex ${\mathbf{x}}_{t_{n}}$ ``` is sampled from ```latex $\mathcal{N}({\mathbf{x}}; t_n^2 {\bm{I}})$ ```.

## Consistency Distillation Proof

The full proof of Theorem 1 proceeds by induction. From ```latex $\mathcal{L}_\text{CD}^N({\bm{\theta}}, {\bm{\theta}}; {\bm{\phi}}) = 0$ ```, we derive that ```latex ${\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}}, t_{n+1}) \equiv {\bm{f}}_{{\bm{\theta}}}(\hat{{\mathbf{x}}}_{t_n}^{\bm{\phi}}, t_n)$ ```. Defining the error vector ```latex ${\bm{e}}_{n} \coloneqq {\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n}}, t_{n}) - {\bm{f}}({\mathbf{x}}_{t_n}, t_n; {\bm{\phi}})$ ```, we establish the recursion:

```latex
$$\left\lVert{\bm{e}}_{n+1}\right\rVert_2 \leq \left\lVert{\bm{e}}_{n}\right\rVert_2 + L\cdot O((t_{n+1} - t_n)^{p+1})$$
```

Since ```latex ${\bm{e}}_1 = \bm{0}$ ``` (due to the boundary condition), induction yields ```latex $\left\lVert{\bm{e}}_{n}\right\rVert_2 \leq O((\Delta t)^p)(T-\epsilon) = O((\Delta t)^p)$ ```.

## Consistency Training Proof

**Lemma 1.** Let ```latex ${\mathbf{x}}\sim p_\text{data}({\mathbf{x}})$ ```, ```latex ${\mathbf{x}}_t \sim \mathcal{N}({\mathbf{x}}; t^2 {\bm{I}})$ ```, and ```latex $p_t({\mathbf{x}}_t) = p_\text{data}({\mathbf{x}}) \otimes \mathcal{N}(\bm{0}, t^2{\bm{I}})$ ```. We have:

```latex
$$\nabla \log p_t({\mathbf{x}}_t) = -\mathbb{E}\left[\frac{{\mathbf{x}}_t - {\mathbf{x}}}{t^2} \mid {\mathbf{x}}_t \right]$$
```

The proof of Theorem 2 uses Taylor expansion and Lemma 1 to show that ```latex $\mathcal{L}_\text{CD}^N({\bm{\theta}}, {\bm{\theta}}^{-}; {\bm{\phi}}) = \mathcal{L}_\text{CT}^N({\bm{\theta}}, {\bm{\theta}}^{-}) + o(\Delta t)$ ```, establishing that the consistency training loss approximates the distillation loss up to a remainder that vanishes faster than ```latex $\Delta t$ ```.

# Appendix: Continuous-Time Extensions

The consistency distillation and consistency training objectives can be generalized to hold for infinite time steps (```latex $N\to\infty$ ```) under suitable conditions.

## Consistency Distillation in Continuous Time

**Theorem 3.** Under regularity conditions, with the Euler solver, we have:

```latex
$$\lim_{N \to \infty} (N-1)^2 \mathcal{L}_\text{CD}^N({\bm{\theta}}, {\bm{\theta}}; {\bm{\phi}}) = \mathcal{L}_\text{CD}^\infty({\bm{\theta}}, {\bm{\theta}}; {\bm{\phi}})$$
```

where:

```latex
$$\mathcal{L}_\text{CD}^\infty({\bm{\theta}}, {\bm{\theta}}; {\bm{\phi}}) = \frac{1}{2} \mathbb{E}\left[\frac{\lambda(t)}{[(\tau^{-1})'(t)]^2} \left(\frac{\partial {\bm{f}}_{\bm{\theta}}({\mathbf{x}}_t, t)}{\partial t} - t \frac{\partial {\bm{f}}_{\bm{\theta}}({\mathbf{x}}_t, t)}{\partial {\mathbf{x}}_t} {\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t}, t)\right)^{\mkern-1.5mu\mathsf{T}}{\bm{G}}({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_t, t)) \left(\frac{\partial {\bm{f}}_{\bm{\theta}}({\mathbf{x}}_t, t)}{\partial t} - t \frac{\partial {\bm{f}}_{\bm{\theta}}({\mathbf{x}}_t, t)}{\partial {\mathbf{x}}_t} {\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t}, t)\right)\right]$$
```

When ```latex $d({\mathbf{x}}, {\mathbf{y}}) = \left\lVert{\mathbf{x}}- {\mathbf{y}}\right\rVert_2^2$ ```, this simplifies to:

```latex
$$\mathcal{L}_\text{CD}^{\infty} ({\bm{\theta}}, {\bm{\theta}}; {\bm{\phi}}) = \mathbb{E}\left[\frac{\lambda(t)}{[(\tau^{-1})'(t)]^2}\left\lVert\frac{\partial {\bm{f}}_{\bm{\theta}}({\mathbf{x}}_t, t)}{\partial t} - t \frac{\partial {\bm{f}}_{\bm{\theta}}({\mathbf{x}}_t, t)}{\partial {\mathbf{x}}_t} {\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t}, t)\right\rVert^2_2 \right]$$
```

This continuous-time objective requires computing Jacobian-vector products, which can be slow in frameworks that do not support forward-mode automatic differentiation.

For the ```latex $\ell_1$ ``` metric, a separate result (Theorem 4) shows:

```latex
$$\lim_{N \to \infty} (N-1) \mathcal{L}_\text{CD}^N({\bm{\theta}}, {\bm{\theta}}; {\bm{\phi}}) = \mathbb{E}\left[\frac{\lambda(t)}{(\tau^{-1})'(t)}\left\lVert t \frac{\partial {\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t}, t)}{\partial {\mathbf{x}}_{t}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t}, t)  - \frac{\partial {\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t}, t)}{\partial t}\right\rVert_1\right]$$
```

When ```latex ${\bm{\theta}}^- = \operatorname{stopgrad}({\bm{\theta}})$ ```, a "pseudo-objective" can be derived (Theorem 5) whose gradient matches the gradient of ```latex $\mathcal{L}_\text{CD}^N$ ``` in the limit of ```latex $N\to\infty$ ```. This pseudo-objective is only meaningful in terms of its gradient---one cannot track training by monitoring its value, but can still apply gradient descent to distill consistency models.

## Consistency Training in Continuous Time

**Theorem 6.** Under regularity conditions, if ```latex ${\bm{\theta}}^{-} = \operatorname{stopgrad}({\bm{\theta}})$ ```, we have:

```latex
$$\lim_{N \to \infty} (N-1)\nabla_{\bm{\theta}}\mathcal{L}_\text{CD}^N({\bm{\theta}}, {\bm{\theta}}^{-}; {\bm{\phi}}) = \lim_{N \to \infty} (N-1)\nabla_{\bm{\theta}}\mathcal{L}_\text{CT}^N({\bm{\theta}}, {\bm{\theta}}^{-}) = \nabla_{\bm{\theta}}\mathcal{L}_\text{CT}^\infty({\bm{\theta}}, {\bm{\theta}}^{-})$$
```

where:

```latex
$$\mathcal{L}_\text{CT}^{\infty} ({\bm{\theta}}, {\bm{\theta}}^{-}) \coloneqq \mathbb{E}\left[\frac{\lambda(t)}{(\tau^{-1})'(t)} {\bm{f}}_{\bm{\theta}}({\mathbf{x}}_t, t) ^{\mkern-1.5mu\mathsf{T}}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^-}({\mathbf{x}}_t, t)) \left(\frac{\partial {\bm{f}}_{{\bm{\theta}}^-}({\mathbf{x}}_t, t)}{\partial t} + \frac{\partial {\bm{f}}_{{\bm{\theta}}^-}({\mathbf{x}}_t, t)}{\partial {\mathbf{x}}_t} \cdot \frac{{\mathbf{x}}_t - {\mathbf{x}}}{t}\right)\right]$$
```

Note that ```latex $\mathcal{L}_\text{CT}^\infty({\bm{\theta}}, {\bm{\theta}}^{-})$ ``` does not depend on the diffusion model parameter ```latex ${\bm{\phi}}$ ``` and hence can be optimized without any pre-trained diffusion models.

## Experimental Verifications

[IMAGE: Comparing discrete consistency distillation/training algorithms with continuous counterparts.]

Experimental comparison on CIFAR-10 shows that discrete-time consistency distillation outperforms continuous-time consistency distillation, possibly due to the larger variance in continuous-time objectives. However, continuous-time CT outperforms discrete-time CT with the same LPIPS metric, likely due to the bias in discrete-time CT since ```latex $\Delta t > 0$ ```.

# Appendix: Additional Experimental Details

### Hyperparameters

| Hyperparameter | CIFAR-10 CD | CIFAR-10 CT | ImageNet 64x64 CD | ImageNet 64x64 CT | LSUN 256x256 CD | LSUN 256x256 CT |
|---|---|---|---|---|---|---|
| Learning rate | 4e-4 | 4e-4 | 8e-6 | 8e-6 | 1e-5 | 1e-5 |
| Batch size | 512 | 512 | 2048 | 2048 | 2048 | 2048 |
| ```latex $\mu$ ``` | 0 | | 0.95 | | 0.95 | |
| ```latex $\mu_0$ ``` | | 0.9 | | 0.95 | | 0.95 |
| ```latex $s_0$ ``` | | 2 | | 2 | | 2 |
| ```latex $s_1$ ``` | | 150 | | 200 | | 150 |
| N | 18 | | 40 | | 40 | |
| ODE solver | Heun | | Heun | | Heun | |
| EMA decay rate | 0.9999 | 0.9999 | 0.999943 | 0.999943 | 0.999943 | 0.999943 |
| Training iterations | 800k | 800k | 600k | 800k | 600k | 1000k |
| Mixed-Precision (FP16) | No | No | Yes | Yes | Yes | Yes |
| Dropout probability | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Number of GPUs | 8 | 8 | 64 | 64 | 64 | 64 |

### Model Architectures

We follow song2021scorebased and dhariwal2021diffusion for model architectures. Specifically, we use the NCSN++ architecture in song2021scorebased for all CIFAR-10 experiments, and take the corresponding network architectures from dhariwal2021diffusion when performing experiments on ImageNet ```latex $64\times 64$ ```, LSUN Bedroom ```latex $256\times 256$ ``` and LSUN Cat ```latex $256\times 256$ ```.

### Parameterization for Consistency Models

We use the same architectures for consistency models as those used for EDMs. The only difference is we slightly modify the skip connections in EDM to ensure the boundary condition holds for consistency models. In EDM [karras2022edm], authors choose:

```latex
$$c_\text{skip}(t) = \frac{\sigma_\text{data}^2}{t^2 + \sigma_\text{data}^2},\quad c_\text{out}(t) = \frac{\sigma_\text{data} t }{\sqrt{\sigma_\text{data}^2 + t^2}}$$
```

where ```latex $\sigma_\text{data} = 0.5$ ```. However, this choice does not satisfy the boundary condition when ```latex $\epsilon \neq 0$ ```. To remedy this, we modify them to:

```latex
$$c_\text{skip}(t) = \frac{\sigma_\text{data}^2}{(t-\epsilon)^2 + \sigma_\text{data}^2},\quad c_\text{out}(t) = \frac{\sigma_\text{data} (t-\epsilon) }{\sqrt{\sigma_\text{data}^2 + t^2}}$$
```

which clearly satisfies ```latex $c_\text{skip}(\epsilon) = 1$ ``` and ```latex $c_\text{out}(\epsilon) = 0$ ```.

### Schedule Functions for Consistency Training

Consistency training requires specifying schedule functions ```latex $N(\cdot)$ ``` and ```latex $\mu(\cdot)$ ```. Throughout our experiments, we use:

```latex
$$N(k) = \left\lceil \sqrt{\frac{k}{K} ((s_1 + 1)^2 - s_0^2) + s_0^2} - 1 \right\rceil + 1$$
$$\mu(k) = \exp\left(\frac{s_0 \log \mu_0}{N(k)}\right)$$
```

where ```latex $K$ ``` denotes the total number of training iterations, ```latex $s_0$ ``` denotes the initial discretization steps, ```latex $s_1 > s_0$ ``` denotes the target discretization steps at the end of training, and ```latex $\mu_0 > 0$ ``` denotes the EMA decay rate at the beginning of model training.

### Training Details

In both consistency distillation and progressive distillation, we distill EDMs [karras2022edm]. We trained these EDMs ourselves according to the specifications given in karras2022edm. We used the Rectified Adam optimizer [liu2019variance], with no learning rate decay or warm-up, and no weight decay. We also applied EMA to the weights of the online consistency models. When using the LPIPS metric on CIFAR-10 and ImageNet ```latex $64\times 64$ ```, we rescale images to resolution ```latex $224\times 224$ ``` with bilinear upsampling before feeding them to the LPIPS network. We performed horizontal flips for data augmentation for all models and on all datasets. We trained all models on a cluster of Nvidia A100 GPUs.

# Appendix: Additional Results on Zero-Shot Image Editing

**Zero-Shot Editing Algorithm:** Given consistency model ```latex ${\bm{f}}_{\bm{\theta}}(\cdot, \cdot)$ ```, sequence of time points ```latex $t_1 > t_2 > \cdots > t_{N}$ ```, reference image ```latex ${\mathbf{y}}$ ```, invertible linear transformation ```latex ${\bm{A}}$ ```, and binary image mask ```latex $\bm{\Omega}$ ```: First compute ```latex ${\mathbf{y}}\gets {\bm{A}}^{-1}[({\bm{A}}{\mathbf{y}}) \odot (1 - \bm{\Omega}) + \bm{0} \odot \bm{\Omega}]$ ```. Sample ```latex ${\mathbf{x}}\sim \mathcal{N}({\mathbf{y}}, t_1^2 {\bm{I}})$ ```, compute ```latex ${\mathbf{x}}\gets {\bm{f}}_{\bm{\theta}}({\mathbf{x}}, t_1)$ ```, apply ```latex ${\mathbf{x}}\gets {\bm{A}}^{-1}[({\bm{A}}{\mathbf{y}}) \odot (1 - \bm{\Omega}) + ({\bm{A}}{\mathbf{x}}) \odot \bm{\Omega}]$ ```. Then for each subsequent ```latex $n$ ```: sample ```latex ${\mathbf{x}}\sim \mathcal{N}({\mathbf{x}}, (t_n^2 - \epsilon^2) {\bm{I}})$ ```, compute ```latex ${\mathbf{x}}\gets {\bm{f}}_{\bm{\theta}}({\mathbf{x}}, t_n)$ ```, apply ```latex ${\mathbf{x}}\gets {\bm{A}}^{-1}[({\bm{A}}{\mathbf{y}}) \odot (1-\bm{\Omega}) + ({\bm{A}}{\mathbf{x}}) \odot \bm{\Omega}]$ ```. Return ```latex ${\mathbf{x}}$ ```.

All zero-shot image editing tasks (inpainting, colorization, super-resolution, stroke-guided generation) can be performed via appropriate choices of ```latex ${\bm{A}}$ ``` and ```latex $\bm{\Omega}$ ``` in the above algorithm.

For **inpainting**, ```latex ${\mathbf{y}}$ ``` is an image where missing pixels are masked out, ```latex $\bm{\Omega}$ ``` marks missing pixels, and ```latex ${\bm{A}}$ ``` is the identity transformation.

For **colorization**, an orthogonal matrix ```latex ${\bm{Q}}$ ``` whose first column is proportional to the grayscale weights ```latex $(0.2989, 0.5870, 0.1140)$ ``` is used to define ```latex ${\bm{A}}$ ```, converting the problem to an inpainting task in a decoupled color space.

For **super-resolution**, a similar orthogonal decomposition strategy is used, where ```latex ${\bm{Q}}$ ``` has its first column equal to ```latex $(\nicefrac{1}{p}, \nicefrac{1}{p}, \cdots, \nicefrac{1}{p})$ ``` for patch size ```latex $p$ ```.

For **stroke-guided image generation** (SDEdit), ```latex ${\bm{A}}= {\bm{I}}$ ``` and ```latex $\bm{\Omega}$ ``` is all ones.

For **denoising**, given an image perturbed with ```latex $\mathcal{N}(\bm{0}; \sigma^2 {\bm{I}})$ ```, as long as ```latex $\sigma \in [\epsilon, T]$ ```, we simply evaluate ```latex ${\bm{f}}_{\bm{\theta}}({\mathbf{x}}, \sigma)$ ```.

For **interpolation**, given two samples ```latex ${\mathbf{x}}_1 = {\bm{f}}_{\bm{\theta}}({\mathbf{z}}_1, T)$ ``` and ```latex ${\mathbf{x}}_2 = {\bm{f}}_{\bm{\theta}}({\mathbf{z}}_2, T)$ ```, we use spherical linear interpolation:

```latex
$${\mathbf{z}}= \frac{\sin[(1-\alpha) \psi]}{\sin(\psi)} {\mathbf{z}}_1 + \frac{\sin(\alpha \psi)}{\sin(\psi)}{\mathbf{z}}_2$$
```

where ```latex $\alpha \in [0, 1]$ ``` and ```latex $\psi = \arccos(\frac{{\mathbf{z}}_1^{\mkern-1.5mu\mathsf{T}}{\mathbf{z}}_2}{\left\lVert{\mathbf{z}}_1\right\rVert_2 \left\lVert{\mathbf{z}}_2\right\rVert_2})$ ```, then evaluate ```latex ${\bm{f}}_{\bm{\theta}}({\mathbf{z}}, T)$ ```.
