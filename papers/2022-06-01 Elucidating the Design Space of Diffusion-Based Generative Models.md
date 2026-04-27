# Abstract 

We argue that the theory and practice of diffusion-based generative models are currently unnecessarily convoluted and seek to remedy the situation by presenting a design space that clearly separates the concrete design choices. This lets us identify several changes to both the sampling and training processes, as well as preconditioning of the score networks. Together, our improvements yield new state-of-the-art FID of 1.79 for CIFAR-10 in a class-conditional setting and 1.97 in an unconditional setting, with much faster sampling (35 network evaluations per image) than prior designs. To further demonstrate their modular nature, we show that our design changes dramatically improve both the efficiency and quality obtainable with pre-trained score networks from previous work, including improving the FID of a previously trained ImageNet-64 model from 2.07 to near-SOTA 1.55, and after re-training with our proposed improvements to a new SOTA of 1.36.

# Introduction

Diffusion-based generative models [SohlDickstein2015] have emerged as a powerful new framework for neural image synthesis, in both unconditional [Ho2020; Nichol2021a; Song2021sde] and conditional [Ho2021cascaded; Nichol2021b; Nichol2021a; Preechakul2021diffusion; Ramesh2022; Rombach2021highresolution; Saharia2021; Song2021sde] settings, even surpassing the quality of GANs [Goodfellow2014] in certain situations [Dhariwal2021]. They are also rapidly finding use in other domains such as audio [Kong2021; Popov2021] and video [Ho2022] generation, image segmentation [Baranchuk2022; Wolleb2022] and language translation [Nachmani2021]. As such, there is great interest in applying these models and improving them further in terms of image/distribution quality, training cost, and generation speed.

The literature on these models is dense on theory, and derivations of sampling schedule, training dynamics, noise level parameterization, etc., tend to be based as directly as possible on theoretical frameworks, which ensures that the models are on a solid theoretical footing. However, this approach has a danger of obscuring the available design space --- a proposed model may appear as a tightly coupled package where no individual component can be modified without breaking the entire system.

As our first contribution, we take a look at the theory behind these models from a practical standpoint, focusing more on the "tangible" objects and algorithms that appear in the training and sampling phases, and less on the statistical processes from which they might be derived. The goal is to obtain better insights into how these components are linked together and what degrees of freedom are available in the design of the overall system. We focus on the broad class of models where a neural network is used to model the score [Hyvarinen05] of a noise level dependent marginal distribution of the training data corrupted by Gaussian noise. Thus, our work is in the context of *denoising score matching* [Vincent11].

Our second set of contributions concerns the sampling processes used to synthesize images using diffusion models. We identify the best-performing time discretization for sampling, apply a higher-order Runge--Kutta method for the sampling process, evaluate different sampler schedules, and analyze the usefulness of stochasticity in the sampling process. The result of these improvements is a significant drop in the number of sampling steps required during synthesis, and the improved sampler can be used as a drop-in replacement with several widely used diffusions models [Nichol2021a; Song2021sde].

The third set of contributions focuses on the training of the score-modeling neural network. While we continue to rely on the commonly used network architectures (DDPM [Ho2020], NCSN [Song2019gradients]), we provide the first principled analysis of the preconditioning of the networks' inputs, outputs, and loss functions in a diffusion model setting and derive best practices for improving the training dynamics. We also suggest an improved distribution of noise levels during training, and note that non-leaking augmentation [Karras2020ada] --- typically used with GANs --- is beneficial for diffusion models as well.

Taken together, our contributions enable significant improvements in result quality, e.g., leading to record FIDs of 1.79 for CIFAR-10 [Krizhevsky2009cifar] and 1.36 for ImageNet [Deng2009imagenet] in 64`$\times$`64 resolution. With all key ingredients of the design space explicitly tabulated, we believe that our approach will allow easier innovation on the individual components, and thus enable more extensive and targeted exploration of the design space of diffusion models. Our implementation and pre-trained models are available at <https://github.com/NVlabs/edm>

# Expressing diffusion models in a common framework 

Let us denote the data distribution by `$p_\text{data}(\boldsymbol{x})$`, with standard deviation `$\sigma_\text{data}$`, and consider the family of mollified distributions `$p(\boldsymbol{x}; \sigma)$` obtained by adding i.i.d. Gaussian noise of standard deviation `$\sigma$` to the data. For `$\sigma_\text{max}\gg\sigma_\text{data}$`, `$p(\boldsymbol{x}; \sigma_\text{max})$` is practically indistinguishable from pure Gaussian noise. The idea of diffusion models is to randomly sample a noise image `$\boldsymbol{x}_0 \sim \mathcal{N}(\mathbf{0}, \sigma_\text{max}^2 \mathbf{I})$`, and sequentially denoise it into images `$\boldsymbol{x}_i$` with noise levels `$\sigma_0 = \sigma_{\text{max}}> \sigma_1 > \dots > \sigma_N = 0$` so that at each noise level `$\boldsymbol{x}_i \sim p(\boldsymbol{x}_i; \sigma_i)$`. The endpoint `$\boldsymbol{x}_N$` of this process is thus distributed according to the data.

Song et al. [Song2021sde] present a stochastic differential equation (SDE) that maintains the desired distribution `$p$` as sample `$\boldsymbol{x}$` evolves over time. This allows the above process to be implemented using a stochastic solver that both removes and adds noise at each iteration. They also give a corresponding "probability flow" ordinary differential equation (ODE) where the only source of randomness is the initial noise image `$\boldsymbol{x}_0$`. Contrary to the usual order of treatment, we begin by examining the ODE, as it offers a fruitful setting for analyzing sampling trajectories and their discretizations. The insights carry over to stochastic sampling, which we reintroduce as a generalization in Section [4].


[IMAGE: denoising/cifar10u-seed84-noisy.jpg] [IMAGE: denoising/cifar10u-seed84-oracle.jpg]


**Figure caption:** Denoising score matching on CIFAR-10. **(a)** Images from the training set corrupted with varying levels of additive Gaussian noise. High levels of noise lead to oversaturated colors; we normalize the images for cleaner visualization. **(b)** Optimal denoising result from minimizing Eq. [eq:score] analytically (see Appendix 8.3). With increasing noise level, the result approaches dataset mean.


#### ODE formulation.

A probability flow ODE [Song2021sde] continuously increases or reduces noise level of the image when moving forward or backward in time, respectively. To specify the ODE, we must first choose a schedule `$\sigma(t)$` that defines the desired noise level at time `$t$`. For example, setting `$\sigma(t)\propto\sqrt{t}$` is mathematically natural, as it corresponds to constant-speed heat diffusion [Fourier1822]. However, we will show in Section [sec:deterministic\] that the choice of schedule has major practical implications and should not be made on the basis of theoretical convenience.

The defining characteristic of the probability flow ODE is that evolving a sample from time `$t_a$` to `$t_b$` (either forward or backward in time) yields a sample . Following previous work [Song2021sde], this requirement is satisfied (see Appendix [8.1] and [8.2]) by 
```latex
$$
\mathrm{d}\boldsymbol{x}= -\dot\sigma(t) ~\sigma(t) ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p \big( \boldsymbol{x}; \sigma(t) \big) ~\mathrm{d}t,$$
```
 where the dot denotes a time derivative. `$\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p(\boldsymbol{x}; \sigma)$` is the *score function* [Hyvarinen05], a vector field that points towards higher density of data at a given noise level. Intuitively, an infinitesimal forward step of this ODE nudges the sample away from the data, at a rate that depends on the change in noise level. Equivalently, a backward step nudges the sample towards the data distribution.

#### Denoising score matching.

The score function has the remarkable property that it does not depend on the generally intractable normalization constant of the underlying density function `$p(\boldsymbol{x}; \sigma)$` [Hyvarinen05], and thus can be much easier to evaluate. Specifically, if `$D(\boldsymbol{x};\sigma)$` is a denoiser function that minimizes the expected `$L_2$` denoising error for samples drawn from `$p_\text{data}$` separately for every `$\sigma$`, i.e., 
```latex
$$\mathbb{E}_{\boldsymbol{y}\sim p_\text{data}} \mathbb{E}_{\boldsymbol{n}\sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})} \lVert D(\boldsymbol{y}+ \boldsymbol{n}; \sigma) - \boldsymbol{y}\rVert^2_2
,\hspace*{2mm}\text{then}\hspace*{2mm}
\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p(\boldsymbol{x}; \sigma) = \big( D(\boldsymbol{x}; \sigma) - \boldsymbol{x}\big) / \sigma^2 ,


}$$
```
 where `$\boldsymbol{y}$` is a training image and `$\boldsymbol{n}$` is noise. In this light, the score function isolates the noise component from the signal in `$\boldsymbol{x}$`, and Eq. [eq:ode\] amplifies (or diminishes) it over time. Figure [1] illustrates the behavior of ideal `$D$` in practice. The key observation in diffusion models is that `$D(\boldsymbol{x};\sigma)$` can be implemented as a neural network `$D_\theta(\boldsymbol{x};\sigma)$` trained according to Eq. [eq:score\]. Note that `$D_\theta$` may include additional pre- and post-processing steps, such as scaling `$\boldsymbol{x}$` to an appropriate dynamic range; we will return to such *preconditioning* in Section [5].

#### Time-dependent signal scaling.

Some methods (see Appendix [9.1]) introduce an additional scale schedule `$s(t)$` and consider `$\boldsymbol{x}= s(t) \hat\boldsymbol{x}$` to be a scaled version of the original, non-scaled variable `$\hat\boldsymbol{x}$`. This changes the time-dependent probability density, and consequently also the ODE solution trajectories. The resulting ODE is a generalization of Eq. [eq:ode\]: 
```latex
$$
\mathrm{d}\boldsymbol{x}= \left[ \frac{\dot s(t)}{s(t)} ~\boldsymbol{x}-s(t)^2 ~\dot\sigma(t) ~\sigma(t) ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\left(\frac{\boldsymbol{x}}{s(t)}; \sigma(t)\right) \right] ~\mathrm{d}t.$$
```
 Note that we explicitly undo the scaling of `$\boldsymbol{x}$` when evaluating the score function to keep the definition of `$p(\boldsymbol{x}; \sigma)$` independent of `$s(t)$`.

#### Solution by discretization.

The ODE to be solved is obtained by substituting Eq. [eq:scoredenoiser\] into Eq. [eq:odescale\] to define the point-wise gradient, and the solution can be found by numerical integration, i.e., taking finite steps over discrete time intervals. This requires choosing both the integration scheme (e.g., Euler or a variant of Runge--Kutta), as well as the discrete sampling times `$\{t_0, t_1, \dots, t_N\}$`. Many prior works rely on Euler's method, but we show in Section [sec:deterministic\] that a 2^nd^ order solver offers a better computational tradeoff. For brevity, we do not provide a separate pseudocode for Euler's method applied to our ODE here, but it can be extracted from Algorithm [alg:heun\] by omitting lines 6--8.

#### Putting it together.

Table [tab:specifics\] presents formulas for reproducing deterministic variants of three earlier methods in our framework. These methods were chosen because they are widely used and achieve state-of-the-art performance, but also because they were derived from different theoretical foundations. Some of our formulas appear quite different from the original papers as indirection and recursion have been removed; see Appendix [9] for details. The main purpose of this reframing is to bring into light all the independent components that often appear tangled together in previous work. In our framework, there are no implicit dependencies between the components --- any choices (within reason) for the individual formulas will, in principle, lead to a functioning model. In other words, changing one component does not necessitate changes elsewhere in order to, e.g., maintain the property that the model converges to the data in the limit. In practice, some choices and combinations will of course work better than others.

# Improvements to deterministic sampling


Improving the output quality and/or decreasing the computational cost of sampling are common topics in diffusion model research (e.g., [Dockhorn2022damped; Jolicoeur2021; Liu2022pseudo; Lu2022dpm; Luhman2021speed; Nichol2021a; Salimans2022; Vahdat2021; Watson2022fastsample; Watson2021efficientsample; Zhang2022exp]). Our hypothesis is that the choices related to the sampling process are largely independent of the other components, such as network architecture and training details. In other words, the training procedure of `$D_\theta$` should not dictate `$\sigma(t)$`, `$s(t)$`, and `$\{t_i\}$`, nor vice versa; from the viewpoint of the sampler, `$D_\theta$` is simply a black box [Watson2022fastsample; Watson2021efficientsample]. We test this by evaluating different samplers on three *pre-trained* models, each representing a different theoretical framework and model family. We first measure baseline results for these models using their original sampler implementations, and then bring these samplers into our unified framework using the formulas in Table [tab:specifics\], followed by our improvements. This allows us to evaluate different practical choices and propose general improvements to the sampling process that are applicable to all models.

We evaluate the "DDPM cont. (VP)" and "NCSN cont. (VE)" models by Song et al. [Song2021sde] trained on unconditional CIFAR-10 [Krizhevsky2009cifar] at 32`$\times$`32, corresponding to the variance preserving (VP) and variance exploding (VE) formulations [Song2021sde], originally inspired by DDPM [Ho2020] and SMLD [Song2019gradients]. We also evaluate the "ADM (dropout)" model by Dhariwal and Nichol [Dhariwal2021] trained on class-conditional ImageNet [Deng2009imagenet] at 64`$\times$`64, corresponding to the improved DDPM (iDDPM) formulation [Nichol2021a]. This model was trained using a discrete set of `$M=1000$` noise levels. Further details are given in Appendix [9].


**Figure caption:** Comparison of deterministic sampling methods using three pre-trained models. For each curve, the dot indicates the lowest NFE whose FID is within 3% of the lowest observed FID.


We evaluate the result quality in terms of Fréchet inception distance (FID) [Heusel2017] computed between 50,000 generated images and all available real images. Figure [2] shows FID as a function of neural function evaluations (NFE), i.e., how many times `$D_\theta$` is evaluated to produce a single image. Given that the sampling process is dominated entirely by the cost of `$D_\theta$`, improvements in NFE translate directly to sampling speed. The original deterministic samplers are shown in blue, and the reimplementations of these methods in our unified framework (orange) yield similar but consistently better results. The differences are explained by certain oversights in the original implementations as well as our more careful treatment of discrete noise levels in the case of DDIM; see Appendix [9]. Note that our reimplementations are fully specified by Algorithm [alg:heun\] and Table [tab:specifics\], even though the original codebases are structured very differently from each other.

#### Discretization and higher-order integrators.

Solving an ODE numerically is necessarily an approximation of following the true solution trajectory. At each step, the solver introduces *truncation error* that accumulates over the course of `$N$` steps. The local error generally scales superlinearly with respect to step size, and thus increasing `$N$` improves the accuracy of the solution.

The commonly used Euler's method is a first order ODE solver with `$\mathcal{O}(h^2)$` local error with respect to step size `$h$`. Higher-order Runge--Kutta methods [Suli2003] scale more favorably but require multiple evaluations of `$D_\theta$` per step. Linear multistep methods have also been recently proposed for sampling diffusion models [Liu2022pseudo; Zhang2022exp]. Through extensive tests, we have found Heun's 2^nd^ order method [Ascher1998] (a.k.a. improved Euler, trapezoidal rule) --- previously explored in the context of diffusion models by Jolicoeur-Martineau et al. [Jolicoeur2021] --- to provide an excellent tradeoff between truncation error and NFE. As illustrated in Algorithm [alg:heun\], it introduces an additional correction step for `$\boldsymbol{x}_{i+1}$` to account for change in `$\mathrm{d}\boldsymbol{x}/ \mathrm{d}t$` between `$t_i$` and `$t_{i+1}$`. This correction leads to `$\mathcal{O}(h^3)$` local error at the cost of one additional evaluation of `$D_\theta$` per step. Note that stepping to `$\sigma=0$` would result in a division by zero, so we revert to Euler's method in this case. We discuss the general family of 2^nd^ order solvers in Appendix [10.2].


\[heun\]  Deterministic sampling using Heun's 2^nd^ order method with arbitrary `$\sigma(t)$` and `$s(t)$`.


1.1

ic


The time steps `$\{t_i\}$` determine how the step sizes and thus truncation errors are distributed between different noise levels. We provide a detailed analysis in Appendix [10.1], concluding that the step size should decrease monotonically with decreasing `$\sigma$` and it does not need to vary on a per-sample basis. We adopt a parameterized scheme where the time steps are defined according to a sequence of noise levels `$\{\sigma_i\}$`, i.e., `$t_i=\sigma^{-1}(\sigma_i)$`. We set `$\sigma_{i<N} = (Ai + B)^\rho$` and select the constants `$A$` and `$B$` so that `$\sigma_0 = \sigma_\text{max}$` and `$\sigma_{N-1} = \sigma_\text{min}$`, which gives 
```latex
$$
\sigma_{i<N} = \big( {\sigma_\text{max}}^\frac{1}{\rho} + {\textstyle\frac{i}{N-1}} ( {\sigma_\text{min}}^\frac{1}{\rho} - {\sigma_\text{max}}^\frac{1}{\rho} ) \big)^\rho \hspace*{3mm}\text{and}\hspace*{3mm}\sigma_N = 0 .$$
```
 Here `$\rho$` controls how much the steps near `$\sigma_\text{min}$` are shortened at the expense of longer steps near `$\sigma_\text{max}$`. Our analysis in Appendix [10.1] shows that setting `$\rho=3$` nearly equalizes the truncation error at each step, but that `$\rho$` in range of 5 to 10 performs much better for sampling images. This suggests that errors near `$\sigma_\text{min}$` have a large impact. We set `$\rho=7$` for the remainder of this paper.

Results for Heun's method and Eq. [eq:discretization\] are shown as the green curves in Figure [2]. We observe consistent improvement in all cases: Heun's method reaches the same FID as Euler's method with considerably lower NFE.

#### Trajectory curvature and noise schedule.

The shape of the ODE solution trajectories is defined by functions `$\sigma(t)$` and `$s(t)$`. The choice of these functions offers a way to reduce the truncation errors discussed above, as their magnitude can be expected to scale proportional to the curvature of `$\mathrm{d}\boldsymbol{x}/ \mathrm{d}t$`. We argue that the best choice for these functions is `$\sigma(t)=t$` and `$s(t)=1$`, which is also the choice made in DDIM [Song2020ddim]. With this choice, the ODE of Eq. [eq:odescale\] simplifies to `$\mathrm{d}\boldsymbol{x}/ \mathrm{d}t= \big( \boldsymbol{x}- D(\boldsymbol{x}; t) \big) / t$` and `$\sigma$` and `$t$` become interchangeable.

An immediate consequence is that at any `$\boldsymbol{x}$` and `$t$`, a single Euler step to `$t=0$` yields the denoised image `$D_\theta(\boldsymbol{x}; t)$`. The tangent of the solution trajectory therefore always points towards the denoiser output. This can be expected to change only slowly with the noise level, which corresponds to largely linear solution trajectories. The 1D ODE sketch of Figure [3]c supports this intuition; the solution trajectories approach linear at both large and small noise levels, and have substantial curvature in only a small region in between. The same effect can be seen with real data in Figure [1]b, where the change between different denoiser targets occurs in a relatively narrow `$\sigma$` range. With the advocated schedule, this corresponds to high ODE curvature being limited to this same range.

The effect of setting `$\sigma(t)=t$` and `$s(t)=1$` is shown as the red curves in Figure [2]. As DDIM already employs these same choices, the red curve is identical to the green one for ImageNet-64. However, VP and VE benefit considerably from switching away from their original schedules.


**Figure caption:** A sketch of ODE curvature in 1D where *p*<sub>data</sub> is two Dirac peaks at **x** = ±1. Horizontal *t* axis is chosen to show *σ* ∈ [0, 25] in each plot, with insets showing *σ* ∈ [0, 1] near the data. Example local gradients are shown with black arrows. **(a)** Variance preserving ODE of Song et al.  has solution trajectories that flatten out to horizontal lines at large *σ*. Local gradients start pointing towards data only at small *σ*. **(b)** Variance exploding variant has extreme curvature near data and the solution trajectories are curved everywhere. **(c)** With the schedule used by DDIM  and us, as *σ* increases the solution trajectories approach straight lines that point towards the mean of data. As *σ* → 0, the trajectories become linear and point towards the data manifold.


#### Discussion.

The choices that we made in this section to improve deterministic sampling are summarized in the *Sampling* part of Table [tab:specifics\]. Together, they reduce the NFE needed to reach high-quality results by a large factor: 7.3`$\times$` for VP, 300`$\times$` for VE, and 3.2`$\times$` for DDIM, corresponding to the highlighted NFE values in Figure [2]. In practice, we can generate 26.3 high-quality CIFAR-10 images per second on a single NVIDIA V100. The consistency of improvements corroborates our hypothesis that the sampling process is orthogonal to how each model was originally trained. As further validation, we show results for the adaptive RK45 method [Dormand1980] using our schedule as the dashed black curves in Figure [2]; the cost of this sophisticated ODE solver outweighs its benefits.

# Stochastic sampling 

Deterministic sampling offers many benefits, e.g., the ability to turn real images into their corresponding latent representations by inverting the ODE. However, it tends to lead to worse output quality [Song2020ddim; Song2021sde] than stochastic sampling that injects fresh noise into the image in each step. Given that ODEs and SDEs recover the same distributions in theory, what exactly is the role of stochasticity?

#### Background. 

The SDEs of Song et al. [Song2021sde] can be generalized [Huang2021; Zhang2021] as a sum of the probability flow ODE of Eq. [eq:ode\] and a time-varying *Langevin diffusion* SDE [Grenander1994] (see Appendix [8.5]): 
```latex
$$
  \mathrm{d}\boldsymbol{x}_{\pm} =
    \underbrace{-\dot\sigma(t) \sigma(t) \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}; \sigma(t) \big) \,\mathrm{d}t}_{\text{probability flow ODE (Eq.~\ref{eq:ode})}}\,\pm\,
    \underbrace{
      \underbrace{\beta(t) \sigma(t)^2 \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}; \sigma(t) \big) \,\mathrm{d}t}_{\text{deterministic noise decay}} +
      \underbrace{\sqrt{2 \beta(t)} \sigma(t) \,\mathrm{d}\omega_t}_{\text{noise injection}}
    }_{\text{Langevin diffusion SDE}} ,$$
```
 where `$\omega_t$` is the standard Wiener process. `$\mathrm{d}\boldsymbol{x}_+$` and `$\mathrm{d}\boldsymbol{x}_-$` are now separate SDEs for moving forward and backward in time, related by the time reversal formula of Anderson [Anderson1982]. The Langevin term can further be seen as a combination of a deterministic score-based denoising term and a stochastic noise injection term, whose net noise level contributions cancel out. As such, `$\beta(t)$` effectively expresses the relative rate at which existing noise is replaced with new noise. The SDEs of Song et al. [Song2021sde] are recovered with the choice `$\beta(t) = {\dot \sigma(t)}/{\sigma(t)}$`, whereby the score vanishes from the forward SDE.


\[stochastic\]  Our stochastic sampler with `$\sigma(t)=t$` and `$s(t)=1$`.


1.1

ic


This perspective reveals why stochasticity is helpful in practice: The implicit Langevin diffusion drives the sample towards the desired marginal distribution at a given time, actively correcting for any errors made in earlier sampling steps. On the other hand, approximating the Langevin term with discrete SDE solver steps introduces error in itself. Previous results [Bao2022analytic; Jolicoeur2021; Song2020ddim; Song2021sde] suggest that non-zero `$\beta(t)$` is helpful, but as far as we can tell, the implicit choice for `$\beta(t)$` in Song et al. [Song2021sde] enjoys no special properties. Hence, the optimal amount of stochasticity should be determined empirically.

#### Our stochastic sampler.

We propose a stochastic sampler that combines our 2^nd^ order deterministic ODE integrator with explicit Langevin-like "churn" of adding and removing noise. A pseudocode is given in Algorithm [alg:stochastic\]. At each step `$i$`, given the sample `$\boldsymbol{x}_i$` at noise level `$t_i$` (`$=\sigma(t_i)$`), we perform two sub-steps. First, we add noise to the sample according to a factor `$\gamma_i\ge0$` to reach a higher noise level . Second, from the increased-noise sample , we solve the ODE backward from to `$t_{i+1}$` with a single step. This yields a sample `$\boldsymbol{x}_{i+1}$` with noise level `$t_{i+1}$`, and the iteration continues. We stress that this is not a general-purpose SDE solver, but a sampling procedure tailored for the specific problem. Its correctness stems from the alternation of two sub-steps that each maintain the correct distribution (up to truncation error in the ODE step). The predictor-corrector sampler of Song et al. [Song2021sde] has a conceptually similar structure to ours.

To analyze the main difference between our method and Euler--Maruyama, we first note a subtle discrepancy in the latter when discretizing Eq. [eq:sde\]. One can interpret Euler--Maruyama as first adding noise and then performing an ODE step, not from the intermediate state after noise injection, but assuming that `$\boldsymbol{x}$` and `$\sigma$` remained at the initial state at the beginning of the iteration step. In our method, the parameters used to evaluate `$D_\theta$` on line 7 of Algorithm [alg:stochastic\] correspond to the state after noise injection, whereas an Euler--Maruyama -like method would use instead of . In the limit of `$\Delta_t$` approaching zero there may be no difference between these choices, but the distinction appears to become significant when pursuing low NFE with large steps.

#### Practical considerations.

Increasing the amount of stochasticity is effective in correcting errors made by earlier sampling steps, but it has its own drawbacks. We have observed (see Appendix [11.1]) that excessive Langevin-like addition and removal of noise results in gradual loss of detail in the generated images with all datasets and denoiser networks. There is also a drift toward oversaturated colors at very low and high noise levels. We suspect that practical denoisers induce a slightly non-conservative vector field in Eq. [eq:scoredenoiser\], violating the premises of Langevin diffusion and causing these detrimental effects. Notably, our experiments with analytical denoisers (such as the one in Figure [1]b) have not shown such degradation.

If the degradation is caused by flaws in `$D_\theta(\boldsymbol{x}; \sigma)$`, they can only be remedied using heuristic means during sampling. We address the drift toward oversaturated colors by only enabling stochasticity within a specific range of noise levels `$t_i \in [S_\text{tmin}, S_\text{tmax}]$`. For these noise levels, we define `$\gamma_i = S_\text{churn}/ N$`, where `$S_\text{churn}$` controls the overall amount of stochasticity. We further clamp `$\gamma_i$` to never introduce more new noise than what is already present in the image. Finally, we have found that the loss of detail can be partially counteracted by setting `$S_\text{noise}$` slightly above `$1$` to inflate the standard deviation for the newly added noise. This suggests that a major component of the hypothesized non-conservativity of `$D_\theta(\boldsymbol{x};\sigma)$` is a tendency to remove slightly too much noise --- most likely due to regression toward the mean that can be expected to happen with any `$L_2$`-trained denoiser [Lehtinen2018].


**Figure caption:** Evaluation of our stochastic sampler (Algorithm [alg:stochastic]). The purple curve corresponds to optimal choices for {*S*<sub>churn</sub>, *S*<sub>tmin</sub>, *S*<sub>tmax</sub>, *S*<sub>noise</sub>}; orange, blue, and green correspond to disabling the effects of *S*<sub>tmin,tmax</sub> and/or *S*<sub>noise</sub>. The red curves show reference results for our deterministic sampler (Algorithm [alg:heun]), equivalent to setting *S*<sub>churn</sub> = 0. The dashed black curves correspond to the original stochastic samplers from previous work: Euler–Maruyama  for VP, predictor-corrector  for VE, and iDDPM  for ImageNet-64. The dots indicate lowest observed FID.


#### Evaluation.

Figure [4] shows that our stochastic sampler outperforms previous samplers [Jolicoeur2021; Nichol2021a; Song2021sde] by a significant margin, especially at low step counts. Jolicoeur-Martineau et al. [Jolicoeur2021] use a standard higher-order adaptive SDE solver [Roberts2012sde] and its performance is a good baseline for such solvers in general. Our sampler has been tailored to the use case by, e.g., performing noise injection and ODE step sequentially, and it is not adaptive. It is an open question if adaptive solvers can be a net win over a well-tuned fixed schedule in sampling diffusion models.

Through sampler improvements alone, we are able to bring the ImageNet-64 model that originally achieved FID 2.07 [Dhariwal2021] to 1.55 that is very close to the state-of-the-art; previously, FID 1.48 has been reported for cascaded diffusion [Ho2021cascaded], 1.55 for classifier-free guidance [Ho2021classifierfree], and 1.52 for StyleGAN-XL [Sauer2022]. While our results showcase the potential gains achievable through sampler improvements, they also highlight the main shortcoming of stochasticity: For best results, one must make several heuristic choices --- either implicit or explicit --- that depend on the specific model. Indeed, we had to find the optimal values of `$\{S_\text{churn}, S_\text{tmin}, S_\text{tmax}, S_\text{noise}\}$` on a case-by-case basis using grid search (Appendix [11.2]). This raises a general concern that using stochastic sampling as the primary means of evaluating model improvements may inadvertently end up influencing the design choices related to model architecture and training.

# Preconditioning and training 

There are various known good practices for training neural networks in a supervised fashion. For example, it is advisable to keep input and output signal magnitudes fixed to, e.g., unit variance, and to avoid large variation in gradient magnitudes on a per-sample basis [Bishop1995book; Huang2020normalize]. Training a neural network to model `$D$` directly would be far from ideal ---  for example, as the input `$\boldsymbol{x}=\boldsymbol{y}+\boldsymbol{n}$` is a combination of clean signal `$\boldsymbol{y}$` and noise `$\boldsymbol{n}\sim\mathcal{N}(\mathbf{0},\sigma^2 \mathbf{I})$`, its magnitude varies immensely depending on noise level `$\sigma$`. For this reason, the common practice is to not represent `$D_\theta$` as a neural network directly, but instead train a different network `$F_\theta$` from which `$D_\theta$` is derived.

Previous methods [Nichol2021a; Song2020ddim; Song2021sde] address the input scaling via a `$\sigma$`-dependent normalization factor and attempt to precondition the output by training `$F_\theta$` to predict `$\boldsymbol{n}$` scaled to unit variance, from which the signal is then reconstructed via `$D_\theta(\boldsymbol{x};\sigma) = \boldsymbol{x}-\sigma F_\theta(\cdot)$`. This has the drawback that at large `$\sigma$`, the network needs to fine-tune its output carefully to cancel out the existing noise `$\boldsymbol{n}$` exactly and give the output at the correct scale; note that any errors made by the network are amplified by a factor of `$\sigma$`. In this situation, it would seem much easier to predict the expected output `$D(\boldsymbol{x}; \sigma)$` directly. In the same spirit as previous parameterizations that adaptively mix signal and noise (e.g., [Dockhorn2022damped; Salimans2022; Vahdat2021]), we propose to precondition the neural network with a `$\sigma$`-dependent skip connection that allows it to estimate either `$\boldsymbol{y}$` or `$\boldsymbol{n}$`, or something in between. We thus write `$D_\theta$` in the following form: 
```latex
$$D_\theta(\boldsymbol{x}; \sigma) = c_\text{skip}(\sigma) ~\boldsymbol{x}+ c_\text{out}(\sigma) ~F_\theta \big( c_\text{in}(\sigma) ~\boldsymbol{x}; ~c_\text{noise}(\sigma) \big) ,
$$
```
 where `$F_\theta$` is the neural network to be trained, `$c_\text{skip}(\sigma)$` modulates the skip connection, `$c_\text{in}(\sigma)$` and `$c_\text{out}(\sigma)$` scale the input and output magnitudes, and `$c_\text{noise}(\sigma)$` maps noise level `$\sigma$` into a conditioning input for `$F_\theta$`. Taking a weighted expectation of Eq. [eq:score\] over the noise levels gives the overall training loss , where , , and . The probability of sampling a given noise level `$\sigma$` is given by `$p_\text{train}(\sigma)$` and the corresponding weight is given by `$\lambda(\sigma)$`. We can equivalently express this loss with respect to the raw network output `$F_\theta$` in Eq. [eq:preconditioning\]: 
```latex
$$\mathbb{E}_{\sigma, \boldsymbol{y}, \boldsymbol{n}} \Big[
    \underbrace{\lambda(\sigma) ~ c_\text{out}(\sigma)^2}_{\text{effective weight}}
    \big\Vert
      \underbrace{F_\theta \big( c_\text{in}(\sigma) \cdot (\boldsymbol{y}+ \boldsymbol{n}); c_\text{noise}(\sigma) \big)}_{\text{network output}} -
      \underbrace{\tfrac{1}{c_\text{out}(\sigma)} \big(\boldsymbol{y}- c_\text{skip}(\sigma) \cdot (\boldsymbol{y}+ \boldsymbol{n}) \big)}_{\text{effective training target}}
    \big\Vert^2_2 \Big] .
  $$
```
 This form reveals the effective training target of `$F_\theta$`, allowing us to determine suitable choices for the preconditioning functions from first principles. As detailed in Appendix [8.6], we derive our choices shown in Table [tab:specifics\] by requiring network inputs and training targets to have unit variance (`$c_\text{in}$`, `$c_\text{out}$`), and amplifying errors in `$F_\theta$` as little as possible (`$c_\text{skip}$`). The formula for `$c_\text{noise}$` is chosen empirically.


Table [tab:TrainingTable\] shows FID for a series of training setups, evaluated using our deterministic sampler from Section [sec:deterministic\]. We start with the baseline training setup of Song et al. [Song2021sde], which differs considerably between the VP and VE cases; we provide separate results for each (config \textsc{a}). To obtain a more meaningful point of comparison, we re-adjust the basic hyperparameters (config \textsc{b}) and improve the expressive power of the model (config \textsc{c}) by removing the lowest-resolution layers and doubling the capacity of the highest-resolution layers instead; see Appendix [12.3] for further details. We then replace the original choices of `$\{c_\text{in}, c_\text{out}, c_\text{noise}, c_\text{skip}\}$` with our preconditioning (config \textsc{d}), which keeps the results largely unchanged --- except for VE that improves considerably at 64`$\times$`64 resolution. Instead of improving FID per se, the main benefit of our preconditioning is that it makes the training more robust, enabling us to turn our focus on redesigning the loss function without adverse effects.

#### Loss weighting and sampling.

Eq. [eq:precloss\] shows that training `$F_\theta$` as preconditioned in Eq. [eq:preconditioning\] incurs an effective per-sample loss weight of `$\lambda(\sigma)c_\text{out}(\sigma)^2$`. To balance the effective loss weights, we set `$\lambda(\sigma)=1/c_\text{out}(\sigma)^2$`, which also equalizes the initial training loss over the entire `$\sigma$` range as shown in Figure [5]a (green curve). Finally, we need to select `$p_\text{train}(\sigma)$`, i.e., how to choose noise levels during training. Inspecting the per-`$\sigma$` loss after training (blue and orange curves) reveals that a significant reduction is possible only at intermediate noise levels; at very low levels, it is both difficult and irrelevant to discern the vanishingly small noise component, whereas at high levels the training targets are always dissimilar from the correct answer that approaches dataset average. Therefore, we target the training efforts to the relevant range using a simple log-normal distribution for `$p_\text{train}(\sigma)$` as detailed in Table [tab:specifics\] and illustrated in Figure [5]a (red curve).

Table [tab:TrainingTable\] shows that our proposed `$p_\text{train}$` and `$\lambda$` (config \textsc{e}) lead to a dramatic improvement in FID in all cases when used in conjunction with our preconditioning (config \textsc{d}). In concurrent work, Choi et al. [Choi2022] propose a similar scheme to prioritize noise levels that are most relevant w.r.t. forming the perceptually recognizable content of the image. However, they only consider the choice of `$\lambda$` in isolation, which results in a smaller overall improvement.

#### Augmentation regularization.

To prevent potential overfitting that often plagues diffusion models with smaller datasets, we borrow an augmentation pipeline from the GAN literature [Karras2020ada]. The pipeline consists of various geometric transformations (see Appendix [12.2]) that we apply to a training image prior to adding noise. To prevent the augmentations from leaking to the generated images, we provide the augmentation parameters as a conditioning input to `$F_\theta$`; during inference we set the them to zero to guarantee that only non-augmented images are generated. Table [tab:TrainingTable\] shows that data augmentation provides a consistent improvement (config \textsc{f}) that yields new state-of-the-art FIDs of 1.79 and 1.97 for conditional and unconditional CIFAR-10, beating the previous records of 1.85 [Sauer2022] and 2.10 [Vahdat2021].

#### Stochastic sampling revisited.


**Figure caption:** **(a)** Observed initial (green) and final loss per noise level, representative of the the 32×32 (blue) and 64×64 (orange) models considered in this paper. The shaded regions represent the standard deviation over 10k random samples. Our proposed training sample density is shown by the dashed red curve. **(b)** Effect of *S*<sub>churn</sub> on unconditional CIFAR-10 with 256 steps (NFE = 511). For the original training setup of Song et al. , stochastic sampling is highly beneficial (blue, green), while deterministic sampling (*S*<sub>churn</sub> = 0) leads to relatively poor FID. For our training setup, the situation is reversed (orange, red); stochastic sampling is not only unnecessary but harmful. **(c)** Effect of *S*<sub>churn</sub> on class-conditional ImageNet-64 with 256 steps (NFE = 511). In this more challenging scenario, stochastic sampling turns out to be useful again. Our training setup improves the results for both deterministic and stochastic sampling.


Interestingly, the relevance of stochastic sampling appears to diminish as the model itself improves, as shown in Figure [5]b,c. When using our training setup in CIFAR-10 (Figure [5]b), the best results were obtained with deterministic sampling, and any amount of stochastic sampling was detrimental.

#### ImageNet-64.

As a final experiment, we trained a class-conditional ImageNet-64 model from scratch using our proposed training improvements. This model achieved a new state-of-the-art FID of 1.36 compared to the previous record of 1.48 [Ho2021cascaded]. We used the ADM architecture [Dhariwal2021] with no changes, and trained it using our config \textsc{e} with minimal tuning; see Appendix [12.3] for details. We did not find overfitting to be a concern, and thus chose to not employ augmentation regularization. As shown in Figure [5]c, the optimal amount of stochastic sampling was much lower than with the pre-trained model, but unlike with CIFAR-10, stochastic sampling was clearly better than deterministic sampling. This suggests that more diverse datasets continue to benefit from stochastic sampling.

# Conclusions 

Our approach of putting diffusion models to a common framework exposes a modular design. This allows a targeted investigation of individual components, potentially helping to better cover the viable design space. In our tests this let us simply replace the samplers in various earlier models, drastically improving the results. For example, in ImageNet-64 our sampler turned an average model (FID 2.07) to a challenger (1.55) for the previous SOTA model (1.48) [Ho2021cascaded], and with training improvements achieved SOTA FID of 1.36. We also obtained new state-of-the-art results on CIFAR-10 while using only 35 model evaluations, deterministic sampling, and a small network. The current high-resolution diffusion models rely either on separate super-resolution steps [Ho2021cascaded; Nichol2021b; Ramesh2022], subspace projection [Jing2022], very large networks [Dhariwal2021; Song2021sde], or hybrid approaches [Preechakul2021diffusion; Rombach2021highresolution; Vahdat2021] --- we believe that our contributions are orthogonal to these extensions. That said, many of our parameter values may need to be re-adjusted for higher resolution datasets. Furthermore, we feel that the precise interaction between stochastic sampling and the training objective remains an interesting question for future work.

#### Societal impact.

Our advances in sample quality can potentially amplify negative societal effects when used in a large-scale system like DALL`$\cdot$`E 2, including types of disinformation or emphasizing sterotypes and harmful biases [Mishkin2022risks]. The training and sampling of diffusion models needs a lot of electricity; our project consumed `$\sim$`250MWh on an in-house cluster of NVIDIA V100s.

**Appendices**

# Additional results 


[IMAGE: grids-img64c-sampling/ode-repro-seed0.jpg] [IMAGE: grids-img64c-sampling/ode-ours-seed8.jpg]


[IMAGE: grids-img64c-sampling/sde-repro-seed1.jpg] [IMAGE: grids-img64c-sampling/sde-ours-seed6.jpg]


**Figure caption:** Results for different samplers on class-conditional ImageNet  at 64×64 resolution, using the pre-trained model by Dhariwal and Nichol . The cases correspond to dots in Figures 2c and 4c.


[IMAGE: grids-img64c-training/ode-ours-seed9.jpg]


[IMAGE: grids-img64c-training/sde-ours-seed2.jpg]


**Figure caption:** Results for our training configuration on class-conditional ImageNet  at 64×64 resolution, using our deterministic and stochastic samplers.


[IMAGE: grids-cifar10u-sampling/ddpmpp-orig-ode-repro-seed2.jpg] [IMAGE: grids-cifar10u-sampling/ncsnpp-orig-ode-repro-seed2.jpg]


[IMAGE: grids-cifar10u-sampling/ddpmpp-orig-ode-ours-seed2.jpg] [IMAGE: grids-cifar10u-sampling/ncsnpp-orig-ode-ours-seed2.jpg]


[IMAGE: grids-cifar10u-sampling/ddpmpp-orig-sde-repro-seed0.jpg] [IMAGE: grids-cifar10u-sampling/ncsnpp-orig-sde-repro-seed1.jpg]


[IMAGE: grids-cifar10u-sampling/ddpmpp-orig-sde-ours-seed4.jpg] [IMAGE: grids-cifar10u-sampling/ncsnpp-orig-sde-ours-seed8.jpg]


**Figure caption:** Results for different samplers on unconditional CIFAR-10  at 32×32 resolution, using the pre-trained models by Song et al. . The cases correspond to dots in Figures 2a,b and 4a,b.


[IMAGE: grids-cifar10u-training/ddpmpp-orig-ode-ours-seed7.jpg] [IMAGE: grids-cifar10u-training/ncsnpp-orig-ode-ours-seed7.jpg]


[IMAGE: grids-cifar10u-training/ddpmpp-augm-ode-ours-seed7.jpg] [IMAGE: grids-cifar10u-training/ncsnpp-augm-ode-ours-seed7.jpg]


**Figure caption:** Results for different training configurations on unconditional CIFAR-10  at 32×32 resolution, using our deterministic sampler with the same set of latent codes (**x**<sub>0</sub>) in each case.


[IMAGE: grids-cifar10c-training/ddpmpp-orig-ode-ours-seed8.jpg] [IMAGE: grids-cifar10c-training/ncsnpp-orig-ode-ours-seed8.jpg]


[IMAGE: grids-cifar10c-training/ddpmpp-augm-ode-ours-seed8.jpg] [IMAGE: grids-cifar10c-training/ncsnpp-augm-ode-ours-seed8.jpg]


**Figure caption:** Results for different training configurations on class-conditional CIFAR-10  at 32×32 resolution, using our deterministic sampler with the same set of latent codes (**x**<sub>0</sub>) in each case.


[IMAGE: grids-ffhq64-training/ddpmpp-orig-ode-ours-seed2.jpg] [IMAGE: grids-ffhq64-training/ncsnpp-orig-ode-ours-seed2.jpg]


[IMAGE: grids-ffhq64-training/ddpmpp-augm-ode-ours-seed2.jpg] [IMAGE: grids-ffhq64-training/ncsnpp-augm-ode-ours-seed2.jpg]


[IMAGE: grids-afhq64-training/ddpmpp-orig-ode-ours-seed2.jpg] [IMAGE: grids-afhq64-training/ncsnpp-orig-ode-ours-seed2.jpg]


[IMAGE: grids-afhq64-training/ddpmpp-augm-ode-ours-seed2.jpg] [IMAGE: grids-afhq64-training/ncsnpp-augm-ode-ours-seed2.jpg]


**Figure caption:** Results for different training configurations on FFHQ  and AFHQv2  at 64×64 resolution, using our deterministic sampler with the same set of latent codes (**x**<sub>0</sub>) in each case.


[IMAGE: nfe-sweep/img64c-dhariwal-orig-ode-ours-seed46.jpg] [IMAGE: nfe-sweep/cifar10c-ddpmpp-augm-ode-ours-seed53.jpg]


[IMAGE: nfe-sweep/ffhq64-ddpmpp-augm-ode-ours-seed12.jpg] [IMAGE: nfe-sweep/afhq64-ddpmpp-augm-ode-ours-seed32.jpg]


**Figure caption:** Image quality and FID as a function of NFE using our deterministic sampler. At 32×32 resolution, reasonable image quality is reached around NFE = 13, but FID keeps improving until NFE = 35. At 64×64 resolution, reasonable image quality is reached around NFE = 19, but FID keeps improving until NFE = 79.


Figure [6] presents generated images for class-conditional ImageNet-64 [Deng2009imagenet] using the pre-trained ADM model by Dhariwal and Nichol [Dhariwal2021]. The original DDIM [Song2020ddim] and iDDPM [Nichol2021a] samplers are compared to ours in both deterministic and stochastic settings (Sections [sec:deterministic\] and [4]). Figure [7] shows the corresponding results that we obtain by training the model from scratch using our improved training configuration (Section [5]).

The original samplers and training configurations by Song et al. [Song2021sde] are compared to ours in Figures [8] and [9] (unconditional CIFAR-10 [Krizhevsky2009cifar]), Figure [10] (class-conditional CIFAR-10), and Figure [11] (FFHQ [Karras2018stylegan] and AFHQv2 [Choi2020afhq]). For ease of comparison, the same latent codes `$\boldsymbol{x}_0$` are used for each dataset/scenario across different training configurations and ODE choices. Figure [12] shows generated image quality with various NFE when using deterministic sampling.

Tables [tab:OdeTable\] and [tab:SdeTable\] summarize the numerical results on deterministic and stochastic sampling methods in various datasets, previously shown as functions of NFE in Figures [2] and [4].

# Derivation of formulas 

## Original ODE / SDE formulation from previous work 

Song et al. [Song2021sde] define their forward SDE (Eq. 5 in [Song2021sde]) as 
```latex
$$\mathrm{d}\boldsymbol{x}= \boldsymbol{f}(\boldsymbol{x}, t) ~\mathrm{d}t + g(t) ~\mathrm{d}\omega_t
  ,$$
```
 where `$\omega_t$` is the standard Wiener process and `$\boldsymbol{f}(\cdot, t): \mathbb{R}^d \rightarrow \mathbb{R}^d$` and `$g(\cdot): \mathbb{R} \rightarrow \mathbb{R}$` are the drift and diffusion coefficients, respectively, where `$d$` is the dimensionality of the dataset. These coefficients are selected differently for the variance preserving (VP) and variance exploding (VE) formulations, and `$\boldsymbol{f}(\cdot)$` is always of the form `$\boldsymbol{f}(\boldsymbol{x}, t) = f(t) ~\boldsymbol{x}$`, where `$f(\cdot): \mathbb{R} \rightarrow \mathbb{R}$`. Thus, the SDE can be equivalently written as 
```latex
$$
  \mathrm{d}\boldsymbol{x}= f(t) ~\boldsymbol{x}~\mathrm{d}t + g(t) ~\mathrm{d}\omega_t
  .$$
```


The perturbation kernels of this SDE (Eq. 29 in [Song2021sde]) have the general form 
```latex
$$
  p_{0t}\big( \boldsymbol{x}(t) ~|~ \boldsymbol{x}(0) \big) = \mathcal{N} \big( \boldsymbol{x}(t); ~s(t) ~\boldsymbol{x}(0), ~s(t)^2 ~\sigma(t)^2 ~\mathbf{I}\big)
  ,$$
```
 where `$\mathcal{N}(\boldsymbol{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma})$` denotes the probability density function of `$\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$` evaluated at `$\boldsymbol{x}$`, 
```latex
$$
  s(t) = \exp\left( \int_0^t f(\xi) ~\mathrm{d}\xi \right)
  ,
  \hspace{4mm}\text{and}\hspace{4mm}
  \sigma(t) = \sqrt{\int_0^t \frac{g(\xi)^2}{s(\xi)^2} ~\mathrm{d}\xi}
  .$$
```


The marginal distribution `$p_t(\boldsymbol{x})$` is obtained by integrating the perturbation kernels over `$\boldsymbol{x}(0)$`: 
```latex
$$
  p_t(\boldsymbol{x}) = \int_{\mathbb{R}^d} p_{0t}(\boldsymbol{x}~|~ \boldsymbol{x}_0) ~p_\text{data}(\boldsymbol{x}_0) ~\mathrm{d}\boldsymbol{x}_0
  .$$
```


Song et al. [Song2021sde] define the probability flow ODE (Eq. 13 in [Song2021sde]) so that it obeys this same `$p_t(\boldsymbol{x})$`: 
```latex
$$
  \mathrm{d}\boldsymbol{x}= \left[ f(t) ~\boldsymbol{x}- \tfrac{1}{2} ~g(t)^2 ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p_t(\boldsymbol{x}) \right] ~\mathrm{d}t
  .$$
```


## Our ODE formulation (Eq. [eq:ode\] and Eq. [eq:odescale\]) 

The original ODE formulation (Eq. [eq:songode\]) is built around the functions `$f$` and `$g$` that correspond directly to specific terms that appear in the formula; the properties of the marginal distribution (Eq. [eq:songscale\]) can only be derived indirectly based on these functions. However, `$f$` and `$g$` are of little practical interest in themselves, whereas the marginal distributions are of utmost importance in terms of training the model in the first place, bootstrapping the sampling process, and understanding how the ODE behaves in practice. Given that the idea of the probability flow ODE is to match a particular set of marginal distributions, it makes sense to treat the marginal distributions as first-class citizens and define the ODE directly based on `$\sigma(t)$` and `$s(t)$`, eliminating the need for `$f(t)$` and `$g(t)$`.

Let us start by expressing the marginal distribution of Eq. [eq:songmarginal\] in closed form: 
```latex
$$\begin{aligned}
  p_t(\boldsymbol{x}) &=& \int_{\mathbb{R}^d} p_{0t}(\boldsymbol{x}~|~ \boldsymbol{x}_0) ~p_\text{data}(\boldsymbol{x}_0) ~\mathrm{d}\boldsymbol{x}_0 \\
  &=& \int_{\mathbb{R}^d} p_\text{data}(\boldsymbol{x}_0) ~\Big[ \mathcal{N} \big( \boldsymbol{x}; ~s(t) ~\boldsymbol{x}_0, ~s(t)^2 ~\sigma(t)^2 ~\mathbf{I}\big) \Big] ~\mathrm{d}\boldsymbol{x}_0 \\
  &=& \int_{\mathbb{R}^d} p_\text{data}(\boldsymbol{x}_0) ~\Big[ s(t)^{-d} ~\mathcal{N} \big( \boldsymbol{x}/ s(t); ~\boldsymbol{x}_0, ~\sigma(t)^2 ~\mathbf{I}\big) \Big] ~\mathrm{d}\boldsymbol{x}_0 \\
  &=& s(t)^{-d} \int_{\mathbb{R}^d} p_\text{data}(\boldsymbol{x}_0) ~\mathcal{N} \big( \boldsymbol{x}/ s(t); ~\boldsymbol{x}_0, ~\sigma(t)^2 ~\mathbf{I}\big) ~\mathrm{d}\boldsymbol{x}_0 \\
  &=& s(t)^{-d} ~\Big[ p_\text{data}\ast \mathcal{N} \big( \mathbf{0}, ~\sigma(t)^2 ~\mathbf{I}\big) \Big] \big( \boldsymbol{x}/ s(t) \big)
  ,
\end{aligned}$$
```
 where `$p_a \ast p_b$` denotes the convolution of probability density functions `$p_a$` and `$p_b$`. The expression inside the brackets corresponds to a mollified version of `$p_\text{data}$` obtained by adding i.i.d. Gaussian noise to the samples. Let us denote this distribution by `$p(\boldsymbol{x}; \sigma)$`: 
```latex
$$
  p(\boldsymbol{x}; \sigma) = p_\text{data}\ast \mathcal{N} \big( \mathbf{0}, ~\sigma(t)^2 ~\mathbf{I}\big)
  \hspace{5mm}\text{and}\hspace{5mm}
  p_t(\boldsymbol{x}) = s(t)^{-d} ~p\big( \boldsymbol{x}/ s(t); \sigma(t) \big)
  .$$
```


We can now express the probability flow ODE (Eq. [eq:songode\]) using `$p(\boldsymbol{x}; \sigma)$` instead of `$p_t(\boldsymbol{x})$`: 
```latex
$$\begin{aligned}
  \mathrm{d}\boldsymbol{x}&=& \left[ f(t) \boldsymbol{x}- \tfrac{1}{2} ~g(t)^2 ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log \big[ p_t(\boldsymbol{x}) \big] \right] ~\mathrm{d}t \\
  &=& \left[ f(t) \boldsymbol{x}- \tfrac{1}{2} ~g(t)^2 ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log \big[ s(t)^{-d} ~p\big( \boldsymbol{x}/ s(t); \sigma(t) \big) \big] \right] ~\mathrm{d}t \\
  &=& \left[ f(t) \boldsymbol{x}- \tfrac{1}{2} ~g(t)^2 ~\big[ \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log s(t)^{-d} + \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}/ s(t); \sigma(t) \big) \big] \right] ~\mathrm{d}t \\
  &=& \left[ f(t) \boldsymbol{x}- \tfrac{1}{2} ~g(t)^2 ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}/ s(t); \sigma(t) \big) \right] ~\mathrm{d}t
  
  .
\end{aligned}$$
```


Next, let us rewrite `$f(t)$` in terms of `$s(t)$` based on Eq. [eq:songscale\]: 
```latex
$$\begin{aligned}
  \exp\left( \int_0^t f(\xi) ~\mathrm{d}\xi \right) &=& s(t) \\
  \int_0^t f(\xi) ~\mathrm{d}\xi &=& \log s(t) \\
  \mathrm{d}\bigg[ \int_0^t f(\xi) ~\mathrm{d}\xi \bigg] \big/ \mathrm{d}t &=& \mathrm{d}\big[ \log s(t) \big] / \mathrm{d}t \\
  f(t) &=& \dot s(t) / s(t)
  
  .
\end{aligned}$$
```


Similarly, we can also rewrite `$g(t)$` in terms of `$\sigma(t)$`: 
```latex
$$\begin{aligned}
  \sqrt{\int_0^t \frac{g(\xi)^2}{s(\xi)^2} ~\mathrm{d}\xi} &=& \sigma(t) \\
  \int_0^t \frac{g(\xi)^2}{s(\xi)^2} ~\mathrm{d}\xi &=& \sigma(t)^2 \\
  \mathrm{d}\bigg[ \int_0^t \frac{g(\xi)^2}{s(\xi)^2} ~\mathrm{d}\xi \bigg] \big/ \mathrm{d}t &=& \mathrm{d}\big[ \sigma(t)^2 \big] / \mathrm{d}t \\
  g(t)^2 / s(t)^2 &=& 2 ~\dot\sigma(t) ~\sigma(t) \\
  g(t) / s(t) &=& \sqrt{2 ~\dot\sigma(t) ~\sigma(t)} \\
  g(t) &=& s(t) ~\sqrt{2 ~\dot\sigma(t) ~\sigma(t)}
  
  .
\end{aligned}$$
```


Finally, substitute `$f$` (Eq. [eq:tempf\]) and `$g$` (Eq. [eq:tempg\]) into the ODE of Eq. [eq:tempode\]: 
```latex
$$\begin{aligned}
  \mathrm{d}\boldsymbol{x}&=& \left[ [f(t)] ~\boldsymbol{x}- \tfrac{1}{2} ~[g(t)]^2 ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}/ s(t); \sigma(t) \big) \right] ~\mathrm{d}t \\
  &=& \bigg[ \big[ \dot s(t) / s(t) \big] ~\boldsymbol{x}- \tfrac{1}{2} ~\Big[ s(t) \sqrt{2 ~\dot\sigma(t) ~\sigma(t)} \Big]^2 ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}/ s(t); \sigma(t) \big) \bigg] ~\mathrm{d}t \\
  &=& \bigg[ \big[ \dot s(t) / s(t) \big] ~\boldsymbol{x}- \tfrac{1}{2} ~\Big[ 2 ~s(t)^2 ~\dot\sigma(t) ~\sigma(t) \Big] ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}/ s(t); \sigma(t) \big) \bigg] ~\mathrm{d}t \\
  %
  &=& \left[ \frac{\dot s(t)}{s(t)} ~\boldsymbol{x}-s(t)^2 ~\dot\sigma(t) ~\sigma(t) ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\left(\frac{\boldsymbol{x}}{s(t)}; \sigma(t)\right) \right] ~\mathrm{d}t
  .
\end{aligned}$$
```


Thus we have obtained Eq. [eq:odescale\] in the main paper, and Eq. [eq:ode\] is recovered by setting `$s(t) = 1$`: 
```latex
$$\mathrm{d}\boldsymbol{x}= -\dot\sigma(t) ~\sigma(t) ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}; \sigma(t) \big) ~\mathrm{d}t
  .$$
```


Our formulation (Eq. [eq:odescale\]) highlights the fact that every realization of the probability flow ODE is simply a reparameterization of the same canonical ODE; changing `$\sigma(t)$` corresponds to reparameterizing `$t$`, whereas changing `$s(t)$` corresponds to reparameterizing `$\boldsymbol{x}$`.

## Denoising score matching (Eq. [eq:score\] and Eq. [eq:scoredenoiser\]) 

For the sake of completeness, we derive the connection between score matching and denoising for a finite dataset. For a more general treatment and further background on the topic, see Hyvärinen [Hyvarinen05] and Vincent [Vincent11].

Let us assume that our training set consists of a finite number of samples `$\{\boldsymbol{y}_1, \dots, \boldsymbol{y}_Y\}$`. This implies `$p_\text{data}(\boldsymbol{x})$` is represented by a mixture of Dirac delta distributions: 
```latex
$$p_\text{data}(\boldsymbol{x}) = \frac{1}{Y} \sum_{i=1}^Y \delta \big( \boldsymbol{x}- \boldsymbol{y}_i \big)
  ,$$
```
 which allows us to also express `$p(\boldsymbol{x}; \sigma)$` in closed form based on Eq. [eq:psigma\]: 
```latex
$$\begin{aligned}
  p(\boldsymbol{x}; \sigma) &=& p_\text{data}\ast \mathcal{N} \big( \mathbf{0}, ~\sigma(t)^2 ~\mathbf{I}\big) \\
  &=& \int_{\mathbb{R}^d} p_\text{data}(\boldsymbol{x}_0) ~\mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{x}_0, ~\sigma^2 ~\mathbf{I}\big) ~\mathrm{d}\boldsymbol{x}_0 \\
  &=& \int_{\mathbb{R}^d} \Bigg[ \frac{1}{Y} \sum_{i=1}^Y \delta \big( \boldsymbol{x}_0 - \boldsymbol{y}_i \big) \Bigg] \mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{x}_0, ~\sigma^2 ~\mathbf{I}\big) ~\mathrm{d}\boldsymbol{x}_0 \\
  &=& \frac{1}{Y} \sum_{i=1}^Y \int_{\mathbb{R}^d} \mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{x}_0, ~\sigma^2 ~\mathbf{I}\big) ~\delta \big( \boldsymbol{x}_0 - \boldsymbol{y}_i \big) ~\mathrm{d}\boldsymbol{x}_0 \\
  &=& \frac{1}{Y} \sum_{i=1}^Y \mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}\big)
  
  .
\end{aligned}$$
```


Let us now consider the denoising score matching loss of Eq. [eq:score\]. By expanding the expectations, we can rewrite the formula as an integral over the noisy samples `$\boldsymbol{x}$`: 
```latex
$$\begin{aligned}
  \mathcal{L}(D; \sigma) &=& \mathbb{E}_{\boldsymbol{y}\sim p_\text{data}} ~\mathbb{E}_{\boldsymbol{n}\sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})} ~\big\lVert D(\boldsymbol{y}+ \boldsymbol{n}; \sigma) - \boldsymbol{y}\big\rVert^2_2 \\
  &=& \mathbb{E}_{\boldsymbol{y}\sim p_\text{data}} ~\mathbb{E}_{\boldsymbol{x}\sim \mathcal{N}(\boldsymbol{y}, \sigma^2 \mathbf{I})} ~\big\lVert D(\boldsymbol{x}; \sigma) - \boldsymbol{y}\big\rVert^2_2 \\
  &=& \mathbb{E}_{\boldsymbol{y}\sim p_\text{data}} \int_{\mathbb{R}^d} \mathcal{N}(\boldsymbol{x}; ~\boldsymbol{y}, ~\sigma^2 ~\mathbf{I}) ~\big\lVert D(\boldsymbol{x}; \sigma) - \boldsymbol{y}\big\rVert^2_2 ~\mathrm{d}\boldsymbol{x}\\
  &=& \frac{1}{Y} \sum_{i=1}^Y \int_{\mathbb{R}^d} \mathcal{N}(\boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}) ~\big\lVert D(\boldsymbol{x}; \sigma) - \boldsymbol{y}_i \big\rVert^2_2 ~\mathrm{d}\boldsymbol{x}\\
  &=& \int_{\mathbb{R}^d} \underbrace{\frac{1}{Y} \sum_{i=1}^Y \mathcal{N}(\boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}) ~\big\lVert D(\boldsymbol{x}; \sigma) - \boldsymbol{y}_i \big\rVert^2_2}_{=: ~ \mathcal{L}(D; \boldsymbol{x}, \sigma)}  ~\mathrm{d}\boldsymbol{x}
  %
  
  .
\end{aligned}$$
```


Eq. [eq:slicedloss\] means that we can minimize `$\mathcal{L}(D; \sigma)$` by minimizing `$\mathcal{L}(D; \boldsymbol{x}, \sigma)$` independently for each `$\boldsymbol{x}$`: 
```latex
$$D(\boldsymbol{x}; \sigma) = \mathop{\mathrm{arg\,min}}_{D(\boldsymbol{x}; \sigma)} \mathcal{L}(D; \boldsymbol{x}, \sigma)
  .$$
```
 This is a convex optimization problem; its solution is uniquely identified by setting the gradient w.r.t. `$D(\boldsymbol{x}; \sigma)$` to zero: 
```latex
$$\begin{aligned}
  \mathbf{0}&=& \nabla_{D(\boldsymbol{x}; \sigma)} \Big[ \mathcal{L}(D; \boldsymbol{x}, \sigma) \Big] \\
  \mathbf{0}&=& \nabla_{D(\boldsymbol{x}; \sigma)} \Bigg[ \frac{1}{Y} \sum_{i=1}^Y \mathcal{N}(\boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}) ~\big\lVert D(\boldsymbol{x}; \sigma) - \boldsymbol{y}_i \big\rVert^2_2 \Bigg] \\
  \mathbf{0}&=& \sum_{i=1}^Y \mathcal{N}(\boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}) ~\nabla_{D(\boldsymbol{x}; \sigma)} \Big[ \big\lVert D(\boldsymbol{x}; \sigma) - \boldsymbol{y}_i \big\rVert^2_2 \Big] \\
  \mathbf{0}&=& \sum_{i=1}^Y \mathcal{N}(\boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}) ~\Big[ 2 ~D(\boldsymbol{x}; \sigma) - 2~\boldsymbol{y}_i \Big] \\
  \mathbf{0}&=& \Bigg[ \sum_{i=1}^Y \mathcal{N}(\boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}) \Bigg] D(\boldsymbol{x}; \sigma) - \sum_{i=1}^Y \mathcal{N}(\boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}) ~\boldsymbol{y}_i \\
  D(\boldsymbol{x}; \sigma) &=& \frac{ \sum_i \mathcal{N}(\boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}) ~\boldsymbol{y}_i }{ \sum_i \mathcal{N}(\boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}) }
  
  ,
\end{aligned}$$
```
 which gives a closed-form solution for the ideal denoiser `$D(\boldsymbol{x}; \sigma)$`. Note that Eq. [eq:idealdenoiser\] is feasible to compute in practice for small datasets --- we show the results for CIFAR-10 in Figure [1]b.

Next, let us consider the score of the distribution `$p(\boldsymbol{x}; \sigma)$` defined in Eq. [eq:diracpsigma\]: 
```latex
$$\begin{aligned}
  \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p(\boldsymbol{x}; \sigma) &=& \frac{\nabla_{\hspace{-0.5mm}\boldsymbol{x}}p(\boldsymbol{x}; \sigma)}{p(\boldsymbol{x}; \sigma)} \\
  &=& \frac{ \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\Big[ \frac{1}{Y} \sum_i \mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}\big) \Big] }{ \Big[ \frac{1}{Y} \sum_i \mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}\big) \Big] } \\
  &=& \frac{ \sum_i \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}\big) }{ \sum_i \mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}\big) }
  
  .
\end{aligned}$$
```


We can simplify the numerator of Eq. [eq:diracscoretemp\] further: 
```latex
$$\begin{aligned}
  \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}\big) &=& \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\Bigg[ \big( 2 \pi \sigma^2 \big)^{-\frac{d}{2}} ~\exp \frac{\lVert \boldsymbol{x}- \boldsymbol{y}_i \rVert_2^2}{-2 ~\sigma^2} \Bigg] \\
  &=& \big( 2 \pi \sigma^2 \big)^{-\frac{d}{2}} ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\Bigg[ \exp \frac{\lVert \boldsymbol{x}- \boldsymbol{y}_i \rVert_2^2}{-2 ~\sigma^2} \Bigg] \\
  &=& \Bigg[\big( 2 \pi \sigma^2 \big)^{-\frac{d}{2}} \exp \frac{\lVert \boldsymbol{x}- \boldsymbol{y}_i \rVert_2^2}{-2 ~\sigma^2} \Bigg] ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\Bigg[ \frac{\lVert \boldsymbol{x}- \boldsymbol{y}_i \rVert_2^2}{-2 ~\sigma^2} \Bigg] \\
  &=& \mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}\big) ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\Bigg[ \frac{\lVert \boldsymbol{x}- \boldsymbol{y}_i \rVert_2^2}{-2 ~\sigma^2} \Bigg] \\
  &=& \mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}\big) \bigg[ \frac{\boldsymbol{y}_i - \boldsymbol{x}}{\sigma^2} \bigg]
  .
\end{aligned}$$
```


Let us substitute the result back to Eq. [eq:diracscoretemp\]: 
```latex
$$\begin{aligned}
  \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p(\boldsymbol{x}; \sigma) &=& \frac{ \sum_i \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}\big) }{ \sum_i \mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}\big) } \\
  &=& \frac{ \sum_i \mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}\big) \Big[ \frac{\boldsymbol{y}_i - \boldsymbol{x}}{\sigma^2} \Big] }{ \sum_i \mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}\big) } \\
  &=& \Bigg( \frac{ \sum_i \mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}\big) \boldsymbol{y}_i }{ \sum_i \mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}\big) } - \boldsymbol{x}\Bigg) \big/ \sigma^2
  
  .
\end{aligned}$$
```


Notice that the fraction in Eq. [eq:diracscore\] is identical to Eq. [eq:idealdenoiser\]. We can thus equivalently write Eq. [eq:diracscore\] as 
```latex
$$\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p(\boldsymbol{x}; \sigma) = \big( D(\boldsymbol{x}; ~\sigma) - \boldsymbol{x}\big) / \sigma^2
  ,$$
```
 which matches Eq. [eq:scoredenoiser\] in the main paper.

## Evaluating our ODE in practice (Algorithm [alg:heun\]) 

Let us consider `$\boldsymbol{x}$` to be a scaled version of an original, non-scaled variable `$\hat\boldsymbol{x}$` and substitute `$\boldsymbol{x}= s(t) ~\hat\boldsymbol{x}$` into the score term that appears in our scaled ODE (Eq. [eq:odescale\]): 
```latex
$$\begin{aligned}
  && \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}/ s(t); \sigma(t) \big) \\
  &=& \nabla_{[ s(t) \hat \boldsymbol{x}]} \log p\big( [s(t) ~\hat\boldsymbol{x}] / s(t); \sigma(t) \big) \\
  &=& \nabla_{s(t) \hat \boldsymbol{x}} \log p\big( \hat\boldsymbol{x}; \sigma(t) \big) \\
  &=& \tfrac{1}{s(t)} \nabla_{\hat\boldsymbol{x}} \log p\big( \hat\boldsymbol{x}; \sigma(t) \big)
  .
\end{aligned}$$
```


We can further rewrite this with respect to `$D(\cdot)$` using Eq. [eq:scoredenoiser\]: 
```latex
$$\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}/ s(t); \sigma(t) \big) ~=~ \tfrac{1}{s(t) \sigma(t)^2} \Big( D\big( \hat\boldsymbol{x}; \sigma(t) \big) - \hat\boldsymbol{x}\Big)
  
  .$$
```


Let us now substitute Eq. [eq:scaledscore\] into Eq. [eq:odescale\], approximating the ideal denoiser `$D(\cdot)$` with our trained model `$D_\theta(\cdot)$`: 
```latex
$$\begin{aligned}
  \mathrm{d}\boldsymbol{x}&=& \left[ \dot s(t) ~\boldsymbol{x}/ s(t) - s(t)^2 ~\dot\sigma(t) ~\sigma(t) ~\Big[ \tfrac{1}{s(t) \sigma(t)^2} \Big( D_\theta \big( \hat\boldsymbol{x}; \sigma(t) \big) - \hat\boldsymbol{x}\Big) \Big] \right] ~\mathrm{d}t \\
  &=& \left[ \tfrac{\dot s(t)}{s(t)} ~\boldsymbol{x}- \tfrac{\dot\sigma(t) s(t)}{\sigma(t)} \Big( D_\theta \big( \hat\boldsymbol{x}; \sigma(t) \big) - \hat\boldsymbol{x}\Big) \right] ~\mathrm{d}t
  .
\end{aligned}$$
```


Finally, backsubstitute `$\hat\boldsymbol{x}= \boldsymbol{x}/ s(t)$`: 
```latex
$$\begin{aligned}
  \mathrm{d}\boldsymbol{x}&=& \left[ \tfrac{\dot s(t)}{s(t)} ~\boldsymbol{x}- \tfrac{\dot\sigma(t) s(t)}{\sigma(t)} \Big( D_\theta \big( [\hat\boldsymbol{x}]; \sigma(t) \big) - [\hat\boldsymbol{x}] \Big) \right] ~\mathrm{d}t \\
  &=& \left[ \tfrac{\dot s(t)}{s(t)} ~\boldsymbol{x}- \tfrac{\dot\sigma(t) s(t)}{\sigma(t)} \Big( D_\theta \big( [\boldsymbol{x}/ s(t)]; \sigma(t) \big) - [\boldsymbol{x}/ s(t)] \Big) \right] ~\mathrm{d}t \\
  &=& \left[ \tfrac{\dot s(t)}{s(t)} ~\boldsymbol{x}- \tfrac{\dot\sigma(t) s(t)}{\sigma(t)} D_\theta \big( \boldsymbol{x}/ s(t); \sigma(t) \big) + \tfrac{\dot\sigma(t)}{\sigma(t)} ~\boldsymbol{x}\right] ~\mathrm{d}t \\
  &=& \left[ \left( \tfrac{\dot\sigma(t)}{\sigma(t)} + \tfrac{\dot s(t)}{s(t)} \right) \boldsymbol{x}- \tfrac{\dot\sigma(t) s(t)}{\sigma(t)} D_\theta \big( \boldsymbol{x}/ s(t); \sigma(t) \big) \right] ~\mathrm{d}t
  
  .
\end{aligned}$$
```


We can equivalenty write Eq. [eq:practicalode\] as 
```latex
$$\mathrm{d}\boldsymbol{x}/ \mathrm{d}t = \bigg( \frac{\dot\sigma(t)}{\sigma(t)} + \frac{\dot s(t)}{s(t)} \bigg) \boldsymbol{x}- \frac{\dot\sigma(t) s(t)}{\sigma(t)} D_\theta \bigg( \frac{\boldsymbol{x}}{s(t)}; \sigma(t) \bigg)
  ,$$
```
 matching lines 4 and 7 of Algorithm [alg:heun\].

## Our SDE formulation (Eq. [eq:sde\]) 

We derive the SDE of Eq. [eq:sde\] by the following strategy:

- The desired marginal densities `$p\big( \boldsymbol{x}; \sigma(t) \big)$` are convolutions of the data density `$p_\text{data}$` and an isotropic Gaussian density with standard deviation `$\sigma(t)$` (see Eq. [eq:psigma\]). Hence, considered as a function of the time `$t$`, the density evolves according to a heat diffusion PDE with time-varying diffusivity. As a first step, we find this PDE.

- We then use the Fokker--Planck equation to recover a family of SDEs for which the density evolves according to this PDE. Eq. [eq:sde\] is obtained from a suitable parametrization of this family.

### Generating the marginals by heat diffusion

We consider the time evolution of a probability density `$q(\boldsymbol{x}, t)$`. Our goal is to find a PDE whose solution with the initial value `$q(\boldsymbol{x}, 0) := p_\text{data}(\boldsymbol{x})$` is `$q(\boldsymbol{x}, t) = p\big( \boldsymbol{x}, \sigma(t) \big)$`. That is, the PDE should reproduce the marginals we postulate in Eq. [eq:psigma\].

The desired marginals are convolutions of `$p_\text{data}$` with isotropic normal distributions of time-varying standard deviation `$\sigma(t)$`, and as such, can be generated by the heat equation with time-varying diffusivity `$\kappa(t)$`. The situation is most conveniently analyzed in the Fourier domain, where the marginal densities are simply pointwise products of a Gaussian function and the transformed data density. To find the diffusivity that induces the correct standard deviations, we first write down the heat equation PDE: 
```latex
$$\frac{\partial q(\boldsymbol{x}, t)}{\partial t} = \kappa(t) {\Delta_{\boldsymbol{x}}}q(\boldsymbol{x}, t)
  
  .$$
```


The Fourier transformed counterpart of Eq. [eq:pde\], where the transform is taken along the `$\boldsymbol{x}$`-dimension, is given by 
```latex
$$\frac{\partial \hat q(\boldsymbol{\nu}, t)}{\partial t} = - \kappa(t) |\boldsymbol{\nu}|^2 \hat q(\boldsymbol{\nu}, t)
  
  .$$
```


The target solution `$q(\boldsymbol{x}, t)$` and its Fourier transform `$\hat q(\boldsymbol{\nu}, t)$` are given by Eq. [eq:psigma\]: 
```latex
$$\begin{aligned}
  q(\boldsymbol{x}, t) &=& p\big( \boldsymbol{x}; \sigma(t) \big) = p_\text{data}(\boldsymbol{x}) \ast \mathcal{N}\big( \mathbf{0}, ~\sigma(t)^2 ~\mathbf{I}\big) \\
  \hat q(\boldsymbol{\nu}, t) &=& \hat{p}_\text{data}(\boldsymbol{\nu}) ~\exp\Big( {-}\tfrac{1}{2} ~|\boldsymbol{\nu}|^2 ~\sigma(t)^2 \Big)
  .
\end{aligned}$$
```


Differentiating the target solution along the time axis, we have 
```latex
$$\begin{aligned}
  \frac{\partial \hat q(\boldsymbol{\nu}, t)}{\partial t} &=& - \dot\sigma(t) \sigma(t) ~|\boldsymbol{\nu}|^2 ~ \hat{p}_\text{data}(\boldsymbol{\nu}) ~\exp\Big( {-}\tfrac{1}{2} ~|\boldsymbol{\nu}|^2 ~\sigma(t)^2 \Big) \\
  &=& - \dot \sigma(t) \sigma(t) ~|\boldsymbol{\nu}|^2 ~\hat q(\boldsymbol{\nu},t)
  
  .
\end{aligned}$$
```


Eqs. [eq:pdeft\] and [eq:pdesolutionftd\] share the same left hand side. Equating them allows us to solve for `$\kappa(t)$` that generates the desired evolution: 
```latex
$$\begin{aligned}
  - \kappa(t) |\boldsymbol{\nu}|^2 \hat q(\boldsymbol{\nu}, t) &=& - \dot \sigma(t) \sigma(t) ~ |\boldsymbol{\nu}|^2 ~ \hat q(\boldsymbol{\nu},t) \\
  \kappa(t) &=& \dot \sigma(t) \sigma(t)
  .
\end{aligned}$$
```


To summarize, the desired marginal densities corresponding to noise levels `$\sigma(t)$` are generated by the PDE 
```latex
$$\frac{\partial q(\boldsymbol{x}, t)}{\partial t} = \dot \sigma(t) \sigma(t) {\Delta_{\boldsymbol{x}}}q(\boldsymbol{x}, t)
  $$
```
 from the initial density `$q(\boldsymbol{x}, 0) = p_\text{data}(\boldsymbol{x})$`.

### Derivation of our SDE

Given an SDE 
```latex
$$\mathrm{d}\boldsymbol{x}= \boldsymbol{f}(\boldsymbol{x}, t) ~ \mathrm{d}t ~ + ~ \boldsymbol{g}(\boldsymbol{x}, t) ~ \mathrm{d}\omega_t
  
  ,$$
```
 the Fokker--Planck PDE describes the time evolution of its solution probability density `$r(\boldsymbol{x}, t)$` as 
```latex
$$\frac{\partial r(\boldsymbol{x}, t)}{\partial t} = -\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\cdot \big( \boldsymbol{f}(\boldsymbol{x},t) ~r(\boldsymbol{x},t) \big) + \tfrac{1}{2} \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\nabla_{\hspace{-0.5mm}\boldsymbol{x}}: \big( \mathbf{D}(\boldsymbol{x}, t) ~r(\boldsymbol{x}, t) \big)
  ,$$
```
 where `$\mathbf{D}_{ij} = \sum_k \boldsymbol{g}_{ik} \boldsymbol{g}_{jk}$` is the *diffusion tensor*. We consider the special case `$\boldsymbol{g}(\boldsymbol{x}, t) = g(t) ~\mathbf{I}$` of `$\boldsymbol{x}$`-independent white noise addition, whereby the equation simplifies to 
```latex
$$\frac{\partial r(\boldsymbol{x}, t)}{\partial t} = -\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\cdot \big( \boldsymbol{f}(\boldsymbol{x},t) ~r(\boldsymbol{x},t) \big) + \tfrac{1}{2} ~g(t)^2 ~{\Delta_{\boldsymbol{x}}}r(\boldsymbol{x}, t)
  
  .$$
```


We are seeking an SDE whose solution density is described by the PDE in Eq. [eq:marginalpde\]. Setting `$r(\boldsymbol{x}, t) = q(\boldsymbol{x}, t)$` and equating Eqs. [eq:fokkerplanck\] and [eq:marginalpde\], we find the sufficient condition that the SDE must satisfy 
```latex
$$\begin{aligned}
  -\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\cdot \big( \boldsymbol{f}(\boldsymbol{x},t) ~q(\boldsymbol{x},t) \big) + \tfrac{1}{2} ~g(t)^2 ~{\Delta_{\boldsymbol{x}}}q(\boldsymbol{x}, t) &=& \dot\sigma(t) ~\sigma(t) ~{\Delta_{\boldsymbol{x}}}q(\boldsymbol{x}, t) \\
  \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\cdot \big( \boldsymbol{f}(\boldsymbol{x},t) ~q(\boldsymbol{x},t) \big) &=& \Big( \tfrac{1}{2} ~g(t)^2 - \dot\sigma(t) ~\sigma(t) \Big) ~{\Delta_{\boldsymbol{x}}}q(\boldsymbol{x}, t)
  .
\end{aligned}$$
```


Any choice of functions `$\boldsymbol{f}(\boldsymbol{x},t)$` and `$g(t)$` satisfying this equation constitute a sought after SDE. Let us now find a specific family of such solutions. The key idea is given by the identity `$\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\cdot \nabla_{\hspace{-0.5mm}\boldsymbol{x}}= {\Delta_{\boldsymbol{x}}}$`. Indeed, if we set `$\boldsymbol{f}(\boldsymbol{x},t) ~q(\boldsymbol{x},t) = \upsilon(t) ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}q(\boldsymbol{x},t)$` for any choice of `$\upsilon(t)$`, the term `${\Delta_{\boldsymbol{x}}}q(\boldsymbol{x},t)$` appears on both sides and cancels out: 
```latex
$$\begin{aligned}
  \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\cdot \big( \upsilon(t) ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}q(\boldsymbol{x},t) \big) &=& \Big( \tfrac{1}{2} ~g(t)^2 - \dot\sigma(t) ~\sigma(t) \Big) ~{\Delta_{\boldsymbol{x}}}q(\boldsymbol{x}, t) \\
  \upsilon(t) ~{\Delta_{\boldsymbol{x}}}q(\boldsymbol{x},t) &=& \Big( \tfrac{1}{2} ~g(t)^2 - \dot\sigma(t) ~\sigma(t) \Big) ~{\Delta_{\boldsymbol{x}}}q(\boldsymbol{x}, t) \\
  \upsilon(t) &=& \tfrac{1}{2} ~g(t)^2 - \dot\sigma(t) ~\sigma(t)
  .
\end{aligned}$$
```


The stated `$\boldsymbol{f}(\boldsymbol{x},t)$` is in fact proportional to the score function, as the formula matches the gradient of the logarithm of the density: 
```latex
$$\begin{aligned}
  \boldsymbol{f}(\boldsymbol{x}, t) &=& \upsilon(t) ~\frac{\nabla_{\hspace{-0.5mm}\boldsymbol{x}}q(\boldsymbol{x},t)}{q(\boldsymbol{x},t)} \\
  &=& \upsilon(t) ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log q(\boldsymbol{x}, t) \\
  &=& \Big( \tfrac{1}{2} ~g(t)^2 - \dot\sigma(t) ~\sigma(t) \Big) ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log q(\boldsymbol{x}, t)
  .
\end{aligned}$$
```


Substituting this back into Eq. [eq:sdegeneral\] and writing `$p(\boldsymbol{x}; \sigma(t))$` in place of `$q(\boldsymbol{x},t)$`, we recover a family of SDEs whose solution densities have the desired marginals with noise levels `$\sigma(t)$` for any choice of `$g(t)$`: 
```latex
$$\mathrm{d}\boldsymbol{x}= \Big( \tfrac{1}{2} ~g(t)^2 - \dot\sigma(t) ~\sigma(t) \Big) ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}; \sigma(t) \big) ~\mathrm{d}t ~+~ g(t) ~\mathrm{d}\omega_t
  .$$
```


The free parameter `$g(t)$` effectively specifies the rate of noise replacement at any given time instance. The special case choice of `$g(t) = 0$` corresponds to the probability flow ODE. The parametrization by `$g(t)$` is not particularly intuitive, however. To obtain a more interpretable parametrization, we set `$g(t) = \sqrt{2 ~\beta(t)} ~\sigma(t)$`, which yields the (forward) SDE of Eq. [eq:sde\] in the main paper: 
```latex
$$\mathrm{d}\boldsymbol{x}_{+} =
    -\dot\sigma(t) \sigma(t) \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}; \sigma(t) \big) \,\mathrm{d}t\, + \,
      \beta(t) \sigma(t)^2 \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}; \sigma(t) \big) \,\mathrm{d}t+
      \sqrt{2 \beta(t)} \sigma(t) \,\mathrm{d}\omega_t
  
  .$$
```


The noise replacement is now proportional to the standard deviation `$\sigma(t)$` of the noise, with the proportionality factor `$\beta(t)$`. Indeed, expanding the score function in the middle term according to Eq. [eq:scoredenoiser\] yields `$\beta(t) ~\big[ D\big( \boldsymbol{x};\sigma(t) \big) - \boldsymbol{x}\big] ~\mathrm{d}t$`, which changes `$\boldsymbol{x}$` proportionally to the negative noise component; the stochastic term injects new noise at the same rate. Intuitively, scaling the magnitude of Langevin exploration according to the current noise standard deviation is a reasonable baseline, as the data manifold is effectively "spread out" by this amount due to the blurring of the density.

The *reverse* SDE used in denoising diffusion is simply obtained by applying the time reversal formula of Anderson [Anderson1982] (as stated in Eq. 6 of Song et al. [Song2021sde]) on Eq. [eq:sdeforward\]; the entire effect of the reversal is a change of sign in the middle term.

The scaled generalization of the SDE can be derived using a similar approach as with the ODE previously. As such, the derivation is omitted here.

## Our preconditioning and training (Eq. [eq:precloss\]) 

Following Eq. [eq:score\], the denoising score matching loss for a given denoiser `$D_\theta$` on a given noise level `$\sigma$` is given by 
```latex
$$\mathcal{L}(D_\theta; \sigma) = \mathbb{E}_{\boldsymbol{y}\sim p_\text{data}} ~\mathbb{E}_{\boldsymbol{n}\sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})} ~\big\lVert D_\theta(\boldsymbol{y}+ \boldsymbol{n}; \sigma) - \boldsymbol{y}\big\rVert^2_2
  .$$
```


We obtain overall training loss by taking a weighted expectation of `$\mathcal{L}(D_\theta; \sigma)$` over the noise levels: 
```latex
$$\begin{aligned}
  \mathcal{L}(D_\theta) &=& \mathbb{E}_{\sigma \sim p_\text{train}} \big[ \lambda(\sigma) ~\mathcal{L}(D_\theta; \sigma) \big] \\
  &=& \mathbb{E}_{\sigma \sim p_\text{train}} ~\Big[ \lambda(\sigma) ~\mathbb{E}_{\boldsymbol{y}\sim p_\text{data}} ~\mathbb{E}_{\boldsymbol{n}\sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})} ~\big\lVert D_\theta(\boldsymbol{y}+ \boldsymbol{n}; \sigma) - \boldsymbol{y}\big\rVert^2_2 \Big] \\
  &=& \mathbb{E}_{\sigma \sim p_\text{train}} ~\mathbb{E}_{\boldsymbol{y}\sim p_\text{data}} ~\mathbb{E}_{\boldsymbol{n}\sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})}  ~\Big[ \lambda(\sigma) ~\big\lVert D_\theta(\boldsymbol{y}+ \boldsymbol{n}; \sigma) - \boldsymbol{y}\big\rVert^2_2 \Big] \\
  &=& \mathbb{E}_{\sigma, \boldsymbol{y}, \boldsymbol{n}} \Big[ \lambda(\sigma) ~\big\lVert D_\theta(\boldsymbol{y}+ \boldsymbol{n}; \sigma) - \boldsymbol{y}\big\rVert^2_2 \Big]
  
  ,
\end{aligned}$$
```
 where the noise levels are distributed according to `$\sigma \sim p_\text{train}$` and weighted by `$\lambda(\sigma)$`.

Using our definition of `$D_\theta(\cdot)$` from Eq. [eq:preconditioning\], we can further rewrite `$\mathcal{L}(D_\theta)$` as 
```latex
$$\begin{aligned}
  && \mathbb{E}_{\sigma, \boldsymbol{y}, \boldsymbol{n}} \Big[ \lambda(\sigma) \big\lVert c_\text{skip}(\sigma) (\boldsymbol{y}{+} \boldsymbol{n}) + c_\text{out}(\sigma) F_\theta\big($c_\text{in}(\sigma) (\boldsymbol{y}{+} \boldsymbol{n}); c_\text{noise}(\sigma)$\big) - \boldsymbol{y}\big\rVert^2_2 \Big]  \\
  &=& \mathbb{E}_{\sigma, \boldsymbol{y}, \boldsymbol{n}} \Big[ \lambda(\sigma) \big\lVert c_\text{out}(\sigma) F_\theta\big($c_\text{in}(\sigma) (\boldsymbol{y}{+} \boldsymbol{n}); c_\text{noise}(\sigma)$\big) - \big( \boldsymbol{y}- c_\text{skip}(\sigma) (\boldsymbol{y}+ \boldsymbol{n}) \big) \big\rVert^2_2 \Big] \\
  &=& \mathbb{E}_{\sigma, \boldsymbol{y}, \boldsymbol{n}} \Big[ \lambda(\sigma) c_\text{out}(\sigma)^2 \big\lVert F_\theta\big($c_\text{in}(\sigma) (\boldsymbol{y}{+} \boldsymbol{n}); c_\text{noise}(\sigma)$\big) - \tfrac{1}{c_\text{out}(\sigma)} \big( \boldsymbol{y}- c_\text{skip}(\sigma) (\boldsymbol{y}{+} \boldsymbol{n}) \big) \big\rVert^2_2 \Big] \\
  &=& \mathbb{E}_{\sigma, \boldsymbol{y}, \boldsymbol{n}} \Big[ w(\sigma) ~\big\lVert F_\theta\big($c_\text{in}(\sigma) (\boldsymbol{y}{+} \boldsymbol{n}); c_\text{noise}(\sigma)$\big) - F_\text{target}(\boldsymbol{y}, \boldsymbol{n}; \sigma) \big\rVert^2_2 \Big]
  ,
\end{aligned}$$
```
 which matches Eq. [eq:precloss\] and corresponds to traditional supervised training of `$F_\theta$` using standard `$L_2$` loss with effective weight `$w(\cdot)$` and target `$F_\text{target}(\cdot)$` given by 
```latex
$$w(\sigma) = \lambda(\sigma) ~c_\text{out}(\sigma)^2
  \hspace{4mm}\text{and}\hspace{4mm}
  F_\text{target}(\boldsymbol{y}, \boldsymbol{n}; \sigma) = \tfrac{1}{c_\text{out}(\sigma)} \big( \boldsymbol{y}- c_\text{skip}(\sigma) (\boldsymbol{y}+ \boldsymbol{n}) \big)
  ,$$
```


We can now derive formulas for `$c_\text{in}(\sigma)$`, `$c_\text{out}(\sigma)$`, `$c_\text{skip}(\sigma)$`, and `$\lambda(\sigma)$` from first principles, shown in the "Ours" column of Table [tab:specifics\].

First, we require the training inputs of `$F_\theta(\cdot)$` to have unit variance: 
```latex
$$\begin{aligned}
  \mathop{\mathrm{Var}}_{\boldsymbol{y}, \boldsymbol{n}} \big[ c_\text{in}(\sigma) (\boldsymbol{y}+ \boldsymbol{n}) \big] &=& 1 \\
  c_\text{in}(\sigma)^2 ~\mathop{\mathrm{Var}}_{\boldsymbol{y}, \boldsymbol{n}} \big[ \boldsymbol{y}+ \boldsymbol{n}\big] &=& 1 \\
  c_\text{in}(\sigma)^2 \big( \sigma_\text{data}^2 + \sigma^2 \big) &=& 1 \\
  c_\text{in}(\sigma) &=& 1 \big/ \sqrt{\sigma^2 + \sigma_\text{data}^2}
  .
\end{aligned}$$
```


Second, we require the effective training target `$F_\text{target}$` to have unit variance: 
```latex
$$\begin{aligned}
  \mathop{\mathrm{Var}}_{\boldsymbol{y}, \boldsymbol{n}} \big[ F_\text{target}(\boldsymbol{y}, \boldsymbol{n}; \sigma) \big] &=& 1 \\
  \mathop{\mathrm{Var}}_{\boldsymbol{y}, \boldsymbol{n}} \Big[ \tfrac{1}{c_\text{out}(\sigma)} \big( \boldsymbol{y}- c_\text{skip}(\sigma) (\boldsymbol{y}+ \boldsymbol{n}) \big) \Big] &=& 1 \\
  \tfrac{1}{c_\text{out}(\sigma)^2} \mathop{\mathrm{Var}}_{\boldsymbol{y}, \boldsymbol{n}} \big[ \boldsymbol{y}- c_\text{skip}(\sigma) (\boldsymbol{y}+ \boldsymbol{n}) \big] &=& 1 \\
  c_\text{out}(\sigma)^2 &=& \mathop{\mathrm{Var}}_{\boldsymbol{y}, \boldsymbol{n}} \big[ \boldsymbol{y}- c_\text{skip}(\sigma) (\boldsymbol{y}+ \boldsymbol{n}) \big] \\
  c_\text{out}(\sigma)^2 &=& \mathop{\mathrm{Var}}_{\boldsymbol{y}, \boldsymbol{n}} \Big[ \big( 1 - c_\text{skip}(\sigma) \big) ~\boldsymbol{y}+ c_\text{skip}(\sigma) ~\boldsymbol{n}\Big] \\
  c_\text{out}(\sigma)^2 &=& \big( 1 - c_\text{skip}(\sigma) \big)^2 ~\sigma_\text{data}^2 + c_\text{skip}(\sigma)^2 ~\sigma^2
  
  .
\end{aligned}$$
```


Third, we select `$c_\text{skip}(\sigma)$` to minimize `$c_\text{out}(\sigma)$`, so that the errors of `$F_\theta$` are amplified as little as possible: 
```latex
$$c_\text{skip}(\sigma) = \mathop{\mathrm{arg\,min}}_{c_\text{skip}(\sigma)} c_\text{out}(\sigma)
  .$$
```
 Since `$c_\text{out}(\sigma) \ge 0$`, we can equivalently write 
```latex
$$c_\text{skip}(\sigma) = \mathop{\mathrm{arg\,min}}_{c_\text{skip}(\sigma)} c_\text{out}(\sigma)^2
  .$$
```
 This is a convex optimization problem; its solution is uniquely identified by setting the derivative w.r.t. `$c_\text{skip}(\sigma)$` to zero: 
```latex
$$\begin{aligned}
  0 &=& \mathrm{d}\big[ c_\text{out}(\sigma)^2 \big] / \mathrm{d}c_\text{skip}(\sigma) \\
  0 &=& \mathrm{d}\Big[ \big(1 - c_\text{skip}(\sigma)\big)^2 ~\sigma_\text{data}^2 + c_\text{skip}(\sigma)^2 ~\sigma^2 \Big] / \mathrm{d}c_\text{skip}(\sigma) \\
  0 &=& \sigma_\text{data}^2 ~\mathrm{d}\Big[ \big(1 - c_\text{skip}(\sigma)\big)^2 \Big] / \mathrm{d}c_\text{skip}(\sigma) + \sigma^2 ~\mathrm{d}\big[c_\text{skip}(\sigma)^2 \big] / \mathrm{d}c_\text{skip}(\sigma) \\
  0 &=& \sigma_\text{data}^2 ~\big[ 2 ~c_\text{skip}(\sigma) - 2 \big] + \sigma^2 ~\big[ 2 ~c_\text{skip}(\sigma) \big] \\
  0 &=& \big( \sigma^2 + \sigma_\text{data}^2 \big) ~c_\text{skip}(\sigma) - \sigma_\text{data}^2 \\
  c_\text{skip}(\sigma) &=& \sigma_\text{data}^2 / \big( \sigma^2 + \sigma_\text{data}^2 \big)
  
  .
\end{aligned}$$
```


We can now substitute Eq. [eq:cskip\] into Eq. [eq:coutpre\] to complete the formula for `$c_\text{out}(\sigma)$`: 
```latex
$$\begin{aligned}
  c_\text{out}(\sigma)^2 &=& \big( 1 - \big[ c_\text{skip}(\sigma) \big] \big)^2 ~\sigma_\text{data}^2 + \big[ c_\text{skip}(\sigma) \big]^2 ~\sigma^2 \\
  c_\text{out}(\sigma)^2 &=& \bigg( 1 - \bigg[ \frac{\sigma_\text{data}^2}{\sigma^2 + \sigma_\text{data}^2} \bigg] \bigg)^2 ~\sigma_\text{data}^2 + \bigg[ \frac{\sigma_\text{data}^2}{\sigma^2 + \sigma_\text{data}^2} \bigg]^2 ~\sigma^2 \\
  c_\text{out}(\sigma)^2 &=& \bigg[ \frac{\sigma^2 ~\sigma_\text{data}}{\sigma^2 + \sigma_\text{data}^2} \bigg]^2 + \bigg[ \frac{\sigma_\text{data}^2 ~\sigma}{\sigma^2 + \sigma_\text{data}^2} \bigg]^2 \\
  c_\text{out}(\sigma)^2 &=& \frac{\big( \sigma^2 ~\sigma_\text{data}\big)^2 + \big( \sigma_\text{data}^2 ~\sigma \big)^2}{\big( \sigma^2 + \sigma_\text{data}^2 \big)^2} \\
  c_\text{out}(\sigma)^2 &=& \frac{(\sigma \cdot \sigma_\text{data})^2 ~\big( \sigma^2 + \sigma_\text{data}^2 \big)}{\big( \sigma^2 + \sigma_\text{data}^2 \big)^2} \\
  c_\text{out}(\sigma)^2 &=& \frac{(\sigma \cdot \sigma_\text{data})^2}{\sigma^2 + \sigma_\text{data}^2} \\
  c_\text{out}(\sigma) &=& \sigma \cdot \sigma_\text{data}\big/ \sqrt{\sigma^2 + \sigma_\text{data}^2}
  .
\end{aligned}$$
```


Fourth, we require the effective weight `$w(\sigma)$` to be uniform across noise levels: 
```latex
$$\begin{aligned}
  w(\sigma) &=& 1 \\
  \lambda(\sigma) ~c_\text{out}(\sigma)^2 &=& 1 \\
  \lambda(\sigma) &=& 1 / c_\text{out}(\sigma)^2 \\
  \lambda(\sigma) &=& 1 \big/ \bigg[ \frac{\sigma \cdot \sigma_\text{data}}{\sqrt{\sigma^2 + \sigma_\text{data}^2}} \bigg]^2 \\
  \lambda(\sigma) &=& 1 \big/ \bigg[ \frac{(\sigma \cdot \sigma_\text{data})^2}{\sigma^2 + \sigma_\text{data}^2} \bigg] \\
  \lambda(\sigma) &=& \big( \sigma^2 + \sigma_\text{data}^2 \big) / (\sigma \cdot \sigma_\text{data})^2
  .
\end{aligned}$$
```


We follow previous work and initialize the output layer weights to zero. Consequently, upon initialization `$F_\theta(\cdot) = 0$` and the expected value of the loss at each noise level is `$1$`. This can be seen by substituting the choices of `$\lambda(\sigma)$` and `$c_\text{skip}(\sigma)$` into Eq. [eq:lossexpanded\], considered at a fixed `$\sigma$`: 
```latex
$$\begin{aligned}
  && \mathbb{E}_{\boldsymbol{y}, \boldsymbol{n}} \Big[ \lambda(\sigma) \big\lVert c_\text{skip}(\sigma) (\boldsymbol{y}{+} \boldsymbol{n}) + c_\text{out}(\sigma) F_\theta\big($c_\text{in}(\sigma) (\boldsymbol{y}{+} \boldsymbol{n}); c_\text{noise}(\sigma)$\big) - \boldsymbol{y}\big\rVert^2_2 \Big] \\
  %
  &=& \mathbb{E}_{\boldsymbol{y}, \boldsymbol{n}} \bigg[ \frac{\sigma^2 + \sigma_\text{data}^2}{(\sigma \cdot \sigma_\text{data})^2} \bigg\lVert \frac{\sigma_\text{data}^2}{\sigma^2 + \sigma_\text{data}^2} (\boldsymbol{y}{+} \boldsymbol{n}) - \boldsymbol{y}\bigg\rVert^2_2 \bigg] \\
  &=& \mathbb{E}_{\boldsymbol{y}, \boldsymbol{n}} \bigg[ \frac{\sigma^2 + \sigma_\text{data}^2}{(\sigma \cdot \sigma_\text{data})^2} \bigg\lVert \frac{\sigma_\text{data}^2 \boldsymbol{n}- \sigma^2 \boldsymbol{y}}{\sigma^2 + \sigma_\text{data}^2} \bigg\rVert^2_2 \bigg] \\
  &=& \mathbb{E}_{\boldsymbol{y}, \boldsymbol{n}} \bigg[ \frac{1}{\sigma^2 + \sigma_\text{data}^2} \bigg\lVert \frac{\sigma_\text{data}}{\sigma} \boldsymbol{n}- \frac{\sigma}{\sigma_\text{data}} \boldsymbol{y}\bigg\rVert^2_2 \bigg] \\
  &=& \frac{1}{\sigma^2 + \sigma_\text{data}^2} \mathbb{E}_{\boldsymbol{y}, \boldsymbol{n}} \bigg[   \frac{\sigma_\text{data}^2}{\sigma^2} \langle \boldsymbol{n}, \boldsymbol{n}\rangle + \frac{\sigma^2}{\sigma_\text{data}^2} \langle \boldsymbol{y}, \boldsymbol{y}\rangle - 2 \langle \boldsymbol{y}, \boldsymbol{n}\rangle \bigg] \\
  &=& \frac{1}{\sigma^2 + \sigma_\text{data}^2} \bigg[ \frac{\sigma_\text{data}^2}{\sigma^2} \underbrace{\mathop{\mathrm{Var}}(\boldsymbol{n})}_{= \sigma^2} + \frac{\sigma^2}{\sigma_\text{data}^2} \underbrace{\mathop{\mathrm{Var}}(\boldsymbol{y})}_{= \sigma_\text{data}^2} - 2 \underbrace{\mathop{\mathrm{Cov}}(\boldsymbol{y}, \boldsymbol{n})}_{=0} \bigg] \\
  &=& 1
\end{aligned}$$
```


# Reframing previous methods in our framework 

In this section, we derive the formulas shown in Table [tab:specifics\] for previous methods, discuss the corresponding original samplers and pre-trained models, and detail the practical considerations associated with using them in our framework.

In practice, the original implementations of these methods differ considerably in terms of the definitions of model inputs and outputs, dynamic range of image data, scaling of `$\boldsymbol{x}$`, and interpretation of `$\sigma$`. We eliminate this variation by standardizing on a unified setup where the model always matches our definition of `$F_\theta$`, image data is always represented in the continuous range `$[-1, 1]$`, and the details of `$\boldsymbol{x}$` and `$\sigma$` are always in agreement with Eq. [eq:odescale\].

We minimize the accumulation of floating point round-off errors by always executing Algorithms [alg:heun\] and [alg:stochastic\] at double precision (`float64`). However, we still execute the network `$F_\theta(\cdot)$` at single precision (`float32`) to minimize runtime and remain faithful to previous work in terms of network architecture.

## Variance preserving formulation 

### VP sampling

Song et al. [Song2021sde] define the VP SDE (Eq. 32 in [Song2021sde]) as 
```latex
$$\mathrm{d}\boldsymbol{x}= -\tfrac{1}{2} ~\Big( \beta_\text{min}+ t ~\big( \beta_\text{max}- \beta_\text{min}\big) \Big) ~\boldsymbol{x}~\mathrm{d}t + \sqrt{ \beta_\text{min}+ t ~\big( \beta_\text{max}- \beta_\text{min}\big) } ~\mathrm{d}\omega_t
  ,$$
```
 which matches Eq. [eq:songsde\] with the following choices for `$f$` and `$g$`: 
```latex
$$
  f(t) = -\tfrac{1}{2} ~\beta(t)
  ,\hspace{4mm}
  g(t) = \sqrt{\beta(t)}
  ,\hspace{4mm}\text{and}\hspace{4mm}
  \beta(t) = \big( \beta_\text{max}- \beta_\text{min}\big) ~t + \beta_\text{min}
  .$$
```


Let `$\alpha(t)$` denote the integral of `$\beta(t)$`: 
```latex
$$\begin{aligned}
  \alpha(t) &=& \int_0^t \beta(\xi) ~\mathrm{d}\xi \\
  &=& \int_0^t \Big[ \big( \beta_\text{max}- \beta_\text{min}\big) ~\xi + \beta_\text{min}\Big] ~\mathrm{d}\xi \\
  &=& \tfrac{1}{2} ~\big( \beta_\text{max}- \beta_\text{min}\big) ~t^2 + \beta_\text{min}~t \\
  &=& \tfrac{1}{2} ~\beta_\text{d}~t^2 + \beta_\text{min}~t
  ,
\end{aligned}$$
```
 where `$\beta_\text{d}= \beta_\text{max}- \beta_\text{min}$`. We can now obtain the formula for `$\sigma(t)$` by substituting Eq. [eq:vpfg\] into Eq. [eq:songscale\]: 
```latex
$$\begin{aligned}
  \sigma(t) &=& \sqrt{\int_0^t \frac{\big[ g(\xi) \big]^2}{\big[ s(\xi) \big]^2} ~\mathrm{d}\xi} \\
  &=& \sqrt{\int_0^t \frac{\big[ \sqrt{\beta(\xi)} \big]^2}{\big[ 1 / \sqrt{e^{\alpha(\xi)}} \big]^2} ~\mathrm{d}\xi} \\
  &=& \sqrt{\int_0^t \frac{\beta(\xi)}{1 / e^{\alpha(\xi)}} ~\mathrm{d}\xi} \\
  &=& \sqrt{\int_0^t \dot\alpha(\xi) ~e^{\alpha(\xi)} ~\mathrm{d}\xi} \\
  &=& \sqrt{e^{\alpha(t)} - e^{\alpha(0)}} \\
  &=& \sqrt{e^{\frac{1}{2} \beta_\text{d}t^2 + \beta_\text{min}t} - 1}
  
  ,
\end{aligned}$$
```
 which matches the "Schedule" row of Table [tab:specifics\]. Similarly for `$s(t)$`: 
```latex
$$\begin{aligned}
  s(t) &=& \exp\left( \int_0^t \big[ f(\xi) \big] ~\mathrm{d}\xi \right) \\
  &=& \exp\left( \int_0^t \big[ -\tfrac{1}{2} ~\beta(\xi) \big] ~\mathrm{d}\xi \right) \\
  &=& \exp\left( -\tfrac{1}{2} \left[ \int_0^t \beta(\xi) ~\mathrm{d}\xi \right] \right) \\
  &=& \exp\left( -\tfrac{1}{2} ~\alpha(t) \right) \\
  &=& 1 / \sqrt{e^{\alpha(t)}} \\
  &=& 1 / \sqrt{e^{\frac{1}{2} \beta_\text{d}t^2 + \beta_\text{min}t}}
  
  ,
\end{aligned}$$
```
 which matches the "Scaling" row of Table [tab:specifics\]. We can equivalently write Eq. [eq:vpscale\] in a slightly simpler form by utilizing Eq. [eq:vpsigma\]: 
```latex
$$s(t) = 1 / \sqrt{\sigma(t)^2 + 1}
  
  .$$
```


Song et al. [Song2021sde] choose to distribute the sampling time steps `$\{t_0, \dots, t_{N-1}\}$` at uniform intervals within `$[\epsilon_\text{s}, 1]$`. This corresponds to setting 
```latex
$$t_{i<N} = 1 + \tfrac{i}{N-1}(\epsilon_\text{s} - 1)
  ,$$
```
 which matches the "Time steps" row of Table [tab:specifics\].

Finally, Song et al. [Song2021sde] set `$\beta_\text{min}= 0.1$`, `$\beta_\text{max}= 20$`, and `$\epsilon_\text{s} = 10^{-3}$` (Appendix C in [Song2021sde]), and choose to represent images in the range `$[-1, 1]$`. These choices are readily compatible with our formulation and are reflected by the "Parameters" section of Table [tab:specifics\].

### VP preconditioning

In the VP case, Song et al. [Song2021sde] approximate the score of `$p_t(\boldsymbol{x})$` of Eq. [eq:songmarginal\] as [^1] 
```latex
$$\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p_t(\boldsymbol{x}) ~\approx~ \underbrace{{-}\tfrac{1}{\bar\sigma(t)} ~F_\theta\big( \boldsymbol{x}; ~(M{-}1)t \big)}_{\mathop{\mathrm{score}}(\boldsymbol{x}; F_\theta, t)}
  
  ,$$
```
 where `$M = 1000$`, `$F_\theta$` denotes the network, and `$\bar\sigma(t)$` corresponds to the standard deviation of the perturbation kernel of Eq. [eq:songperturbation\].

Let us expand the definitions of `$p_t(\boldsymbol{x})$` and `$\bar\sigma(t)$` from Eqs. [eq:psigma\] and [eq:songperturbation\], respectively, and substitute `$\boldsymbol{x}= s(t) \hat\boldsymbol{x}$` to obtain the corresponding formula with respect to the non-scaled variable `$\hat\boldsymbol{x}$`: 
```latex
$$\begin{aligned}
  \nabla_{\boldsymbol{x}} \log \big[ p\big( \boldsymbol{x}/ s(t); \sigma(t) \big) \big] &\approx& {-}\tfrac{1}{[s(t) \sigma(t)]} ~F_\theta\big( \boldsymbol{x}; ~(M{-}1)t \big) \\
  \nabla_{[s(t) \hat\boldsymbol{x}]} \log p\big( [s(t) ~\hat\boldsymbol{x}] / s(t); \sigma(t) \big) &\approx& {-}\tfrac{1}{s(t) \sigma(t)} ~F_\theta\big( [s(t) ~\hat\boldsymbol{x}]; ~(M{-}1)t \big) \\
  \tfrac{1}{s(t)} \nabla_{\hat\boldsymbol{x}} \log p\big( \hat\boldsymbol{x}; \sigma(t) \big) &\approx& {-}\tfrac{1}{s(t) \sigma(t)} ~F_\theta\big( s(t) ~\hat\boldsymbol{x}; ~(M{-}1)t \big) \\
  \nabla_{\hat\boldsymbol{x}} \log p\big( \hat\boldsymbol{x}; \sigma(t) \big) &\approx& {-}\tfrac{1}{\sigma(t)} ~F_\theta\big( s(t) ~\hat\boldsymbol{x}; ~(M{-}1)t \big)
  .
\end{aligned}$$
```


We can now replace the left-hand side with Eq. [eq:scoredenoiser\] and expand the definition of `$s(t)$` from Eq. [eq:vpscalesimple\]: 
```latex
$$\begin{aligned}
  \Big[ \Big( D\big( \hat\boldsymbol{x}; \sigma(t) \big) - \hat\boldsymbol{x}\Big) / \sigma(t)^2 \Big] &\approx& {-}\tfrac{1}{\sigma(t)} ~F_\theta\big( s(t) ~\hat\boldsymbol{x}; ~(M{-}1)t \big) \\
  D\big( \hat\boldsymbol{x}; \sigma(t) \big) &\approx& \hat\boldsymbol{x}- \sigma(t) ~F_\theta\big( s(t) ~\hat\boldsymbol{x}; ~(M{-}1)t \big) \\
  D\big( \hat\boldsymbol{x}; \sigma(t) \big) &\approx& \hat\boldsymbol{x}- \sigma(t) ~F_\theta\bigg( \bigg[ \tfrac{1}{\sqrt{\sigma(t)^2 + 1}} \bigg] ~\hat\boldsymbol{x}; ~(M{-}1)t \bigg)
  ,
\end{aligned}$$
```
 which can be further expressed in terms of `$\sigma$` by replacing `$\sigma(t) \rightarrow \sigma$` and `$t \rightarrow \sigma^{-1}(\sigma)$`: 
```latex
$$D(\hat\boldsymbol{x}; \sigma) ~\approx~ \hat\boldsymbol{x}- \sigma ~F_\theta\Big( \tfrac{1}{\sqrt{\sigma^2 + 1}} ~\hat\boldsymbol{x}; ~(M{-}1) ~\sigma^{-1}(\sigma) \Big)
  
  .$$
```


We adopt the right-hand side of Eq. [eq:vpprecondtemp\] as the definition of `$D_\theta$`, obtaining 
```latex
$$D_\theta(\hat\boldsymbol{x}; \sigma) = \underbrace{1~\cdot}_{c_\text{skip}}\hat\boldsymbol{x}~\underbrace{-~\sigma}_{c_\text{out}} \,\cdot ~F_\theta\Big( \underbrace{\tfrac{1}{\sqrt{\sigma^2 + 1}}}_{c_\text{in}} \,\cdot~\hat\boldsymbol{x}; ~\underbrace{(M{-}1)~\sigma^{-1}(\sigma)}_{c_\text{noise}} \Big)
  
  ,$$
```
 where `$c_\text{skip}$`, `$c_\text{out}$`, `$c_\text{in}$`, and `$c_\text{noise}$` match the "Network and preconditioning" section of Table [tab:specifics\].

### VP training

Song et al. [Song2021sde] define their training loss as [^2] 
```latex
$$\mathbb{E}_{t \sim \mathcal{U}(\epsilon_\text{t}, 1), \boldsymbol{y}\sim p_\text{data}, \bar\boldsymbol{n}\sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \Big[ \big\lVert \bar\sigma(t) ~\mathop{\mathrm{score}}\big( s(t) ~\boldsymbol{y}+ \bar\sigma(t) ~\bar\boldsymbol{n}; ~F_\theta, t \big) + \bar\boldsymbol{n}\big\rVert^2_2 \Big]
  ,$$
```
 where the definition of `$\mathop{\mathrm{score}}(\cdot)$` is the same as in Eq. [eq:vpprecondorig\]. Let us simplify the formula by substituting `$\bar\sigma(t) = s(t) \sigma(t)$` and `$\bar\boldsymbol{n}= \boldsymbol{n}/ \sigma(t)$`, where `$\boldsymbol{n}\sim \mathcal{N}(\mathbf{0}, \sigma(t)^2 \mathbf{I})$`: 
```latex
$$\begin{aligned}
  && \mathbb{E}_{t, \boldsymbol{y}, \bar\boldsymbol{n}} \Big[ \big\lVert s(t) \sigma(t) ~\mathop{\mathrm{score}}\big( s(t) ~\boldsymbol{y}+ [s(t)\sigma(t)] ~\bar\boldsymbol{n}; ~F_\theta, t \big) + \bar\boldsymbol{n}\big\rVert^2_2 \Big] \\
  &=& \mathbb{E}_{t, \boldsymbol{y}, \boldsymbol{n}} \Big[ \big\lVert s(t) \sigma(t) ~\mathop{\mathrm{score}}\big( s(t) ~\boldsymbol{y}+ s(t)\sigma(t) ~[\boldsymbol{n}/ \sigma(t)]; ~F_\theta, t \big) + [\boldsymbol{n}/ \sigma(t)] \big\rVert^2_2 \Big] \\
  &=& \mathbb{E}_{t, \boldsymbol{y}, \boldsymbol{n}} \Big[ \big\lVert s(t) \sigma(t) ~\mathop{\mathrm{score}}\big( s(t) ~(\boldsymbol{y}+ \boldsymbol{n}); ~F_\theta, t \big) + \boldsymbol{n}/ \sigma(t) \big\rVert^2_2 \Big]
  
  .
\end{aligned}$$
```


We can express `$\mathop{\mathrm{score}}(\cdot)$` in terms of `$D_\theta(\cdot)$` by combining Eqs. [eq:vpprecondorig\], [eq:vpscalesimple\], and [eq:scaledscore\]: 
```latex
$$\mathop{\mathrm{score}}\big( s(t) ~\boldsymbol{x}; F_\theta, t \big) ~=~ \tfrac{1}{s(t) \sigma(t)^2} \Big( D_\theta \big( \boldsymbol{x}; \sigma(t) \big) - \boldsymbol{x}\Big)
  .$$
```


Substituting this back into Eq. [eq:vplosstemp\] gives 
```latex
$$\begin{aligned}
  && \mathbb{E}_{t, \boldsymbol{y}, \boldsymbol{n}} \Big[ \big\lVert s(t) \sigma(t) ~\Big[ \tfrac{1}{s(t) \sigma(t)^2} \Big( D_\theta \big( \boldsymbol{y}+ \boldsymbol{n}; \sigma(t) \big) - (\boldsymbol{y}+ \boldsymbol{n}) \Big) \Big] + \tfrac{1}{\sigma(t)} ~\boldsymbol{n}\big\rVert^2_2 \Big] \\
  &=& \mathbb{E}_{t, \boldsymbol{y}, \boldsymbol{n}} \Big[ \big\lVert \tfrac{1}{\sigma(t)} \Big( D_\theta \big( \boldsymbol{y}+ \boldsymbol{n}; \sigma(t) \big) - (\boldsymbol{y}+ \boldsymbol{n}) \Big) + \tfrac{1}{\sigma(t)} ~\boldsymbol{n}\big\rVert^2_2 \Big] \\
  &=& \mathbb{E}_{t, \boldsymbol{y}, \boldsymbol{n}} \Big[ \tfrac{1}{\sigma(t)^2} ~\big\lVert D_\theta \big( \boldsymbol{y}+ \boldsymbol{n}; \sigma(t) \big) - \boldsymbol{y}\big\rVert^2_2 \Big]
  .
\end{aligned}$$
```


We can further express this in terms of `$\sigma$` by replacing `$\sigma(t) \rightarrow \sigma$` and `$t \rightarrow \sigma^{-1}(\sigma)$`: 
```latex
$$\underbrace{\mathbb{E}_{\sigma^{-1}(\sigma) \sim \mathcal{U}(\epsilon_\text{t}, 1)}}_{p_\text{train}} \mathbb{E}_{\boldsymbol{y}, \boldsymbol{n}} \Big[ \underbrace{\tfrac{1}{\sigma^2}}_{\lambda} \big\lVert D_\theta \big( \boldsymbol{y}+ \boldsymbol{n}; \sigma \big) - \boldsymbol{y}\big\rVert^2_2 \Big]
  
  ,$$
```
 which matches Eq. [eq:totalloss\] with the choices for `$p_\text{train}$` and `$\lambda$` shown in the "Training" section of Table [tab:specifics\].

### VP practical considerations

The pre-trained VP model that we use on CIFAR-10 corresponds to the "DDPM++ cont. (VP)" checkpoint [^3] provided by Song et al. [Song2021sde]. It contains a total of 62 million trainable parameters and supports a continuous range of noise levels `$\sigma \in \big[ \sigma(\epsilon_\text{t}), \sigma(1) \big] \approx [0.001, 152]$`, i.e., wider than our preferred sampling range `$[0.002, 80]$`. We import the model directly as `$F_\theta(\cdot)$` and run Algorithms [alg:heun\] and [alg:stochastic\] using the definitions in Table [tab:specifics\].

In Figure [2]a, the differences between the original sampler (blue) and our reimplementation (orange) are explained by oversights in the implementation of Song et al. [Song2021sde], also noted by Jolicoeur-Martineau et al. [Jolicoeur2021] (Appendix D in [Jolicoeur2021]). First, the original sampler employs an incorrect multiplier [^4] in the Euler step: it multiplies `$\mathrm{d}\boldsymbol{x}/ \mathrm{d}t$` by `$-1 / N$` instead of `$(\epsilon_\text{s} - 1) / (N - 1)$`. Second, it either overshoots or undershoots on the last step by going from `$t_{N-1} = \epsilon_\text{s}$` to `$t_N = \epsilon_\text{s} - 1 / N$`, where `$t_N < 0$` when `$N < 1000$`. In practice, this means that the generated images contain noticeable noise that becomes quite severe with, e.g., `$N = 128$`. Our formulation avoids these issues, because the step sizes in Algorithm [alg:heun\] are computed consistently from `$\{t_i\}$` and `$t_N = 0$`.

## Variance exploding formulation 

### VE sampling in theory

Song et al. [Song2021sde] define the VE SDE (Eq. 30 in [Song2021sde]) as 
```latex
$$\mathrm{d}\boldsymbol{x}= \sigma_\text{min}\bigg( \frac{\sigma_\text{max}}{\sigma_\text{min}} \bigg)^t \sqrt{2 \log \frac{\sigma_\text{max}}{\sigma_\text{min}}} ~\mathrm{d}\omega_t
  ,$$
```
 which matches Eq. [eq:songsde\] with 
```latex
$$
  f(t) = 0
  ,\hspace{4mm}
  g(t) = \sigma_\text{min}\sqrt{2\log\sigma_\text{d}} ~\sigma_\text{d}^t
  ,\hspace{4mm}\text{and}\hspace{4mm}
  \sigma_\text{d}= \sigma_\text{max}/ \sigma_\text{min}
  .$$
```


The VE formulation does not employ scaling, which can be easily seen from Eq. [eq:songscale\]: 
```latex
$$s(t) = \exp\left( \int_0^t \big[ f(\xi) \big] ~\mathrm{d}\xi \right) = \exp\left( \int_0^t \big[ 0 \big] ~\mathrm{d}\xi \right) = \exp(0) = 1
  .$$
```


Substituting Eq. [eq:vefg\] into Eq. [eq:songscale\] suggests the following form for `$\sigma(t)$`: 
```latex
$$\begin{aligned}
  \sigma(t) &=& \sqrt{\int_0^t \frac{\big[ g(\xi) \big]^2}{\big[ s(\xi) \big]^2} ~\mathrm{d}\xi} \\
  &=& \sqrt{\int_0^t \frac{\big[ \sigma_\text{min}\sqrt{2\log\sigma_\text{d}} ~\sigma_\text{d}^\xi \big]^2}{\big[ 1 \big]^2} ~\mathrm{d}\xi} \\
  &=& \sqrt{\int_0^t \sigma_\text{min}^2 ~\big[ 2\log\sigma_\text{d}\big] ~\big[ \sigma_\text{d}^{2\xi} \big] ~\mathrm{d}\xi} \\
  &=& \sigma_\text{min}\sqrt{\int_0^t \Big[ \log \big( \sigma_\text{d}^2 \big) \Big] ~\Big[ \big( \sigma_\text{d}^2 \big)^\xi \Big] ~\mathrm{d}\xi} \\
  &=& \sigma_\text{min}\sqrt{\big( \sigma_\text{d}^2 \big)^t - \big( \sigma_\text{d}^2 \big)^0} \\
  &=& \sigma_\text{min}\sqrt{\sigma_\text{d}^{2t} - 1}
  
  .
\end{aligned}$$
```


Eq. [eq:fakevesigma\] is consistent with the perturbation kernel reported by Song et al. (Eq. 29 in [Song2021sde]). However, we note that this does not fulfill their intended definition of `$\sigma(t) = \sigma_\text{min}~\big( \tfrac{\sigma_\text{max}}{\sigma_\text{min}} \big)^t$` (Appendix C in [Song2021sde]).

### VE sampling in practice

The original implementation [^5] of Song et al. [Song2021sde] uses reverse diffusion predictor [^6] to integrate discretized reverse probability flow [^7] of discretized VE SDE [^8]. Put together, these yield the following update rule for `$\boldsymbol{x}_{i+1}$`: 
```latex
$$
  \boldsymbol{x}_{i+1} = \boldsymbol{x}_i + \tfrac{1}{2} ~\big( \bar\sigma_i^2 - \bar\sigma_{i+1}^2 \big) ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log \bar p_i (\boldsymbol{x})
  ,$$
```
 where 
```latex
$$\bar\sigma_{i<N} = \sigma_\text{min}~\bigg( \frac{\sigma_\text{max}}{\sigma_\text{min}} \bigg)^{1 - i / (N-1)}
  \hspace{4mm}\text{and}\hspace{4mm}
  \bar\sigma_N = 0
  .$$
```


Interestingly, Eq. [eq:vestep\] is identical to the Euler iteration of our ODE with the following choices: 
```latex
$$s(t) = 1
  ,\hspace{4mm}
  \sigma(t) = \sqrt{t}
  ,\hspace{4mm}\text{and}\hspace{4mm}
  t_i = \bar\sigma_i^2
  .$$
```


These formulas match the "Sampling" section of Table [tab:specifics\], and their correctness can be verified by substituting them into line 5 of Algorithm [alg:heun\]: 
```latex
$$\begin{aligned}
  \boldsymbol{x}_{i+1} &=& \boldsymbol{x}_i + (t_{i+1} - t_i) ~\boldsymbol{d}_i \\
  &=& \boldsymbol{x}_i + (t_{i+1} - t_i) \bigg[ \bigg( \frac{\dot\sigma(t)}{\sigma(t)} + \frac{\dot s(t)}{s(t)} \bigg) \boldsymbol{x}- \frac{\dot\sigma(t) s(t)}{\sigma(t)} D \bigg( \frac{\boldsymbol{x}}{s(t)}; \sigma(t) \bigg) \bigg] \\
  &=& \boldsymbol{x}_i + (t_{i+1} - t_i) \bigg[ \frac{\dot\sigma(t)}{\sigma(t)} ~\boldsymbol{x}- \frac{\dot\sigma(t)}{\sigma(t)} ~D \big( \boldsymbol{x}; \sigma(t) \big) \bigg] \\
  &=& \boldsymbol{x}_i - (t_{i+1} - t_i) ~\dot\sigma(t) ~\sigma(t) \bigg[ \Big( D \big( \boldsymbol{x}; \sigma(t) \big) - \boldsymbol{x}\Big) \big/ \sigma(t)^2 \bigg] \\
  &=& \boldsymbol{x}_i - (t_{i+1} - t_i) ~\dot\sigma(t) ~\sigma(t) ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}; \sigma(t) \big) \\
  &=& \boldsymbol{x}_i - (t_{i+1} - t_i) \Big[ \tfrac{1}{2\sqrt{t}} \Big] \Big[ \sqrt{t} \Big] ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}; \sigma(t) \big) \\
  &=& \boldsymbol{x}_i + \tfrac{1}{2} ~(t_i - t_{i+1}) ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}; \sigma(t) \big) \\
  &=& \boldsymbol{x}_i + \tfrac{1}{2} ~\big( \bar\sigma_i^2 - \bar\sigma_{i+1}^2 \big) ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}; \sigma(t) \big)
  ,
\end{aligned}$$
```
 which is made identical to Eq. [eq:vestep\] by the choice `$\bar p_i(\boldsymbol{x}) = p\big( \boldsymbol{x}; \sigma(t_i) \big)$`.

Finally, Song et al. [Song2021sde] set `$\sigma_\text{min}= 0.01$` and `$\sigma_\text{max}= 50$` for CIFAR-10 (Appendix C in [Song2021sde]), and choose to represent their images in the range `$[0, 1]$` to match previous SMLD models. Since our standardized range `$[-1, 1]$` is twice as large, we must multiply `$\sigma_\text{min}$` and `$\sigma_\text{max}$` by 2`$\times$` to compensate. The "Parameters" section of Table [tab:specifics\] reflects these adjusted values.

### VE preconditioning

In the VE case, Song et al. [Song2021sde] approximate the score of `$p_t(\boldsymbol{x})$` of Eq. [eq:songmarginal\] directly as [^9] 
```latex
$$\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p_t(\boldsymbol{x}) ~\approx~ \bar{F}_\theta\big( \boldsymbol{x}; \sigma(t) \big)
  
  ,$$
```
 where the network `$\bar{F}_\theta$` is designed to include additional pre- [^10] and [^11] postprocessing [^12] steps: 
```latex
$$\bar{F}_\theta\big( \boldsymbol{x}; \sigma \big) ~=~ \tfrac{1}{\sigma} ~F_\theta\big( 2 \boldsymbol{x}{-} 1; \log(\sigma) \big)
  
  .$$
```
 For consistency, we handle the pre- and postprocessing using `$\{c_\text{skip}, c_\text{out}, c_\text{in}, c_\text{noise}\}$` as opposed to baking them into the network itself.

We cannot use Eqs. [eq:vescoreorig\] and [eq:veprecondorig\] directly in our framework, however, because they assume that the images are represented in range `$[0, 1]$`. In order to use `$[-1, 1]$` instead, we replace `$p_t(\boldsymbol{x}) \rightarrow p_t(2 \boldsymbol{x}{-} 1)$`, `$\boldsymbol{x}\rightarrow \tfrac{1}{2} \boldsymbol{x}+ \tfrac{1}{2}$` and `$\sigma \rightarrow \tfrac{1}{2} \sigma$`: 
```latex
$$\begin{aligned}
  \nabla_{[\frac{1}{2} \boldsymbol{x}+ \frac{1}{2}]} \log p_t \big( 2 \big[ \tfrac{1}{2} \boldsymbol{x}+ \tfrac{1}{2} \big] {-} 1 \big) &\approx& \tfrac{1}{[\frac{1}{2} \sigma]} ~F_\theta\big( 2 \big[ \tfrac{1}{2} \boldsymbol{x}+ \tfrac{1}{2} \big] {-} 1; \log \big[ \tfrac{1}{2} \sigma \big] \big) \\
  2 ~\nabla_{\boldsymbol{x}} \log p_t(\boldsymbol{x}) &\approx& \tfrac{2}{\sigma} ~F_\theta\Big( \boldsymbol{x}; \log \big( \tfrac{1}{2} \sigma \big) \Big) \\
  \nabla_{\boldsymbol{x}} \log p(\boldsymbol{x}; \sigma) &\approx& \tfrac{1}{\sigma} ~F_\theta\Big( \boldsymbol{x}; \log \big( \tfrac{1}{2} \sigma \big) \Big)
  
  .
\end{aligned}$$
```


We can now express the model in terms of `$D_\theta(\cdot)$` by replacing the left-hand side of Eq. [eq:veprecondtemp\] with Eq. [eq:scoredenoiser\]: 
```latex
$$\begin{aligned}
  \Big( D_\theta \big( \boldsymbol{x}; \sigma \big) - \boldsymbol{x}\Big) / \sigma^2 &=& \tfrac{1}{\sigma} ~F_\theta\Big( \boldsymbol{x}; \log\big( \tfrac{1}{2} \sigma \big) \Big)  \\
  D_\theta \big( \boldsymbol{x}; \sigma \big) &=& \underbrace{1~\cdot}_{c_\text{skip}} \boldsymbol{x}+ \underbrace{\sigma~\cdot}_{c_\text{out}} F_\theta\Big(\underbrace{1~\cdot}_{c_\text{in}} \boldsymbol{x}; ~\underbrace{\log \big( \tfrac{1}{2} \sigma \big)}_{c_\text{noise}} \Big)
  
  ,
\end{aligned}$$
```
 where `$c_\text{skip}$`, `$c_\text{out}$`, `$c_\text{in}$`, and `$c_\text{noise}$` match the "Network and preconditioning" section of Table [tab:specifics\].

### VE training

Song et al. [Song2021sde] define their training loss similarly for VP and VE, so we can reuse Eq. [eq:vplosstemp\] by borrowing the definition of `$\mathop{\mathrm{score}}(\cdot)$` from Eq. [eq:vedenoisertemp\]: 
```latex
$$\begin{aligned}
  && \mathbb{E}_{t, \boldsymbol{y}, \boldsymbol{n}} \Big[ \big\lVert s(t) \sigma(t) ~\mathop{\mathrm{score}}\big( s(t) ~(\boldsymbol{y}+ \boldsymbol{n}); ~F_\theta, t \big) + \boldsymbol{n}/ \sigma(t) \big\rVert^2_2 \Big] \\
  &=& \mathbb{E}_{t, \boldsymbol{y}, \boldsymbol{n}} \Big[ \big\lVert \sigma(t) ~\mathop{\mathrm{score}}\big( \boldsymbol{y}+ \boldsymbol{n}; ~F_\theta, t \big) + \boldsymbol{n}/ \sigma(t) \big\rVert^2_2 \Big] \\
  &=& \mathbb{E}_{t, \boldsymbol{y}, \boldsymbol{n}} \Big[ \big\lVert \sigma(t) ~\Big[ \Big( D_\theta \big( \boldsymbol{y}+ \boldsymbol{n}; \sigma(t) \big) - (\boldsymbol{y}+ \boldsymbol{n}) \Big) / \sigma(t)^2 \Big] + \boldsymbol{n}/ \sigma(t) \big\rVert^2_2 \Big] \\
  &=& \mathbb{E}_{t, \boldsymbol{y}, \boldsymbol{n}} \Big[ \tfrac{1}{\sigma(t)^2} ~\big\lVert D_\theta \big( \boldsymbol{y}+ \boldsymbol{n}; \sigma(t) \big) - \boldsymbol{y}\big\rVert^2_2 \Big]
  
  .
\end{aligned}$$
```


For VE training, the original implementation [^13] defines `$\sigma(t) = \sigma_\text{min}~\big( \tfrac{\sigma_\text{max}}{\sigma_\text{min}} \big)^t$`. We can thus rewrite Eq. [eq:velosstemp\] as 
```latex
$$\underbrace{\mathbb{E}_{\ln(\sigma) \sim \mathcal{U}( \ln(\sigma_\text{min}), \ln(\sigma_\text{max}))}}_{p_\text{train}} \mathbb{E}_{\boldsymbol{y}, \boldsymbol{n}} \Big[ \underbrace{\tfrac{1}{\sigma^2}}_{\lambda} \big\lVert D_\theta \big( \boldsymbol{y}+ \boldsymbol{n}; \sigma \big) - \boldsymbol{y}\big\rVert^2_2 \Big]
  ,$$
```
 which matches Eq. [eq:totalloss\] with the choices for `$p_\text{train}$` and `$\lambda$` shown in the "Training" section of Table [tab:specifics\].

### VE practical considerations

The pre-trained VE model that we use on CIFAR-10 corresponds to the "NCSN++ cont. (VE)" checkpoint [^14] provided by Song et al. [Song2021sde]. It contains a total of 63 million trainable parameters and supports a continuous range of noise levels `$\sigma \in \big[ \sigma(\epsilon_\text{t}), \sigma(1) \big] \approx [0.02, 100]$`. This is narrower than our preferred sampling range `$[0.002, 80]$`, so we set `$\sigma_\text{min}= 0.02$` in all related experiments. Note that this limitation is lifted by our training improvements in config \textsc{e}, so we revert back to using `$\sigma_\text{min}= 0.002$` with configs \textsc{e} and \textsc{f} in Table [tab:TrainingTable\]. When importing the model, we remove the pre- and postprocessing steps shown in Eq. [eq:veprecondorig\] to stay consistent with the definition of `$F_\theta(\cdot)$` in Eq. [eq:veprecond\]. With these changes, we can run Algorithms [alg:heun\] and [alg:stochastic\] using the definitions in Table [tab:specifics\].

In Figure [2]b, the differences between the original sampler (blue) and our reimplementation (orange) are explained by floating point round-off errors that the original implementation suffers from at high step counts. Our results are more accurate in these cases because we represent `$\boldsymbol{x}_i$` at double precision in Algorithm [alg:heun\].

## Improved DDPM and DDIM 

### DDIM ODE formulation 

Song et al. [Song2020ddim] make the observation that their deterministic DDIM sampler can be expressed as Euler integration of the following ODE (Eq. 14 in [Song2020ddim]): 
```latex
$$
  \mathrm{d}\boldsymbol{x}(t) = \epsilon_\theta^{(t)} \left( \frac{\boldsymbol{x}(t)}{\sqrt{\sigma(t)^2 + 1}} \right) ~\mathrm{d}\sigma(t)
  ,$$
```
 where `$\boldsymbol{x}(t)$` is a scaled version of the iterate that appears in their discrete update formula (Eq. 10 in [Song2020ddim]) and `$\epsilon_\theta$` is a model trained to predict the normalized noise vector, i.e., `$\epsilon_\theta^{(t)}\big( \boldsymbol{x}(t) / \sqrt{\sigma(t)^2 + 1} \big) \approx \boldsymbol{n}(t) / \sigma(t)$` for `$\boldsymbol{x}(t) = \boldsymbol{y}(t) + \boldsymbol{n}(t)$`. In our formulation, `$D_\theta$` is trained to approximate the clean signal, i.e., `$D_\theta\big( \boldsymbol{x}(t); \sigma(t) \big) \approx \boldsymbol{y}$`, so we can reinterpret `$\epsilon_\theta$` in terms of `$D_\theta$` as follows: 
```latex
$$\begin{aligned}
  \boldsymbol{n}(t) &=& \boldsymbol{x}(t) - \boldsymbol{y}(t) \\
  \big[ \boldsymbol{n}(t) / \sigma(t) \big] &=& \big( \boldsymbol{x}(t) - \big[ \boldsymbol{y}(t) \big] \big) / \sigma(t) \\
  \epsilon_\theta^{(t)} \big( \boldsymbol{x}(t) / \sqrt{\sigma(t)^2 + 1} \big) &=& \big( \boldsymbol{x}(t) - D_\theta \big( \boldsymbol{x}(t); \sigma(t) \big) \big) / \sigma(t)
  .
\end{aligned}$$
```


Assuming ideal `$\epsilon(\cdot)$` and `$D(\cdot)$` in `$L_2$` sense, we can further simplify the above formula using Eq. [eq:scoredenoiser\]: 
```latex
$$\begin{aligned}
  \epsilon^{(t)} \big( \boldsymbol{x}(t) / \sqrt{\sigma(t)^2 + 1} \big) &=& \big( \boldsymbol{x}(t) - D \big( \boldsymbol{x}(t); \sigma(t) \big) \big) / \sigma(t)  \\
  &=& -\sigma(t) ~\Big[ \Big( D \big( \boldsymbol{x}(t); \sigma(t) \big) - \boldsymbol{x}(t) \Big) / \sigma(t)^2 \Big] \\
  &=& -\sigma(t) ~\nabla_{\boldsymbol{x}(t)} \log p\big( \boldsymbol{x}(t); \sigma(t) \big)
  
  .
\end{aligned}$$
```


Substituting Eq. [eq:ddimeps\] back into Eq. [eq:ddimode\] gives 
```latex
$$\mathrm{d}\boldsymbol{x}(t) = -\sigma(t) ~\nabla_{\boldsymbol{x}(t)} \log p\big( \boldsymbol{x}(t); \sigma(t) \big) ~\mathrm{d}\sigma(t)
  ,$$
```
 which we can further simplify by setting `$\sigma(t) = t$`: 
```latex
$$\mathrm{d}\boldsymbol{x}= -t ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}; \sigma(t) \big) ~\mathrm{d}t
  .$$
```
 This matches our Eq. [eq:odescale\] with `$s(t) = 1$` and `$\sigma(t) = t$`, reflected by the "Sampling" section of Table [tab:specifics\].

### iDDPM time step discretization

The original DDPM formulation of Ho et al. [Ho2020] defines the forward process (Eq. 2 in [Ho2020]) as a Markov chain that gradually adds Gaussian noise to `$\bar\boldsymbol{x}_0 \sim p_\text{data}$` according to a discrete variance schedule `$\{\beta_1, \dots, \beta_T\}$`: 
```latex
$$q(\bar\boldsymbol{x}_t ~|~ \bar\boldsymbol{x}_{t-1}) = \mathcal{N}\big( \bar\boldsymbol{x}_t; ~\sqrt{1 - \beta_t} ~\bar\boldsymbol{x}_{t-1}, ~\beta_t ~\mathbf{I}\big)
  .$$
```


The corresponding transition probability from `$\bar\boldsymbol{x}_0$` to `$\bar\boldsymbol{x}_t$` (Eq. 4 in [Ho2020]) is given by 
```latex
$$
  q(\bar\boldsymbol{x}_t ~|~ \bar\boldsymbol{x}_0) = \mathcal{N}\big( \bar\boldsymbol{x}_t; ~\sqrt{\bar\alpha_t} ~\bar\boldsymbol{x}_0, ~(1 - \bar\alpha_t) ~\mathbf{I}\big)
  ,\hspace{4mm}\text{where}\hspace{4mm}
  \bar\alpha_t = \prod_{s=1}^t ~(1 - \beta_s)
  .$$
```


Ho et al. [Ho2020] define `$\{\beta_t\}$` based on a linear schedule and then calculate the corresponding `$\{\bar\alpha_t\}$` from Eq. [eq:ddpmtrans\]. Alternatively, one can also define `$\{\bar\alpha_t\}$` first and then solve for `$\{\beta_t\}$`: 
```latex
$$\begin{aligned}
  
  \bar\alpha_t &=& \prod_{s=1}^t ~(1 - \beta_s) \\
  \bar\alpha_t &=& \bar\alpha_{t-1} ~(1 - \beta_t) \\
  \beta_t &=& 1 - \frac{\bar\alpha_t}{\bar\alpha_{t-1}}
  .
\end{aligned}$$
```


The improved DDPM formulation of Nichol and Dhariwal [Nichol2021a] employs a cosine schedule for `$\bar\alpha_t$` (Eq. 17 in [Nichol2021a]), defined as 
```latex
$$\bar\alpha_t = \frac{f(t)}{f(0)}
  ,\hspace{4mm}\text{where}\hspace{4mm}
  f(t) = \cos^2 \bigg( \frac{t/T + s}{1 + s} \cdot \frac{\pi}{2} \bigg)
  ,$$
```
 where `$s = 0.008$`. In their implementation [^15], however, Nichol et al. leave out the division by `$f(0)$` and simply define [^16] 
```latex
$$
  \bar\alpha_t = \cos^2 \bigg( \frac{t/T + s}{1 + s} \cdot \frac{\pi}{2} \bigg)
  .$$
```


To prevent singularities near `$t = T$`, they also clamp `$\beta_t$` to `$0.999$`. We can express the clamping in terms of `$\bar\alpha_t$` by utilizing Eq. [eq:ddpmtrans\] and Eq. [eq:ddpmbeta\]: 
```latex
$$\begin{aligned}
  
  \bar\alpha'_t &=& \prod_{s=1}^t ~\big( 1 - [\beta'_s] \big) \\
  &=& \prod_{s=1}^t ~\Big( 1 - \min\big( [\beta_s], ~0.999) \Big) \\
  &=& \prod_{s=1}^t ~\bigg( 1 - \min\bigg( 1 - \frac{\bar\alpha_s}{\bar\alpha_{s-1}}, ~0.999 \bigg) \bigg) \\
  &=& \prod_{s=1}^t ~\max\bigg( \frac{\bar\alpha_s}{\bar\alpha_{s-1}}, ~0.001 \bigg)
  .
\end{aligned}$$
```


Let us now reinterpret the above formulas in our unified framework. Recall from Table [tab:specifics\] that we denote the original iDDPM sampling steps by `$\{u_j\}$` in the order of descending noise level `$\sigma(u_j)$`, where `$j \in \{0, \dots, M\}$`. To harmonize the notation of Eq. [eq:ddpmtrans\], Eq. [eq:ddpmalpha\], and Eq. [eq:ddpmclamp\], we thus have to replace `$T \longrightarrow M$` and `$t \longrightarrow M-j$`: 
```latex
$$\begin{aligned}
  q(\bar\boldsymbol{x}_j ~|~ \bar\boldsymbol{x}_M) &=& \mathcal{N}\big( \bar\boldsymbol{x}_j; ~\sqrt{\bar\alpha'_j} ~\bar\boldsymbol{x}_M, ~(1 - \bar\alpha'_j) ~\mathbf{I}\big)  , \\[2mm]
  \bar\alpha_j &=& \cos^2 \bigg( \frac{(M - j) / M + C_2}{1 + C_2} \cdot \frac{\pi}{2} \bigg)  ,\hspace{4mm}\text{and} \\
  \bar\alpha'_j &=& \prod_{s=M-1}^j ~\max\bigg( \frac{\bar\alpha_j}{\bar\alpha_{j+1}}, ~C_1 \bigg) ~=~ \bar\alpha'_{j+1} ~\max\bigg( \frac{\bar\alpha_j}{\bar\alpha_{j+1}}, ~C_1 \bigg) 
  ,
\end{aligned}$$
```
 where the constants are `$C_1 = 0.001$` and `$C_2 = 0.008$`.

We can further simplify Eq. [eq:ddpmalphanew\]: 
```latex
$$\begin{aligned}
  \bar\alpha_j &=& \cos^2 \bigg( \frac{(M - j) / M + C_2}{1 + C_2} \cdot \frac{\pi}{2} \bigg) \\
  &=& \cos^2 \bigg( \frac{\pi}{2} ~\frac{(1 + C_2) - j / M}{1 + C_2} \bigg)\\
  &=& \cos^2 \bigg( \frac{\pi}{2} - \frac{\pi}{2} ~\frac{j}{M (1 + C_2)} \bigg)\\
  &=& \sin^2 \bigg( \frac{\pi}{2} ~\frac{j}{M (1 + C_2)} \bigg)
  ,
\end{aligned}$$
```
 giving the formula shown in the "Parameters" section of Table [tab:specifics\].

To harmonize the definitions of `$\boldsymbol{x}$` and `$\bar\boldsymbol{x}$`, we must match the perturbation kernel of Eq. [eq:songperturbation\] with the transition probability of Eq. [eq:ddpmtransnew\] for each time step `$t = u_j$`: 
```latex
$$\begin{aligned}
  p_{0t}\big( \boldsymbol{x}(u_j) ~|~ \boldsymbol{x}(0) \big) &=& q(\bar\boldsymbol{x}_j ~|~ \bar\boldsymbol{x}_M) \\
  \mathcal{N} \big( \boldsymbol{x}(u_j); ~s(t) ~\boldsymbol{x}(0), ~s(u_j)^2 ~\sigma(u_j)^2 ~\mathbf{I}\big) &=& \mathcal{N}\left( \bar\boldsymbol{x}_j; ~\sqrt{\bar\alpha'_j} ~\bar\boldsymbol{x}_M, ~\big( 1 - \bar\alpha'_j \big) ~\mathbf{I}\right)
  .
\end{aligned}$$
```


Substituting `$s(t) = 1$` and `$\sigma(t) = t$` from Appendix [9.3.1](#app:ddim), as well as `$\bar\boldsymbol{x}_M = \boldsymbol{x}(0)$`: 
```latex
$$\mathcal{N} \big( \boldsymbol{x}(u_j); ~\boldsymbol{x}(0), ~u_j^2 ~\mathbf{I}\big) = \mathcal{N}\left( \bar\boldsymbol{x}_j; ~\sqrt{\bar\alpha'_j} ~\boldsymbol{x}(0), ~\big( 1 - \bar\alpha'_j \big) ~\mathbf{I}\right)
  .$$
```


We can match the means of these two distributions by defining `$\bar\boldsymbol{x}_j = \sqrt{\bar\alpha'_j} ~\boldsymbol{x}(u_j)$`: 
```latex
$$\begin{aligned}
  \mathcal{N} \big( \boldsymbol{x}(u_j); ~\boldsymbol{x}(0), ~u_j^2 ~\mathbf{I}\big) &=& \mathcal{N}\left( \sqrt{\bar\alpha'_j} ~\boldsymbol{x}(u_j); ~\sqrt{\bar\alpha'_j} ~\boldsymbol{x}(0), ~\big( 1 - \bar\alpha'_j \big) ~\mathbf{I}\right) \\
  &=& \mathcal{N}\bigg( \boldsymbol{x}(u_j); ~\boldsymbol{x}(0), ~\frac{1 - \bar\alpha'_j}{\bar\alpha'_j} ~\mathbf{I}\bigg)
  .
\end{aligned}$$
```


Matching the variances and solving for `$\bar\alpha'_j$` gives 
```latex
$$\begin{aligned}
  u_j^2 &=& (1 - \bar\alpha'_j) ~/~ \bar\alpha'_j \\
  u_j^2 ~\bar\alpha'_j &=& 1 - \bar\alpha'_j \\
  u_j^2 ~\bar\alpha'_j + \bar\alpha'_j &=& 1 \\
  (u_j^2 + 1) ~\bar\alpha'_j &=& 1 \\
  \bar\alpha'_j &=& 1 ~/~ (u_j^2 + 1)
  .
\end{aligned}$$
```


Finally, we can expand the left-hand side using Eq. [eq:ddpmclampnew\] and solve for `$u_{j-1}$`: 
```latex
$$\begin{aligned}
  \bar\alpha'_{j+1} ~\max(\bar\alpha_j / \bar\alpha_{j+1}, ~C_1) &=& 1 ~/~ (u_j^2 + 1) \\
  \bar\alpha'_j ~\max(\bar\alpha_{j-1} / \bar\alpha_j, ~C_1) &=& 1 ~/~ (u_{j-1}^2 + 1) \\
  \big[ 1 ~/~ (u_j^2 + 1) \big] ~\max(\bar\alpha_{j-1} / \bar\alpha_j, ~C_1) &=& 1 ~/~ (u_{j-1}^2 + 1) \\
  \max(\bar\alpha_{j-1} / \bar\alpha_j, ~C_1) ~(u_{j-1}^2 + 1) &=& u_j^2 + 1 \\
  u_{j-1}^2 + 1 &=& (u_j^2 + 1) ~/~ \max(\bar\alpha_{j-1} / \bar\alpha_j, ~C_1) \\
  u_{j-1} &=& \sqrt{\frac{u_j^2 + 1}{\max(\bar\alpha_{j-1} / \bar\alpha_j, ~C_1)} - 1}
  ,
\end{aligned}$$
```
 giving a recurrence formula for `$\{u_j\}$`, bootstrapped by `$u_M = 0$`, that matches the "Time steps" row of Table [tab:specifics\].

### iDDPM preconditioning and training

We can solve `$D_\theta(\cdot)$` from Eq. [eq:ddimdenoiser\] by substituting `$\sigma(t) = t$` from Appendix [9.3.1](#app:ddim): 
```latex
$$\begin{aligned}
  \epsilon_\theta^{(j)} \left( \boldsymbol{x}/ \sqrt{\sigma^2 + 1} \right) &=& \big( \boldsymbol{x}- D_\theta(\boldsymbol{x}; \sigma) \big) / \sigma \\
  D_\theta(\boldsymbol{x}; \sigma) &=& \boldsymbol{x}- \sigma ~\epsilon_\theta^{(j)} \left( \boldsymbol{x}/ \sqrt{\sigma^2 + 1} \right)
  .
\end{aligned}$$
```


We choose to define `$F_\theta(\cdot; j) = \epsilon_\theta^{(j)}(\cdot)$` and solve `$j$` from `$\sigma$` by finding the nearest `$u_j$`: 
```latex
$$D_\theta(\boldsymbol{x}; \sigma) = \underbrace{1~\cdot}_{c_\text{skip}} \boldsymbol{x}~\underbrace{-~\sigma}_{c_\text{out}} \,\cdot ~F_\theta\Big( \underbrace{\tfrac{1}{\sqrt{\sigma^2 + 1}}}_{c_\text{in}} \,\cdot~\boldsymbol{x}; ~\underbrace{\mathop{\mathrm{arg\,min}}_j |u_j - \sigma|}_{c_\text{noise}} \Big)
  
  ,$$
```
 where `$c_\text{skip}$`, `$c_\text{out}$`, `$c_\text{in}$`, and `$c_\text{noise}$` match the "Network and preconditioning" section of Table [tab:specifics\].

Note that Eq. [eq:iddpmprecond\] is identical to the VP preconditioning formula in Eq. [eq:vpprecond\]. Furthermore, Nichol and Dhariwal [Nichol2021a] define their main training loss `$L_\text{simple}$` (Eq. 14 in [Nichol2021a]) the same way as Song et al. [Song2021sde], with `$\sigma$` drawn uniformly from `$\{u_j\}$`. Thus, we can reuse Eq. [eq:vploss\] with `$\sigma = u_j$`, `$j \sim \mathcal{U}(0, M-1)$`, and `$\lambda(\sigma) = 1 / \sigma^2$`, matching the "Training" section of Table [tab:specifics\]. In addition to `$L_\text{simple}$`, Nichol and Dhariwal [Nichol2021a] also employ a secondary loss term `$L_\text{vlb}$`; we refer the reader to Section 3.1 in [Nichol2021a] for details.

### iDDPM practical considerations

The pre-trained iDDPM model that we use on ImageNet-64 corresponds to the "ADM (dropout)" checkpoint [^17] provided by Dhariwal and Nichol [Dhariwal2021]. It contains 296 million trainable parameters and supports a discrete set of `$M = 1000$` noise levels `$\sigma \in \{u_j\} \approx \{$`20291, 642, 321, 214, 160, 128, 106, 92, 80, 71, `$\dots$`, 0.0064`$\}$`. The fact that we can only evaluate `$F_\theta$` these specific choices of `$\sigma$` presents three practical challenges:

1.  In the context of DDIM, we must choose how to resample `$\{u_j\}$` to yield `$\{t_i\}$` for `$N \ne M$`. Song et al. [Song2020ddim] employ a simple resampling scheme where `$t_i = u_{k \cdot i}$` for resampling factor `$k \in \mathbb{Z}^+$`. This scheme, however, requires that `$1000 \equiv 0 \pmod{N}$`, which limits the possible choices for `$N$` considerably. Nichol and Dhariwal [Nichol2021a], on the other hand, employ a more flexible scheme where `$t_i = u_j$` with `$j = \lfloor (M - 1) / (N - 1) \cdot i \rfloor$`. We note, however, that in practice the values of `$u_{j<8}$` are considerably larger than our preferred `$\sigma_\text{max}= 80$`. We choose to skip these values by defining `$j = \lfloor j_0 + (M - 1 - j_0) / (N - 1) \cdot i \rfloor$` with `$j_0 = 8$`, matching the "Time steps" row in Table [tab:specifics\]. In Figure [2]c, the differences between the original sampler (blue) and our reimplementation (orange) are explained by this choice.

2.  In the context of our time step discretization (Eq. [eq:discretization\]), we must ensure that `$\sigma_i \in \{u_j\}$`. We accomplish this by rounding each `$\sigma_i$` to its nearest supported counterpart, i.e., `$\sigma_i \gets u_{\mathop{\mathrm{arg\,min}}_j |u_j - \sigma_i|}$`, and setting `$\sigma_\text{min}= 0.0064 ~\approx~ u_{N-1}$`. This is sufficient, because Algorithm [alg:heun\] only evaluates `$D_\theta(\cdot; \sigma)$` with `$\sigma \in \{\sigma_{i<N}\}$`.

3.  In the context of our stochastic sampler, we must ensure that `$\hat t_i \in \{u_j\}$`. We accomplish this by replacing line 5 of Algorithm [alg:stochastic\] with `$\hat t_i \gets u_{\mathop{\mathrm{arg\,min}}_j |u_j - (t_i + \gamma_i t_i)|}$`.

With these changes, we are able to import the pre-trained model directly as `$F_\theta(\cdot)$` and run Algorithms [alg:heun\] and [alg:stochastic\] using the definitions in Table [tab:specifics\]. Note that the model outputs both and `$\Sigma_\theta(\cdot)$`, as described in Section 3.1 of [Nichol2021a]; we use only the former and ignore the latter.

# Further analysis of deterministic sampling 

## Truncation error analysis and choice of discretization parameters 

As discussed in Section [sec:deterministic\], the fundamental reason why diffusion models tend to require a large number of sampling steps is that any numerical ODE solver is necessarily an approximation; the larger the steps, the farther away we drift from the true solution at each step. Specifically, given the value of `$\boldsymbol{x}_{i-1}$` at time step `$i-1$`, the solver approximates the true `$\boldsymbol{x}^*_i$` as `$\boldsymbol{x}_i$`, resulting in local truncation error `$\boldsymbol{\tau}_i = \boldsymbol{x}^*_i - \boldsymbol{x}_i$`. The local errors get accumulated over the `$N$` steps, ultimately leading to global truncation error `$\boldsymbol{e}_N$`.

Euler's method is a first order ODE solver, meaning that `$\boldsymbol{\tau}_i = \mathcal{O}\left(h_i^2\right)$` for any sufficiently smooth `$\boldsymbol{x}(t)$`, where `$h_i = |t_i - t_{i-1}|$` is the local step size [Suli2003]. In other words, there exist some `$C$` and `$H$` such that `$||\boldsymbol{\tau}_i|| < C h_i^2$` for every `$h_i < H$`, i.e., halving `$h_i$` reduces `$\boldsymbol{\tau}_i$` by 4`$\times$`. Furthermore, if we assume that `$D_\theta$` is Lipschitz continuous --- which is true for all network architectures considered in this paper --- the global truncation error is bounded by `$||\boldsymbol{e}_N|| \le E \max_i ||\boldsymbol{\tau}_i||$`, where the value of `$E$` depends on `$N$`, `$t_0$`, `$t_N$`, and the Lipschitz constant [Suli2003]. Thus, reducing the global error for given `$N$`, which in turn enables reducing `$N$` itself, boils down to choosing the solver and `$\{t_i\}$` so that `$\max_i ||\boldsymbol{\tau}_i||$` is minimized.


**Figure caption:** **(a)** Local truncation error (*y*-axis) at different noise levels (*x*-axis) using Euler’s method with the VE-based CIFAR-10 model. Each curve corresponds to a different time step discretization, defined for *N* = 64 and a specific choice for the polynomial exponent *ρ*. The values represent the root mean square error (RMSE) between one Euler iteration and a sequence of multiple smaller Euler iterations, representing the ground truth. The shaded regions, barely visible at low *σ*, represent standard deviation over different latents **x**<sub>0</sub>. **(b)** Corresponding error curves for Heun’s 2<sup>nd</sup> order method (Algorithm [alg:heun]). **(c)** FID (*y*-axis) as a function of the polynomial exponent (*x*-axis) for different models, measured using Heun’s 2<sup>nd</sup> order method. The shaded regions indicate the range of variation between the lowest and highest observed FID, and the dots indicate the value of *ρ* that we use in all other experiments.


To gain insight on how the local truncation error behaves in practice, we measure the values of `$\boldsymbol{\tau}_i$` over different noise levels using the VE-based CIFAR-10 model. For a given noise level, we set `$t_i = \sigma^{-1}(\sigma_i)$` and choose some `$t_{i-1} > t_i$` depending on the case. We then sample `$\boldsymbol{x}_{i-1}$` from `$p(\boldsymbol{x}; \sigma_{i-1})$` and estimate the true `$\boldsymbol{x}^*_i$` by performing 200 Euler steps over uniformly selected subintervals between `$t_{i-1}$` and `$t$`. Finally, we plot the mean and standard deviation of the root mean square error (RMSE), i.e., `$||\boldsymbol{\tau}_i|| / \scriptstyle\sqrt{\dim\boldsymbol{\tau}}$`, as a function of `$\sigma_i$`, averaged over 200 random samples of `$\boldsymbol{x}_{i-1}$`. Results for Euler's method are shown in Figure [13]a, where the blue curve corresponds to uniform step size `$h_\sigma = 1.25$` with respect to `$\sigma$`, i.e., `$\sigma_{i-1} = \sigma_i + h_\sigma$` and `$t_{i-1} = \sigma^{-1}(\sigma_{i-1})$`. We see that the error is very large (`$\text{RMSE} \approx 0.56$`) for low noise levels (`$\sigma_i \le 0.5$`) and considerably smaller for high noise levels. This is in line with the common intuition that, in order to reduce `$\boldsymbol{e}_N$`, the step size should be decreased monotonically with decreasing `$\sigma$`. Each curve is surrounded by a shaded region that indicates standard deviation, barely visible at low values of `$\sigma$`. This indicates that `$\boldsymbol{\tau}_i$` is nearly constant with respect to `$\boldsymbol{x}_{i-1}$`, and thus there would be no benefit in varying `$\{t_i\}$` schedule on a per-sample basis.

A convenient way to vary the local step size depending on the noise level is to define `$\{\sigma_i\}$` as a linear resampling of some monotonically increasing, unbounded warp function `$w(z)$`. In other words, `$\sigma_{i<N} = w(A i + B)$` and `$\sigma_N = 0$`, where constants `$A$` and `$B$` are selected so that `$\sigma_0 = \sigma_\text{max}$` and `$\sigma_{N-1} = \sigma_\text{min}$`. In practice, we set `$\sigma_\text{min}= \max(\sigma_\text{lo}, 0.002)$` and `$\sigma_\text{max}= \min(\sigma_\text{hi}, 80)$`, where `$\sigma_\text{lo}$` and `$\sigma_\text{hi}$` are the lowest and highest noise levels supported by a given model, respectively; we have found these choices to perform reasonably well in practice. Now, to balance `$\boldsymbol{\tau}_i$` between low and high noise levels, we can, for example, use a polynomial warp function `$w(z) = z^\rho$` parameterized by the exponent `$\rho$`. This choice leads to the following formula for `$\{\sigma_i\}$`: 
```latex
$$
\sigma_{i<N} = \left( {\sigma_\text{max}}^\frac{1}{\rho} + \frac{i}{N-1} \left( {\sigma_\text{min}}^\frac{1}{\rho} - {\sigma_\text{max}}^\frac{1}{\rho} \right) \right)^\rho, \sigma_N = 0,$$
```
 which reduces to uniform discretization when `$\rho=1$` and gives more and more emphasis to low noise levels as `$\rho$` increases.[^18]

Based on the value of `$\sigma_i$`, we can now compute `$\sigma_{i-1} = \big( \sigma_i^{1 / \rho} - A \big)^\rho$`, which enables us to visualize `$\boldsymbol{\tau}_i$` for different choices of `$\rho$` in Figure [13]a. We see that increasing `$\rho$` reduces the error for low noise levels (`$\sigma < 10$`) while increasing it for high noise levels (`$\sigma > 10$`). Approximate balance is achieved at `$\rho=2$`, but RMSE remains relatively high (`$\sim0.03$`), meaning that Euler's method drifts away from the correct result by several ULPs at each step. While the error could be reduced by increasing `$N$`, we would ideally like the RMSE to be well below 0.01 even with low step counts.

Heun's method introduces an additional correction step for `$\boldsymbol{x}_{i+1}$` to account for the fact that `$\mathrm{d}\boldsymbol{x}/ \mathrm{d}t$` may change between `$t_i$` and `$t_{i+1}$`; Euler's method assumes it to be constant. The correction leads to cubic convergence of the local truncation error, i.e., `$\boldsymbol{\tau}_i = \mathcal{O}\left(h_i^3\right)$`, at the cost of one additional evaluation of `$D_\theta$` per step. We discuss the general family of Heun-like schemes later in Appendix [10.2]. Figure [13]b shows local truncation error for Heun's method using the same setup as Figure [13]a. We see that the differences in `$||\boldsymbol{\tau}_i||$` are generally more pronounced, which is to be expected given the quadratic vs. cubic convergence of the two methods. Cases where Euler's method has low RMSE tend to have even lower RMSE with Heun's method, and vice versa for cases with high RMSE. Most remarkably, the red curve shows almost constant `$\text{RMSE} \in [0.0030, 0.0045]$`. This means that the combination of Eq. [eq:discretizationII\] and Heun's method is, in fact, very close to optimal with `$\rho=3$`.

Thus far, we have only considered the raw numerical error, i.e., component-wise deviation from the true result in RGB space. The raw numerical error is relevant for certain use cases, e.g., image manipulation where the ODE is first evaluated in the direction of increasing `$t$` and then back to `$t=0$` again --- in this case, `$||\boldsymbol{e}_N||$` directly tells us how much the original image degrades in the process and we can use `$\rho=3$` to minimize it. Considering the generation of novel images from scratch, however, it is reasonable to expect different noise levels to introduce different kinds of errors that may not necessarily be on equal footing considering their perceptual importance. We investigate this in Figure [13]c, where we plot FID as a function of `$\rho$` for different models and different choices of `$N$`. Note that the ImageNet-64 model was only trained for a discrete set of noise levels; in order to use it with Eq. [eq:discretizationII\], we round each `$t_i$` to its nearest supported counterpart, i.e., `$t'_i = u_{\mathop{\mathrm{arg\,min}}_j |u_j - t_i|}$`.

From the plot, we can see that even though `$\rho=3$` leads to relatively good FID, it can be reduced further by choosing `$\rho > 3$`. This corresponds to intentionally introducing error at high noise levels to reduce it at low noise levels, which makes intuitive sense because the value of `$\sigma_\text{max}$` is somewhat arbitrary to begin with --- increasing `$\sigma_\text{max}$` can have a large impact on `$||\boldsymbol{e}_N||$`, but it does not affect the resulting image distribution nearly as much. In general, we have found `$\rho=7$` to perform reasonably well in all cases, and use this value in all other experiments.

## General family of 2^nd^ order Runge--Kutta variants 

Heun's method illustrated in Algorithm [alg:heun\] belongs to a family of explicit two-stage 2^nd^ order Runge--Kutta methods, each having the same computational cost. A common parameterization [Suli2003] of this family is, 
```latex
$$\boldsymbol{d}_i = f(\boldsymbol{x}_i;t_i)\ \ \ \textrm{;} \ \ \ \boldsymbol{x}_{i+1} = \boldsymbol{x}_i + h\Big[\Big(1-{\tfrac{1}{2\alpha}}\Big)\boldsymbol{d}_i+{\tfrac{1}{2\alpha}}f(\boldsymbol{x}_i + \alpha h \boldsymbol{d}_i;t_i+\alpha h)\Big]\textrm{,}$$
```
 where `$h=t_{i+1}-t_i$` and `$\alpha$` is a parameter that controls where the additional gradient is evaluated and how much it influences the step taken. Setting `$\alpha=1$` corresponds to Heun's method, and `$\alpha=\tfrac{1}{2}$` and `$\alpha=\tfrac{2}{3}$` yield so-called midpoint and Ralston methods, respectively. All these variants differ in the kind of approximation error they incur due to the geometry of the underlying function `$f$`.

To establish the optimal `$\alpha$` in our use case, we ran a separate series of experiments. According to the results, it appears that `$\alpha=1$` is very close to being optimal. Nonetheless, the experimentally best choice was `$\alpha=1.1$` that performed slightly better, even though values greater than one are theoretically hard to justify as they overshoot the target `$t_{i+1}$`. As we have no good explanation for this observation and cannot tell if it holds in general, we chose not to make `$\alpha$` a new hyperparameter and instead fixed it to `$1$`, corresponding exactly to Heun's method. Further analysis is left as future work, including the possibility of having `$\alpha$` vary during sampling.

An additional benefit of setting `$\alpha=1$` is that it makes it possible to use pre-trained neural networks `$D_\theta(\boldsymbol{x};\sigma)$` that have been trained only for specific values of `$\sigma$`. This is because a Heun step evaluates the additional gradient at exactly `$t_{i+1}$` unlike the other 2^nd^ order variants. Hence it is sufficient to ensure that each `$t_i$` corresponds to a value of `$\sigma$` that the network was trained for.


\[alpha\]  Deterministic sampling using general 2^nd^ order Runge--Kutta, `$\sigma(t)=t$` and `$s(t)=1$`.


1.1

ic


Algorithm [alg:alpha\] shows the pseudocode for a general 2^nd^ order solver parameterized by `$\alpha$`. For clarity, the pseudocode assumes the specific choices of `$\sigma(t)=t$` and `$s(t)=1$` that we advocate in Section [sec:deterministic\]. Note that the fallback to Euler step (line 11) can occur only when `$\alpha \ge 1$`.

# Further results with stochastic sampling 

## Image degradation due to excessive stochastic iteration 


[IMAGE: degradation/cifar10u-ddpmpp-orig-seed08-nadj1.000.jpg]
[IMAGE: degradation/cifar10u-ddpmpp-orig-seed08-nadj1.007.jpg]


[IMAGE: degradation/img64c-dhariwal-orig-seed00-nadj1.000.jpg]
[IMAGE: degradation/img64c-dhariwal-orig-seed00-nadj1.003.jpg]


**Figure caption:** Gradual image degradation with repeated addition and removal of noise. We start with a random image drawn from *p*(**x**; *σ*) (first column) and run Algorithm [alg:stochastic] for a certain number of steps (remaining columns) with fixed . Each row corresponds to a specific choice of *σ* (indicated in the middle) that we keep fixed throughout the entire process. We visualize the results after running them through the denoiser, i.e., *D*<sub>*θ*</sub>(**x**<sub>*i*</sub>; *σ*).


Figure [14] illustrates the image degradation caused by excessive Langevin iteration (Section [4], "Practical considerations"). These images are generated by doing a specified number of iterations at a fixed noise level `$\sigma$` so that at each iteration an equal amount of noise is added and removed. In theory, Langevin dynamics should bring the distribution towards the ideal distribution `$p(\boldsymbol{x};\sigma)$` but as noted in Section [4], this holds only if the denoiser `$D_\theta(\boldsymbol{x};\sigma)$` induces a conservative vector field in Eq. [eq:scoredenoiser\].

As seen in the figure, it is clear that the image distribution suffers from repeated iteration in all cases, although the exact failure mode depends on dataset and noise level. For low noise levels (below `$0.2$` or so), the images tend to oversaturate starting at 2k iterations and become fully corrupted after that. Our heuristic of setting `$S_\text{tmin}> 0$` is designed to prevent stochastic sampling altogether at very low noise levels to avoid this effect.

For high noise levels, we can see that iterating without the standard deviation correction, i.e., when `$S_\text{noise}=1.000$`, the images tend to become more abstract and devoid of color at high iteration counts; this is especially visible in the 10k column of CIFAR-10 where the images become mostly black and white with no discernible backgrounds. Our heuristic inflation of standard deviation by setting `$S_\text{noise}> 1$` counteracts this tendency efficiently, as seen in the corresponding images on the right hand side of the figure. Notably, this still does not fix the oversaturation and corruption at low noise levels, suggesting multiple sources for the detrimental effects of excessive iteration. Further research will be required to better understand the root causes of these observed effects.


**Figure caption:** Ablations of our stochastic sampler (Algorithm [alg:stochastic]) parameters using pre-trained networks of Song et al.  and Dhariwal and Nichol . Each curve shows FID (*y*-axis) as a function of *S*<sub>churn</sub> (*x*-axis) for *N* = 256 steps (NFE = 511). The dashed red lines correspond to our deterministic sampler (Algorithm [alg:heun]), equivalent to setting *S*<sub>churn</sub> = 0. The purple curves correspond to optimal choices for {*S*<sub>tmin</sub>, *S*<sub>tmax</sub>, *S*<sub>noise</sub>}, found separately for each case using grid search. Orange, blue, and green correspond to disabling the effects of *S*<sub>tmin,tmax</sub> and/or *S*<sub>noise</sub>. The shaded regions indicate the range of variation between the lowest and highest observed FID.


Figure [15] presents the output quality of our stochastic sampler in terms of FID as a function of `$S_\text{churn}$` at fixed NFE, using pre-trained networks of Song et al. [Song2021sde] and Dhariwal and Nichol [Dhariwal2021]. Generally, for each case and combination of our heuristic corrections, there is an optimal amount of stochasticity after which the results start to degrade. It can also be seen that regardless of the value of `$S_\text{churn}$`, the best results are obtained by enabling all corrections, although whether `$S_\text{noise}$` or `$S_\text{tmin,tmax}$` is more important depends on the case.

## Stochastic sampling parameters 

N NC@find@z

::: tabu
\|c\|@x@x@\|@y@y@\|z\| & & &\
& VP & VE & Pre-trained & Our model &\
& 30 & 80 & 80 & 40 & 0,10,20,30,`$\dots$`,70,80,90,100\
& 0.01 & 0.05 & 0.05 & 0.05 & 0,0.005,0.01,0.02,`$\dots$`,1,2,5,10\
& 1 & 1 & 50 & 50 & 0.2,0.5,1,2,`$\dots$`,10,20,50,80\
& 1.007 & 1.007 & 1.003 & 1.003 & 1.000,1.001,`$\dots$`,1.009,1.010\

Table [tab:StochasticParams\] lists the values for `$S_\text{churn}$`, `$S_\text{tmin}$`, `$S_\text{tmax}$`, and `$S_\text{noise}$` that we used in our stochastic sampling experiments. These were determined with a grid search over the combinations listed in the rightmost column. It can be seen that the optimal parameters depend on the case; better understanding of the degradation phenomena will hopefully give rise to more direct ways of handling the problem in the future.

# Implementation details 

We implemented our techniques in a newly written codebase, based loosely on the original implementations by Song et al.[^19] [Song2021sde], Dhariwal and Nichol[^20] [Dhariwal2021], and Karras et al.[^21] [Karras2021alias]. We performed extensive testing to verify that our implementation produced exactly the same results as previous work, including samplers, pre-trained models, network architectures, training configurations, and evaluation. We ran all experiments using PyTorch 1.10.0, CUDA 11.4, and CuDNN 8.2.0 on NVIDIA DGX-1's with 8 Tesla V100 GPUs each.

Our implementation and pre-trained models are available at <https://github.com/NVlabs/edm>

## FID calculation 

We calculate FID [Heusel2017] between 50,000 generated images and all available real images, without any augmentation such as `$x$`-flips. We use the pre-trained Inception-v3 model provided with StyleGAN3 [^22] [Karras2021alias] that is, in turn, a direct PyTorch translation of the original TensorFlow-based model [^23]. We have verified that our FID implementation produces identical results compared to Dhariwal and Nichol [Dhariwal2021] and Karras et al. [Karras2021alias]. To reduce the impact of random variation, typically in the order of `$\pm$`2%, we compute FID three times in each experiment and report the minimum. We also highlight the difference between the highest and lowest achieved FID in Figures [4], [5]b, [13]c, and [15].

## Augmentation regularization 

In Section [5], we propose to combat overfitting of `$D_\theta$` using conditional augmentation. We build our augmentation pipeline around the same concepts that were originally proposed by Karras et al. [Karras2020ada] in the context of GANs. In practice, we employ a set of 6 geometric transformations; we have found other types of augmentations, such as color corruption and image-space filtering, to be consistently harmful for diffusion-based models.

The details of our augmentation pipeline are shown in Table [tab:AugmentPipe\]. We apply the augmentations independently to each training image `$\boldsymbol{y}\sim p_\text{data}$` prior to adding the noise `$\boldsymbol{n}\sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$`. First, we determine whether to enable or disable each augmentation based on a weighted coin toss. The probability of enabling a given augmentation ("Prob." column) is fixed to 12% for CIFAR-10 and 15% for FFHQ and AFHQv2, except for `$x$`-flips that are always enabled. We then draw 8 random parameters from their corresponding distributions ("Parameters" column); if a given augmentation is disabled, we override the associated parameters with zero. Based on these, we construct a homogeneous 2D transformation matrix based on the parameters ("Transformation" column). This transformation is applied to the image using the implementation of [Karras2020ada] that employs 2`$\times$` supersampled high-quality Wavelet filters. Finally, we construct a 9-dimensional conditioning input vector ("Conditioning" column) and feed it to the denoiser network, in addition to the image and noise level inputs.

::: tabu
\|l\|l\|l\|c\|l\|l\| Augmentation & Transformation & Parameters & Prob. & Conditioning & Constants\
`$x$`-flip & `$\raisebox{0.1mm}{\textsc{Scale2D}}\big(1 - 2 a_0, ~1\big)$` & `$a_0 \sim \mathcal{U}\{0, 1\}$` & 100% & `$a_0$` & `$= 12$`%\
`$y$`-flip & `$\raisebox{0.1mm}{\textsc{Scale2D}}\big(1, ~1 - 2 a_1\big)$` & `$a_1 \sim \mathcal{U}\{0, 1\}$` & `$A_\text{prob}$` & `$a_1$` & or `$15$`%\
Scaling & `$\raisebox{0.1mm}{\textsc{Scale2D}}\big(  (A_\text{scale})^{a_2},$` & `$a_2 \sim \mathcal{N}(0, 1)$` & `$A_\text{prob}$` & `$a_2$` & `$= 2^{0.2}$`\
& `$(A_\text{scale})^{a_2}\big)$` & & & &\
Rotation & `$\raisebox{0.1mm}{\textsc{Rotate2D}}\big({-}a_3\big)$` & `$a_3 \sim \mathcal{U}(-\pi, \pi)$` & `$A_\text{prob}$` & `$\cos a_3 - 1$` &\
& & & & `$\sin a_3$` &\
Anisotropy & `$\raisebox{0.1mm}{\textsc{Rotate2D}}\big(a_4\big)$` & `$a_4 \sim \mathcal{U}(-\pi, \pi)$` & `$A_\text{prob}$` & `$a_5 ~\cos a_4$` & `$= 2^{0.2}$`\
& `$\raisebox{0.1mm}{\textsc{Scale2D}}\big(  (A_\text{aniso})^{a_5},$` & `$a_5 \sim \mathcal{N}(0, 1)$` & & `$a_5 ~\sin a_4$` &\
& `$1/(A_\text{aniso})^{a_5}\big)$` & & & &\
& `$\raisebox{0.1mm}{\textsc{Rotate2D}}\big({-}a_4\big)$` & & & &\
Translation & `$\raisebox{0.1mm}{\textsc{Translate2D}}\big((A_\text{trans}) a_6,$` & `$a_6 \sim \mathcal{N}(0, 1)$` & `$A_\text{prob}$` & `$a_6$` & `$= 1/8$`\
& `$(A_\text{trans}) a_7\big)$` & `$a_7 \sim \mathcal{N}(0, 1)$` & & `$a_7$` &\

The role of the conditioning input is to present the network with a set of auxiliary tasks; in addition to the main task of modeling `$p(\boldsymbol{x}; \sigma)$`, we effectively ask the network to also model an infinite set of distributions `$p(\boldsymbol{x}; \sigma, \boldsymbol{a})$` for each possible choice of the augmentation parameters `$\boldsymbol{a}$`. These auxiliary tasks provide the network with a large variety of unique training samples, preventing it from overfitting to any individual sample. Still, the auxiliary tasks appear to be beneficial for the main task; we speculate that this is because the denoising operation itself is similar for every choice of `$\boldsymbol{a}$`.

We have designed the conditioning input so that zero corresponds to the case where no augmentations were applied. During sampling, we simply set `$\boldsymbol{a} = \mathbf{0}$` to obtain results consistent with the main task. We have not observed any leakage between the auxiliary tasks and the main task; the generated images exhibit no traces of out-of-domain geometric transformations even with `$A_\text{prob} = 100$`%. In practice, this means that we are free to choose the constants `$\{A_\text{prob}, A_\text{scale}, A_\text{aniso}, A_\text{trans}\}$` any way we like as long as the results improve. Horizontal flips serve as an interesting example. Most of the prior work augments the training set with random `$x$`-flips, which is beneficial for most datasets but has the downside that any text or logos may appear mirrored in the generated images. With our non-leaky augmentations, we get the same benefits without the downsides by executing the `$x$`-flip augmentation with 100% probability. Thus, we rely exclusively on our augmentation scheme and disable dataset `$x$`-flips to ensure that the generated images stay true to the original distribution.

## Training configurations 

NC@find@x

::: tabu
\|l\|@x@x@\|@x@x@\|@x@\| & & & ImagetNet\
& & & & &\
Number of GPUs & 4 & 8 & 4 & 8 & 32\
Duration & 200 & 200 & 200 & 200 & 2500\
Minibatch size & 128 & 512 & 128 & 256 & 4096\
Gradient clipping & & --& & --& --\
Mixed-precision & -- & -- & -- & -- &\
Learning rate & 2 & 10 & 2 & 2 & 1\
LR ramp-up & 0.64 & 10 & 0.64 & 10 & 10\
EMA half-life & 0.89 / 0.9 & 0.5 & 0.89 / 0.9 & 0.5 & 50\
& & & & &\
Dropout probability & 10% & 13% & 10% & 5% / 25% & 10%\
& & & & &\
Channel multiplier & 128 & 128 & 128 & 128 & 192\
Channels per resolution & 1-2-2-2 & 2-2-2 & 1-1-2-2-2 & 1-2-2-2 & 1-2-3-4\
Dataset `$x$`-flips & & --& & --& --\
Augment probability & --& 12% & --& 15% & --\

Table [tab:TrainingParams\] shows the exact set of hyperparameters that we used in our training experiments reported in Section [5]. We will first detail the configurations used with CIFAR-10, FFHQ, and AFHQv2, and then discuss the training of our improved ImageNet model.

Config \textsc{a} of Table [tab:TrainingTable\] ("Baseline") corresponds to the original setup of Song et al. [Song2021sde] for the two cases (VP and VE), and config \textsc{f} ("Ours") corresponds to our improved setup. We trained each model until a total of 200 million images had been drawn from the training set, abbreviated as "200 Mimg" in Table [tab:TrainingParams\]; this corresponds to a total of `$\sim$`400,000 training iterations using a batch size of 512. We saved a snapshot of the model every 2.5 million images and reported results for the snapshot that achieved the lowest FID according to our deterministic sampler with NFE `$=$` 35 or NFE `$=$` 79, depending on the resolution.

In config \textsc{b}, we re-adjust the basic hyperparameters to enable faster training and obtain a more meaningful point of comparison. Specifically, we increase the parallelism from 4 to 8 GPUs and batch size from 128 to 512 or 256, depending on the resolution. We also disable gradient clipping, i.e., forcing `$\lVert \mathrm{d}\mathcal{L}(D_\theta) / \mathrm{d}\theta \rVert_2 \le 1$`, that we found to provide no benefit in practice. Furthermore, we increase the learning rate from 0.0002 to 0.001 for CIFAR-10, ramping it up during the first 10 million images, and standardize the half-life of the exponential moving average of `$\theta$` to 0.5 million images. Finally, we adjust the dropout probability for each dataset as shown in Table [tab:TrainingParams\] via a full grid search at 1% increments. Our total training time is approximately 2 days for CIFAR-10 at 32`$\times$`32 resolution and 4 days for FFHQ and AFHQv2 at 64`$\times$`64 resolution.

In config \textsc{c}, we improve the expressive power of the model by removing the 4`$\times$`4 layers and doubling the capacity of the 16`$\times$`16 layers instead; we found the former to mainly contribute to overfitting, whereas the latter were critical for obtaining high-quality results. The original models of Song et al. [Song2021sde] employ 128 channels at 64`$\times$`64 (where applicable) and 32`$\times$`32, and 256 channels at 16`$\times$`16, 8`$\times$`8, and 4`$\times$`4. We change these numbers to 128 channels at 64`$\times$`64 (where applicable), and 256 channels at 32`$\times$`32, 16`$\times$`16, and 8`$\times$`8. We abbreviate these counts in Table [tab:TrainingParams\] as multiples of 128, listed from the highest resolution to the lowest. In practice, this rebalancing reduces the total number of trainable parameters slightly, resulting in `$\sim$`56 million parameters for each model at 32`$\times$`32 resolution and `$\sim$`62 million parameters at 64`$\times$`64 resolution.

In config \textsc{d}, we replace the original preconditioning with our improved formulas ("Network and preconditioning" section in Table [tab:specifics\]). In config \textsc{e}, we do the same for the noise distribution and loss weighting ("Training" section in Table [tab:specifics\]). Finally, in config \textsc{f}, we enable augmentation regularization as discussed in Appendix [12.2]. The other hyperparameters remain the same as in config \textsc{c}.

With ImageNet-64, it is necessary to train considerably longer compared to the other datasets in order to reach state-of-the-art results. To reduce the training time, we employed 32 NVIDIA Ampere GPUs (4 nodes) with a batch size of 4096 (128 per GPU) and utilized the high-performance Tensor Cores via mixed-precision FP16/FP32 training. In practice, we store the trainable parameters as FP32 but cast them to FP16 when evaluating `$F_\theta$`, except for the embedding and self-attention layers, where we found the limited exponent range of FP16 to occasionally lead to stability issues. We trained the model for two weeks, corresponding to `$\sim$`2500 million images drawn from the training set and `$\sim$`600,000 training iterations, using learning rate 0.0001, exponential moving average of 50 million images, and the same model architecture and dropout probability as Dhariwal and Nichol [Dhariwal2021]. We did not find overfitting to be a concern, and thus chose to not employ augmentation regularization.

## Network architectures 

NC@find@x

::: tabu
\|l\|@x@x@x@\| & DDPM & NCSN & ADM\
& & &\
Resampling filter & Box & Bilinear & Box\
Noise embedding & Positional & Fourier & Positional\
Skip connections in encoder & --& Residual & --\
Skip connections in decoder & --& --& --\
Residual blocks per resolution & 4 & 4 & 3\
Attention resolutions & {16} & {16} & {32, 16, 8}\
Attention heads & 1 & 1 & 6-9-12\
Attention blocks in encoder & 4 & 4 & 9\
Attention blocks in decoder & 2 & 2 & 13\

As a result of our training improvements, the VP and VE cases become otherwise identical in config \textsc{f} except for the network architecture; VP employs the DDPM architecture while VE employs NCSN, both of which were originally proposed by Song et al. [Song2021sde]. These architectures correspond to relatively straightforward variations of the same U-net backbone with three differences, as illustrated in Table [tab:NetworkDetails\]. First, DDPM employs box filter `$[1, 1]$` for the upsampling and downsampling layers whereas NCSN employs bilinear filter `$[1, 3, 3, 1]$`. Second, DDPM inherits its positional encoding scheme for the noise level directly from DDPM [Ho2020] whereas NCSN replaces it with random Fourier features [Tancik2020fourier]. Third, NCSN incorporates additional residual skip connections from the input image to each block in the encoder, as explained in Appendix H of [Song2021sde] ("progressive growing architectures").

For class conditioning and augmentation regularization, we extend the original DDPM and NCSN arhictectures by introducing two optional conditioning inputs alongside the noise level input. We represent class labels as one-hot encoded vectors that we first scale by , where `$C$` is the total number of classes, and then feed through a fully-connected layer. For the augmentation parameters, we feed the conditioning inputs of Appendix [12.2] through a fully-connected layer as-is. We then combine the resulting feature vectors with the original noise level conditioning vector through elementwise addition.

For class-conditional ImageNet-64, we use the ADM architecture of Dhariwal and Nichol [Dhariwal2021] with no changes. The model has a total of `$\sim$`296 million trainable parameters. As detailed in Tables [tab:TrainingParams\] and [tab:NetworkDetails\], the most notable differences to DDPM include the use of a slightly shallower model (3 residual blocks per resolution instead of 4) with considerably more channels (e.g., 768 in the lowest resolution instead of 256), more self-attention layers interspersed throughout the network (22 instead of 6), and the use of multi-head attention (e.g., 12 heads in the lowest resolution). We feel that the precise impact of architectural choices remains an interesting question for future work.

## Licenses 

Datasets:

- MIT license

- Creative Commons BY-NC-SA 4.0 license

- Creative Commons BY-NC 4.0 license

- The license status is unclear

Pre-trained models:

- Apache V2.0 license

- MIT license

- Apache V2.0 license