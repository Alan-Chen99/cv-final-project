# Abstract

Predictions of global climate models typically operate on coarse spatial scales due to the large computational costs of climate simulations. This has led to a considerable interest in methods for statistical downscaling, a similar process to super-resolution in the computer vision context, to provide more local and regional climate information. In this work, we apply conditional normalizing flows to the task of climate variable downscaling. We showcase its successful performance on an ERA5 water content dataset for different upsampling factors. Additionally, we show that the method allows us to assess the predictive uncertainty in terms of standard deviation from the fitted conditional distribution mean.

# Introduction

In climate modeling, simulations are typically run at coarse spatial resolution due to computational constraints. However, it is often of interest to obtain accurate predictions about the earth's climate not only on global but also on local scales, for example to guide local adaptation to precipitation or temperature trends. To fill this gap, statistical downscaling methods have increasingly been used to derive high-resolution information from low-resolution input. Early works have used Convolutional Neural Networks (CNNs) for climate variable downscaling [Chen_2021; mital2022; harilal2021; Sha2020; geiss2020; liu2020; cheng2020]. However, these purely deterministic methods fail to capture the ill-determined nature of the problem -- for the same low-resolution image, there exist many possible fine-scale realizations. Capturing such stochasticity is important in order to improve the accuracy of local scale predictions. Therefore, Generative Adversarial Networks (GANs) have become a widely used method in super-resolution and climate variable downscaling [Wang_2018_ECCV_Workshops; watson2020investigating; Climalign; Singh2019DownscalingNW; harder2022generating; watson2020investigating; Chaudhuri2020]. However, such methods lack latent-space encoders and are known to suffer from mode collapse. It is hard to assess whether they are overfitting or generalizing. In climate variable downscaling, we require estimating a density as close to the true high-resolution pixel distribution as possible, as high-frequency details are of main importance. In this field, recently exact likelihood methods such as diffusion models have been applied [wan2023debias] for climate variable downscaling by leveraging theory from optimal transport. First, the data is debiased and then a diffusion model is used for upsampling. In [Climalign], the authors use normalizing flows for aligning the latent variables to a reference representation after performing statistical downscaling using a GAN architecture. Current state-of-the-art work [yang2023fourier] uses Fourier neural operators to learn a mapping between high and low-resolution climate data for arbitrary resolution downscaling.

In this work, we introduce the use of Conditional Normalizing Flows [winkler2019learning] (CNFs) for stochastic climate variable downscaling. They are particularly desirable, since we can tractably compute likelihood values, their sampling procedure is efficient, and we are able to assess predictive uncertainty due to its probabilistic interpretation. Unlike other generative models where predictive uncertainty is often computed over an ensemble of different runs of weight initializations, or using techniques such as dropout, we are able to directly evaluate the predictive uncertainty of the CNF by computing the standard deviation from the fitted distribution mean.

#### Contributions

Our main contributions can be summarized as follows:

- We show for the first time how to apply conditional normalizing flows to the task of climate variable downscaling.

- We verify that CNF makes it possible to evaluate predictive uncertainty, by computing uncertainty maps from the standard deviation of sampled outputs.

# Background

**Normalizing flows** represent a function as the composition of simpler invertible functions ```latex $f(\mathbf{z}) = f^K \circ f^{K-1} \circ \cdots \circ f^1 (\mathbf{z})$ ``` which yield the transformed random variables ```latex $\mathbf{z}^K \leftarrow ... \leftarrow \mathbf{z}^1 \leftarrow \mathbf{z}^0$ ``` as intermediates after applying the transformations ```latex $f^1$ ``` through ```latex $f^k$ ```. The functions ```latex $f^k: \mathbb{R}^d \longmapsto \mathbb{R}^d$ ``` are defined such that ```latex $f(\mathbf{z}^0) =\mathbf{y}$ ``` with ```latex $\mathbf{y}\in \mathbb{R}^d$ ```. All transformations ```latex $f^k$ ``` are invertible and differentiable, making it possible to computing the Jacobian determinant. Then, by applying the *Change of Variables Formula* we can model the density:

```latex
$$p_y({\mathbf{y}}) = p_{\mathbf{z}}(f({\mathbf{y}})) \begin{vmatrix} \det \frac{\partial f({\mathbf{y}})}{\partial {\mathbf{y}}} \end{vmatrix}$$
```

where ```latex ${\mathbf{y}}$ ``` is our input data at training time mapping to latent variable ```latex ${\mathbf{z}}$ ```. This allows us to formulate a model for the marginal likelihood ```latex $p_{{\mathbf{y}}}({\mathbf{y}})$ ``` that can be computed tractably and optimized on the negative log-likelihood. We propose to learn conditional likelihoods using conditional normalizing flows [winkler2019learning] for the task of super-resolution on climate data. Take as input the low-resolution image ```latex ${\mathbf{x}}\in \mathcal{X}$ ``` and as target the high-resolution image ```latex ${\mathbf{y}}\in \mathcal{Y}$ ```. We learn a distribution ```latex $p_{Y|X}({\mathbf{y}}| {\mathbf{x}})$ ``` using a conditional prior ```latex $p_{Z|X}({\mathbf{z}}| {\mathbf{x}})$ ``` and a mapping ```latex $f_\phi: {\mathcal{Y}}\times {\mathcal{X}}\to {\mathcal{Z}}$ ```, which is bijective in ```latex ${\mathcal{Y}}$ ``` and ```latex ${\mathcal{Z}}$ ```. The likelihood of this model is then defined as:

```latex
$$p_{Y|X}({\mathbf{y}}| {\mathbf{x}}) = p_{Z|X}({\mathbf{z}}| {\mathbf{x}})  \left\lvert \frac{\partial {\mathbf{z}}}{\partial {\mathbf{y}}} \right\rvert = p_{Z|X}(f_{\phi}({\mathbf{y}}, {\mathbf{x}}) | {\mathbf{x}})  \left\lvert \frac{\partial f_{\phi}({\mathbf{y}}, {\mathbf{x}})}{\partial {\mathbf{y}}} \right\rvert.$$
```

Notice that the difference between Equations [eq:change-of-variables] and [eq:cnf] is that all distributions are conditional, and the flow has a conditioning argument of ```latex ${\mathbf{x}}$ ```. The generative process or in our case super-resolving an image from ```latex ${\mathbf{x}}$ ``` to ```latex ${\mathbf{y}}$ ``` is described by first sampling ```latex ${\mathbf{z}}\sim  p_{Z|X}({\mathbf{z}}| {\mathbf{x}})$ ``` from a simple base density with its parameters conditioned on ```latex $\mathbf{x}$ ``` (for us this is a diagonal Gaussian) and then passing it through a sequence of invertible mappings ```latex $f^{-1}_{\phi}({\mathbf{z}}; {\mathbf{x}})$ ``` to obtain a predicted super-resolved image ```latex $\hat{{\mathbf{y}}}$ ```.

# Experiments

#### ERA5 Hourly Water Content Dataset:

This reanalysis dataset measures Total Column Water (TWC) provided in ```latex $\frac{kg}{m^2}$ ```. It describes the vertical integral of the total amount of atmospheric water content, that is, cloud water, water vapor, and cloud ice, but not precipitation. We use the same water content dataset as described in [harder2022generating] who perform physically consistent downsampling to create the low-resolution image counterparts. The dataset includes 40,000 training samples, with 10,000 for validation and 10,000 for testing. Similar as before, for preprocessing, we transform the input data values Z by ```latex $X=\frac{Z-\min{Z}}{\max{Z}-\min{Z}}$ ``` such that they lie within range [0,1].

#### Experimental setup:

For all experiments, we train the conditioned spatio-temporal flow with a learning rate of 2e-4 using a step-wise learning rate scheduler with a decay rate of 0.5 after every 200,000th parameter update step. We used the Adam optimizer [kingma2014method] with exponential moving average and coefficients of running averages of gradients and its square are set to ```latex $\beta=(0.9,0.99)$ ```. We train the model with an architecture of 3 scales and 2 flow steps per scale for 35 epochs.

## Qualitative Evaluation

Figure 1 and 2 display super-resolution results predicted by the conditional normalizing flow on the hourly water content and daily temperature datasets for upsampling factors of 2 and 4 respectively. The method is able to generalize over images in the test set, where each test sample conveys very different water content distributions. However, in regions with high intensity values, there is greater absolute error in predicting the correct pixel values than for regions with low intensity values. This may arise simply because the same percentage error results in a larger absolute error in such regions.

[IMAGE: Super resolution results on the ERA5 water content TCW test data for 2x upsampling. Samples are taken from the CNF with tau = 0.8. Shows ground truth, super-resolved samples, and absolute error.]

[IMAGE: Super resolution results on the ERA5 TCW water content test data for 4x upsampling. Samples are taken from the CNF with tau = 0.8. Shows ground truth, super-resolved samples, and absolute error.]

## Quantitative Evaluation

Table 1 shows the quantitative results of our method compared to a GAN architecture and bicubic interpolation. We added a perceptual Mean Squared Error loss between the predictions and ground truth image to improve sample quality. It can be seen that the generative approach outperforms the bicubic baseline. For the two times upsampling task, the super-resolution GAN outperforms the CNF on all metrics except the Continuous Ranked Probability Score (CRPS).

## Sample Uncertainty

[IMAGE: The top row depicts the ground truth, conditional mean, different high-resolution realizations for one low-resolution image and computed standard deviation from the conditional mean for a 2x upsampling factor. The bottom row displays the same experiment for an upsampling factor of 4x.]

One of the main advantages of normalizing flows is the ability to generate multiple samples for one initial condition. In our case, this would mean generating multiple high-resolution realizations for the same low-resolution image. Figure 3 visualizes the standard deviation computed across twenty samples from the model for one low-resolution image. For convenience, we plotted only four predicted samples to compare with. It can be seen that in areas of high variance and finer texture regions, the standard deviation is generally higher. In applications such as flood risk estimation, this may be highly advantageous, since we deliberately want to have a model which is able to capture anomalies in the water content distribution.

# Conclusion

In this work, we have shown the successful application of conditional normalizing flows to climate variables providing physically consistent results. The proposed method provides the advantage of density estimation and efficient sampling, and is able to model the stochasticity inherent in the relationships among fine and coarse spatial scales of climate variables. Additionally, we have shown that the method allows us to compute uncertainty maps in terms of standard deviation computed from the distribution mean.
