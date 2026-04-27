# Abstract

Climate change is intensifying rainfall extremes, making high-resolution precipitation projections crucial for society to better prepare for impacts such as flooding. However, current Global Climate Models (GCMs) operate at spatial resolutions too coarse for localized analyses. To address this limitation, deep learning-based statistical downscaling methods offer promising solutions, providing high-resolution precipitation projections with a moderate computational cost. In this work, we introduce a bias-informed conditional diffusion model for statistical downscaling of precipitation. Specifically, our model leverages a conditional diffusion approach to learn distribution priors from large-scale, high-resolution precipitation datasets. The long-tail distribution of precipitation poses a unique challenge for training diffusion models; to address this, we apply gamma correction during preprocessing. Additionally, to correct biases in the downscaled results, we employ a guided-sampling strategy to enhance bias correction. Our experiments demonstrate that the proposed model achieves highly accurate results in an ```latex $8 \times$ ``` downscaling setting, outperforming previous deterministic methods. The code and dataset are available at [Github](https://github.com/RoseLV/research_super-resolution).

**Keywords:** Deep learning, Denoising Diffusion Probability Models, Statistical Downscaling, Climate Modeling

# Introduction

Due to climate change, there is a growing demand for reliable weather and climate simulation at local scales [kumar2023modern]. The current de facto technique Global Climate Models (GCMs) [heavens2013studying] can simulate the Earth's response to varying atmospheric greenhouse gas (GHG) emissions scenarios. However, the outputs from GCMs are often coarse and lack the granularity [gutowski2020ongoing] needed to understand local weather patterns and their impacts on specific sectors such as agriculture, water resources, and food security.

One of the foundational methods is statistical downscaling, which enhances coarse climate model outputs by predicting fine-resolution data based on statistical relationships between low-resolution and high-resolution observations. This approach shares similarities with image super-resolution techniques in computer vision. A historically significant image super-resolution method is the Super-Resolution Convolutional Neural Network (SRCNN). Although SRCNN has been widely adopted and dominated the field for nearly a decade, it suffers from limitations such as a restricted receptive field and insufficient network depth, which impede its ability to capture global context and effectively extract features [luo2016understanding]. Additionally, SRCNN struggles with poor adaptation to long-tail data distributions [feldman2020neural], such as precipitation, making it unsuitable for tasks like 8x downscaling.

Recently, denoised diffusion models [ho2020denoising; song2020score] have emerged as the dominant deep generative models for images due to their comprehensive coverage of data distributions and high-quality outputs. Improved Denoising Diffusion Probabilistic Models (DDPMs) [nichol2021improved] are generative models that produce high-quality samples, achieve competitive log-likelihoods with simple modifications, enable faster sampling through learned variances, and scale effectively with model capacity and compute. Furthermore, diffusion models outperform state-of-the-art generative models like GANs [goodfellow2020generative] in image synthesis tasks by leveraging improved architectures and classifier guidance for enhanced fidelity and diversity, achieving superior FID scores across multiple resolutions while maintaining efficient sampling and better distribution coverage [dhariwal2021diffusion].

In this work, we aim to adapt conditional diffusion models for the task of precipitation downscaling. However, precipitation downscaling has its unique challenges. **1)** The distribution of precipitation is inherently non-normal, posing difficulties for traditional deep learning models to capture its long-tail characteristics. **2)** Bias is a critical metric for precipitation downscaling, as small inaccuracies can significantly impact downstream applications. To tackle these issues, we introduce a Bias-aware Guided Sampling (BGS) approach to systematically reduce bias during the downscaling process, improving the overall accuracy and reliability of the generated high-resolution precipitation data.

# Method

In this section, we first introduce the definition and statistical downscaling. Then we explain how to train a conditioned denoised diffusion model for statistical downscaling. Finally, we introduce the motivation and formulation of our two core innovations: Gamma Correction and Bias-aware Guided Sampling.

[IMAGE: Overall pipeline of the proposed framework: In the training phase, a noise-corrupted high-resolution image, a low-resolution image, and a topography image are concatenated and input into a U-Net model. The U-Net is trained to predict the noise in the corrupted high-resolution precipitation image. In the prediction phase, the trained U-Net iteratively denoises precipitation images, transforming pure noise into high-resolution precipitation. Our bias-informed sampling strategy quantifies and reduces the bias between the corrupted high-resolution precipitation and the low-resolution input at each denoising step. (Fig9.png)]

## Problem Setting and Conditional Diffusion Models

Statistical downscaling involves generating high-resolution climate variables from low-resolution counterparts using statistical methods. This process closely resembles the image super-resolution task in computer vision [saharia2022image].

We are given a dataset of low-high resolution precipitation pairs, denoted ```latex $\mathcal{D} = \{\boldsymbol{x}_i, \boldsymbol{y}_i\}^N_{i=1}$ ```, which represent samples drawn from an unknown conditional distribution ```latex $p(\boldsymbol{y}|\boldsymbol{x})$ ```. Here ```latex $\boldsymbol{y}$ ``` represents high-resolution precipitation and ```latex $\boldsymbol{x}$ ``` represents low-resolution precipitation. Due to the high correlation between topography information and precipitation [daly1994statistical], we concatenate it with ```latex $\boldsymbol{x}$ ``` as the input for our model.

The conditional diffusion model generates a high resolution precipitation ```latex $\boldsymbol{y}_0$ ``` in ```latex $T$ ``` refinement steps. Starting with an initial noise precipitation ```latex $\boldsymbol{y}_T \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$ ```, the model progressively refines the precipitation through successive iterations ```latex $(\boldsymbol{y}_{T-1}, \boldsymbol{y}_{T-2},...,\boldsymbol{y}_0)$ ``` using learned conditional transition distributions ```latex $p_{\theta}(\boldsymbol{y}_{t-1}|\boldsymbol{y}_t, \boldsymbol{x})$ ``` such that ```latex $\boldsymbol{y}_0 \sim p(\boldsymbol{y}|\boldsymbol{x})$ ```.

## Gamma correction

Distribution of precipitation naturally deviates from Gaussian distribution, which is favored by deep learning models. Inspired by the similarity between distributions of precipitation and low-light images [li2023pixel], we apply Gamma Correction on both low and high resolution precipitation data in preprocessing stage. The equation of Gamma Correction is

```latex
$$\hat{a} = a^\gamma, \gamma = 0.15$$
```

This approach addresses the issue of large precipitation areas dominating the optimization process, thereby effectively enhancing the model's performance.

## Bias-aware Guided Sampling (BGS)

Classifier-guided sampling is a technique used in [dhariwal2021diffusion]. This approach can improve the performance of diffusion model without slowing down the training process. We propose to incorporate this technique in our sampling process.

While the diffusion process conditions on low-resolution image ```latex $\boldsymbol{x}$ ```, we assume that the information of ```latex $\boldsymbol{x}$ ``` is not fully leveraged due to indirect connection between ```latex $\boldsymbol{x}$ ``` and ```latex $\boldsymbol{y}$ ```. Moreover, the ```latex $\boldsymbol{x}$ ``` comes from GCMs, which are expert models with minimum bias from real climate variables. To fully leverage this prior knowledge, we propose a Bias-aware Guided Sampling approach [dhariwal2021diffusion]. Specifically, we use a L2 norm ```latex $f = \|\boldsymbol{y}_t - \boldsymbol{x}\|_2$ ``` to quantify the bias between ```latex $\boldsymbol{y}_t$ ``` and ```latex $\boldsymbol{x}$ ``` and use its gradients to steer the generation process at each diffusion step. Formally,

```latex
$$\boldsymbol{y}_{t-1} = \frac{1}{\sqrt{\alpha_t}}(\boldsymbol{y}_t - \frac{1-\alpha_t}{\sqrt{1 - \Bar{\alpha}_t}}\epsilon_{\theta}(\boldsymbol{y}_t, \boldsymbol{x}, t)) - w \nabla f$$
```

where ```latex $w = 100$ ``` is a hyperparameter we searched by experiments.

# Experiment and results

## Dataset

The high-resolution precipitation data (4km) is from PRISM Climate data [daly2008physiographically]. The low-resolution precipitation data (32km) is bilinearly downsampled 8x from the high-resolution data. We use the data between 2000-2018 for training and 2019-2022 for evaluation.

| Methods | RMSE (mm/day) | Corr | Bias (mm/day) |
|---|---|---|---|
| Interpolation | 3.134 | 0.939 | **-0.0053** |
| SRCNN | 5.921 | 0.828 | 0.0660 |
| Ours w/o BGS, Topo | 3.019 | 3.943 | 0.0544 |
| Ours w/o BGS | 2.991 | 0.944 | -0.0548 |
| Ours | **2.972** | **0.945** | -0.0389 |

Table: Evaluation of downscaling methods on PRISM dataset

[IMAGE: One example of 16x16 to 128x128 downscaling of precipitation. (results.png)]

## Evaluation metrics

Following DeepSD [vandal2017deepsd], we evaluate our method and baselines using three metrics: RMSE, correlation, and bias. RMSE measures the average error, correlation assesses the linear relationship and bias quantifies systematic errors. Both RMSE and bias are in mm/day. The interpolation method uses bilinear interpolation to upscale the original low-resolution image to high resolution, serving as our baseline. SRCNN performs poorly in the evaluation metrics, with a significantly increased RMSE and noticeably low correlation, highlighting its limitations in effectively downscaling precipitation data.

## Implementation details

Our code is based on improved diffusion [nichol2021improved]. We treat both the high and low resolution precipitations as ```latex $128 \times 128$ ``` grayscale images. All the precipitations are cropped from the same region for convenience. We use AdamW [loshchilov2017decoupled] and learning rate 3e-4. We train our model using one A100 for 1000 epochs, which takes around one day.

## Quantitative and qualitative results

The quantitative results are presented in Table 1. As shown, our method outperforms the deterministic baseline SRCNN [dong2014learning; vandal2017deepsd], and both the incorporation of topography information (Topo) and Bias-aware Guided Sampling (BGS) prove effective in this setting.

As shown in Figure 2, while SRCNN struggles to generate high-resolution precipitation under an ```latex $8 \times$ ``` super-resolution setting. The distribution and visual performance of the SRCNN image significantly deviate from the high-resolution ground truth, while our method generates physically and visually plausible results with fine details.

# Conclusion

In this work, we proposed a bias-informed conditional diffusion model for precipitation downscaling, addressing the challenges posed by the long-tail distribution of precipitation and biases in low-resolution input data. By incorporating gamma correction during preprocess and introducing Bias-aware Guided Sampling (BGS), our method achieved significant improvements in accuracy and bias correction compared to baseline methods. The experimental results demonstrated the effectiveness of our approach in producing high-resolution precipitation maps with fine details, making it a promising solution for localized climate impact studies. Future work will explore extending this framework to other climate variables and integrating physical constraints for enhanced generalizability.
