# Denoising Diffusion Models for High-Resolution Microscopy Image Restoration

## Abstract

Advances in microscopy imaging enable researchers to visualize structures at the nanoscale level thereby unraveling intricate details of biological organization. However, challenges such as image noise, photobleaching of fluorophores, and low tolerability of biological samples to high light doses remain, restricting temporal resolutions and experiment durations. Reduced laser doses enable longer measurements at the cost of lower resolution and increased noise, which hinders accurate downstream analyses. Here we train a denoising diffusion probabilistic model (DDPM) to predict high-resolution images by conditioning the model on low-resolution information. Additionally, the probabilistic aspect of the DDPM allows for repeated generation of images that tend to further increase the signal-to-noise ratio. We show that our model achieves a performance that is better or similar to the previously best-performing methods, across four highly diverse datasets. Importantly, while any of the previous methods show competitive performance for some, but not all datasets, our method consistently achieves high performance across all four data sets, suggesting high generalizability.

## 1 Introduction

High-resolution microscopy imaging impacts biology by revealing the detailed structure of living systems, providing a crucial basis for visualization, analysis, and interpretation. However, obtaining detailed microscopy images often requires immoderate imaging conditions, such as high light intensity. This can lead to photobleaching and phototoxicity, compromising the integrity of biological samples and limiting the duration over which observations can be made. Reducing the light intensity during imaging can mitigate these effects, but often at the cost of higher noise, obscuring important details in the imaging data.

Denoising techniques have thus become essential in biological and medical microscopy applications, where preserving sample integrity while obtaining high-resolution images is often paramount. Important examples include resolving the temporal evolution of sub-cellular organization such as organelles, revealing their morphology, or identifying condition-dependent changes of these structures (e.g., stress or drugs). Traditional denoising methods, such as Gaussian filtering or wavelet transforms, often fall short in preserving fine details while removing noise. Recently, deep learning-based approaches have shown significant improvements in image-denoising tasks, also in the field of microscopy.

Denoising Diffusion Probabilistic Models (DDPMs) [Ho et al. 2020] are powerful models that generate images from noisy inputs by iteratively removing noise in a diffusion process. Motivated by their ability to generate fine-scale images, these models may be well-suited for denoising in biology, where reconstruction of detailed structures is crucial. Indeed, DDPMs have been recently shown to effectively remove signal-dependent and -independent noise in medical and biological imaging data. However, it is currently unclear how well DDPMs perform in denoising fluorescence microscopy data, especially since the noise characteristics of such data can differ significantly from that of other data types.

Here we demonstrate that DDPMs can be highly effective in denoising a diverse range of fluorescence microscopy datasets, resolving fine structural details of the samples being studied. We use DDPMs to model the complex noise characteristics of different types of low-light microscopy images and use these models for high-quality image restoration, enabling longer imaging periods without sacrificing the sample quality. Additionally, we show that computing an average across several denoised reconstructions, exploiting the DDPM's stochasticity, can further enhance the performance significantly. We systematically test the denoising performance using four different datasets acquired through stimulated emission depletion (STED, datasets 1 and 2), confocal and Airyscan super-resolution (dataset 3), and single example and averaged confocal (dataset 4) fluorescence microscopy. These datasets vary considerably regarding the acquisition process, the samples being imaged (e.g., mitochondria or zebrafish), the sample condition (live vs fixed), the noise levels and structure, and the procedure for obtaining the low- and high-noise examples. This diversity of imaging conditions allows us to test the robustness of the proposed denoising method across different fluorescence microscopy applications. Importantly, our approach shows a performance that is higher than, or at least similar to current benchmark models across *all* tested datasets, demonstrating its broad applicability and robustness. In fact, no other benchmark model performs consistently as high across datasets as our proposed DDPM. Overall, we find that our DDPMs architecture provides a highly competitive method for denoising fluorescence microscopy data, and integrating such models into the microscopy imaging workflow could pave the way for more accurate and less invasive imaging practices in the future.

Our contributions are as follows:

- We introduce a DDPM architecture for fluorescence microscopy image denoising that achieves competitive performance across diverse datasets.
- We suggest a repeated sampling scheme that increases the signal-to-noise ratio, building on the stochastic denoising process of the DDPM.
- We publish two novel, challenging denoising datasets containing STED images of fixed-cell microtubules and live-cell mitochondria.

## 2 Related Work

Diffusion models have emerged as a powerful tool for several computer vision tasks, showing greater training stability and superior image quality compared to previous generative models. In the medical and biological domain DDPMs have been applied to segmentation, anomaly detection, image-to-image translation, molecule generation, or 2/3D generation. Several studies have proposed using DDPMs in microscopy, to e.g. predict 3D cellular structure from 2D images, reconstruct 3D biomolecule structure in Cryo-EM data, generate super-resolution images, or design drug molecules.

In recent years, deep learning methods have replaced classical denoising methods due to their better performance. One popular self-supervised denoising method is Noise2Void [Krull et al. 2019], where pixel-wise independent noise is assumed such that nearby pixels within a single example provide useful information for denoising. Pix2pix [Isola et al. 2018] was introduced as a general-purpose framework for image-to-image translation tasks, using conditional generative adversarial networks (cGANs). Several works extended pix2pix to image denoising tasks. Additionally, the widely-used content-aware image restoration (CARE) network [Weigert et al. 2018] incorporates a U-Net architecture to denoise low-resolution fluorescence data. More recently, the UNet-RCAN [Ebrahimi et al. 2023] first restores contextual features using a U-Net, and then leverages the ability of Residual Channel Attention Networks (RCAN) to reconstruct super-resolution images. Diffusion models have also been used for denoising ultrasounds, CT or PET images, MRI data, retinal images, or EM data. Chaudhary et al. recently proposed a DDPM for denoising fluorescence microscopy data using unpaired samples. However, while not relying on paired samples is advantageous, the performance of methods using paired samples is often higher. Whereas all of the above-mentioned methods significantly enhance image quality, they still show various limitations, such as blurriness, hallucinations, low signal-to-noise ratio, or excessive smoothing of the sample.

## 3 Methods

### 3.1 Data

We here use several microscopy datasets to test the performance of DDPMs for image denoising. Specifically, we use two novel datasets containing stimulated emission depletion (STED) microscopy images of microtubules (immunostained for alpha-tubulin) and of mitochondria that we publish alongside this paper. Further, we use two open-source datasets containing high- and low-resolution images of ex-vivo synapses [Xu et al. 2023], and confocal images of zebrafish embryos [Zhang et al. 2019]. These datasets differ across microscopy types, noise levels, sample types, and ground truth data generation, providing a challenging generalization task for denoising methods in fluorescence microscopy.

#### 3.1.1 Fixed-cell microtubules and live-cell mitochondria datasets

STED imaging of both fixed-cell microtubules (immunostained for alpha-Tubulin) and live-cell mitochondria (stained with transient HaloTag ligand Hy4-SiR for TOM20) were performed with an Abberior expert line microscope (Abberior Instruments, Germany). The setup uses an Olympus IX83 body where the imaging was done using a UPLXAPO 60x NA 1.42 oil immersion objective. A 640 nm excitation laser was used to acquire sample images (both confocal and STED imaging). In the case of STED images, depletion was performed with a 775 nm laser with a donut PSF (for planar and long-term imaging) with a delay of 750 ps and fluorescence photons between 750 ps and 8.75 ns were detected between every laser pulse. Both 640 nm and 775 nm lasers were pulsed at 40 MHz. Fluorescence was collected in the spectral range of 650 nm to 760 nm using an avalanche photodiode (APD). The pixel size used for the microtubule dataset was 25 nm and for the mitochondria dataset was 20 nm. The low-intensity images were measured by changing different parameters that influence the noise in a STED image, such as the excitation laser intensity, the number of lines integrated into the image, and the pixel dwell time. The depletion laser intensity was kept constant for the low- and high-intensity images to keep the resolution information intact.

One particularity of the microtubules and the mitochondria datasets is that, due to the difference in light dosage, the pixel distributions of low- and high-resolution images cover very distinct ranges. The low-resolution images contain only few pixel values, which poses an extra challenge for any denoising algorithm.

**Table 1: Measurement conditions of STED microtubules and mitochondria image dataset.** Note: Dwell times given are the total dwell time and take into account the number of line integrations.

| Intensity | Excitation (uW) | Depletion (mW) | Dwell time (us) | Light dosage to low-intensity |
|---|---|---|---|---|
| Low-tubulin | 1.5 | 60 | 1.5 | 1 |
| High-tubulin | 10.6 | 60 | 25 | 17 |
| Low-mitoc. | 1.5 | 174 | 2 | 1 |
| High-mitoc. | 8 | 174 | 20 | 10 |

#### 3.1.2 Synapse dataset

Additionally, we use the data published by Xu et al. [2023] containing low-resolution confocal images and high-resolution Airyscan imaging ground-truth (GT) from tissue slices of different cortical regions of transgenic mice. First, the high-resolution volumes were acquired immediately after the corresponding low-resolution images to reduce registration errors. The authors additionally curated the quality of the low-resolution images to replicate the image quality of *in vivo* two-photon data.

#### 3.1.3 Zebrafish dataset

Finally, we use one of the partitions from the open-source Fluorescence Microscopy Denoising dataset [Zhang et al. 2019]. The partition we employ consists of confocal images of fixed zebrafish embryos [EGFP labeled Tg(sox10:megfp) zebrafish at 2 days post fertilization]. Each of the fields of view (FOVs) was captured 50 times, each exhibiting a different noise realization. Authors provide images with different noise levels, generated by averaging $S$ noisy raw images. With an increasing number of images used for averaging, the peak signal-to-noise ratio (PSNR) of the averaged images increases, making the denoising task more simple. Here we employ the most difficult case, using $S=1$ as raw images and $S=50$ as ground-truth images.

### 3.2 Conditional denoising diffusion probabilistic models

We here follow the work proposed by Saharia et al. [Palette, 2022] to adapt denoising diffusion probabilistic models (DDPMs) [Ho et al. 2020] to a conditional image generation model.

Consider a data set of input-output (i.e. high-low noise) image pairs $(\mathbf{x}_i, \mathbf{y}_i)_{i=1}^N$ drawn from an unknown conditional distribution $p(\mathbf{y}|\mathbf{x})$. We aim to approximate $p(\mathbf{y}|\mathbf{x})$ using a stochastic iterative refinement process conditioned on a source image $\mathbf{x}$ to generate a target image $\mathbf{y}$. Specifically, the conditioned DDPM is trained to generate a target image $\mathbf{y}_0$ in $T$ steps, starting from an image of isotropic Gaussian noise $\mathbf{y}_T \sim \mathcal{N}(0, \mathbf{I})$. Via $T$ successive iterations $t$, the model computes $\mathbf{y}_0 \sim p(\mathbf{y} | \mathbf{x})$ using learned conditional transition distributions $p_\theta (\mathbf{y}_{t-1} | \mathbf{y}_t, \mathbf{x})$, where $\theta$ are the model parameters.

In the *forward diffusion process* Gaussian noise is gradually added to the signal via a fixed Markov chain $q(\mathbf{y}_t | \mathbf{y}_{t-1})$. Specifically, by reparameterizing the variance and merging the Gaussian noise, we can sample $\mathbf{y}_t$ at any step $t$ as:

$$\mathbf{y}_t = \sqrt{\bar{\alpha}_t} \mathbf{y}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$$

where $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$, and $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i \in (0,1)$ determines the variance of the noise added in each iteration, according to the variance schedule $\{\alpha_i\}^T_{i=1}$.

Then, the above process is reversed by the *reverse diffusion process*. The conditional DDPM employed here uses a reverse Markov chain conditioned on $\mathbf{x}$ to iteratively recover the signal $\mathbf{y}_0$ from noise $\mathbf{y}_T$. A denoising model $\boldsymbol{\epsilon}_\theta$ that follows from $p_\theta$ is trained to predict the noise $\boldsymbol{\epsilon}$ using the conditioning source image $\mathbf{x}$, a noisy target image $\mathbf{y}_t$, and additional conditioning on the statistics for the noise variance $\bar{\alpha}_t$.

The learned inference process is defined as a conditional transition distribution $p_\theta(\mathbf{y}_{t-1}|\mathbf{y}_t, \mathbf{x})$. The target image $\mathbf{y}_0$ is then approximated by $\hat{\mathbf{y}}_0$:

$$\hat{\mathbf{y}}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} \left(\mathbf{y}_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}_\theta(\mathbf{x}, \mathbf{y}_t, \bar{\alpha}_t)\right)$$

We optimize the parameters $\theta$ of the noise predictor model $\boldsymbol{\epsilon}_\theta$ by defining the learning objective $L$ as:

$$\mathbb{E}_{(\mathbf{x},\mathbf{y}_0)} \mathbb{E}_{\boldsymbol{\epsilon}, \bar{\alpha}_t}\left[\left\|\boldsymbol{\epsilon}_\theta (\mathbf{x}, \sqrt{\bar{\alpha}_t} \mathbf{y}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}, \bar{\alpha}_t) - \boldsymbol{\epsilon} \right\|_2^2\right]$$

where $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$, $(\mathbf{x},\mathbf{y}_0)$ is sampled from the training set, $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$ given $\alpha_i$ defined by the variance schedule $\{\alpha_i\}^T_{i=1}$ and $t$ uniformly sampled from $[1,T]$.

Recall that our goal is to learn the conditioned transition distributions $p_\theta (\mathbf{y}_{t-1}|\mathbf{y}_t)$ in the reverse diffusion process:

$$p_\theta (\mathbf{y}_{t-1}|\mathbf{y}_t)= \mathcal{N} \left(\mathbf{y}_{t-1}; \mu_\theta (\mathbf{x}, \mathbf{y}_t, \bar{\alpha}_t), \Sigma_\theta(\mathbf{x}, \mathbf{y}_{t}, \bar{\alpha}_t)\right)$$

Instead of learning the diagonal variance $\Sigma_\theta$, we fix it as proposed by Ho et al. [2020]:

$$\Sigma_\theta (\mathbf{x}, \mathbf{y}_{t}, \bar{\alpha}_t) = (1-\bar{\alpha}_t) \mathbf{I}$$

and parametrize the mean $\mu_\theta$ as:

$$\mu_\theta (\mathbf{x}, \mathbf{y}_t, \bar{\alpha}_t) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{y}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}, \mathbf{y}_t, \bar{\alpha}_t)\right)$$

which together allows for an iterative refinement in the following form:

$$\mathbf{y}_{t-1} \leftarrow \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{y}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}, \mathbf{y}_t, \bar{\alpha}_t)\right) + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}$$

### 3.3 Model architecture

The underlying structure of the DDPM is a U-Net [Ronneberger et al. 2015]. We adjust the conditional DDPM architecture of Saharia et al. [Palette, 2022] to make the model more robust to different types of fluorescence microscopy data. Our changes are inspired by recently proposed improvements [Karras et al. 2024], which identified weaknesses in the training dynamics of the traditional ADM architecture [Dhariwal & Nichol 2021]. Every resolution level of the U-Net consists of two blocks with convolutional layers for downsampling and transposed convolutional layers for upsampling, followed by self-attention at resolution 32 with 32 heads. The residual branch uses two convolutional layers, each preceded by a SiLU nonlinearity. Foremost, all operations, such as convolutions, activations, concatenation, and summation, are modified such that the expectation value of their magnitudes is preserved.

Each denoising step is conditioned by the noise level information, which is encoded by an auxiliary embedding network into Fourier features by applying random frequencies and phases to the noise level information, as opposed to the ADM's positional embedding scheme that employs a sinusoidal encoding. To condition on low-resolution images, each denoising step receives as input a concatenation of the conditioning low-resolution image $\mathbf{x}$ and the prediction $\mathbf{y}_t$ of the current time step $t$, sampled from a zero-mean isotropic Gaussian distribution at the first time step.

### 3.4 Model training and evaluation

We used image flipping, rotation, and Gaussian filtering as data augmentation techniques for training the DDPM. We trained the DDPM with AdamW optimizer, a batch size of 8, an initial learning rate of $2\text{e-}4$, and use a cosine annealing schedule to adjust the learning rate during training. The epoch with the lowest mean absolute error (MAE) in the validation set is used to select the best epoch for testing. We used a cosine-based variance schedule with an offset $s = 8\text{e-}3$, and set $T = 200$. All DDPMs were implemented in PyTorch and trained with one GPU NVIDIA A100.

During inference, we utilized the stochasticity of the DDPMs and generated several predictions using the same condition input but different initial noise inputs. We then averaged across the predictions to remove any random noise not removed by the DDPM. We henceforth refer to the averaging of DDPM predictions as DDPM-avg.

### 3.5 Quality control metrics

We compute the mean absolute error (MAE), the peak signal-to-noise ratio (PSNR), the multiscale structural similarity index measure (MS-SSIM), and the learned perceptual image patch similarity (LPIPS) between each ground truth image and reconstruction. Whereas the MAE measures the difference in pixel intensities, the PSNR quantifies the logarithmic peak error. The MS-SSIM additionally assesses the luminance, contrast, and structural information. LPIPS approximates the perceptual similarity between two images as would be indicated by humans. Additionally, we report the Pearson correlation coefficient, the resolution, and the normalized root mean square error (NRMSE) in the Supplement.

### 3.6 Benchmarks

We benchmark the performance of the conditioned DDPM against commonly used methods: Noise2Void (N2V), pix2pix, UNet-RCAN, and CARE. For CARE and N2V probabilistic versions exist. However, we did not find a performance increase, even when averaging multiple output instances and thus report only results using the non-probabilistic versions.

The microtubules dataset contains very different pixel distributions between the low- and high-intensity images. To adjust for this effect it was sufficient for pix2pix and the DDPMs to clip the pixel values of the reconstruction to the range [0, 255], and cast the result to 8-bit format. However, we observed a significant difference between the predicted and ground truth pixel distribution of high-resolution images for N2V, CARE, and UNet-RCAN, which strongly affected performance. Therefore, for these models, we first clipped the pixel values to the range [0, 255], and then rescaled them to the ground truth pixel distribution of the training set using a linear transformation.

## 4 Results and Experiments

We use conditioned denoising diffusion probabilistic models (DDPM) to denoise different types of microscopy datasets. Specifically, we test our method using i) a novel dataset of low- and high-intensity STED images of fixed microtubules, ii) a second novel dataset of STED images of living mitochondria, iii) a publicly available dataset of synapses in mouse brain acquired with low- and high-resolution microscopes, and iv) another publicly available dataset of zebrafish imaged with confocal microscopy and different noise levels. We report results for a single generated prediction using the DDPM as well as an average across 15 such generated predictions (DDPM-avg). We benchmark the performance of our model to several previous methods (Noise2Void, pix2pix, UNet-RCAN, and CARE) and compare the performance between methods using the MAE, PSNR, MS-SSIM, and LPIPS.

### 4.1 Denoising STED images of microtubules

First, we train the conditioned DDPM to denoise low-intensity STED images of fixed microtubules. Despite aligning the low- and high-resolution image pairs, all models predict a small shift in the reconstruction indicated by an offset in the peaks of the signal; thus we re-aligned the predictions to the GT for all models. Note that a highly accurate alignment is needed to compute a valid pixel-wise loss but can be omitted during practice. We observe that the DDPM accurately learns the pixel distribution of the target images and reconstructs the microtubule structures. The DDPM and DDPM-avg outperform several previous methods (Noise2Void, pix2pix, and UNet-RCAN) in all evaluation metrics ($p < .001$ using Mood's median test). In addition, DDPM-avg achieves a similar performance to CARE for all evaluation metrics ($p > .43$). The signal profiles of the prediction align well with the ground truth signal profile across all models. In particular, the DDPM and pix2pix most closely preserve the peaks and troughs, whereas Noise2Void underpredicts the pixel intensities, and CARE overpredicts the peaks of the data that correspond to microtubule structures. The DDPM preserves the fine structures between the long microtubule structures that are removed by pix2pix and Noise2Void, which can be problematic when the imaged structures are small or lie orthogonal to the imaging plane. Further, the structures denoised with the DDPM are more sharp, when averaged across several examples compared to the single example, which is also reflected by the significant increase in performance when averaging across an increasing number of denoised examples. As illustrated by the difference maps, the errors in the predicted pixel intensities of the DDPM-avg are small and not systematic along the microtubule structures. In contrast, all benchmark models show correlated errors along the microtubules, indicated by pronounced red (overprediction) and blue (underprediction) lines.

[IMAGE: Figure 1 - Conditioned DDPMs outperform several previous methods in denoising STED images. A) Performance comparison based on several evaluation metrics. B) Pixel intensity profiles along a dashed yellow line. C) Representative low-intensity image (Raw) from the test dataset, the corresponding high-resolution version (GT), and the results of the proposed method. D) Pixel-wise difference between ground truth and reconstruction for each model.]

**Table 2: Benchmarking the conditioned DDPM with the STED fixed-cell microtubule data set.** We report the median value of several performance metrics (MAE, PSNR, MS-SSIM, LPIPS) across models.

| Model | MAE ↓ | PSNR (dB) ↑ | MS-SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|
| Raw | 17.6 | 20.88 | 0.10 | 0.71 |
| Noise2Void | 13.26 | 24.32 | 0.72 | 0.22 |
| Pix2pix | 5.38 | 30.48 | 0.84 | 0.14 |
| UNet-RCAN | 12.73 | 23.40 | 0.80 | 0.24 |
| CARE | 4.11 | 32.58 | **0.89** | 0.20 |
| **DDPM** | 4.71 | 31.61 | 0.84 | **0.11** |
| **DDPM-avg** | **3.99** | **32.81** | **0.89** | 0.19 |

### 4.2 Denoising STED images of live-cell mitochondria

An important application of denoising in microscopy is live-cell imaging, as strong light exposure and photobleaching can strongly impact molecular biological processes. As one application, we therefore generate a dataset containing low- and high-resolution image pairs of living mitochondria and train the DDPM to predict the high-resolution data. Our proposed DDPM-avg achieves the highest performance across all metrics and only weakly underpredicts the true pixel distribution. Note that only DDPM, CARE, and UNet-RCAN are able to reconstruct the mitochondria's outer membrane, but also predict structures to be smoother than in the GT. The predictions of Noise2Void and pix2pix are pixelated and often fail to enhance the mitochondria structures compared to the background. Note that all models fail to make predictions that are biologically fully plausible, as evident, for instance, by the 'open' membranes of some mitochondria. Again, CARE achieves the most similar results to the DDPM, both visually and based on the metrics, and the UNet-RCAN overpredicts the pixel intensities. Interestingly, Noise2Void achieves very high performance across several evaluation metrics, but visually shows a poor performance. Averaging across several reconstructions again increases the performance in most metrics, except for LPIPS, where a single prediction of the DDPM achieves the highest performance.

[IMAGE: Figure 2 - Conditioned DDPMs outperform all benchmark models in denoising STED images of live-cell mitochondria.]

**Table 3: Benchmarking the conditioned DDPM on the live-cell mitochondria dataset.**

| Model | MAE ↓ | PSNR (dB) ↑ | MS-SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|
| Raw | 10.12 | 25.29 | 0.25 | 0.42 |
| Noise2Void | 3.06 | 34.78 | 0.77 | 0.32 |
| Pix2pix | 3.76 | 33.2 | 0.73 | **0.08** |
| UNet-RCAN | 39.25 | 13.81 | 0.62 | 0.61 |
| CARE | 3.79 | 32.79 | 0.78 | 0.12 |
| **DDPM** | 3.33 | 34.1 | 0.77 | **0.08** |
| **DDPM-avg** | **2.72** | **35.88** | **0.81** | 0.28 |

### 4.3 Denoising microscopy images of synapses in the mouse brain

Next, we tested our model on confocal and super-resolution images of synapses in the mouse brain. Here, the model doesn't only have to denoise the image but also has to predict a signal across microscopy types, i.e., from confocal to super-resolution Airyscan quality. Our proposed conditioned DDPM successfully reconstructs the synaptic structures similarly to or better than previous methods. Whereas for the tubulin dataset CARE shows the most similar results to the DDPM, here only UNet-RCAN achieves similar performance. Most methods predict blurry synapses with weak signal strengths for the bright synapses, indicating a high level of prediction uncertainty. From the predictions and error maps we observe a clear improvement of the DDPM in the background prediction with respect to CARE and Noise2Void; plus lower pixel error in contrast to pix2pix, and similar errors to UNet-RCAN in the restored intensities of the synapses. Interestingly, averaging across several reconstructions does not improve the performance significantly in this data set. Note that, in contrast to the error maps from the other datasets, all models struggle to accurately predict the synapse and/or background pixel values. This suggests further refinement is required in applications studying the changes in synapse strength and size, which is a task of central interest in neuroscience.

[IMAGE: Figure 3 - Conditioned DDPMs outperform several previous methods in denoising confocal images of synapses in the mouse brain.]

**Table 4: Benchmarking the conditioned DDPM on the synapse dataset.**

| Model | MAE ↓ | PSNR (dB) ↑ | MS-SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|
| Raw | 27.35 | 18.61 | 0.60 | 0.62 |
| Noise2Void | 26.87 | 18.74 | 0.60 | 0.61 |
| Pix2pix | 6.16 | 24.43 | 0.77 | 0.19 |
| UNet-RCAN | 5.66 | **26.11** | **0.81** | 0.19 |
| CARE | 12.67 | 23.35 | 0.69 | 0.24 |
| **DDPM** | 5.45 | 25.61 | 0.80 | **0.17** |
| **DDPM-avg** | **5.09** | 25.96 | **0.81** | 0.18 |

### 4.4 Denoising confocal images of zebrafish embryos

Additionally, we trained our model to denoise confocal images of zebrafish embryos. The DDPM-avg outperforms all benchmark models in MAE, PSNR, and MS-SSIM, whereas the DDPM is best for LPIPS. In this dataset the demarcation of sample structure and background is good for all methods, except the UNet-RCAN. The averaging across several reconstructions eliminates noisy elements but also fine-grained structures in the data which are challenging for all models. Nevertheless, for this particular dataset, averaging implied a major improvement in the performance, except for the LPIPS, which might be a reflection of the higher smoothing. The state-of-the-art denoising performance indicates that DDPMs effectively generalize to Poisson noise in microscopy data.

[IMAGE: Figure 4 - Conditioned DDPMs outperform previous methods in denoising confocal images of zebrafish embryos.]

**Table 5: Benchmarking the conditioned DDPM with the confocal zebrafish dataset.**

| Model | MAE ↓ | PSNR (dB) ↑ | MS-SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|
| Raw | 8.43 | 25.28 | 0.86 | 0.52 |
| Noise2Void | 2.76 | 33.72 | 0.96 | 0.13 |
| Pix2pix | 3.27 | 32.01 | 0.95 | 0.10 |
| UNet-RCAN | 7.44 | 27.45 | 0.90 | 0.23 |
| CARE | 3.10 | 32.74 | 0.96 | 0.14 |
| **DDPM** | 2.98 | 32.62 | 0.95 | **0.09** |
| **DDPM-avg** | **2.43** | **34.62** | **0.97** | 0.15 |

## 5 Discussion

In order to reduce phototoxicity while maintaining image quality in fluorescence microscopy, we explore the potential of image-conditioned diffusion models to denoise microscopy data in different datasets. We show the effectiveness of DDPMs in restoring (i) microtubule structures, (ii) living mitochondria (both acquired with STED microscopy), (iii) synaptic structures (confocal and high-resolution Airyscan imaging), and (iv) zebrafish embryos (confocal imaging).

The stochastic property of DDPMs allows to repeatedly generate slightly different predictions from a single low-resolution input. We leverage this property by estimating averaged reconstructions from 15 individual predictions which increases the performance for three datasets significantly (microtubules, mitochondria, and zebrafish). Further, computing uncertainty maps from several predictions highlights the pixel-wise prediction uncertainty of the DDPM.

We compare the performance of the DDPM with several benchmark models: Noise2Void, pix2pix, UNet-RCAN, and CARE; all of which are based on supervised learning, except for Noise2Void. We find that for all four datasets, our proposed DDPMs achieve similarly high and sometimes higher performance than the best-performing benchmark model. CARE, UNet-RCAN, and the DDPM denoise the structural features well enough for further biological inspection. The DDPM-avg best predicts the pixel intensities across datasets, followed by CARE or UNet-RCAN. However, compared to the DDPM the predictions of CARE are often more blurry and the UNet-RCAN tends to overestimate the pixel intensities. In general, the DDPM not only denoises the sample structures more precisely but also better demarcates the background from the sample. Importantly, the DDPM is consistently among the best (top two) performing models across all tested datasets, whereas the performance of the benchmark models varies considerably across datasets. These results suggest a high degree of generalizability of DDPMs for denoising fluorescence microscopy data across different biological systems and microscopy conditions.

We compare the performance of all models using several different performance metrics. The DDPM-avg is always the best-performing model for the MAE and MS-SSIM and at least second best for PSNR. LPIPS is always higher for single than for averaged reconstructions with the DDPM, suggesting that averaging reduces the perceptual similarity.

We here test several microscopy techniques and four widely used denoising methods as benchmarks. We acknowledge that other important microscopy techniques (e.g. widefield imaging, structured illumination microscopy) and denoising methods (e.g., ZS-DeconvNet, Noise2Fast) exist.

While DDPMs achieve the highest performance across several datasets, the costs of training and application are higher than for the other benchmark models. Future work could test the performance of the different denoising methods on biologically relevant downstream tasks, such as synapse detection or tracking and quantification of changes in fluorescence intensity, morphology, or motility of cell organelles across time.

## 6 Conclusion

In this work, we explore the applicability of DDPMs to denoise fluorescence microscopy images across different imaging modalities, ranging from STED and Airyscan to confocal microscopy, and a diversity of conditions, such as light-dosage, and cross-modality acquisition. We leverage recent improvements in the architecture of DDPMs to obtain a highly reliable training process. Also, we further increase the performance of the DDPM by averaging multiple high-resolution predictions conditioned on the same low-resolution image, thereby leveraging the stochastic nature of DDPMs. Overall, DDPMs perform better or similarly to current state-of-the-art denoising approaches. Foremost, their effectiveness is consistently high across the four datasets we test, suggesting this method is broadly applicable to a wide range of microscopy imaging data.

## Supplementary Materials

### S1 Datasets

#### S1.1 Data pre-processing

For the microtubules and synapse datasets, we used the itk library v5.4rc1 to rigidly align the pairs of low- and high-resolution images. The registered image contains padded pixels, while the reference image does not. Thus, to avoid the models from learning misleading information, we used the resulting transformation to reproduce the padding in the reference image. For all datasets, images were cropped into patches of size 256 x 256 pixels in a non-overlapping fashion.

#### S1.2 Dataset partitioning

**Table S1: Dataset description and pre-processing.** For each dataset, we report the number of fields of view (FOVs), the image sizes, as well as the number of FOVs for the train, validation and test partitions. For the zebrafish dataset, the same sample is consecutively captured 50 times, exhibiting different noise realizations. Thus, we report in parenthesis the number of different sub-FOVs (after dividing the original into patches) before using all noise realizations.

| Dataset | # FOVs | Orig. image size (px) | Train | Validation | Test |
|---|---|---|---|---|---|
| Microtubules | 104 | 2560 x 2560 | 1272 | 89 | 265 |
| Mitochondria | 345 | 600 x 600 | 2646 | 153 | 306 |
| Synapses | 24 | 550 x 550 x 20 | 1198 | 56 | 112 |
| Zebrafish | 20 | 512 x 512 | 3600 (72) | 200 (4) | 200 (4) |

#### S1.3 Diversity

**Table S2: Differences between denoising datasets.** We test the denoising performance using four diverse datasets. These datasets vary along the sample and imaging type, the cell condition (fixed vs. live cells), as well as how the raw and ground truth data were generated.

| # | Sample type | Imaging type | Condition | Raw | Ground Truth |
|---|---|---|---|---|---|
| 1 | Microtubules | STED | Fixed | Low-light dose | High-light dose |
| 2 | Mitochondria | STED | Living | Low-light dose | High-light dose |
| 3 | Synapses | Confocal | Fixed | Confocal | Super-resolution |
| 4 | Zebrafish | Confocal | Fixed | Single images | Avg. of 50 images |

### S2 Additional quality control metrics

The mean absolute error (MAE) between the ground truth image $y$ and reconstructed image $\hat{y}$ captures the general offset in pixel values and is calculated as:

$$MAE(y, \hat{y}) = |y - \hat{y}|$$

The Normalized Root Mean Square Error (NRMSE) compares the pixel values of the reconstruction to the ground truth image. NRMSE normalizes the Root MSE to account for the scale of the data:

$$NRMSE(y, \hat{y}) = \frac{\sqrt{MSE(y, \hat{y})}}{\overline{y}}$$

The peak signal-to-noise ratio (PSNR) quantifies the quality of reconstructed images using a logarithmic measure of the peak error between $y$ and $\hat{y}$. The PSNR value is expressed in decibels (dB):

$$PSNR(y, \hat{y}) = 10 \log_{10}\frac{L^2}{MSE}$$

where $L=255$ is the maximum possible pixel value.

The structural similarity index measure (SSIM) was designed to improve PSNR or MAE by also incorporating differences in luminance $l(y, \hat{y})$, contrast $c(y, \hat{y})$, and structural information $s(y, \hat{y})$:

$$SSIM(y, \hat{y}) = [l(y, \hat{y})]^\alpha \cdot [c(y, \hat{y})]^\beta \cdot [s(y, \hat{y})]^\gamma$$

where $\alpha$, $\beta$, and $\gamma$ define the relative importance of the three components (all set to 1). The multi-scale SSIM (MS-SSIM) additionally evaluates the structural similarity across various scales to capture both fine details and coarse structures.

The *resolution*, as defined by Descloux et al. [2019], assesses the resolution of individual images based on decorrelation analysis. The core idea is to examine how the frequency components of the image decorrelate as the distance between them increases. The correlation coefficient is computed as:

$$d(r) = \frac{\iint Re\{I(\mathbf{k}) I_n(\mathbf{k}) M(\mathbf{k}; r)\}dk_x dk_y}{\sqrt{\iint |I(\mathbf{k})|^2 dk_x dk_y \iint |I_n(\mathbf{k})M(\mathbf{k}; r)|^2 dk_x dk_y}}$$

The resolution is then defined as:

$$R = \frac{2 \times \text{pixel size}}{\max [r_0, \dots, r_{N_g}]}$$

For denoising tasks, we compute the relative resolution:

$$\bar{R} = \frac{R_{\hat{y}}}{R_y}$$

where values close to 1 indicate similar resolution between the high-intensity image and the prediction.

The learned perceptual image patch similarity (LPIPS) assesses the perceptual similarity between images using feature representations extracted from a pre-trained deep neural network (AlexNet). The LPIPS value ranges from 0 (high perceptual similarity) to 1 (low perceptual similarity).

### S3 Benchmark models

- **Noise2Void** - TensorFlow implementation from the authors. Epochs: 100, batch size: 32, initial learning rate: 2e-4. All other parameters use the default.
- **pix2pix** - Implementation from ZeroCostDL4Mic. Epochs: 5, batch size: 1, initial learning rate: 2e-4.
- **UNet-RCAN** - Default settings. Max epochs: 200, initial learning rate: 1e-4, batch size: 1.
- **CARE** - Implementation from ZeroCostDL4Mic. Epochs: 1000, batch size: 8, initial learning rate: 4e-4.

### S4 Averaging across many reconstructions

To improve the performance of the DDPM and remove any noise that was not removed by the denoising process, we employ an averaging strategy. Specifically, we generate several images using the same conditioning input but different inference runs. We consistently observe an increase in performance across several metrics when averaging, except for LPIPS, and in some cases resolution. This might be explained by the smoothing effect of averaging which removes fine-grained structures. Moreover, we observe the performance saturating with approximately 10 averaged samples.

[IMAGE: Figure S1 - Averaging across samples improves the performance for most metrics for the DDPM.]

#### S4.1 Uncertainty maps

We benefit from the repeated sampling strategy to enhance the interpretability of the model. In particular, repeated sampling captures the variability of the model, thus reflecting its uncertainty in restoring certain areas of the image. We approximate the uncertainty based on the pixel-wise standard deviation:

$$\sqrt{\frac{\sum_{i=1}^{N}\left({\hat{y}^i - \bar{\hat{y}}}\right)^2}{255^2 N}}$$

where $N = 15$ is the number of times we repeat the sampling, $\bar{\hat{y}}$ is the average of the multiple predicted samples, and $255^2$ is a normalization factor.

The entropy-based uncertainty map $S = (s_{jk})$ is:

$$s_{jk} = - \sum_{m=1}^{M} p_m \log p_m$$

where $M$ is the number of unique pixel values at location $(j, k)$ among the single image predictions, and $p_m$ is the probability of the $m$-th unique pixel value at that location.

[IMAGE: Figure S2 - Uncertainty maps based on repeated sampling strategy with DDPM. For each dataset (A: microtubules, B: mitochondria, C: synapse, D: zebrafish), showing low- and high-resolution images, and the resulting uncertainty maps based on pixel-wise standard deviation and entropy.]

When computing uncertainty as the pixel-wise standard deviation, many high uncertainty regions correspond to the brighter areas of the low-resolution images. The model shows the highest uncertainty for the synapse dataset, whereas the mitochondria dataset has the lowest uncertainty values. In particular, for mitochondria, the model is most uncertain in predicting the membrane, an area which is inherently ambiguous in the noisy data.

In contrast, uncertainty regions for the entropy-based formulation go beyond bright areas, and also include very noisy background regions. On the zebrafish images, high uncertainty is observed in regions with visibly fine-grained details in the high-resolution image, that are ambiguous in the low-resolution image due to overlaid noise. In both uncertainty formulations, smooth regions in the noisy images are characterized by high-confidence values.

### S5 Results on additional metrics

**Table S3: Benchmarking the conditioned DDPM with additional metrics on novel datasets.** We report the median value of NRMSE (lower is better) and Pearson correlation (higher is better).

| | Microtubule | | Mitochondria | |
|---|---|---|---|---|
| Model | NRMSE | Corr. | NRMSE | Corr. |
| Raw | 0.99 | 0.46 | 0.97 | 0.40 |
| Noise2Void | 0.65 | 0.87 | 0.32 | 0.90 |
| Pix2pix | 0.35 | 0.88 | 0.40 | 0.83 |
| UNet-RCAN | 0.90 | **0.92** | 3.76 | **0.92** |
| CARE | 0.26 | **0.92** | 0.42 | 0.90 |
| **DDPM** | 0.29 | 0.89 | 0.36 | 0.87 |
| **DDPM-avg** | **0.25** | **0.92** | **0.30** | **0.92** |

**Table S4: Benchmarking the conditioned DDPM with additional metrics on external datasets.**

| | Synapse | | Zebrafish | |
|---|---|---|---|---|
| Model | NRMSE | Corr. | NRMSE | Corr. |
| Raw | 1.33 | 0.60 | 0.70 | 0.74 |
| Noise2Void | 1.32 | 0.61 | 0.27 | 0.94 |
| Pix2pix | 0.69 | 0.77 | 0.32 | 0.91 |
| UNet-RCAN | **0.58** | **0.83** | 0.55 | 0.94 |
| CARE | 0.74 | **0.83** | 0.31 | **0.95** |
| **DDPM** | 0.61 | 0.80 | 0.30 | 0.92 |
| **DDPM-avg** | **0.58** | 0.81 | **0.24** | **0.95** |

#### S5.1 Reconstruction resolution

**Table S5: Resolution across models and datasets.** We report the median of image resolution in nm, and the resolution ratio with respect to ground-truth (GT) resolution.

| Model | Microtubule r / r ratio | Mitochondria r / r ratio | Synapse r / r ratio | Zebrafish r / r ratio |
|---|---|---|---|---|
| Raw | 128.60 / 1.3 | 3563.64 / 11.91 | 143.14 / 0.49 | 5297.4 / 6.82 |
| GT | 98.80 / 1.00 | 299.24 / 1.00 | 293.33 / 1.00 | 776.70 / 1.00 |
| Noise2Void | 107.85 / 1.09 | 111.36 / 0.37 | **147.04** / 0.50 | 1141.8 / 1.47 |
| Pix2pix | **88.45** / 0.90 | 149.62 / 0.50 | 230.58 / 0.79 | **730.05** / 0.94 |
| UNet-RCAN | 118.35 / 1.20 | **76.72** / 0.27 | 385.88 / 1.32 | 1031.70 / 1.33 |
| CARE | 119.75 / 1.21 | 137.54 / 0.46 | 363.20 / 1.24 | 772.35 / 0.99 |
| DDPM | 97.6 / 0.99 | 177.38 / 0.59 | 330.18 / 1.13 | 831.60 / 1.07 |
| DDPM-avg | 115.28 / 1.17 | 110.74 / 0.37 | 363.20 / 1.24 | 777.75 / 1.00 |

### S6 Model architecture

[IMAGE: Figure S3 - U-Net architecture adapted from Karras et al. 2024. A) Three main parts: auxiliary embedding network, encoder blocks, and decoding blocks. B) Network receives noisy image concatenated to conditioning image, processed by encoder and decoder blocks with skip connections.]

#### S6.1 Timestep embedding

As in Karras et al. [2024], the original ADM timestep embedding layer is replaced with Fourier features:

$$MPFourier(a) = \begin{bmatrix} \sqrt{2} \cos(2\pi(f_1 a + \varphi_1)) \\ \sqrt{2} \cos(2\pi(f_2 a + \varphi_2)) \\ \vdots \\ \sqrt{2} \cos(2\pi(f_N a + \varphi_N)) \end{bmatrix}$$

where $f_i \sim \mathcal{N}(0,1)$, $\varphi \sim \mathcal{U}(0,1)$, and $a = \bar{\alpha}_t$ is a scalar defined as a function of the noise level $t$ and the variance schedule. In the feature vector, $\sqrt{2}$ is the scaling factor that enables magnitude preservation, followed by a linear transformation with learnable parameters, a magnitude-preserving sum operator, and a magnitude-preserving SiLU non-linearity.
