# Palette: Image-to-Image Diffusion Models

## Abstract

This paper develops a unified framework for image-to-image translation based on conditional diffusion models and evaluates this framework on four challenging image-to-image translation tasks, namely colorization, inpainting, uncropping, and JPEG restoration. Our simple implementation of image-to-image diffusion models outperforms strong GAN and regression baselines on all tasks, without task-specific hyper-parameter tuning, architecture customization, or any auxiliary loss or sophisticated new techniques needed. We uncover the impact of an L2 vs. L1 loss in the denoising diffusion objective on sample diversity, and demonstrate the importance of self-attention in the neural architecture through empirical studies. Importantly, we advocate a unified evaluation protocol based on ImageNet, with human evaluation and sample quality scores (FID, Inception Score, Classification Accuracy of a pre-trained ResNet-50, and Perceptual Distance against original images). We expect this standardized evaluation protocol to play a role in advancing image-to-image translation research. Finally, we show that a generalist, multi-task diffusion model performs as well or better than task-specific specialist counterparts. Check out https://diffusion-palette.github.io/ for an overview of the results and code.

## 1 Introduction

[IMAGE: Figure 1 - Image-to-image diffusion models are able to generate high-fidelity output across tasks (colorization, inpainting, uncropping, JPEG restoration) without task-specific customization or auxiliary loss.]

Many problems in vision and image processing can be formulated as image-to-image translation. Examples include restoration tasks, like super-resolution, colorization, and inpainting, as well as pixel-level image understanding tasks, such as instance segmentation and depth estimation. Many such tasks are complex inverse problems, where multiple output images are consistent with a single input. A natural approach to image-to-image translation is to learn the conditional distribution of output images given the input, using deep generative models that can capture multi-modal distributions in the high-dimensional space of images.

[IMAGE: Figure 2 - Given the central 256x256 pixels, we extrapolate to the left and right in steps of 128 pixels (2x8 applications of 50% Palette uncropping), to generate the final 256x2304 panorama.]

Generative Adversarial Networks (GANs) [goodfellow2014generative; radford2015unsupervised] have emerged as the model family of choice for many image-to-image tasks [isola-cvpr-2017]; they are capable of generating high fidelity outputs, are broadly applicable, and support efficient sampling. Nevertheless, GANs can be challenging to train [arjovsky-arxiv-2017; gulrajani2017improved], and often drop modes in the output distribution [metz2016unrolled; ravuri2019classification]. Autoregressive Models [oord2016conditional; parmar2018image], VAEs [Kingma2013; vahdat2021nvae], and Normalizing Flows [dinh2016density; Kingma2018] have seen success in specific applications, but arguably, have not established the same level of quality and generality as GANs.

Diffusion and score-based models [sohl2015deep; song-arxiv-2020; ho2020denoising] have received a surge of recent interest [cai-eccv-2020; song-iclr-2021; hoogeboom2021argmax; vahdat2021score; kingma2021variational; austin2021structured], resulting in several key advances in modeling continuous data. On speech synthesis, diffusion models have achieved human evaluation scores on par with SoTA autoregressive models [chen-iclr-2021; chen-interspeech-2021; kong-arxiv-2020]. On the class-conditional ImageNet generation challenge they have outperformed strong GAN baselines in terms of FID scores [dhariwal2021diffusion; ho-arxiv-2021]. On image super-resolution, they have delivered impressive face enhancement results, outperforming GANs [saharia2021image]. Despite these results, it is not clear whether diffusion models rival GANs in offering a versatile and general framework for image manipulation.

This paper investigates the general applicability of *Palette*, our implementation of image-to-image diffusion models, to a suite of distinct and challenging tasks, namely colorization, inpainting, uncropping, and JPEG restoration. We show that Palette, with no task-specific architecture customization, nor changes to hyper-parameters or the loss, delivers high-fidelity outputs across all four tasks. It outperforms task-specific baselines and a strong regression baseline with an identical neural architecture. Importantly, we show that a single *generalist* Palette model, trained on colorization, inpainting and JPEG restoration, outperforms a task-specific JPEG model and achieves competitive performance on the other tasks.

We study key components of Palette, including the denoising loss function and the neural net architecture. We find that while ```latex $L_2$ ``` [ho2020denoising] and ```latex $L_1$ ``` [chen-iclr-2021] losses in the denoising objective yield similar sample-quality scores, ```latex $L_2$ ``` leads to a higher degree of diversity in model samples, whereas ```latex $L_1$ ``` [chen-iclr-2021] produces more conservative outputs. We also find that removing self-attention layers from the U-Net architecture of Palette, to build a fully convolutional model, hurts performance. Finally, we advocate a standardized evaluation protocol for inpainting, uncropping, and JPEG restoration based on ImageNet [deng2009imagenet], and we report sample quality scores for several baselines. We hope this benchmark will help advance image-to-image translation research.

## 2 Related work

Our work is inspired by Pix2Pix [isola-cvpr-2017], which explored myriad image-to-image translation tasks with GANs. GAN-based techniques have also been proposed for image-to-image problems like unpaired translation [zhu2017unpaired], unsupervised cross-domain generation [taigman2016unsupervised], multi-domain translation [choi2018stargan], and few shot translation [liu2019few]. Nevertheless, existing GAN models are sometimes unsuccessful in holistically translating images with consistent structural and textural regularity.

Diffusion models [sohl2015deep] recently emerged with impressive results on image generation [ho2020denoising; ho-arxiv-2021; dhariwal2021diffusion], audio synthesis [chen-iclr-2021; kong2020diffwave], and image super-resolution [saharia2021image; Kadkhodaie2021], as well as unpaired image-to-image translation [sasaki2021unit] and image editing [meng2021sdedit; sinha2021d2c]. Our conditional diffusion models build on these recent advances, showing versatility on a suite of image-to-image translation tasks.

Most diffusion models for inpainting and other linear inverse problems have adapted unconditional models for use in conditional tasks [sohl2015deep; song-iclr-2021; meng2021sdedit]. This has the advantage that only one model need be trained. However, unconditional tasks are often more difficult than conditional tasks. We cast Palette as a conditional model, opting for multitask training should one want a single model for multiple tasks.

Early **inpainting** approaches [bertalmio2000image; barnes2009PAR; he2012statistics; hays2007scene] work well on textured regions but often fall short in generating semantically consistent structure. GANs are widely used but often require auxiliary objectives on structures, context, edges, contours and hand-engineered features [iizuka2017globally; yu2018generative; yu2019free; nazeri2019edgeconnect; yi2020contextual; liu2020rethinking; kim2021zoomtoinpaint], and they lack diversity in their outputs [zheng2019pluralistic; zhao2021large].

**Image uncropping** (a.k.a. outpainting) is considered more challenging than inpainting as it entails generating open-ended content with less context. Early methods relied on retrieval [kopf2012quality; wang2014biggerpicture; shan2014uncrop]. GAN-based methods are now predominant [teterwak2019boundless], but are often domain-specific [yang2019very; bowen2021oconet; wang2019srn; cheng2021out; lin2021infinitygan]. We show that conditional diffusion models trained on large datasets reliably address both inpainting and uncropping across image domains.

**Colorization** is a well-studied task [coltran; guadarrama2017pixcolor; royer-arxiv-2017; ardizzone-arxiv-2017], requiring a degree of scene understanding, which makes it a natural choice for self-supervised learning [larsson2016learning]. Challenges include diverse colorization [deshpande2017learning], respecting semantic categories [zhang2016colorful], and producing high-fidelity color [guadarrama2017pixcolor]. While some prior work makes use of specialized auxiliary classification losses, we find that generic image-to-image diffusion models work well without task-specific specialization.

**JPEG restoration** (aka. JPEG artifact removal) is the nonlinear inverse problem of removing compression artifacts. [dong2015compression] applied deep CNN architectures for JPEG restoration, and [galteri2017deep; galteri2019deep] successfully applied GANs for artifact removal, but they have been restricted to quality factors above 10. We show the effectiveness of Palette in removing compression artifacts for quality factors as low as 5.

**Multi-task training** is a relatively under-explored area in image-to-image translation. [guocheng2019trinity; yu2018crafting] train simultaneously on multiple tasks, but they focus primarily on enhancement tasks like deblurring, denoising, and super-resolution, and they use smaller modular networks. Several works have also dealt with simultaneous training over multiple degradations on a single task, e.g., multi-scale super-resolution [kim2016deeply], and JPEG restoration on multiple quality factors [galteri2019deep; liu2018multi]. With Palette we take a first step toward building multi-task image-to-image diffusion models for a wide variety of tasks.

## 3 Palette

Diffusion models [sohl2015deep; ho2020denoising] convert samples from a standard Gaussian distribution into samples from an empirical data distribution through an iterative denoising process. Conditional diffusion models [chen-iclr-2021; saharia2021image] make the denoising process conditional on an input signal. Image-to-image diffusion models are conditional diffusion models of the form ```latex $p(\mathbf{y} \mid \mathbf{x})$ ```, where both ```latex $\mathbf{x}$ ``` and ```latex $\mathbf{y}$ ``` are images, e.g., ```latex $\mathbf{x}$ ``` is a grayscale image and ```latex $\mathbf{y}$ ``` is a color image. These models have been applied to image super-resolution [saharia2021image; nichol2021improved]. We study the general applicability of image-to-image diffusion models on a broad set of tasks.

For a detailed treatment of diffusion models, please see Appendix A. Here, we briefly discuss the denoising loss function. Given a training output image ```latex $\mathbf{y}$ ```, we generate a noisy version ```latex $\tilde{\mathbf{y}}$ ```, and train a neural network ```latex $f_\theta$ ``` to denoise ```latex $\tilde{\mathbf{y}}$ ``` given ```latex $\mathbf{x}$ ``` and a noise level indicator ```latex $\gamma$ ```, for which the loss is:

```latex
$$\mathbb{E}_{(\mathbf{x}, \mathbf{y})} \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)} \mathbb{E}_{\gamma}\, \bigg\lVert f_\theta(\mathbf{x},\, \underbrace{\sqrt{\gamma} \,\mathbf{y} + \sqrt{1-\gamma}\, \boldsymbol{\epsilon}}_{\tilde{\mathbf{y}}}, \,\gamma) - \boldsymbol{\epsilon}\, \bigg\rVert^{p}_p$$
```

[chen-iclr-2021] and [saharia2021image] suggest using the ```latex $L_1$ ``` norm, i.e., ```latex $p=1$ ```, whereas the standard formulation is based on the usual ```latex $L_2$ ``` norm [ho2020denoising]. We perform careful ablations below, and analyze the impact of the choice of norm. We find that ```latex $L_1$ ``` yields significantly lower sample diversity compared to ```latex $L_2$ ```. While ```latex $L_1$ ``` may be useful, to reduce potential hallucinations in some applications, here we adopt ```latex $L_2$ ``` to capture the output distribution more faithfully.

**Architecture.** Palette uses a U-Net architecture [ho2020denoising] with several modifications inspired by recent work [song-iclr-2021; saharia2021image; dhariwal2021diffusion]. The network architecture is based on the 256x256 class-conditional U-Net model of [dhariwal2021diffusion]. The two main differences between our architecture and theirs are (i) absence of class-conditioning, and (ii) additional conditioning of the source image via concatenation, following [saharia2021image].

[IMAGE: Figure 3 - Colorization results on ImageNet validation images comparing Grayscale Input, PixColor, ColTran, Regression, Palette (Ours), and Original.]

## 4 Evaluation protocol

Evaluating image-to-image translation models is challenging. Prior work on colorization [zhang2016colorful; guadarrama2017pixcolor; coltran] relied on FID scores and human evaluation for model comparison. Tasks like inpainting [yu2019free; yu2018generative] and uncropping [teterwak2019boundless; wang2019wide] have often heavily relied on qualitative evaluation. For other tasks, like JPEG restoration [dong2015compression; liu2018multi; galteri2019deep], it has been common to use reference-based pixel-level similarity scores such as PSNR and SSIM. It is also notable that many tasks lack a standardized dataset for evaluation, e.g., different test sets with method-specific splits are used for evaluation.

We propose a unified evaluation protocol for inpainting, uncropping, and JPEG restoration on ImageNet [deng2009imagenet], due to its scale, diversity, and public availability. For inpainting and uncropping, existing work has relied on Places2 dataset [zhou2017places] for evaluation. Hence, we also use a standard evaluation setup on Places2 for these tasks. Specifically, we advocate the use of ImageNet ctest10k split proposed by [larsson2016learning] as a standard subset for benchmarking of all image-to-image translation tasks on ImageNet. We also introduce a similar category-balanced 10,950 image subset of Places2 validation set called *places10k*. We further advocate the use of automated metrics that capture both image quality and diversity, in addition to controlled human evaluation. We avoid pixel-level metrics like PSNR and SSIM as they are not reliable measures of sample quality for difficult tasks that require hallucination, like recent super-resolution work, where [ledig2017photo; dahl2017pixel; menon2020pulse] observe that PSNR and SSIM tend to prefer blurry regression outputs, unlike human perception.

We use four automated quantitative measures of sample quality for image-to-image translation: **Inception Score (IS)** [salimans-iclr-2017]; **Frechet Inception Distance (FID)**; **Classification Accuracy (CA)** (top-1) of a pre-trained ResNet-50 classifier; and a simple measure of **Perceptual Distance (PD)**, i.e., Euclidean distance in Inception-v1 feature space (c.f., [DosovitskiyBrox2016]). To facilitate benchmarking on our proposed subsets, we release our model outputs together with other data such as the inpainting masks (see https://bit.ly/eval-pix2pix). For some tasks, we also assess **sample diversity** through pairwise SSIM and LPIPS scores between multiple model outputs. Sample diversity is challenging and has been a key limitation of many existing GAN-based methods [zhu2017multimodal; yang2019diversity].

The ultimate evaluation of image-to-image translation models is **human evaluation**; i.e., whether or not humans can discriminate model outputs from natural images. To this end we use 2-alternative forced choice (2AFC) trials to evaluate the perceptual quality of model outputs against natural images from which we obtained test inputs (c.f., the Colorization Turing Test [zhang2016colorful]). We summarize the results in terms of the **fool rate**, the percentage of human raters who select model outputs over natural images when they were asked "Which image would you guess is from a camera?". (See Appendix C for details.)

## 5 Experiments

We apply Palette to a suite of challenging image-to-image tasks:

1. **Colorization** transforms an input grayscale image to a plausible color image.
2. **Inpainting** fills in user-specified masked regions of an image with realistic content.
3. **Uncropping** extends an input image along one or more directions to enlarge the image.
4. **JPEG restoration** corrects for JPEG compression artifacts, restoring plausible image detail.

We do so without task-specific hyper-parameter tuning, architecture customization, or any auxiliary loss function. Inputs and outputs for all tasks are represented as 256x256 RGB images. Each task presents its own unique challenges. Colorization entails a representation of objects, segmentation and layout, with long-range image dependencies. Inpainting is challenging with large masks, image diversity and cluttered scenes. Uncropping is widely considered even more challenging than inpainting as there is less surrounding context to constrain semantically meaningful generation. While the other tasks are linear in nature, JPEG restoration is a non-linear inverse problem; it requires a good local model of natural image statistics to detect and correct compression artifacts. While previous work has studied these problems extensively, it is rare that a model with no task-specific engineering achieves strong performance in all tasks, beating strong task-specific GAN and regression baselines. Palette uses an ```latex $L_2$ ``` loss for the denoising objective, unless otherwise specified. (Implementation details can be found in Appendix B.)

### 5.1 Colorization

While prior works [zhang2016colorful; coltran] have adopted LAB or YCbCr color spaces to represent output images for colorization, we use the RGB color space to maintain generality across tasks. Preliminary experiments indicated that Palette is equally effective in YCbCr and RGB spaces. We compare Palette with Pix2Pix [isola2017image], PixColor [guadarrama2017pixcolor], and ColTran [coltran]. Palette establishes a new SoTA, outperforming existing works by a large margin. Further, the performance measures (FID, IS, and CA) indicate that Palette outputs are close to being indistinguishable from the original images that were used to create the test greyscale inputs. Surprisingly, our ```latex $L_2$ ``` Regression baseline also outperforms prior task-specific techniques, highlighting the importance of modern architectures and large-scale training, even for a basic Regression model. On human evaluation, Palette improves upon human raters' fool rate of ColTran by more than 10%, approaching an ideal fool rate of 50%.

**Table 1: Colorization quantitative scores and fool rates on ImageNet val set.**

| Model | FID-5K | IS | CA | PD | Fool rate |
|---|---|---|---|---|---|
| *Prior Work* | | | | | |
| pix2pix | 24.41 | - | - | - | - |
| PixColor | 24.32 | - | - | - | 29.90% |
| Coltran | 19.37 | - | - | - | 36.55% |
| *This paper* | | | | | |
| Regression | 17.89 | 169.8 | 68.2% | 60.0 | 39.45% |
| **Palette** | **15.78** | **200.8** | **72.5%** | **46.2** | **47.80%** |
| Original images | 14.68 | 229.6 | 75.6% | 0.0 | - |

### 5.2 Inpainting

We follow [yu2019free] and train inpainting models on free-form generated masks, augmented with simple rectangular masks. To maintain generality of Palette across tasks, in contrast to prior work, we do not pass a binary inpainting mask to the models. Instead, we fill the masked region with standard Gaussian noise, which is compatible with denoising diffusion models. The training loss only considers the masked out pixels, rather than the entire image, to speed up training. We compare Palette with DeepFillv2 [yu2019free], HiFill [yi2020contextual], Photoshop's *Content-aware Fill*, and Co-ModGAN [zhao2021large].

Palette exhibits strong performance across inpainting datasets and mask configurations, outperforming DeepFillv2, HiFill and Co-ModGAN by a large margin. Importantly, like the colorization task above, the FID scores for Palette outputs in the case of 20-30% free-form masks, are extremely close to FID scores on the original images from which we created the masked test inputs.

[IMAGE: Figure 4 - Comparison of inpainting methods on object removal. Baselines: Photoshop's Content-aware Fill, DeepFillv2, HiFill, and Co-ModGAN.]

**Table 2: Quantitative evaluation for free-form and center inpainting on ImageNet and Places2 validation images.**

| Model | FID (ImageNet) | IS | CA | PD | FID (Places2) | PD (Places2) |
|---|---|---|---|---|---|---|
| *20-30% free form* | | | | | | |
| DeepFillv2 | 9.4 | 174.6 | 68.8% | 64.7 | 13.5 | 63.0 |
| HiFill | 12.4 | 157.0 | 65.7% | 86.2 | 15.7 | 92.8 |
| Co-ModGAN | - | - | - | - | 12.4 | 51.6 |
| **Palette (Ours)** | **5.2** | **205.5** | **72.3%** | **27.6** | **11.7** | **35.0** |
| *128x128 center* | | | | | | |
| DeepFillv2 | 18.0 | 135.3 | 64.3% | 117.2 | 15.3 | 96.3 |
| HiFill | 20.1 | 126.8 | 62.3% | 129.7 | 16.9 | 115.4 |
| Co-ModGAN | - | - | - | - | 13.7 | 86.2 |
| **Palette (Ours)** | **6.6** | **173.9** | **69.3%** | **59.5** | **11.9** | **57.3** |
| Original images | 5.1 | 231.6 | 74.6% | 0.0 | 11.4 | 0.0 |

### 5.3 Uncropping

Recent works [teterwak2019boundless; lin2021infinitygan] have shown impressive visual effects by extending (extrapolating) input images along the right border. We train Palette on uncropping in any one of the four directions, or around the entire image border on all four sides. In all cases, we keep the area of the masked region at 50% of the image. Like inpainting, we fill the masked region with Gaussian noise, and keep the unmasked region fixed during inference. We compare Palette with Boundless [teterwak2019boundless] and InfinityGAN [lin2021infinitygan]. From the results, one can see that Palette outperforms baselines on ImageNet and Places2 by a large margin. On human evaluation, Palette has a 40% fool rate, compared to 25% and 15% for Boundless and InfinityGAN.

We further assess the robustness of Palette by generating panoramas through repeated application of left and right uncropping. We observe that Palette is surprisingly robust, generating realistic and coherent outputs even after 8 repeated applications of uncrop.

[IMAGE: Figure 5 - Image uncropping results on Places2 validation images comparing Masked Input, Boundless, InfinityGAN, and Palette (Ours).]

**Table 3: Quantitative scores and human raters' fool rates on uncropping.**

| Model | FID (ImageNet) | IS | CA | PD | FID (Places2) | PD (Places2) |
|---|---|---|---|---|---|---|
| Boundless | 18.7 | 104.1 | 58.8% | 127.9 | 11.8 | 129.3 |
| **Palette (Ours)** | **5.8** | **138.1** | **63.4%** | **85.9** | **3.53** | **103.3** |
| Original images | 2.7 | 250.1 | 76.0% | 0.0 | 2.1 | 0.0 |

### 5.4 JPEG restoration

Finally, we evaluate Palette on the task of removing JPEG compression artifacts, a long standing image restoration problem [dong2015compression; galteri2019deep; liu2018multi]. Like prior work [ehrlich2020quantization; liu2018multi], we train Palette on inputs compressed with various quality factors (QF). While prior work has typically limited itself to a Quality Factor ```latex $\geq$ ``` 10, we increase the difficulty of the task and train on Quality Factors as low as 5, producing severe compression artifacts. Palette exhibits strong performance across all quality factors, outperforming the regression baseline. As expected, the performance gap between Palette and the regression baseline widens with decreasing quality factor. The regression model produces blurry outputs, while Palette produces sharper images.

[IMAGE: Figure 6 - Example of JPEG restoration results comparing Input (QF=5), Regression, Palette (Ours), and Original.]

**Table 4: Quantitative evaluation for JPEG restoration for various Quality Factors (QF).**

| QF | Model | FID-5K | IS | CA | PD |
|---|---|---|---|---|---|
| 5 | Regression | 29.0 | 73.9 | 52.8% | 155.4 |
| 5 | **Palette (Ours)** | **8.3** | **133.6** | **64.2%** | **95.5** |
| 10 | Regression | 18.0 | 117.2 | 63.5% | 102.2 |
| 10 | **Palette (Ours)** | **5.4** | **180.5** | **70.7%** | **58.3** |
| 20 | Regression | 11.5 | 158.7 | 69.7% | 65.4 |
| 20 | **Palette (Ours)** | **4.3** | **208.7** | **73.5%** | **37.1** |
| | Original images | 2.7 | 250.1 | 76.0% | 0.0 |

### 5.5 Self-attention in diffusion model architectures

Self-attention layers [vaswani-nips-2017] have been an important component in recent U-Net architectures for diffusion models [ho2020denoising; dhariwal2021diffusion]. While self-attention layers provide a direct form of global dependency, they prevent generalization to unseen image resolutions. Generalization to new resolutions at test time is convenient for many image-to-image tasks, and therefore previous works have relied primarily on fully convolutional architectures [yu2019free; galteri2019deep].

[IMAGE: Figure 7 - Palette diversity for inpainting, colorization, and uncropping showing Input, Sample 1-4.]

We analyze the impact of these self-attention layers on sample quality for inpainting. We experiment with the following four configurations:

1. **Global Self-Attention**: Baseline configuration with global self-attention layers at 32x32, 16x16 and 8x8 resolutions.
2. **Local Self-Attention**: Local self-attention layers [vaswani2021scaling] at 32x32, 16x16 and 8x8 resolutions, at which feature maps are divided into 4 non-overlapping query blocks.
3. **More ResNet Blocks w/o Self-Attention**: 2x residual blocks at 32x32, 16x16 and 8x8 resolutions allowing deeper convolutions to increase receptive field sizes.
4. **Dilated Convolutions w/o Self-Attention**: ResNet blocks at 32x32, 16x16 and 8x8 resolutions with increasing dilation rates [chen2017rethinking] allowing exponentially increasing receptive fields.

We train models for 500K steps, with a batch size of 512. Global self-attention offers better performance than fully-convolutional alternatives (even with 15% more parameters), re-affirming the importance of self-attention layers for such tasks. Surprisingly, local self-attention performs worse than fully-convolutional alternatives. Sampling speed is slower than GAN models. There is a large overhead for loading models and the initial jit compilation, but for 1000 test images, Palette requires 0.8 sec./image on a TPUv4.

**Table 5: Architecture ablation for inpainting.**

| Model | # Params | FID | IS | PD |
|---|---|---|---|---|
| *Fully Convolutional* | | | | |
| Dilated Convolutions | 624M | 8.0 | 157.5 | 70.6 |
| More ResNet Blocks | 603M | 8.1 | 157.1 | 71.9 |
| *Self-Attention* | | | | |
| Local Self-Attention | 552M | 9.4 | 149.8 | 78.2 |
| **Global Self-Attention** | 552M | **7.4** | **164.8** | **67.1** |

### 5.6 Sample diversity

We next analyze sample diversity of Palette on two tasks, colorization and inpainting. Specifically, we analyze the impact of changing the diffusion loss function ```latex $L_{\text{simple}}$ ``` [ho2020denoising], and compare ```latex $L_1$ ``` vs. ```latex $L_2$ ``` on sample diversity. While existing conditional diffusion models, SR3 [saharia2021image] and WaveGrad [chen-iclr-2021], have found ```latex $L_1$ ``` norm to perform better than the conventional ```latex $L_2$ ``` loss, there has not been a detailed comparison of the two. To quantitatively compare sample diversity, we use multi-scale SSIM [guadarrama2017pixcolor] and the LPIPS diversity score [zhu2017multimodal]. Given multiple generated outputs for each input image, we compute pairwise multi-scale SSIM between the first output sample and the remaining samples. Following [zhu2017multimodal], we also compute LPIPS scores between consecutive pairs of model outputs for a given input image, and then average across all outputs and input images. Lower SSIM and higher LPIPS scores imply more sample diversity. The results clearly show that models trained with the ```latex $L_2$ ``` loss have greater sample diversity than those trained with the ```latex $L_1$ ``` loss.

Interestingly, ```latex $L_1$ ``` and ```latex $L_2$ ``` models yield similar FID scores (i.e., comparable perceptual quality), but ```latex $L_1$ ``` has somewhat lower Perceptual Distance scores than ```latex $L_2$ ```. One can speculate that ```latex $L_1$ ``` models may drop more modes than ```latex $L_2$ ``` models, thereby increasing the likelihood that a single sample from an ```latex $L_1$ ``` model is from the mode containing the corresponding original image, and hence a smaller Perceptual Distance.

**Table 6: Comparison of L_p norm in denoising objective.**

| Model | FID (Inpainting) | PD (Inpainting) | LPIPS (Inpainting) | FID (Colorization) | PD (Colorization) | LPIPS (Colorization) |
|---|---|---|---|---|---|---|
| Diffusion L1 | 3.6 | **41.9** | 0.11 | 3.4 | **45.8** | 0.09 |
| Diffusion L2 | 3.6 | 43.8 | **0.13** | 3.4 | 48.0 | **0.15** |

[IMAGE: Figure 8 - Pairwise multi-scale SSIM for colorization (left) and inpainting (right).]

### 5.7 Multi-task learning

Multi-task training is a natural approach to learning a single model for multiple image-to-image tasks, i.e., blind image enhancement. Another is to adapt an unconditional model to conditional tasks with imputation. For example, [song-iclr-2021] do this for inpainting; in each step of iterative refinement, they denoise the noisy image from the previous step, and then simply replace any pixels in the estimated image ```latex $\mathbf{y}$ ``` with pixels from the observed image regions, then adding noise and proceeding to the next denoising iteration. The re-purposed unconditional model does not perform well, in part because it is hard to learn a good unconditional model on diverse datasets like ImageNet, and also because, during iterative refinement, noise is added to all pixels, including the observed pixels. By contrast, Palette is conditioned directly on noiseless observations for all steps.

[IMAGE: Figure 9 - Comparison of conditional and unconditional diffusion models for inpainting showing Input, Unconditional, Multi-Task, and Task-Specific outputs.]

Multi-task generalist Palette outperforms the task-specific JPEG restoration specialist model, but slightly lags behind task-specific Palette models on inpainting and colorization. The multi-task and task-specific Palette models had the same number of training steps; we expect multi-task performance to improve with more training.

**Table 7: Performance of multi-task Palette on various tasks.**

| Model | FID | IS | CA | PD |
|---|---|---|---|---|
| *Inpainting (128x128 center mask)* | | | | |
| Palette (Task-specific) | **6.6** | **173.9** | **69.3%** | **59.5** |
| Palette (Multi-task) | 6.8 | 165.7 | 68.9% | 65.2 |
| *Colorization* | | | | |
| Regression (Task-specific) | 5.5 | 176.9 | 68.0% | 61.1 |
| Palette (Task-specific) | **3.4** | **212.9** | **72.0%** | **48.0** |
| Palette (Multi-task) | 3.7 | 187.4 | 69.4% | 57.1 |
| *JPEG Restoration (QF = 5)* | | | | |
| Regression (Task-specific) | 29.0 | 73.9 | 52.8% | 155.4 |
| Palette (Task-specific) | 8.3 | 133.6 | 64.2% | 95.5 |
| **Palette (Multi-task)** | **7.0** | **137.8** | **64.7%** | **92.4** |

## 6 Conclusion

We present Palette, a simple, general framework for image-to-image translation. Palette achieves strong results on four challenging image-to-image translation tasks (colorization, inpainting, uncropping, and JPEG restoration), outperforming strong GAN and regression baselines. Unlike many GAN models, Palette produces diverse and high fidelity outputs. This is accomplished without task-specific customization nor optimization instability. We also present a multi-task Palette model, that performs just as well or better over their task-specific counterparts. Further exploration and investigation of multi-task diffusion models is an exciting avenue for future work. This paper shows some of the potential of image-to-image diffusion models, but we look forward to seeing new applications.

## Appendix A: Diffusion Models

Diffusion models comprise a forward diffusion process and a reverse denoising process that is used at generation time. The forward diffusion process is a Markovian process that iteratively adds Gaussian noise to a data point ```latex $\mathbf{y}_0 \equiv \mathbf{y}$ ``` over ```latex $T$ ``` iterations:

```latex
$$q(\mathbf{y}_{t+1} | \mathbf{y}_t) = \mathcal{N}(\mathbf{y}_{t-1} ; \sqrt{\alpha_t} \mathbf{y}_{t-1}, (1 - \alpha_t) I)$$
$$q(\mathbf{y}_{1:T} | \mathbf{y}_0) = \prod_{t=1}^T q(\mathbf{y}_t | \mathbf{y}_{t-1})$$
```

where ```latex $\alpha_t$ ``` are hyper-parameters of the noise schedule. The forward process with ```latex $\alpha_t$ ``` is constructed in a manner where at ```latex $t=T$ ```, ```latex $\mathbf{y}_T$ ``` is virtually indistinguishable from Gaussian noise. We can also marginalize the forward process at each step:

```latex
$$q(\mathbf{y}_t | \mathbf{y}_0) = \mathcal{N}(\mathbf{y}_t ; \sqrt{\gamma_t} \mathbf{y}_0, (1 - \gamma_t) I)$$
```

where ```latex $\gamma_t = \prod_{t'}^t \alpha_{t'}$ ```.

The Gaussian parameterization of the forward process also allows a closed form formulation of the posterior distribution of ```latex $\mathbf{y}_{t-1}$ ``` given ```latex $(\mathbf{y}_0, \mathbf{y}_t)$ ``` as:

```latex
$$q(\mathbf{y}_{t-1} \mid \mathbf{y}_0, \mathbf{y}_t) = \mathcal{N}(\mathbf{y}_{t-1} \mid \boldsymbol{\mu}, \sigma^2 \mathbf{I})$$
```

where ```latex $\boldsymbol{\mu} = \frac{\sqrt{\gamma_{t-1}}(1-\alpha_t)}{1-\gamma_t} \mathbf{y}_0 + \frac{\sqrt{\alpha_t}(1-\gamma_{t-1})}{1-\gamma_t}\mathbf{y}_t$ ``` and ```latex $\sigma^2 = \frac{(1-\gamma_{t-1})(1-\alpha_t)}{1-\gamma_t}$ ```.

**Learning:** Palette learns a reverse process which inverts the forward process. Given a noisy image ```latex $\tilde{\mathbf{y}}$ ```:

```latex
$$\tilde{\mathbf{y}} = \sqrt{\gamma}\, \mathbf{y}_0 + \sqrt{1-\gamma} \,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I})$$
```

the goal is to recover the target image ```latex $\mathbf{y}_0$ ```. We parameterize our neural network model ```latex $f_\theta(x, \tilde{\mathbf{y}}, \gamma)$ ``` to condition on the input ```latex $x$ ```, a noisy image ```latex $\tilde{\mathbf{y}}$ ```, and the current noise level ```latex $\gamma$ ```. Learning entails prediction of the noise vector ```latex $\boldsymbol{\epsilon}$ ``` by optimizing the objective:

```latex
$$\mathbb{E}_{(\mathbf{x}, \mathbf{y})} \mathbb{E}_{\boldsymbol{\epsilon}, \gamma} \bigg\lVert f_\theta(\mathbf{x}, \underbrace{\sqrt{\gamma} \,\mathbf{y}_0 + \sqrt{1-\gamma}\, \boldsymbol{\epsilon}}_{\tilde{\mathbf{y}}}, \gamma) - \boldsymbol{\epsilon}\, \bigg\rVert^{p}_p$$
```

This objective, also known as ```latex $L_{\text{simple}}$ ``` in [ho2020denoising], is equivalent to maximizing a weighted variational lower-bound on the likelihood [ho2020denoising].

**Inference:** Palette performs inference via the learned reverse process. Since the forward process is constructed so the prior distribution ```latex $p(\mathbf{y}_T)$ ``` approximates a standard normal distribution ```latex $\mathcal{N}(\mathbf{y}_T | \mathbf{0}, \mathbf{I})$ ```, the sampling process can start at pure Gaussian noise, followed by ```latex $T$ ``` steps of iterative refinement.

Given ```latex $\mathbf{y}_t$ ```, we approximate ```latex $\mathbf{y}_0$ ``` as:

```latex
$$\hat{\mathbf{y}}_0 = \frac{1}{\sqrt{\gamma_t}} \left( \mathbf{y}_t - \sqrt{1 - \gamma_t}\, f_{\theta}(\mathbf{x}, \mathbf{y}_{t}, \gamma_t) \right)$$
```

We parameterize the mean of ```latex $p_\theta(\mathbf{y}_{t-1} | \mathbf{y}_t, \mathbf{x})$ ``` as:

```latex
$$\mu_{\theta}(\mathbf{x}, \mathbf{y}_{t}, \gamma_t) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{y}_t - \frac{1-\alpha_t}{ \sqrt{1 - \gamma_t}} f_{\theta}(\mathbf{x}, \mathbf{y}_{t}, \gamma_t) \right)$$
```

And we set the variance of ```latex $p_\theta(\mathbf{y}_{t-1}|\mathbf{y}_t, \mathbf{x})$ ``` to ```latex $(1 - \alpha_t)$ ```, a default given by the variance of the forward process [ho2020denoising].

**Algorithm 1: Training a denoising model**
1. Repeat:
   - Sample ```latex $(\mathbf{x}, \mathbf{y}_0) \sim p(\mathbf{x}, \mathbf{y})$ ```
   - Sample ```latex $\gamma \sim p(\gamma)$ ```
   - Sample ```latex $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ ```
   - Take a gradient descent step on ```latex $\nabla_\theta \lVert f_\theta(\mathbf{x}, \sqrt{\gamma} \mathbf{y}_0 + \sqrt{1-\gamma} \boldsymbol{\epsilon}, \gamma) - \boldsymbol{\epsilon} \rVert_p^p$ ```
2. Until converged

**Algorithm 2: Inference in T iterative refinement steps**
1. Sample ```latex $\mathbf{y}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ ```
2. For ```latex $t = T, \ldots, 1$ ```:
   - Sample ```latex $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ ``` if ```latex $t > 1$ ```, else ```latex $\mathbf{z} = \mathbf{0}$ ```
   - ```latex $\mathbf{y}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left( \mathbf{y}_t - \frac{1-\alpha_t}{\sqrt{1-\gamma_t}} f_\theta(\mathbf{x}, \mathbf{y}_t, \gamma_t) \right) + \sqrt{1 - \alpha_t} \mathbf{z}$ ```
3. Return ```latex $\mathbf{y}_0$ ```

With this parameterization, each iteration of the reverse process can be computed as:

```latex
$$\mathbf{y}_{t-1} \leftarrow \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{y}_t - \frac{1-\alpha_t}{ \sqrt{1 - \gamma_t}} f_{\theta}(\mathbf{x}, \mathbf{y}_{t}, \gamma_t) \right) + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_t$$
```

where ```latex $\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ ```. This resembles one step of Langevin dynamics for which ```latex $f_{\theta}$ ``` provides an estimate of the gradient of the data log-density.

## Appendix B: Implementation Details

**Training Details:** We train all models with a mini batch-size of 1024 for 1M training steps. We do not find over fitting to be an issue, and hence use the model checkpoint at 1M steps for reporting the final results. Consistent with previous works [ho2020denoising; saharia2021image], we use standard Adam optimizer with a fixed 1e-4 learning rate and 10k linear learning rate warmup schedule. We use 0.9999 EMA for all our experiments. We do not perform any task-specific hyper-parameter tuning, or architectural modifications.

**Diffusion Hyper-parameters:** Following [saharia2021image; chen-iclr-2021] we use ```latex $\alpha$ ``` conditioning for training Palette. This allows us to perform hyper-parameter tuning over noise schedules and refinement steps for Palette during inference. During training, we use a linear noise schedule of ```latex $(1e^{-6}, 0.01)$ ``` with 2000 time-steps, and use 1000 refinement steps with a linear schedule of ```latex $(1e^{-4}, 0.09)$ ``` during inference.

**Task Specific Details:**

- **Colorization:** We use RGB parameterization for colorization. We use the grayscale image as the source image and train Palette to predict the full RGB image. During training, following [coltran], we randomly select the largest square crop from the image and resize it to 256x256.

- **Inpainting:** We train Palette on a combination of free-form and rectangular masks. For free-form masks, we use Algorithm 1 in [yu2019free]. For rectangular masks, we uniformly sample between 1 and 5 masks. The total area covered by the rectangular masks is kept between 10% to 40% of the image. We randomly sample a free-form mask with 60% probability, and rectangular masks with 40% probability. We do not provide any additional mask channel, and simply fill the masked region with random Gaussian noise. During training, we restrict the ```latex $L_{\text{simple}}$ ``` loss function to the spatial region corresponding to masked regions, and use the model's prediction for only the masked region during inference.

- **Uncropping:** We train the model for image extension along all four directions, or just one direction. In both cases, we set the masked region to 50% of the image. During training, we uniformly choose masking along one side, or masking along all 4 sides.

- **JPEG Restoration:** We train Palette for JPEG restoration on quality factors in (5, 30). Since decompression for lower quality factors is a significantly more difficult task, we use an exponential distribution to sample the quality factor during training. Specifically, the sampling probability of a quality range ```latex $Q$ ``` is set to ```latex $\propto e^{-Q/10}$ ```.

## Appendix C: Additional Experimental Results

### C.1 Colorization

Following prior work [zhang2016colorful; guadarrama2017pixcolor; coltran], we train and evaluate models on ImageNet [deng2009imagenet]. We follow ColTran [coltran] and use the first 5000 images from ImageNet validation set to report performance on standard metrics.

**Table 8: Benchmark numbers on ctest10k ImageNet subset for Image Colorization.**

| Model | FID-10K | IS | CA | PD |
|---|---|---|---|---|
| Palette (L2) | 3.4 | 212.9 | 72.0% | 48.0 |
| Palette (L1) | 3.4 | 215.8 | 71.9% | 45.8 |
| Ground Truth | 2.7 | 250.1 | 76.0% | 0.0 |

**Human Evaluation:** We use controlled human experiments with 2AFC trials. Subjects were asked "Which image would you guess is from a camera?" with 3 or 5 second display times. The fool rate for Palette is close to 50% and higher than baselines in all cases. When subjects are given less time to inspect the images the fool rates are somewhat higher, as expected.

### C.2 Inpainting

We report all inpainting results on 256x256 center cropped images. We train two Palette models: Palette (I) trained on ImageNet, and Palette (I+P) trained on a mixture of ImageNet and Places2. Palette consistently outperforms existing works by a significant margin on all configurations.

**Table 9: Full quantitative evaluation for inpainting on ImageNet and Places2 validation images.**

| Mask Type | Model | FID (ImageNet) | IS | CA | PD | FID (Places2) | PD (Places2) |
|---|---|---|---|---|---|---|---|
| *10-20% Free-Form* | DeepFillv2 | 6.7 | 198.2 | 71.6% | 38.6 | 12.2 | 38.1 |
| | HiFill | 7.5 | 192.0 | 70.1% | 46.9 | 13.0 | 55.1 |
| | **Palette (I)** | **5.1** | **221.0** | **73.8%** | 15.6 | 11.6 | 22.1 |
| | Palette (I+P) | 5.2 | 219.2 | 73.7% | **15.5** | **11.6** | **20.3** |
| *20-30% Free-Form* | DeepFillv2 | 9.4 | 174.6 | 68.8% | 64.7 | 13.5 | 63.0 |
| | HiFill | 12.4 | 157.0 | 65.7% | 86.2 | 15.7 | 92.8 |
| | Co-ModGAN | - | - | - | - | 12.4 | 51.6 |
| | **Palette (I)** | **5.2** | **208.6** | **72.6%** | **27.4** | 11.8 | 37.7 |
| | Palette (I+P) | **5.2** | 205.5 | 72.3% | 27.6 | **11.7** | **35.0** |
| *30-40% Free-Form* | DeepFillv2 | 14.2 | 144.7 | 64.9% | 95.5 | 15.8 | 90.1 |
| | HiFill | 20.9 | 115.6 | 59.4% | 131.0 | 20.1 | 132.0 |
| | **Palette (I)** | **5.5** | **195.2** | **71.4%** | **39.9** | 12.1 | 53.5 |
| | Palette (I+P) | 5.6 | 192.8 | 71.3% | 40.2 | **11.6** | **49.2** |
| *128x128 Center* | DeepFillv2 | 18.0 | 135.3 | 64.3% | 117.2 | 15.3 | 96.3 |
| | HiFill | 20.1 | 126.8 | 62.3% | 129.7 | 16.9 | 115.4 |
| | **Palette (I)** | **6.4** | 173.3 | **69.7%** | **58.8** | 12.2 | 62.8 |
| | Co-ModGAN | - | - | - | - | 13.7 | 86.2 |
| | Palette (I+P) | 6.6 | **173.9** | 69.3% | 59.5 | **11.9** | **57.3** |
| | Ground Truth | 5.1 | 231.6 | 74.6% | 0.0 | 11.4 | 0.0 |

### C.3 Uncropping

We follow a similar setup as inpainting and train Palette on a combined dataset of Places2 and ImageNet. While we train Palette to extend the image in all directions or just one direction, to compare fairly against existing methods we evaluate Palette on extending only the right half of the image.

**Table 10: Comparison with Boundless on top-50 Places2 categories.**

| Model | FID | PD |
|---|---|---|
| Boundless | 28.3 | 115.0 |
| **Palette** | **22.9** | **93.4** |
| Ground Truth | 23.6 | 0.0 |

**Table 11: Comparison with InfinityGAN and Boundless on scenery categories.**

| Model | FID |
|---|---|
| Boundless | 12.7 |
| InfinityGAN | 15.7 |
| **Palette** | **5.6** |

**Human Evaluation:** Palette obtains significantly higher fool rates on all human evaluation runs compared to Boundless and InfinityGAN. Interestingly, when raters are given more time to inspect each pair of images, the fool rates for InfinityGAN and Boundless worsen considerably. Palette, on the other hand, observes approximately similar fool rates.

### C.4 JPEG Restoration

We perform training and evaluation on ImageNet dataset. We compare Palette with a strong Regression baseline which uses an identical architecture.

### C.5 Evaluation and Benchmarking Details

**Benchmark datasets:** For ImageNet evaluation, we use the 10,000 image subset from ImageNet validation set - **ctest10k** introduced by [larsson2016learning]. For Places2, we introduce **places10k**, a 10,950 image subset of Places2 validation set. Similar to ctest10k, we make places10k class balanced with 30 images per class (Places2 dataset has 365 classes/categories in total).

**Metrics:** We report FID, Inception Score, Perceptual Distance, and Classification Accuracy. When computing FID scores, we use the full validation set as the reference distribution (50k images from ImageNet, 36.5k from Places2). For Perceptual Distance, we use Euclidean distance in the pool_3 feature space of pre-trained InceptionV1. We use EfficientNet-B0 top-1 accuracy for Classification Accuracy scores.

## Appendix D: Limitations

While Palette achieves strong results on several image-to-image translation tasks demonstrating the generality and versatility of the emerging diffusion models, there are many important limitations to address. Diffusion models generally require large number of refinement steps during sample generation (e.g. we use 1k refinement steps for Palette throughout the paper) resulting in significantly slower inference compared to GAN based models. This is an active area of research, and several new techniques [nichol2021improved; watson2021learning; jolicoeur2021gotta] have been proposed to reduce the number of refinement steps significantly. Palette's use of group-normalization and self-attention layers prevents its generalizability to arbitrary input image resolutions, limiting its practical usability. Techniques to adapt such models to arbitrary resolutions such as fine-tuning, or patch based inference can be an interesting direction of research. Like other generative models, Palette also suffers from implicit biases, which should be studied and mitigated before deployment in practice.
