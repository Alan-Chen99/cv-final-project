# High-Resolution Image Synthesis with Latent Diffusion Models

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Bjorn Ommer

Ludwig Maximilian University of Munich & IWR, Heidelberg University, Germany; Runway ML

https://github.com/CompVis/latent-diffusion

## Abstract

By decomposing the image formation process into a sequential application of denoising autoencoders, diffusion models (DMs) achieve state-of-the-art synthesis results on image data and beyond. Additionally, their formulation allows for a guiding mechanism to control the image generation process without retraining. However, since these models typically operate directly in pixel space, optimization of powerful DMs often consumes hundreds of GPU days and inference is expensive due to sequential evaluations. To enable DM training on limited computational resources while retaining their quality and flexibility, we apply them in the latent space of powerful pretrained autoencoders. In contrast to previous work, training diffusion models on such a representation allows for the first time to reach a near-optimal point between complexity reduction and detail preservation, greatly boosting visual fidelity. By introducing cross-attention layers into the model architecture, we turn diffusion models into powerful and flexible generators for general conditioning inputs such as text or bounding boxes and high-resolution synthesis becomes possible in a convolutional manner. Our latent diffusion models (LDMs) achieve new state-of-the-art scores for image inpainting and class-conditional image synthesis and highly competitive performance on various tasks, including text-to-image synthesis, unconditional image generation and super-resolution, while significantly reducing computational requirements compared to pixel-based DMs.

## 1 Introduction

Image synthesis is one of the computer vision fields with the most spectacular recent development, but also among those with the greatest computational demands. Especially high-resolution synthesis of complex, natural scenes is presently dominated by scaling up likelihood-based models, potentially containing billions of parameters in autoregressive (AR) transformers. In contrast, the promising results of GANs have been revealed to be mostly confined to data with comparably limited variability as their adversarial learning procedure does not easily scale to modeling complex, multi-modal distributions. Recently, diffusion models, which are built from a hierarchy of denoising autoencoders, have shown to achieve impressive results in image synthesis and beyond, and define the state-of-the-art in class-conditional image synthesis and super-resolution. Moreover, even unconditional DMs can readily be applied to tasks such as inpainting and colorization or stroke-based synthesis, in contrast to other types of generative models. Being likelihood-based models, they do not exhibit mode-collapse and training instabilities as GANs and, by heavily exploiting parameter sharing, they can model highly complex distributions of natural images without involving billions of parameters as in AR models.

### Democratizing High-Resolution Image Synthesis

DMs belong to the class of likelihood-based models, whose mode-covering behavior makes them prone to spend excessive amounts of capacity (and thus compute resources) on modeling imperceptible details of the data. Although the reweighted variational objective aims to address this by undersampling the initial denoising steps, DMs are still computationally demanding, since training and evaluating such a model requires repeated function evaluations (and gradient computations) in the high-dimensional space of RGB images. As an example, training the most powerful DMs often takes hundreds of GPU days (e.g. 150 - 1000 V100 days) and repeated evaluations on a noisy version of the input space render also inference expensive, so that producing 50k samples takes approximately 5 days on a single A100 GPU. This has two consequences for the research community and users in general: Firstly, training such a model requires massive computational resources only available to a small fraction of the field, and leaves a huge carbon footprint. Secondly, evaluating an already trained model is also expensive in time and memory, since the same model architecture must run sequentially for a large number of steps (e.g. 25 - 1000 steps).

To increase the accessibility of this powerful model class and at the same time reduce its significant resource consumption, a method is needed that reduces the computational complexity for both training and sampling. Reducing the computational demands of DMs without impairing their performance is, therefore, key to enhance their accessibility.

### Departure to Latent Space

Our approach starts with the analysis of already trained diffusion models in pixel space. As with any likelihood-based model, learning can be roughly divided into two stages: First is a *perceptual compression* stage which removes high-frequency details but still learns little semantic variation. In the second stage, the actual generative model learns the semantic and conceptual composition of the data (*semantic compression*). We thus aim to first find a *perceptually equivalent, but computationally more suitable space*, in which we will train diffusion models for high-resolution image synthesis.

Following common practice, we separate training into two distinct phases: First, we train an autoencoder which provides a lower-dimensional (and thereby efficient) representational space which is perceptually equivalent to the data space. Importantly, and in contrast to previous work, we do not need to rely on excessive spatial compression, as we train DMs in the learned latent space, which exhibits better scaling properties with respect to the spatial dimensionality. The reduced complexity also provides efficient image generation from the latent space with a single network pass. We dub the resulting model class *Latent Diffusion Models* (LDMs).

A notable advantage of this approach is that we need to train the universal autoencoding stage only once and can therefore reuse it for multiple DM trainings or to explore possibly completely different tasks. This enables efficient exploration of a large number of diffusion models for various image-to-image and text-to-image tasks. For the latter, we design an architecture that connects transformers to the DM's UNet backbone and enables arbitrary types of token-based conditioning mechanisms, see Sec. 3.3.

In sum, our work makes the following **contributions**:

(i) In contrast to purely transformer-based approaches, our method scales more gracefully to higher dimensional data and can thus (a) work on a compression level which provides more faithful and detailed reconstructions than previous work and (b) can be efficiently applied to high-resolution synthesis of megapixel images.

(ii) We achieve competitive performance on multiple tasks (unconditional image synthesis, inpainting, stochastic super-resolution) and datasets while significantly lowering computational costs. Compared to pixel-based diffusion approaches, we also significantly decrease inference costs.

(iii) We show that, in contrast to previous work which learns both an encoder/decoder architecture and a score-based prior simultaneously, our approach does not require a delicate weighting of reconstruction and generative abilities. This ensures extremely faithful reconstructions and requires very little regularization of the latent space.

(iv) We find that for densely conditioned tasks such as super-resolution, inpainting and semantic synthesis, our model can be applied in a convolutional fashion and render large, consistent images of ~1024^2 px.

(v) Moreover, we design a general-purpose conditioning mechanism based on cross-attention, enabling multi-modal training. We use it to train class-conditional, text-to-image and layout-to-image models.

(vi) Finally, we release pretrained latent diffusion and autoencoding models at https://github.com/CompVis/latent-diffusion which might be reusable for various tasks besides training of DMs.

## 2 Related Work

**Generative Models for Image Synthesis.** The high dimensional nature of images presents distinct challenges to generative modeling. Generative Adversarial Networks (GAN) allow for efficient sampling of high resolution images with good perceptual quality, but are difficult to optimize and struggle to capture the full data distribution. In contrast, likelihood-based methods emphasize good density estimation which renders optimization more well-behaved. Variational autoencoders (VAE) and flow-based models enable efficient synthesis of high resolution images, but sample quality is not on par with GANs. While autoregressive models (ARM) achieve strong performance in density estimation, computationally demanding architectures and a sequential sampling process limit them to low resolution images. Because pixel based representations of images contain barely perceptible, high-frequency details, maximum-likelihood training spends a disproportionate amount of capacity on modeling them, resulting in long training times. To scale to higher resolutions, several two-stage approaches use ARMs to model a compressed latent image space instead of raw pixels.

Recently, **Diffusion Probabilistic Models** (DM), have achieved state-of-the-art results in density estimation as well as in sample quality. The generative power of these models stems from a natural fit to the inductive biases of image-like data when their underlying neural backbone is implemented as a UNet. The best synthesis quality is usually achieved when a reweighted objective is used for training. In this case, the DM corresponds to a lossy compressor and allows to trade image quality for compression capabilities. Evaluating and optimizing these models in pixel space, however, has the downside of low inference speed and very high training costs. While the former can be partially addressed by advanced sampling strategies and hierarchical approaches, training on high-resolution image data always requires to calculate expensive gradients. We address both drawbacks with our proposed *LDMs*, which work on a compressed latent space of lower dimensionality. This renders training computationally cheaper and speeds up inference with almost no reduction in synthesis quality.

**Two-Stage Image Synthesis.** To mitigate the shortcomings of individual generative approaches, a lot of research has gone into combining the strengths of different methods into more efficient and performant models via a two stage approach. VQ-VAEs use autoregressive models to learn an expressive prior over a discretized latent space. This approach has been extended to text-to-image generation by learning a joint distribution over discretized image and text representations. More generally, conditionally invertible networks provide a generic transfer between latent spaces of diverse domains. Different from VQ-VAEs, VQGANs employ a first stage with an adversarial and perceptual objective to scale autoregressive transformers to larger images. However, the high compression rates required for feasible ARM training, which introduces billions of trainable parameters, limit the overall performance of such approaches and less compression comes at the price of high computational cost. Our work prevents such trade-offs, as our proposed *LDMs* scale more gently to higher dimensional latent spaces due to their convolutional backbone. Thus, we are free to choose the level of compression which optimally mediates between learning a powerful first stage, without leaving too much perceptual compression up to the generative diffusion model while guaranteeing high-fidelity reconstructions.

While approaches to jointly or separately learn an encoding/decoding model together with a score-based prior exist, the former still require a difficult weighting between reconstruction and generative capabilities and are outperformed by our approach (Sec. 4), and the latter focus on highly structured images such as human faces.

## 3 Method

To lower the computational demands of training diffusion models towards high-resolution image synthesis, we observe that although diffusion models allow to ignore perceptually irrelevant details by undersampling the corresponding loss terms, they still require costly function evaluations in pixel space, which causes huge demands in computation time and energy resources.

We propose to circumvent this drawback by introducing an explicit separation of the compressive from the generative learning phase. To achieve this, we utilize an autoencoding model which learns a space that is perceptually equivalent to the image space, but offers significantly reduced computational complexity.

Such an approach offers several advantages: (i) By leaving the high-dimensional image space, we obtain DMs which are computationally much more efficient because sampling is performed on a low-dimensional space. (ii) We exploit the inductive bias of DMs inherited from their UNet architecture, which makes them particularly effective for data with spatial structure and therefore alleviates the need for aggressive, quality-reducing compression levels as required by previous approaches. (iii) Finally, we obtain general-purpose compression models whose latent space can be used to train multiple generative models and which can also be utilized for other downstream applications such as single-image CLIP-guided synthesis.

### 3.1 Perceptual Image Compression

Our perceptual compression model is based on previous work and consists of an autoencoder trained by combination of a perceptual loss and a patch-based adversarial objective. This ensures that the reconstructions are confined to the image manifold by enforcing local realism and avoids blurriness introduced by relying solely on pixel-space losses such as L2 or L1 objectives.

More precisely, given an image x in R^(H x W x 3) in RGB space, the encoder E encodes x into a latent representation z = E(x), and the decoder D reconstructs the image from the latent, giving x_tilde = D(z) = D(E(x)), where z in R^(h x w x c). Importantly, the encoder *downsamples* the image by a factor f = H/h = W/w, and we investigate different downsampling factors f = 2^m, with m in N.

In order to avoid arbitrarily high-variance latent spaces, we experiment with two different kinds of regularizations. The first variant, *KL-reg.*, imposes a slight KL-penalty towards a standard normal on the learned latent, similar to a VAE, whereas *VQ-reg.* uses a vector quantization layer within the decoder. This model can be interpreted as a VQGAN but with the quantization layer absorbed by the decoder. Because our subsequent DM is designed to work with the two-dimensional structure of our learned latent space z = E(x), we can use relatively mild compression rates and achieve very good reconstructions. This is in contrast to previous works, which relied on an arbitrary 1D ordering of the learned space z to model its distribution autoregressively and thereby ignored much of the inherent structure of z. Hence, our compression model preserves details of x better. The full objective and training details can be found in the supplement.

### 3.2 Latent Diffusion Models

**Diffusion Models** are probabilistic models designed to learn a data distribution p(x) by gradually denoising a normally distributed variable, which corresponds to learning the reverse process of a fixed Markov Chain of length T. For image synthesis, the most successful models rely on a reweighted variant of the variational lower bound on p(x), which mirrors denoising score-matching. These models can be interpreted as an equally weighted sequence of denoising autoencoders epsilon_theta(x_t, t); t = 1...T, which are trained to predict a denoised variant of their input x_t, where x_t is a noisy version of the input x. The corresponding objective can be simplified to:

$$L_{DM} = E_{x, \epsilon \sim N(0,1), t} [ \| \epsilon - \epsilon_\theta(x_t, t) \|_2^2 ]$$

with t uniformly sampled from {1, ..., T}.

**Generative Modeling of Latent Representations.** With our trained perceptual compression models consisting of E and D, we now have access to an efficient, low-dimensional latent space in which high-frequency, imperceptible details are abstracted away. Compared to the high-dimensional pixel space, this space is more suitable for likelihood-based generative models, as they can now (i) focus on the important, semantic bits of the data and (ii) train in a lower dimensional, computationally much more efficient space.

Unlike previous work that relied on autoregressive, attention-based transformer models in a highly compressed, discrete latent space, we can take advantage of image-specific inductive biases that our model offers. This includes the ability to build the underlying UNet primarily from 2D convolutional layers, and further focusing the objective on the perceptually most relevant bits using the reweighted bound, which now reads:

$$L_{LDM} := E_{E(x), \epsilon \sim N(0,1), t} [ \| \epsilon - \epsilon_\theta(z_t, t) \|_2^2 ]$$

The neural backbone epsilon_theta(., t) of our model is realized as a time-conditional UNet. Since the forward process is fixed, z_t can be efficiently obtained from E during training, and samples from p(z) can be decoded to image space with a single pass through D.

### 3.3 Conditioning Mechanisms

Similar to other types of generative models, diffusion models are in principle capable of modeling conditional distributions of the form p(z | y). This can be implemented with a conditional denoising autoencoder epsilon_theta(z_t, t, y) and paves the way to controlling the synthesis process through inputs y such as text, semantic maps or other image-to-image translation tasks.

In the context of image synthesis, however, combining the generative power of DMs with other types of conditionings beyond class-labels or blurred variants of the input image is so far an under-explored area of research.

We turn DMs into more flexible conditional image generators by augmenting their underlying UNet backbone with the cross-attention mechanism, which is effective for learning attention-based models of various input modalities. To pre-process y from various modalities (such as language prompts) we introduce a domain specific encoder tau_theta that projects y to an intermediate representation tau_theta(y) in R^(M x d_tau), which is then mapped to the intermediate layers of the UNet via a cross-attention layer implementing Attention(Q, K, V) = softmax(QK^T / sqrt(d)) . V, with:

$$Q = W_Q^{(i)} \cdot \varphi_i(z_t), \quad K = W_K^{(i)} \cdot \tau_\theta(y), \quad V = W_V^{(i)} \cdot \tau_\theta(y)$$

Here, phi_i(z_t) in R^(N x d_epsilon^i) denotes a (flattened) intermediate representation of the UNet implementing epsilon_theta and W_V^(i) in R^(d x d_epsilon^i), W_Q^(i) in R^(d x d_tau) & W_K^(i) in R^(d x d_tau) are learnable projection matrices.

Based on image-conditioning pairs, we then learn the conditional LDM via:

$$L_{LDM} := E_{E(x), y, \epsilon \sim N(0,1), t} [ \| \epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y)) \|_2^2 ]$$

where both tau_theta and epsilon_theta are jointly optimized via this equation. This conditioning mechanism is flexible as tau_theta can be parameterized with domain-specific experts, e.g. (unmasked) transformers when y are text prompts.

## 4 Experiments

*LDMs* provide means to flexible and computationally tractable diffusion based image synthesis of various image modalities, which we empirically show in the following. Firstly, however, we analyze the gains of our models compared to pixel-based diffusion models in both training and inference. Interestingly, we find that *LDMs* trained in *VQ*-regularized latent spaces sometimes achieve better sample quality, even though the reconstruction capabilities of *VQ*-regularized first stage models slightly fall behind those of their continuous counterparts.

### 4.1 On Perceptual Compression Tradeoffs

This section analyzes the behavior of our LDMs with different downsampling factors f in {1, 2, 4, 8, 16, 32} (abbreviated as *LDM-f*, where *LDM-1* corresponds to pixel-based DMs). To obtain a comparable test-field, we fix the computational resources to a single NVIDIA A100 for all experiments in this section and train all models for the same number of steps and with the same number of parameters.

Sample quality as a function of training progress for 2M steps of class-conditional models on the ImageNet dataset shows that, i) small downsampling factors for *LDM-{1,2}* result in slow training progress, whereas ii) overly large values of f cause stagnating fidelity after comparably few training steps. We attribute this to i) leaving most of perceptual compression to the diffusion model and ii) too strong first stage compression resulting in information loss and thus limiting the achievable quality. *LDM-{4-16}* strike a good balance between efficiency and perceptually faithful results, which manifests in a significant FID gap of 38 between pixel-based diffusion (*LDM-1*) and *LDM-8* after 2M training steps.

Comparing models trained on CelebA-HQ and ImageNet in terms of sampling speed for different numbers of denoising steps with the DDIM sampler and plotting against FID-scores, *LDM-{4-8}* outperform models with unsuitable ratios of perceptual and conceptual compression. Especially compared to pixel-based *LDM-1*, they achieve much lower FID scores while simultaneously significantly increasing sample throughput. In summary, *LDM-4* and *-8* offer the best conditions for achieving high-quality synthesis results.

#### Evaluation Metrics for Unconditional Image Synthesis

| | CelebA-HQ 256x256 | | | | | FFHQ 256x256 | | | |
|---|---|---|---|---|---|---|---|---|---|
| **Method** | FID | Prec. | Recall | | | **Method** | FID | Prec. | Recall |
| DC-VAE | 15.8 | - | - | | | ImageBART | 9.57 | - | - |
| VQGAN+T. (k=400) | 10.2 | - | - | | | U-Net GAN (+aug) | 10.9 (7.6) | - | - |
| PGGAN | 8.0 | - | - | | | UDM | 5.54 | - | - |
| LSGM | 7.22 | - | - | | | StyleGAN | 4.16 | 0.71 | 0.46 |
| UDM | 7.16 | - | - | | | ProjectedGAN | **3.08** | 0.65 | 0.46 |
| *LDM-4* (ours, 500-s) | **5.11** | 0.72 | 0.49 | | | *LDM-4* (ours, 200-s) | 4.98 | **0.73** | **0.50** |

| | LSUN-Churches 256x256 | | | | | LSUN-Bedrooms 256x256 | | | |
|---|---|---|---|---|---|---|---|---|---|
| **Method** | FID | Prec. | Recall | | | **Method** | FID | Prec. | Recall |
| DDPM | 7.89 | - | - | | | ImageBART | 5.51 | - | - |
| ImageBART | 7.32 | - | - | | | DDPM | 4.9 | - | - |
| PGGAN | 6.42 | - | - | | | UDM | 4.57 | - | - |
| StyleGAN | 4.21 | - | - | | | StyleGAN | 2.35 | 0.59 | 0.48 |
| StyleGAN2 | 3.86 | - | - | | | ADM | 1.90 | **0.66** | **0.51** |
| ProjectedGAN | **1.59** | 0.61 | 0.44 | | | ProjectedGAN | **1.52** | 0.61 | 0.34 |
| *LDM-8* (ours, 200-s) | 4.02 | **0.64** | **0.52** | | | *LDM-4* (ours, 200-s) | 2.95 | **0.66** | 0.48 |

### 4.2 Image Generation with Latent Diffusion

We train unconditional models of 256^2 images on CelebA-HQ, FFHQ, LSUN-Churches and LSUN-Bedrooms and evaluate the i) sample quality and ii) their coverage of the data manifold using FID and Precision-and-Recall. On CelebA-HQ, we report a new state-of-the-art FID of 5.11, outperforming previous likelihood-based models as well as GANs. We also outperform LSGM where a latent diffusion model is trained jointly together with the first stage. In contrast, we train diffusion models in a fixed space and avoid the difficulty of weighing reconstruction quality against learning the prior over the latent space.

We outperform prior diffusion based approaches on all but the LSUN-Bedrooms dataset, where our score is close to ADM, despite utilizing half its parameters and requiring 4-times less train resources. Moreover, *LDMs* consistently improve upon GAN-based methods in Precision and Recall, thus confirming the advantages of their mode-covering likelihood-based training objective over adversarial approaches.

### 4.3 Conditional Latent Diffusion

#### 4.3.1 Transformer Encoders for LDMs

By introducing cross-attention based conditioning into LDMs we open them up for various conditioning modalities previously unexplored for diffusion models.

For **text-to-image** image modeling, we train a 1.45B parameter *KL*-regularized *LDM* conditioned on language prompts on LAION-400M. We employ the BERT-tokenizer and implement tau_theta as a transformer to infer a latent code which is mapped into the UNet via (multi-head) cross-attention (Sec. 3.3). This combination of domain specific experts for learning a language representation and visual synthesis results in a powerful model, which generalizes well to complex, user-defined text prompts. For quantitative analysis, we follow prior work and evaluate text-to-image generation on the MS-COCO validation set, where our model improves upon powerful AR and GAN-based methods. We note that applying classifier-free diffusion guidance greatly boosts sample quality, such that the guided *LDM-KL-8-G* is on par with the recent state-of-the-art AR and diffusion models for text-to-image synthesis, while substantially reducing parameter count.

#### Text-Conditional Image Synthesis

| **Method** | FID | IS | N_params | |
|---|---|---|---|---|
| CogView | 27.10 | 18.20 | 4B | self-ranking, rejection rate 0.017 |
| LAFITE | 26.94 | 26.02 | 75M | |
| GLIDE | 12.24 | - | 6B | 277 DDIM steps, c.f.g. s=3 |
| Make-A-Scene | **11.84** | - | 4B | c.f.g for AR models s=5 |
| *LDM-KL-8* | 23.31 | 20.03 | 1.45B | 250 DDIM steps |
| *LDM-KL-8-G* | 12.63 | **30.29** | 1.45B | 250 DDIM steps, c.f.g. s=1.5 |

To further analyze the flexibility of the cross-attention based conditioning mechanism we also train models to synthesize images based on **semantic layouts** on OpenImages, and finetune on COCO.

Lastly, following prior work, we evaluate our best-performing **class-conditional** ImageNet models with f in {4, 8}. Here we outperform the state of the art diffusion model ADM while significantly reducing computational requirements and parameter count.

#### Class-Conditional ImageNet

| **Method** | FID | IS | Precision | Recall | N_params | |
|---|---|---|---|---|---|---|
| BigGAN-deep | 6.95 | 203.6 | **0.87** | 0.28 | 340M | - |
| ADM | 10.94 | 100.98 | 0.69 | **0.63** | 554M | 250 DDIM steps |
| ADM-G | 4.59 | 186.7 | 0.82 | 0.52 | 608M | 250 DDIM steps |
| *LDM-4* (ours) | 10.56 | 103.49 | 0.71 | 0.62 | 400M | 250 DDIM steps |
| *LDM-4-G* (ours) | **3.60** | **247.67** | **0.87** | 0.48 | 400M | 250 steps, c.f.g, s=1.5 |

#### 4.3.2 Convolutional Sampling Beyond 256^2

By concatenating spatially aligned conditioning information to the input of epsilon_theta, *LDMs* can serve as efficient general-purpose image-to-image translation models. We use this to train models for semantic synthesis, super-resolution (Sec. 4.4) and inpainting (Sec. 4.5). For semantic synthesis, we use images of landscapes paired with semantic maps and concatenate downsampled versions of the semantic maps with the latent image representation of a f=4 model (VQ-reg.). We train on an input resolution of 256^2 (crops from 384^2) but find that our model generalizes to larger resolutions and can generate images up to the megapixel regime when evaluated in a convolutional manner.

We exploit this behavior to also apply the super-resolution models and the inpainting models to generate large images between 512^2 and 1024^2. For this application, the signal-to-noise ratio (induced by the scale of the latent space) significantly affects the results. The latter, in combination with classifier-free guidance, also enables the direct synthesis of >256^2 images for the text-conditional *LDM-KL-8-G*.

### 4.4 Super-Resolution with Latent Diffusion

LDMs can be efficiently trained for super-resolution by directly conditioning on low-resolution images via concatenation. In a first experiment, we follow SR3 and fix the image degradation to a bicubic interpolation with 4x-downsampling and train on ImageNet following SR3's data processing pipeline. We use the f=4 autoencoding model pretrained on OpenImages (VQ-reg.) and concatenate the low-resolution conditioning y and the inputs to the UNet, i.e. tau_theta is the identity. Our qualitative and quantitative results show competitive performance and LDM-SR outperforms SR3 in FID while SR3 has a better IS.

#### Super-Resolution Results (4x upscaling on ImageNet-Val, 256^2)

| **Method** | FID | IS | PSNR | SSIM | N_params | Samples/s |
|---|---|---|---|---|---|---|
| Image Regression | 15.2 | 121.1 | **27.9** | **0.801** | 625M | N/A |
| SR3 | 5.2 | **180.1** | 26.4 | 0.762 | 625M | N/A |
| *LDM-4* (ours, 100 steps) | 2.8/4.8 | 166.3 | 24.4 | 0.69 | **169M** | 4.62 |
| *LDM-4* (ours, big, 100 steps) | **2.4**/**4.3** | 174.9 | 24.7 | 0.71 | 552M | 4.5 |
| *LDM-4* (ours, 50 steps, guiding) | 4.4/6.4 | 153.7 | 25.8 | 0.74 | 184M | 0.38 |

A simple image regression model achieves the highest PSNR and SSIM scores; however these metrics do not align well with human perception and favor blurriness over imperfectly aligned high frequency details. Further, we conduct a user study comparing the pixel-baseline with LDM-SR. The results affirm the good performance of LDM-SR. PSNR and SSIM can be pushed by using a post-hoc guiding mechanism and we implement this *image-based guider* via a perceptual loss.

#### User Study

| | SR on ImageNet | | | Inpainting on Places | |
|---|---|---|---|---|---|
| **User Study** | Pixel-DM (f1) | *LDM-4* | | LAMA | *LDM-4* |
| **Task 1:** Preference vs GT | 16.0% | **30.4%** | | 13.6% | **21.0%** |
| **Task 2:** Preference Score | 29.4% | **70.6%** | | 31.9% | **68.1%** |

Since the bicubic degradation process does not generalize well to images which do not follow this pre-processing, we also train a generic model, *LDM-BSR*, by using more diverse degradation.

### 4.5 Inpainting with Latent Diffusion

Inpainting is the task of filling masked regions of an image with new content either because parts of the image are corrupted or to replace existing but undesired content within the image. We evaluate how our general approach for conditional image generation compares to more specialized, state-of-the-art approaches for this task. Our evaluation follows the protocol of LaMa, a recent inpainting model that introduces a specialized architecture relying on Fast Fourier Convolutions.

We first analyze the effect of different design choices for the first stage. In particular, we compare the inpainting efficiency of *LDM-1* (i.e. a pixel-based conditional DM) with *LDM-4*, for both *KL* and *VQ* regularizations, as well as *VQ-LDM-4* without any attention in the first stage (which reduces GPU memory for decoding at high resolutions). For comparability, we fix the number of parameters for all models.

#### Inpainting Efficiency

| **Model** (reg.-type) | train throughput samples/sec | sampling @256 | sampling @512 | train+val hours/epoch | FID@2k epoch 6 |
|---|---|---|---|---|---|
| *LDM-1* (no first stage) | 0.11 | 0.26 | 0.07 | 20.66 | 24.74 |
| *LDM-4* (*KL*, w/ attn) | 0.32 | 0.97 | 0.34 | 7.66 | 15.21 |
| *LDM-4* (*VQ*, w/ attn) | 0.33 | 0.97 | 0.34 | 7.04 | 14.99 |
| *LDM-4* (*VQ*, w/o attn) | 0.35 | 0.99 | 0.36 | 6.66 | 15.95 |

Overall, we observe a speed-up of at least 2.7x between pixel- and latent-based diffusion models while improving FID scores by a factor of at least 1.6x.

#### Inpainting Comparison (30k crops of 512x512 from Places)

| | 40-50% masked | | All samples | |
|---|---|---|---|---|
| **Method** | FID | LPIPS | FID | LPIPS |
| *LDM-4* (ours, big, w/ ft) | **9.39** | 0.246 | **1.50** | 0.137 |
| *LDM-4* (ours, big, w/o ft) | 12.89 | 0.257 | 2.40 | 0.142 |
| *LDM-4* (ours, w/ attn) | 11.87 | 0.257 | 2.15 | 0.144 |
| *LDM-4* (ours, w/o attn) | 12.60 | 0.259 | 2.37 | 0.145 |
| LaMa | 12.00 | **0.24** | 2.21 | **0.14** |
| CoModGAN | 10.40 | 0.26 | 1.82 | 0.15 |
| RegionWise | 21.30 | 0.27 | 4.75 | 0.15 |
| DeepFill v2 | 22.10 | 0.28 | 5.20 | 0.16 |
| EdgeConnect | 30.50 | 0.28 | 8.37 | 0.16 |

The comparison with other inpainting approaches shows that our model with attention improves the overall image quality as measured by FID over that of LaMa. LPIPS between the unmasked images and our samples is slightly higher than that of LaMa. We attribute this to LaMa only producing a single result which tends to recover more of an average image compared to the diverse results produced by our LDM. Additionally in a user study human subjects favor our results over those of LaMa.

Based on these initial results, we also trained a larger diffusion model (*big* in the table) in the latent space of the *VQ*-regularized first stage without attention. Following ADM, the UNet of this diffusion model uses attention layers on three levels of its feature hierarchy, the BigGAN residual block for up- and downsampling and has 387M parameters instead of 215M. After fine-tuning the model for half an epoch at resolution 512^2, it sets a new state of the art FID on image inpainting.

## 5 Limitations & Societal Impact

### Limitations

While LDMs significantly reduce computational requirements compared to pixel-based approaches, their sequential sampling process is still slower than that of GANs. Moreover, the use of LDMs can be questionable when high precision is required: although the loss of image quality is very small in our f=4 autoencoding models, their reconstruction capability can become a bottleneck for tasks that require fine-grained accuracy in pixel space. We assume that our superresolution models are already somewhat limited in this respect.

### Societal Impact

Generative models for media like imagery are a double-edged sword: On the one hand, they enable various creative applications, and in particular approaches like ours that reduce the cost of training and inference have the potential to facilitate access to this technology and democratize its exploration. On the other hand, it also means that it becomes easier to create and disseminate manipulated data or spread misinformation and spam. In particular, the deliberate manipulation of images ("deep fakes") is a common problem in this context, and women in particular are disproportionately affected by it.

Generative models can also reveal their training data, which is of great concern when the data contain sensitive or personal information and were collected without explicit consent. However, the extent to which this also applies to DMs of images is not yet fully understood.

Finally, deep learning modules tend to reproduce or exacerbate biases that are already present in the data. While diffusion models achieve better coverage of the data distribution than e.g. GAN-based approaches, the extent to which our two-stage approach that combines adversarial training and a likelihood-based objective misrepresents the data remains an important research question.

## 6 Conclusion

We have presented latent diffusion models, a simple and efficient way to significantly improve both the training and sampling efficiency of denoising diffusion models without degrading their quality. Based on this and our cross-attention conditioning mechanism, our experiments could demonstrate favorable results compared to state-of-the-art methods across a wide range of conditional image synthesis tasks without task-specific architectures.

## Appendix

### A Changelog

Changes between v2 and v1:
- Updated results on text-to-image synthesis obtained by training a new, larger model (1.45B parameters), including comparison to recent competing methods.
- Updated results on class-conditional synthesis on ImageNet obtained by retraining with a larger batch size. Both the updated text-to-image and class-conditional models now use classifier-free guidance.
- Conducted a user study for inpainting and super-resolution models.
- Added additional figures.

### B Detailed Information on Denoising Diffusion Models

Diffusion models can be specified in terms of a signal-to-noise ratio SNR(t) = alpha_t^2 / sigma_t^2 consisting of sequences (alpha_t) and (sigma_t) which, starting from a data sample x_0, define a forward diffusion process q as:

$$q(x_t | x_0) = N(x_t | \alpha_t x_0, \sigma_t^2 I)$$

with the Markov structure for s < t:

$$q(x_t | x_s) = N(x_t | \alpha_{t|s} x_s, \sigma_{t|s}^2 I)$$

$$\alpha_{t|s} = \alpha_t / \alpha_s$$

$$\sigma_{t|s}^2 = \sigma_t^2 - \alpha_{t|s}^2 \sigma_s^2$$

Denoising diffusion models are generative models p(x_0) which revert this process with a similar Markov structure running backward in time:

$$p(x_0) = \int_z p(x_T) \prod_{t=1}^T p(x_{t-1} | x_t)$$

The evidence lower bound (ELBO) associated with this model then decomposes over the discrete time steps as:

$$-\log p(x_0) \le KL(q(x_T | x_0) | p(x_T)) + \sum_{t=1}^T E_{q(x_t|x_0)} KL(q(x_{t-1} | x_t, x_0) | p(x_{t-1} | x_t))$$

A common choice to parameterize p(x_{t-1} | x_t) is to specify it in terms of the true posterior q(x_{t-1} | x_t, x_0) but with the unknown x_0 replaced by an estimate x_theta(x_t, t):

$$p(x_{t-1} | x_t) := q(x_{t-1} | x_t, x_\theta(x_t, t)) = N(x_{t-1} | \mu_\theta(x_t, t), \sigma_{t|t-1}^2 \frac{\sigma_{t-1}^2}{\sigma_t^2} I)$$

where the mean can be expressed as:

$$\mu_\theta(x_t, t) = \frac{\alpha_{t|t-1} \sigma_{t-1}^2}{\sigma_t^2} x_t + \frac{\alpha_{t-1} \sigma_{t|t-1}^2}{\sigma_t^2} x_\theta(x_t, t)$$

The sum of the ELBO simplifies to:

$$\sum_{t=1}^T E_{N(\epsilon|0,I)} \frac{1}{2}(SNR(t-1) - SNR(t)) \| x_0 - x_\theta(\alpha_t x_0 + \sigma_t \epsilon, t) \|^2$$

Following Ho et al., the reparameterization epsilon_theta(x_t, t) = (x_t - alpha_t x_theta(x_t, t)) / sigma_t expresses the reconstruction term as a denoising objective, and the reweighting which assigns each term the same weight results in the loss of Eq. 1.

### C Image Guiding Mechanisms

An intriguing feature of diffusion models is that unconditional models can be conditioned at test-time. The guiding algorithm for an epsilon-parameterized model with fixed variance reads:

$$\hat{\epsilon} \leftarrow \epsilon_\theta(z_t, t) + \sqrt{1 - \alpha_t^2} \nabla_{z_t} \log p_\Phi(y | z_t)$$

This can be interpreted as an update correcting the "score" epsilon_theta with a conditional distribution log p_Phi(y | z_t).

We re-interpret the guiding distribution p_Phi(y | T(D(z_0(z_t)))) as a general purpose image-to-image translation task given a target image y, where T can be any differentiable transformation adopted to the task at hand, such as the identity, a downsampling operation or similar. As an example, we can assume a Gaussian guider with fixed variance sigma^2 = 1, such that:

$$\log p_\Phi(y | z_t) = -\frac{1}{2} \| y - T(D(z_0(z_t))) \|_2^2$$

becomes an L2 regression objective.

### D Additional Results

#### D.1 Choosing the Signal-to-Noise Ratio for High-Resolution Synthesis

The signal-to-noise ratio induced by the variance of the latent space (i.e. Var(z)/sigma_t^2) significantly affects the results for convolutional sampling. When training a LDM directly in the latent space of a KL-regularized model, this ratio is very high, such that the model allocates a lot of semantic detail early on in the reverse denoising process. In contrast, when rescaling the latent space by the component-wise standard deviation of the latents, the SNR is decreased. The VQ-regularized space has a variance close to 1, such that it does not have to be rescaled.

#### D.2 Full List of all First Stage Models

Complete autoencoder zoo trained on OpenImages, evaluated on ImageNet-Val:

| f | |Z| | c | R-FID | R-IS | PSNR | PSIM | SSIM |
|---|---|---|---|---|---|---|---|
| 16 *VQGAN* | 16384 | 256 | 4.98 | -- | 19.9 | 1.83 | 0.51 |
| 16 *VQGAN* | 1024 | 256 | 7.94 | -- | 19.4 | 1.98 | 0.50 |
| 8 *DALL-E* | 8192 | - | 32.01 | -- | 22.8 | 1.95 | 0.73 |
| 32 | 16384 | 16 | 31.83 | 40.40 | 17.45 | 2.58 | 0.41 |
| 16 | 16384 | 8 | 5.15 | 144.55 | 20.83 | 1.73 | 0.54 |
| 8 | 16384 | 4 | 1.14 | 201.92 | 23.07 | 1.17 | 0.65 |
| 8 | 256 | 4 | 1.49 | 194.20 | 22.35 | 1.26 | 0.62 |
| 4 | 8192 | 3 | 0.58 | 224.78 | 27.43 | 0.53 | 0.82 |
| 4 (no attn) | 8192 | 3 | 1.06 | 221.94 | 25.21 | 0.72 | 0.76 |
| 4 | 256 | 3 | 0.47 | 223.81 | 26.43 | 0.62 | 0.80 |
| 2 | 2048 | 2 | 0.16 | 232.75 | 30.85 | 0.27 | 0.91 |
| 2 | 64 | 2 | 0.40 | 226.62 | 29.13 | 0.38 | 0.90 |
| 32 | KL | 64 | 2.04 | 189.53 | 22.27 | 1.41 | 0.61 |
| 32 | KL | 16 | 7.3 | 132.75 | 20.38 | 1.88 | 0.53 |
| 16 | KL | 16 | 0.87 | 210.31 | 24.08 | 1.07 | 0.68 |
| 16 | KL | 8 | 2.63 | 178.68 | 21.94 | 1.49 | 0.59 |
| 8 | KL | 4 | 0.90 | 209.90 | 24.19 | 1.02 | 0.69 |
| 4 | KL | 3 | 0.27 | 227.57 | 27.53 | 0.55 | 0.82 |
| 2 | KL | 2 | 0.086 | 232.66 | 32.47 | 0.20 | 0.93 |

#### D.3 Layout-to-Image Synthesis

| | COCO 256x256 | OpenImages 256x256 | OpenImages 512x512 |
|---|---|---|---|
| **Method** | FID | FID | FID |
| LostGAN-V2 | 42.55 | - | - |
| OC-GAN | 41.65 | - | - |
| SPADE | 41.11 | - | - |
| VQGAN+T | 56.58 | 45.33 | 48.11 |
| *LDM-8* (100 steps, ours) | 42.06 | - | - |
| *LDM-4* (200 steps, ours) | **40.91** | **32.02** | **35.80** |

Our COCO model reaches the performance of recent state-of-the-art models in layout-to-image synthesis when following their training and evaluation protocol. When finetuning from the OpenImages model, we surpass these works. Our OpenImages model surpasses prior results by a margin of nearly 11 in terms of FID.

#### D.4 Class-Conditional Image Synthesis on ImageNet (Extended)

| **Method** | FID | IS | Precision | Recall | N_params | |
|---|---|---|---|---|---|---|
| SR3 | 11.30 | - | - | - | 625M | - |
| ImageBART | 21.19 | - | - | - | 3.5B | - |
| ImageBART | 7.44 | - | - | - | 3.5B | 0.05 acc. rate |
| VQGAN+T | 17.04 | 70.6 | - | - | 1.3B | - |
| VQGAN+T | 5.88 | **304.8** | - | - | 1.3B | 0.05 acc. rate |
| BigGAN-deep | 6.95 | 203.6 | **0.87** | 0.28 | 340M | - |
| ADM | 10.94 | 100.98 | 0.69 | **0.63** | 554M | 250 DDIM steps |
| ADM-G | 4.59 | 186.7 | 0.82 | 0.52 | 608M | 250 DDIM steps |
| ADM-G, ADM-U | 3.85 | 221.72 | 0.84 | 0.53 | n/a | 2x250 DDIM steps |
| CDM | 4.88 | 158.71 | - | - | n/a | 2x100 DDIM steps |
| *LDM-4* (ours) | 10.56 | 103.49 | 0.71 | 0.62 | 400M | 250 DDIM steps |
| *LDM-4-G* (ours) | **3.60** | 247.67 | **0.87** | 0.48 | 400M | 250 DDIM steps, c.f.g. s=1.5 |

LDM-8 requires significantly fewer parameters and compute requirements to achieve very competitive performance. Similar to previous work, we can further boost the performance by training a classifier on each noise scale and guiding with it. Unlike the pixel-based methods, this classifier is trained very cheaply in latent space.

#### D.5 Super-Resolution (Extended)

| **Method** | FID | IS | PSNR | SSIM |
|---|---|---|---|---|
| Image Regression | 15.2 | 121.1 | **27.9** | **0.801** |
| SR3 | 5.2 | **180.1** | 26.4 | 0.762 |
| *LDM-4* (ours, 100 steps) | **2.8**/4.8 | 166.3 | 24.4 | 0.69 |
| *LDM-4* (ours, 50 steps, guiding) | 4.4/6.4 | 153.7 | 25.8 | 0.74 |
| *LDM-4* (ours, 100 steps, guiding) | 4.4/6.4 | 154.1 | 25.7 | 0.73 |
| *LDM-4* (ours, 100 steps, +15 ep.) | **2.6**/4.6 | 169.76 | 24.4 | 0.69 |
| Pixel-DM (100 steps, +15 ep.) | 5.1/7.1 | 163.06 | 24.1 | 0.59 |

The results demonstrate that LDM achieves better performance while allowing for significantly faster sampling.

**LDM-BSR: General Purpose SR Model via Diverse Image Degradation.** To evaluate generalization of our LDM-SR, we apply it both on synthetic LDM samples from a class-conditional ImageNet model and images crawled from the internet. LDM-SR, trained only with a bicubically downsampled conditioning, does not generalize well to images which do not follow this pre-processing. Hence, to obtain a superresolution model for a wide range of real world images containing complex superpositions of camera noise, compression artifacts, blur and interpolations, we replace the bicubic downsampling operation with the degradation pipeline from BSR-degradation which applies JPEG compression noise, camera sensor noise, different image interpolations for downsampling, Gaussian blur kernels and Gaussian noise in a random order.

### E Implementation Details and Hyperparameters

#### E.1 Implementations of tau_theta for conditional LDMs

For the experiments on text-to-image and layout-to-image synthesis, we implement the conditioner tau_theta as an unmasked transformer which processes a tokenized version of the input y and produces an output zeta := tau_theta(y), where zeta in R^(M x d_tau). More specifically, the transformer is implemented from N transformer blocks consisting of global self-attention layers, layer-normalization and position-wise MLPs:

$$\zeta \leftarrow \text{TokEmb}(y) + \text{PosEmb}(y)$$

for i = 1, ..., N:
- zeta_1 = LayerNorm(zeta)
- zeta_2 = MultiHeadSelfAttention(zeta_1) + zeta
- zeta_3 = LayerNorm(zeta_2)
- zeta = MLP(zeta_3) + zeta_2

zeta = LayerNorm(zeta)

With zeta available, the conditioning is mapped into the UNet via the cross-attention mechanism. We modify the "ablated UNet" architecture and replace the self-attention layer with a shallow (unmasked) transformer consisting of T blocks with alternating layers of (i) self-attention, (ii) a position-wise MLP and (iii) a cross-attention layer.

For the text-to-image model, we rely on a publicly available BERT tokenizer. The layout-to-image model discretizes the spatial locations of the bounding boxes and encodes each box as a (l, b, c)-tuple, where l denotes the (discrete) top-left and b the bottom-right position. Class information is contained in c.

Note that the class-conditional model is also implemented via cross-attention, where tau_theta is a single learnable embedding layer with a dimensionality of 512, mapping classes y to zeta in R^(1x512).

#### Transformer block architecture (replacing self-attention in UNet)

| **input** | R^(h x w x c) |
|---|---|
| LayerNorm | R^(h x w x c) |
| Conv1x1 | R^(h x w x d*n_h) |
| Reshape | R^(hw x d*n_h) |
| x T { SelfAttention, MLP, CrossAttention } | R^(hw x d*n_h) |
| Reshape | R^(h x w x d*n_h) |
| Conv1x1 | R^(h x w x c) |

#### Transformer encoder hyperparameters

| | Text-to-Image | Layout-to-Image |
|---|---|---|
| seq-length | 77 | 92 |
| depth N | 32 | 16 |
| dim | 1280 | 512 |

#### E.2 Hyperparameters

**Unconditional LDMs (Tab. 2 results):**

| | CelebA-HQ 256x256 | FFHQ 256x256 | LSUN-Churches 256x256 | LSUN-Bedrooms 256x256 |
|---|---|---|---|---|
| f | 4 | 4 | 8 | 4 |
| z-shape | 64x64x3 | 64x64x3 | - | 64x64x3 |
| |Z| | 8192 | 8192 | - | 8192 |
| N_params | 274M | 274M | 294M | 274M |
| Channels | 224 | 224 | 192 | 224 |
| Depth | 2 | 2 | 2 | 2 |
| Channel Mult. | 1,2,3,4 | 1,2,3,4 | 1,2,2,4,4 | 1,2,3,4 |
| Attn. resolutions | 32,16,8 | 32,16,8 | 32,16,8,4 | 32,16,8 |
| Batch Size | 48 | 42 | 96 | 48 |
| Iterations | 410k | 635k | 500k | 1.9M |
| Learning Rate | 9.6e-5 | 8.4e-5 | 5.e-5 | 9.6e-5 |

**Conditional LDMs:**

| **Task** | Text-to-Image | Layout (OI) | Layout (COCO) | Class-Label | Super-Res | Inpainting | Semantic |
|---|---|---|---|---|---|---|---|
| **Dataset** | LAION | OpenImages | COCO | ImageNet | ImageNet | Places | Landscapes |
| f | 8 | 4 | 8 | 4 | 4 | 4 | 8 |
| Model Size | 1.45B | 306M | 345M | 395M | 169M | 215M | 215M |
| Channels | 320 | 128 | 192 | 192 | 160 | 128 | 128 |
| Depth | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| Channel Mult. | 1,2,4,4 | 1,2,3,4 | 1,2,4 | 1,2,3,5 | 1,2,2,4 | 1,4,8 | 1,4,8 |
| Batch Size | 680 | 24 | 48 | 1200 | 64 | 128 | 48 |
| Iterations | 390K | 4.4M | 170K | 178K | 860K | 360K | 360K |
| Conditioning | CA | CA | CA | CA | concat | concat | concat |

### F Computational Requirements

| **Method** | Generator Compute | Overall Compute | Inference Throughput | N_params | FID |
|---|---|---|---|---|---|
| **LSUN Churches 256^2** | | | | | |
| StyleGAN2 | 64 | 64 | - | 59M | 3.86 |
| *LDM-8* (ours) | 18 | 18 | 6.80 | 256M | 4.02 |
| **LSUN Bedrooms 256^2** | | | | | |
| ADM (1000 steps) | 232 | 232 | 0.03 | 552M | 1.9 |
| *LDM-4* (ours) | 60 | 55 | 1.07 | 274M | 2.95 |
| **CelebA-HQ 256^2** | | | | | |
| *LDM-4* (ours) | 14.4 | 14.4 | 0.43 | 274M | 5.11 |
| **FFHQ 256^2** | | | | | |
| StyleGAN2 | 32.13 | 32.13 | - | 59M | 3.8 |
| *LDM-4* (ours) | 26 | 26 | 1.07 | 274M | 4.98 |
| **ImageNet 256^2** | | | | | |
| BigGAN-deep | 128-256 | 128-256 | - | 340M | 6.95 |
| ADM (250 steps) | 916 | 916 | 0.12 | 554M | 10.94 |
| ADM-G (250 steps) | 916 | 962 | 0.07 | 608M | 4.59 |
| *LDM-4-G* (ours, c.f.g. s=1.5) | 271 | 271 | 0.4 | 400M | 3.60 |

Compute during training in V100-days. Throughput measured in samples/sec on a single NVIDIA A100.

### G Details on Autoencoder Models

We train all our autoencoder models in an adversarial manner, such that a patch-based discriminator D_psi is optimized to differentiate original images from reconstructions D(E(x)). To avoid arbitrarily scaled latent spaces, we regularize the latent z to be zero centered and obtain small variance by introducing a regularizing loss term L_reg.

We investigate two different regularization methods: (i) a low-weighted Kullback-Leibler-term between q_E(z | x) = N(z; E_mu, E_sigma^2) and a standard normal distribution N(z; 0, 1) as in a standard variational autoencoder, and (ii) regularizing the latent space with a vector quantization layer by learning a codebook of |Z| different exemplars.

To obtain high-fidelity reconstructions we only use a very small regularization for both scenarios, i.e. we either weight the KL term by a factor ~10^-6 or choose a high codebook dimensionality |Z|.

The full objective to train the autoencoding model (E, D) reads:

$$L_{\text{Autoencoder}} = \min_{E, D} \max_\psi (L_{rec}(x, D(E(x))) - L_{adv}(D(E(x))) + \log D_\psi(x) + L_{reg}(x; E, D))$$

**DM Training in Latent Space.** For training diffusion models on the learned latent space, we distinguish two cases:
(i) For a KL-regularized latent space, we sample z = E_mu(x) + E_sigma(x) * epsilon := E(x), where epsilon ~ N(0, 1). When rescaling the latent, we estimate the component-wise variance from the first batch in the data and scale z to have unit standard deviation.
(ii) For a VQ-regularized latent space, we extract z *before* the quantization layer and absorb the quantization operation into the decoder, i.e. it can be interpreted as the first layer of D.
