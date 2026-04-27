# A generative foundation model for an all-in-one seismic processing framework

- **arXiv**: [2502.01111](https://arxiv.org/abs/2502.01111)
- **Submitted**: 2025-02-03
- **Authors**: Shijun Cheng, Randy Harsuko, Tariq Alkhalifah (KAUST)
- **Keywords**: Generative foundation model, Generative diffusion models, Multi-task seismic processing

## Abstract

Seismic data often face challenges in their utilization due to noise contamination, incomplete acquisition, and limited low-frequency information, which hinder accurate subsurface imaging and interpretation. Traditional processing methods rely heavily on task-specific designs to address these challenges and fail to account for the variability of data. To address these limitations, we present a generative seismic foundation model (GSFM), a unified framework based on generative diffusion models (GDMs), designed to tackle multi-task seismic processing challenges, including denoising, backscattered noise attenuation, interpolation, and low-frequency extrapolation. GSFM leverages a pre-training stage on synthetic data to capture the features of clean, complete, and broadband seismic data distributions and applies an iterative fine-tuning strategy to adapt the model to field data. By adopting a target-oriented diffusion process prediction, GSFM improves computational efficiency without compromising accuracy. Synthetic data tests demonstrate GSFM surpasses benchmarks with equivalent architectures in all tasks and achieves performance comparable to traditional pre-training strategies, even after their fine-tuning. Also, field data tests suggest that our iterative fine-tuning approach addresses the generalization limitations of conventional pre-training and fine-tuning paradigms, delivering significantly enhanced performance across diverse tasks. Furthermore, GSFM's inherent probabilistic nature enables effective uncertainty quantification, offering valuable insights into the reliability of processing results.

## Introduction

Seismic processing is an essential step for raw data acquisition to produce high-quality subsurface images [yilmaz2001seismic]. It involves a series of complex and diverse procedures aimed at revealing detailed information about subsurface formations and their physical properties. Due to the highly intricate nature of seismic wave propagation in subsurface media and the interference of acquisition environments, raw acquired data are often degraded by various factors. For example, environmental noise reduces the signal-to-noise ratio, making it challenging to extract valuable signals. Damaged geophones can lead to bad traces in the data, compromising the consistency and completeness of subsequent processing. Low-frequency signals are often week, resulting in the loss of crucial signal components that are vital for accurately characterizing subsurface structures [virieux2009overview]. These factors negatively impact the accuracy of subsequent seismic processing, imaging and inversion. Therefore, various seismic processing steps should be performed to enhance data quality, strengthen signals, and eliminate interferences, thereby achieving reliable subsurface imaging results and accurate geological interpretation.

The conventional seismic processing paradigm generally consists of several key steps designed to address the aforementioned issues and enhance data quality. The first stage is preprocessing, which usually includes denoising to mitigate the impact of noise [abma1995lateral; krohn2008introduction; chen2014random; chen2015random; liu2015signal]. Usually, static correction and normalization are performed to compensate for surface irregularities and variations in amplitude, ensuring consistency in signal phase and amplitude [cox1999static]. Multiple suppression is also often employed to eliminate the impact of multiple reflections, thereby enhancing the clarity of primary reflection signals [verschuur1992adaptive; lopez2015closed]. For areas with incomplete data acquisition, interpolation techniques are used to fill in missing information, thereby improving spatial sampling density and resolution [spitz1991seismic; wang2002seismic; chen2019interpolation]. Moreover, velocity analysis [alkhalifah1995velocity; symes2008migration; fomel2009velocity] and migration [baysal1983reverse; chang1987elastic; zhang2015stable] are central to obtain an image of the subsurface. In this process, accurate velocity models are constructed to reposition seismic reflection events to their true locations, resulting in precise subsurface images [etgen2009overview]. In addition, especially recently, inversion techniques are applied to extract lithological and physical property information from the subsurface, enabling a quantitative description of subsurface formations [tarantola1984inversion; tarantola1986strategy; alkhalifah2014tomography]. These processing steps work together to gradually improve the quality of seismic data, and the resulting image for proper geological interpretation and understanding.

The advantages of the traditional seismic processing paradigm lie in its rigorous theoretical foundation and its extensive application in geophysical exploration [yilmaz2001seismic]. These processing steps have been validated over time and, also, have demonstrated effectiveness in addressing a variety of complex issues while progressively enhancing the quality and reliability of seismic data. Additionally, traditional methods possess strong physical interpretability, enabling clear imaging of subsurface structures and extraction of crucial lithological and physical properties. However, there are also notable limitations associated with traditional methods. Firstly, conventional seismic processing often relies heavily on expert knowledge and experience, requiring frequent parameter adjustments and expert judgment throughout the various steps, to adapt to the various data. In other words, the processing algorithms, other than certain user-defined parameters, are often fixed and not driven by the data. This results in a high professional threshold and a lengthy processing cycle [yu2021deep]. Secondly, given the increasing volume of data, the efficiency and timeliness of traditional methods struggle to meet practical demands, with the data processing and imaging often consuming considerable time and computational resources [hou2021machine]. Lastly, the performance of traditional methods is often insufficiently robust, making them susceptible to noise and the complexities of subsurface media, which hinders their ability to consistently deliver high-quality results [Li2020DLInversion].

To overcome the limitations of traditional methods, neural network (NN)-based seismic processing approaches have gradually gained attention due to their numerous unique advantages [yu2021deep; mousavi2022deep; mousavi2024applications]. For instance, deep learning (DL) methods can automatically learn features from data, thereby reducing the reliance on expert knowledge, while also better meeting the need to handle large volumes of data. Furthermore, NN-based models often exhibit superior performance when processing complex seismic data. Typically, an NN-based seismic processing paradigm involves training a deep NN on a substantial amount of seismic data to approximate the nonlinear relationship between input and target data. Since target data from real-world cases are often inaccessible, a common approach is to train the NN using synthetic data in a supervised learning (SL) manner before applying it to real data [yu2019deep; wang2019deep; dong2019desert; wu2019faultseg3d; wu2020building; zhang2021deep; dong2024can; dong2024seismic]. A significant limitation of this approach arises when the synthetic data distribution poorly represents the real data, leading to considerable performance degradation for the trained network [alkhalifah2022mlreal; zhang2022improving]. Therefore, an alternative method is to use self-supervised learning (SSL) (or unsupervised learning) to eliminate the need for labeled data, enabling the network to be trained directly on real data, which can mitigate the generalization issues [saad2020deep; birnie2021potential; liu2023trace; liu2024self; liu2024gabor; saad2024noise; cheng2024effective; cheng2024self]. However, since the training is performed on each real seismic dataset individually, it is often dataset and task specific and, thus, the overall efficiency is lower compared to networks trained using SL.

Actually, regardless of whether it is in the SL or SSL paradigm, another major issue is that a trained network is often tailored to a specific seismic processing task (SPT). When switching to another task, the network is often trained again from scratch. As mentioned earlier, seismic processing comprises multiple distinct tasks, and training a network from scratch for each task incurs significant time and the computational cost. Consequently, some recent paradigms based on pre-trained models have been proposed, where these models are first pre-trained on large amounts of seismic data using SSL for reconstruction, and then fine-tuned for downstream tasks to improve training efficiency and reduce computational costs. For example, [harsuko2022storseismic] proposed the StorSeismic framework, in which they pre-trained a Transformer model that takes the sequence of seismic shot gathers as input to extract and store features of the seismic data. The pre-trained model is then fine-tuned for multiple SPTs, such as denoising, velocity estimation, first arrival picking, and normal moveout correction, among other tasks. The fine-tuned model demonstrated excellent performance on field data. Similarly, [sheng2023seismic] introduced the Seismic Foundation Model (SFM), employing the Masked Autoencoders approach to pre-train a Transformer on over 2 million large datasets. After pre-training, they extracted the encoder part and connected a simple decoder network for fine-tuning on downstream tasks. SFM exhibited superior performance across tasks like denoising, interpolation, seismic facies classification, geological body recognition, and inversion. Unlike the previous two paradigms, [cheng2024meta] proposed a Meta-Processing framework for multi-task seismic processing that employs meta-learning to extract shared features of seismic data from very limited datasets, thereby providing a robust initialization. This initialization allows for rapid convergence to optimal performance across various SPTs.

We can see that, the core of the pre-training strategy lies in leveraging NNs to learn and extract distributional characteristics of seismic data, enabling these pre-trained networks to achieve rapid convergence and outstanding performance across various downstream SPTs. Therefore, it inspires us that if a network model can effectively capture the distribution characteristics of seismic data, it can significantly enhance its performance in seismic processing. Recently, generative diffusion models (GDMs) have shown substantial potential in seismology due to their powerful ability to learn given data distributions, including applications such as denoising [li2024conditional; xiao2024diffusion; trappolini2024cold], interpolation [wei2023seismic; liu2024generative; wang2024self; wei2024seismic], resolution enhancement [zhang2024seisresodiff], waveform separation [zhang2024conditional], imaging improvement [shi2024generative], and velocity model building [wang2023prior; wang2024controllable; taufik2024learned]. Notably, [durall2023deep] tested GDMs on various SPTs, including demultiple, denoising, and interpolation. They trained GDMs on synthetic data and evaluated it on synthetic data. They presented results of field data testing for the demultiple task, demonstrating competitive outcomes with traditional DL methods. However, significant signal leakage was still observed, which can be attributed to generalization issues arising from the distributional shift between synthetic and field data. In addition, for different SPTs, they trained different GDMs from scratch to accommodate each specific task, which is time-consuming.

In this paper, we propose a generative seismic foundation model (GSFM) framework for various SPTs. This framework is based on the GDM's powerful capability to capture and store the distributional characteristics of seismic data, potentially offering greater expressiveness compared to traditional pre-training methods. Due to the GDM's need for target data distributions, as shown by [durall2023deep], we also train our GDM on synthetic data. However, a significant difference from Durall et al.'s approach is that we train various SPTs simultaneously on a single GDM, such as denoising, backscattered noise attenuation, interpolation, and low-frequency extrapolation. We encode these tasks by introducing different class labels, embedding them into the training process of the GDM, enabling the network to automatically identify and handle various SPTs. Training for multi-task applications is based on the assumption that ideal seismic data (the target) should be clean, complete, and broadband. By training the GDM to capture this ideal distribution, we enable the model to generate the ideal target output from low-quality seismic data. Additionally, we adopt, within the GDM framework, target prediction instead of noise prediction during training to enhance both training stability and inference efficiency. Predicting the target directly aligns the model output with the ideal seismic data distribution, avoiding the iterative denoising process commonly required in conventional GDMs. This design not only simplifies the training process by reducing optimization complexity but also allows us to achieve high-quality results during inference with just a single sampling step. Nevertheless, due to the feature gap between synthetic and field data, we would still face generalization issues when applying the trained model to field data. To address this problem, we propose a strategy to fine-tune our pre-trained GDM on field data using an SSL approach. Specifically, during the initial stage of fine-tuning for each SPT, we use the pre-trained GDM model to directly predict the field data, which is then added to our training dataset. After the GDM model undergoes several iterations of optimization, we iteratively employ the model trained in the previous stage to predict field data and update the training set at fixed intervals. In this way, we gradually shift the distribution captured by the pre-trained GDM from the synthetic domain to the field data distribution, thereby enhancing the model's performance on field data.

Furthermore, the inherent randomness in the initial noise used during the sampling process allows us to generate multiple predictions for the same input condition. This provides a natural mechanism to assess prediction variability. By evaluating the standard deviation of these predictions, we can identify regions of higher uncertainty, which often correspond to areas with greater signal leakage or processing errors. This capability not only helps evaluate the reliability of the model's predictions but also provides valuable feedback to guide further optimization during the fine-tuning process, ensuring robust performance on field data.

The contributions of this paper can be summarized as follows:

- We propose a generative seismic foundation model framework capable of simultaneously performing various SPTs.
- We introduce the use of class label constraints to guide the NN in jointly optimizing different SPTs.
- We propose a strategy to fine-tune the pre-trained foundation model on synthetic data using an SSL approach on field data, thereby overcoming the generalization issues of NNs.
- We leverage the probabilistic nature of GDMs to quantify the uncertainty of the processing product, which helps assess its reliability and helps guide the fine-tuning of the pre-trained model.
- Examples from synthetic and field data demonstrate that our all-in-one seismic processing framework can achieve good processing performance.

## Review of conventional neural network-based seismic processing

Traditional seismic processing methods often rely on explicit physical models and assumptions, which may not be fully applicable in complex media. In contrast, the advantage of NNs lies in their ability to automatically approximate such complex mapping relationships by extracting features from data, without relying heavily on prior assumptions.

Commonly, NN-based seismic processing methods can be viewed as a parameterized function approximator, which adjusts its internal weights through a training process to learn the nonlinear relationship that maps seismic data, ```latex $x_i$ ```, (such as raw noisy data) to the desired output products, ```latex $y_i$ ```, (such as denoised data). During this process, the network optimizes its parameters by minimizing the error between the NN output (prediction) and the target, ```latex $y_i$ ```, to capture key features in the seismic data, which can be represented by the following loss function:

```latex
$$L(\theta) = \frac{1}{N} \sum_{i=1}^N \| \text{NN}_\theta(x_i) - y_i \|^2,$$
```

where ```latex $\theta$ ``` is the set of parameters of the network, ```latex $\text{NN}_\theta(x_i)$ ``` is the predicted output of the network for input ```latex $x_i$ ```, ```latex $y_i$ ``` is the corresponding target output, and ```latex $N$ ``` is the number of training samples. By minimizing the loss function ```latex $L(\theta)$ ```, the network continuously optimizes its parameters to make the predicted results as close as possible to the target output.

To further enhance the performance of conventional NN-based seismic processing, a pre-training and fine-tuning paradigm has been proposed [harsuko2022storseismic; sheng2023seismic]. The pre-training method involves an SSL training by reconstructing masked original seismic data, thus providing a good initial parameter set for downstream tasks. The objective can be expressed as the following loss function minimization problem:

```latex
$$L(\theta) = \frac{1}{N} \sum_{i=1}^N \| \text{NN}_\theta(x_i^{masked}) - x_i \|^2$$
```

where ```latex $\text{NN}_\theta(x_i^{masked})$ ``` is the NN's reconstructed output for the masked input ```latex $x_i^{masked}$ ```, and ```latex $x_i$ ``` is the corresponding original input data. By performing the pre-training phase, the network can learn representations of the basic features of the original seismic data. On this basis, fine-tuning training on labeled datasets allows the network to better adapt to specific task requirements. This paradigm not only significantly improves the generalization performance of the network but also accelerates convergence and reduces the dependence on labeled data, thereby achieving more robust seismic processing under complex conditions.

Despite the notable progress made by the current pre-training and fine-tuning paradigm in seismic data processing, we still face several challenges. First, the pre-training stage often relies on synthetic data, resulting in limited generalizability when addressing the complexities of real data. Meanwhile, due to the differences between pre-training tasks (e.g., reconstruction) and downstream tasks (e.g., denoising, interpolation, or low-frequency extrapolation), the model's performance may be constrained during task transfer, preventing it from fully leveraging the benefits of pre-training phase. Moreover, the dependency on labeled data during fine-tuning stage further restricts the paradigm's applicability when labeled data are scarce.

To address these challenges, we present a framework for a generative seismic foundation model (GSFM) based on generative diffusion models (GDM). This framework aims to capture ideal seismic data distribution features through multi-task learning and generative data distribution modeling, while training on synthetic data for various tasks to enable the model to handle various seismic processing tasks (SPTs) effectively. By incorporating task-specific encoding, the model can automatically identify and manage different SPTs during training. Furthermore, we introduce a gradual transfer strategy using an SSL approach to fine-tune the model on real data, progressively shifting its distribution from synthetic data to real data, thereby improving its performance in practical applications.

## Method

In this section, we will first introduce the fundamental concepts of GDMs. Following that, we will present the framework for a GSFM based on GDMs. We will provide detailed illustrations on how we perform multi-task encoding, pre-training, fine-tuning, and prediction. Finally, we will introduce our network architecture.

### Generative diffusion models

GDMs are gaining attention for its strong capability to produce highly realistic samples. These models initially convert data into pure noise through a forward process and then progressively denoise it to recover the data in the reverse process.

Within the denoising diffusion probabilistic model (DDPM), the forward process is defined as [ho2020denoising]:

```latex
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t} x_{t-1}, (1 - \alpha_t) \mathbf{I}).$$
```

In the forward process, the data sample ```latex $x_0$ ``` gradually transforms into a noisy version ```latex $x_t$ ``` at each step ```latex $t$ ```, controlled by the parameter ```latex $\alpha_t$ ```, with ```latex $\alpha_t = 1 - \beta_t$ ```, where ```latex $\beta_t$ ``` is a small constant specifying the noise variance incrementally introduced at each step. Thus, at each step, noise is added according to the Gaussian distribution ```latex $\mathcal{N}$ ```. This process adds isotropic Gaussian noise, facilitated by the identity matrix ```latex $\mathbf{I}$ ```, and progressively diffuses the original data into a Gaussian noise distribution from step ```latex $t = 1$ ``` to ```latex $t = T$ ```.

To express ```latex $x_t$ ``` directly in terms of ```latex $x_0$ ``` and a noise term, we can use the reparameterization trick:

```latex
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon,$$
```

where ```latex $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ ``` is Gaussian noise, and ```latex $\bar{\alpha}_t = \prod_{i=1}^t (1 - \beta_i)$ ```.

To learn how to reverse this process, the network is trained to predict the noise ```latex $\epsilon$ ``` added to the sample at each step ```latex $t$ ```. Given a noisy sample ```latex $x_t$ ```, the network predicts the noise term ```latex $\epsilon_\theta(x_t, t)$ ```. The objective of the network is to minimize the difference between the actual noise ```latex $\epsilon$ ``` (added during the forward process) and the predicted noise ```latex $\epsilon_\theta(x_t, t)$ ```. This is achieved by minimizing the mean squared error (MSE) between the added noise and the predicted noise:

```latex
$$L(\theta) = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$
```

This objective trains the network to accurately predict the noise added at each step, and to do so, the network needs to store the features of the signal information diffused by this added noise.

Once the network is trained, the reverse process, starting for a noise sample drawn from the Gaussian distribution, uses the network's noise predictions ```latex $\epsilon_\theta(x_t, t)$ ``` to iteratively inject the stored signal and gradually remove noise to reconstruct a sample from the distribution it was trained on. Specifically, starting from pure noise ```latex $x_T$ ```, each step in the reverse process estimates the previous sample ```latex $x_{t-1}$ ``` from ```latex $x_t$ ``` by subtracting the predicted noise component. The reverse sampling equation can be expressed as:

```latex
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z,$$
```

where ```latex $\epsilon_\theta(x_t, t)$ ``` is the noise predicted by the trained network, ```latex $\sigma_t z$ ``` is an optional noise term for stochastic sampling, and ```latex $z \sim \mathcal{N}(0, \mathbf{I})$ ```.

In DDPM, the step-by-step denoising process is implemented through a Markov chain, which requires numerous time steps to gradually remove noise. As a result, the sampling speed of DDPM is relatively slow. In contrast, the denoising diffusion implicit model (DDIM) improves the sampling process of DDPM by removing the dependence on the Markov chain [song2020denoising]. Actually, DDIM and DDPM share the same forward process, which can also control noise introduction through the reparameterization equation. The reverse process of DDIM can be represented by the following sampling equation:

```latex
$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \, \epsilon_\theta(x_t, t) + \sigma_t z,$$
```

with

```latex
$$\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \, \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$
```

giving an estimate of the original data ```latex $x_0$ ``` directly, ```latex $\epsilon_\theta(x_t, t)$ ``` is the noise term predicted by the neural network, and ```latex $z \sim \mathcal{N}(0, \mathbf{I})$ ``` is a random noise term, and ```latex $\sigma_t$ ``` controls the level of this randomness. When ```latex $\sigma_t = 0$ ```, the sampling process becomes deterministic because the random noise term ```latex $\sigma_t z$ ``` is removed. If ```latex $\sigma_t \neq 0$ ```, it introduces a small random term ```latex $\sigma_t z$ ``` in the sampling, allowing for some variation in the generated samples. Since we aim to apply DDIM in seismic processing, we prefer a deterministic approach rather than producing random solutions. Therefore, we will set parameter ```latex $\sigma_t = 0$ ``` in the following.

In DDIM, an additional improvement involves directly training the network to predict the original clean image ```latex $x_0$ ``` rather than focusing on the noise ```latex $\epsilon$ ``` added to ```latex $x_0$ ``` [bansal2024cold]. This approach benefits from effectively leveraging the spatial coherence and semantic information within the image, enabling faster convergence and higher generation quality. In this case, the network's optimization target shifts to minimizing the difference between the predicted image ```latex $x_{0,\theta}(x_t, t)$ ``` and the original image ```latex $x_0$ ```, as follows:

```latex
$$L(\theta) = \mathbb{E}_{x_0, \epsilon, t} \left[ \| x_0 - x_{0,\theta}(x_t, t) \|^2 \right].$$
```

In the ```latex $x_0$ ```-based prediction framework, the reverse sampling equation in DDIM can be simplified to:

```latex
$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} x_{0,\theta}(x_t, t) + \sqrt{1 - \bar{\alpha}_{t-1}} \hat{\epsilon}(x_t, t)$$
```

where ```latex $\hat{\epsilon}(x_t, t)$ ``` is an estimate of the added noise. We can estimate the added noise ```latex $\hat{\epsilon}(x_t, t)$ ``` in terms of ```latex $x_t$ ``` and the network's prediction ```latex $x_{0,\theta}(x_t, t)$ ```, as follows:

```latex
$$\hat{\epsilon}(x_t, t) = \frac{x_t - \sqrt{\bar{\alpha}_t} x_{0,\theta}(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}.$$
```

GDMs demonstrate excellent performance in generating high-quality samples due to its strong capability in capturing distributions and its stepwise denoising approach [rombach2022high]. Specifically, DDIM significantly enhances sampling speed by eliminating the dependency on the Markov chain. Furthermore, by adopting a strategy based on predicting ```latex $x_0$ ```, we are able to further improve both generation quality and sampling speed, which is needed to meet the accuracy and efficiency requirements of seismic processing. Based on this, we introduce the GSFM framework in the following section, which incorporates multi-task learning into GDM to handle a variety of SPTs, including denoising, interpolation, and low-frequency extrapolation in a unified manner.

### Generative seismic foundation model: Pre-training

Our GSFM is adapted from a GDM and employs multi-task simultaneous pre-training on synthetic data, followed by direct fine-tuning on real data. However, in traditional GDMs, the model's input consists of a noisy version of a clean single-channel image, which is used to train the model for stepwise denoising.

To accommodate the needs of multi-task seismic processing, we extend the input of our GSFM to a dual-channel structure. During the pre-training and fine-tuning phases, the dual-channel inputs may contain different content. In this section, we first explain how the dual-channel network inputs are configured during the pre-training phase. Since the network is optimized on synthetic data during the pre-training phase, we can access the labels for different tasks. Therefore, in this phase, the first channel contains a noisy version of the labels (the target complete clean data), while the second channel is used for the corresponding data to be processed, i.e., the degraded data specific to the task. The content of the second channel varies depending on the SPT, enabling the model to adapt flexibly to different tasks based on the input data. Specifically, the dual channels for different specified tasks are as follows:

- **Denoising**: The second channel contains the data contaminated with noise we want the network to learn to remove.
- **Backscattered noise attenuation**: As a special case, the second channel contains data contaminated with backscattered noise.
- **Interpolation**: The second channel contains data with missing traces.
- **Low-frequency extrapolation**: The second channel contains data lacking low-frequency components.

In the pre-training phase, the forward process of our GSFM shares the noise injection formulation of the conventional diffusion model, thereby constructing the content for the first channel of the dual-channel input. For the second channel, we can see that we essentially use the same input data as that used in conventional NN-based seismic processing methods.

To enable simultaneous training for different tasks, we introduce a task encoding label ```latex $c$ ```, allowing the network to identify and distinguish between various SPTs. For the tasks considered in this paper, including denoising, backscattered noise attenuation, interpolation, and low-frequency extrapolation, their class labels ```latex $c$ ``` are defined as 0, 1, 2, and 3, respectively. The embedding method for the task encoding label ```latex $c$ ``` is similar to that used for the step ```latex $t$ ```.

As previously mentioned, setting the GDM network's prediction target to ```latex $x_0$ ``` can enhance generation quality and efficiency. Therefore, during the pre-training and also the following fine-tuning phase, the prediction target is set to ```latex $x_0$ ```. In other words, for different SPTs, the network's prediction target corresponds to their respective labeled data. In this case, our pre-training objective can be expressed as:

```latex
$$L(\theta) = \mathbb{E}_{x_0, x, \epsilon, t, c} \left[ \| x_0 - x_{0,\theta}(x_t, x, t, c) \|^2 \right],$$
```

where ```latex $x$ ``` represents the second channel input serving as the conditional constraint.

Here, we consider the four SPTs described above. However, we emphasize that our framework is flexible and can be extended to accommodate additional SPTs by simply defining the appropriate degraded data format for the second channel and assigning a new class encoding label for each added task. This adaptability allows our GSFM to serve as a versatile foundation for a wide range of seismic processing needs. For example, if our objective is for GSFM to remove surface multiples, we simulate shot gathers with free surface boundary condition to serve as input to the second channel, while having our clean data target modeled using absorbing boundary condition [harsuko2024optimizing].

### Generative seismic foundation model: Fine-tuning

After completing pre-training on synthetic data, our GSFM is directly fine-tuned on real data to enhance its generalization capability for practical applications. During the fine-tuning phase, due to the lack of labels, we employ an SSL-based optimization approach, maintaining the model's adaptability and stability across multi-task seismic processing. To ensure consistency, the fine-tuning process retains the embedding methods for the task encoding label ```latex $c$ ``` during pre-training, enabling the model to continue supporting multi-task learning on real data and improving task transfer efficiency. The prediction target during fine-tuning remains set to ```latex $x_0$ ``` (pseudo-labels), which represents the ideal output for each task.

To accomplish fine-tuning, we perform this process independently for each SPT, using the pre-trained network as the starting point for each task and setting the task encoding label ```latex $c$ ``` to the value corresponding to the desired task. We propose the following three fine-tuning strategies for each SPT:

- **Strategy 1**: The pre-trained model on synthetic data is used directly on the raw field data to generate preliminary processing products, which are then used as pseudo-labels during the fine-tuning phase. In this case, the first channel of the network input is a noisy version of the predicted pseudo-labels, while the second channel takes in shot gathers from the field data.

- **Strategy 2**: The second channel of the network input differs from that of strategy 1. Instead of using the field data, we use a corrupted version (similar to the corruptions applied to the synthetic data) of the pseudo-labels. For example, for the denoising task, additional noise is added to the predicted pseudo-labels. For the backscattered noise attenuation task, backscattered noise is added to the pseudo-labels. For the interpolation task, traces are removed from the pseudo-labels. For the low-frequency extrapolation task, the low-frequency components are filtered out from the pseudo-labels.

- **Strategy 3**: The third strategy is based on strategy 2 and involves iteratively updating the training dataset during the fine-tuning process. Specifically, the fine-tuning process is divided into multiple stages, with each stage consisting of several iterations. In the first stage, we maintain the configuration of strategy 2. In each subsequent stage, the model fine-tuned from the previous stage is used directly on the field data, generating new pseudo-labels. Diffusion process is added to these pseudo-labels to create the input for the first channel, while the second channel contains a further corrupted version of the newly generated pseudo-labels, consistent with strategy 2.

In subsequent experiments, we will test these three fine-tuning strategies to determine which one performs better in enhancing the model's generalization ability and processing performance. Based on our test results, the fine-tuned network induced by strategy 3 provides superior performance. This outcome is expected, as the multi-stage strategy with gradual optimization allows the model to achieve a smooth transition between the feature distributions of synthetic and real data. By updating and further degrading the pseudo-labels at each stage, the model progressively shifts from the synthetic domain to the real data domain during the fine-tuning process. This stepwise adjustment not only makes the model more robust in handling the complexity of real data but also effectively reduces the distribution gap between synthetic and real data, thereby improving the model's generalization capability in real-world tasks.

**Algorithm: Iterative Fine-Tuning with Progressive Pseudo-Labeling for GSFM**

**Input:** Pre-trained GSFM model; Raw field data ```latex $x$ ```, initial pseudo-labels ```latex $x_{\text{pseudo}}$ ```, noise ```latex $\epsilon$ ```; Total stages ```latex $S$ ```, iterations per stage ```latex $N_{\text{stage}}$ ```; Task-specific corruption ```latex $\text{COR}[\cdot]$ ```
**Output:** Fine-tuned GSFM model

1. Load pre-trained GSFM model.
2. Set task-specific label ```latex $c$ ``` according to the seismic processing task.
3. Initialize pseudo-labels ```latex $x_{\text{pseudo}}$ ``` by predicting on field data ```latex $x$ ``` with the pre-trained model.
4. **for** stage ```latex $s = 1$ ``` to ```latex $S$ ``` **do**
5. &nbsp;&nbsp;**for** iteration ```latex $n = 1$ ``` to ```latex $N_{\text{stage}}$ ``` **do**
6. &nbsp;&nbsp;&nbsp;&nbsp;Sample a step ```latex $t$ ``` and add noise to pseudo-labels: ```latex $x_t = \sqrt{\bar{\alpha}_t} x_{\text{pseudo}} + \sqrt{1 - \bar{\alpha}_t} \epsilon$ ```
7. &nbsp;&nbsp;&nbsp;&nbsp;Apply task-specific corruption: ```latex $\hat{x} = \text{COR}[(x_{\text{pseudo}}, c)]$ ```
8. &nbsp;&nbsp;&nbsp;&nbsp;Forward pass: ```latex $x_{0,\theta}(x_t, \hat{x}, t, c) = \text{GSFM}(x_t, \hat{x}, t, c)$ ```
9. &nbsp;&nbsp;&nbsp;&nbsp;Compute loss: ```latex $L(\theta) = \mathbb{E} \left[ \| x_{\text{pseudo}} - x_{0,\theta}(x_t, \hat{x}, t, c) \|^2 \right]$ ```
10. &nbsp;&nbsp;&nbsp;&nbsp;Backpropagate the loss and update model parameters ```latex $\theta$ ```
11. &nbsp;&nbsp;**end for**
12. &nbsp;&nbsp;After ```latex $N_{\text{stage}}$ ``` iterations, generate updated pseudo-labels: ```latex $x_{\text{pseudo}} = \text{GSFM}(\epsilon, x, t, c)$ ```
13. **end for**
14. **Return:** Fine-tuned GSFM model

### Generative seismic foundation model: Predicting

After completing the fine-tuning process, our GSFM is ready to perform predictions for each seismic processing task independently. For each task-specific fine-tuned network, we obtain predictions tailored to the corresponding task. Unlike conventional NN-based seismic processing methods, which typically use a direct mapping approach, our GSFM leverages a generative prediction process due to its foundation in GDM.

Specifically, for each SPT, we begin by assigning the corresponding task encoding label ```latex $c$ ``` to indicate the target task. The network input is structured as follows: The first channel is initialized with random noise ```latex $\epsilon$ ```, while the second channel contains seismic data ```latex $x$ ``` that needs processing. Using the reverse process of GDM, we iteratively denoise the input to generate the desired output ```latex $x_0$ ```. At each step ```latex $t$ ``` in the reverse process, the model estimates ```latex $x_0$ ``` based on the current noisy input ```latex $x_t$ ```, and the reverse step is given by:

```latex
$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \, x_{0,\theta}(x_t, x, t, c) + \sqrt{1 - \bar{\alpha}_{t-1}} \, \hat{\epsilon}(x_t, x, t, c),$$
```

where ```latex $x_{0,\theta}(x_t, x, t, c)$ ``` is the model's prediction of the clean data ```latex $x_0$ ``` given the noisy input ```latex $x_t$ ``` at step ```latex $t$ ``` and task label ```latex $c$ ```. Here, ```latex $\hat{\epsilon}(x_t, x, t, c)$ ``` represents an estimate of the noise component.

The conventional prediction process continues iteratively, with the model starting from a high level of noise and gradually refining the input. The final output at last step, ```latex $x_0$ ```, represents the processed data for the specified task, having been transformed from noise to the desired form through a series of denoising steps. However, considering the efficiency requirements in actual processing, we will only use one time step for the sampling process here. Specifically, we only use the last sampling step, that is, ```latex $t$ ``` is set to 0 to get our final prediction product.

### Network architecture

Our GSFM adopts an enhanced U-Net-based architecture tailored for multi-task seismic processing. This architecture incorporates multi-scale feature extraction, task-specific embeddings, and attention mechanisms to deliver accurate and robust predictions. The main components of the network include convolutional layers, residual blocks, attention blocks, downsampling and upsampling layers, and embeddings for time and task-specific information.

[IMAGE: Network architecture diagram showing (a) overall network structure, (b) time embedding layer, (c) class embedding layer, (d) residual block, (e) attention block]

The GSFM processes dual-channel inputs ```latex $(x_t, x)$ ```, where ```latex $x_t$ ``` represents in training the target data input at timestep ```latex $t$ ```, and ```latex $x$ ``` contains the data to be processed specific to the task. These inputs are first passed through an initial ```latex $3 \times 3$ ``` convolutional layer that maps the two input channels to 64 feature channels, preparing the data for hierarchical processing in the encoder-decoder structure.

The encoder path progressively extracts hierarchical features using a combination of downsampling layers, residual blocks and attention blocks. Each downsampling layer reduces the spatial resolution by a factor of 2 and simultaneously doubles the number of feature channels, enabling the extraction of high-level features at coarser scales. Specifically, after the first downsampling operation, the number of channels increases from 64 to 128. Subsequent downsampling operations further increase the channels to 256 and 512.

The decoder path restores the spatial resolution and reduces the number of channels in a symmetrical manner with respect to the encoder, combining high-level semantic information from the encoder with low-level spatial details via skip connections. Each upsampling layer doubles the spatial resolution and halves the number of feature channels. For example, the number of channels decreases from 512 to 256 after the first upsampling layer. This process continues until the final layer restores the original spatial resolution and reduces the channels back to 64. At the end of the decoder, a final output layer is applied. This layer consists of group normalization, followed by a sigmoid linear unit (SiLU) activation function, and a ```latex $3 \times 3$ ``` convolutional layer that reduces the feature channels to the single-channel prediction ```latex $x_{0,\theta}(x_t, y, t, c)$ ```.

To address the requirements of multi-task processing, GSFM integrates two types of embeddings to guide the network with temporal and task-specific information:

- **Time embedding layer**: The timestep ```latex $t$ ``` is encoded using a sinusoidal positional encoding scheme [vaswani2017attention], which represents temporal information as a combination of sine and cosine functions. The resulting encoded vector is passed through a series of linear transformations and SiLU activation functions, producing the time embedding vector ```latex $t_{emb}$ ```. This embedding vector is injected into the residual blocks to regulate the denoising process across timesteps.

- **Class embedding layer**: Task-specific information is provided through a learnable embedding layer implemented using `torch.nn.Embedding`. The task encoding label ```latex $c$ ``` is mapped to a high-dimensional embedding vector, which is further processed by linear transformations and gaussian error linear unit (GELU) activations, producing the class embedding vector ```latex $c_{emb}$ ```. This embedding vector is incorporated into residual blocks to enable task-specific adaptability.

In our GSFM, feature extraction and refinement rely on the integration of residual and attention blocks:

- **Residual blocks**: Each residual block processes feature maps using a combination of group normalization, SiLU activation functions, and ```latex $3 \times 3$ ``` convolutional layers. Task and time embeddings (```latex $c_{emb}$ ``` and ```latex $t_{emb}$ ```) are incorporated by projecting them through linear layers and adding the resulting vectors to the feature maps. This design enables task- and time-aware feature processing.

- **Attention block**: Attention blocks, which is developed by [vaswani2017attention], are applied to refine the feature maps further. These blocks compute query, key, and value matrices via ```latex $1 \times 1$ ``` convolutional layers and normalize attention scores using a softmax operation. The resulting weighted feature maps are aggregated and processed through another ```latex $1 \times 1$ ``` convolutional layer. This mechanism allows the network to focus on task-relevant regions, improving feature representation for seismic data.

The GSFM leverages downsampling and upsampling layers to capture features across multiple spatial scales. Downsampling layers reduce the spatial resolution of feature maps, enabling the extraction of high-level semantic features, while upsampling layers restore spatial resolution to match the input dimensions. Skip connections link encoder and decoder layers, combining fine-grained spatial details with deep semantic features, thereby improving the accuracy of task-specific predictions.

## Synthetic data examples

In this section, we first introduce the pre-training details of the GSFM, including dataset preparation and training configuration. We then evaluate the pre-trained GSFM's performance on denoising, backscattered noise attenuation, interpolation, and low-frequency extrapolation tasks using synthetic test data.

To assess the effectiveness of the pre-trained GSFM, we provide two comparative experimental benchmarks:

- **Benchmark 1**: Traditional NN-based processing paradigm. This benchmark utilizes conventional NN-based seismic processing methods, employing the networks to approximate the nonlinear relationship between input data and target data. To ensure a fair comparison, Benchmark 1 adopts the same U-Net-based architecture as GSFM, but excludes the time encoding module used in the diffusion model. The network takes single-channel degraded data as input and outputs the corresponding target data.

- **Benchmark 2**: Conventional pre-training and fine-tuning strategy. In this benchmark, we first pre-train a NN on synthetic data using an SSL approach, followed by fine-tuning for denoising, backscattered noise attenuation, and low-frequency extrapolation tasks. Again, to ensure fairness, we use the same U-Net architecture as GSFM but remove both the task encoding and time encoding modules. During the pre-training phase, we use the GSFM dataset for all tasks, constructing the input data using random masking.

### Pre-training configuration

Creating synthetic subsurface models that represent the real Earth remains a challenge. For our purposes, we closely follow the workflow introduced by [ovcharenko2022multi] to generate random velocity models, which have been shown to effectively generalize to real data. Specifically, first, we randomly create 1D compressional wave velocity (```latex $V_p$ ```) profiles using velocity values within our expected range of 1,500 to 4,500 m/s. These 1D profiles are then spread laterally to build 2D laterally homogeneous layered velocity models. Lastly, we apply random elastic transforms to the velocity models to distort them and introduce structures resembling realistic geological phenomena (folding, intrusion, etc.)

Since we aim to establish a foundation model applied to seismic waveforms, it is of utmost importance that the synthetic waveform for the training dataset is as close as a realistic waveform, which justifies the need to use an elastic modeling engine. We use a Pytorch-based seismic modeling and inversion package called Deepwave [richardson_alan_2023] to perform 2D elastic forward modeling on the aforementioned velocity models. The shear wave velocity (```latex $V_s$ ```) is obtained through a fixed ratio of ```latex $V_p /\sqrt{3}$ ```, while the density (```latex $\rho$ ```) is obtained through Gardner's relation [gardner1974formation].

**Table 1: Parameters for modeling of the synthetic pre-training dataset.**

| Parameter | Description | Value |
|-----------|-------------|-------|
| nx | Number of samples in the X axis | 324 |
| nz | Number of samples in the Z axis | 376 |
| dx | Sampling step in the X axis | 25 m |
| dz | Sampling step in the Z axis | 25 m |
| dt | Recording sampling step | 1.6e-2 s |
| nt | Number of recording timesteps | 376 |
| T | Total recording time (nt x dt) | 6.016 s |
| nr | Number of receivers | 324 |
| ds | Receiver spacing | 25 m |

In the pre-training phase, we generate a total of 2456 training samples for each task. As our framework simultaneously trains on four SPTs (denoising, backscattered noise attenuation, interpolation, and low-frequency extrapolation), the overall dataset comprise 9824 training samples. We employ the AdamW optimizer with a fixed learning rate of ```latex $1e\text{-}4$ ``` and a batch size of 5. To enhance the stability of the diffusion model training process, we apply an exponential moving average (EMA) with a rate of 0.999. The pre-training is conducted over 200,000 iterations.

For a fair comparison, the two benchmark models use the same training configuration, except for the EMA mechanism, as their training processes are stable and do not require it. Once pre-training is completed, we evaluate the performance of the pre-trained GSFM and benchmarks on synthetic test data. During inference, to ensure consistency and fairness across all models, we use a single sampling step (i.e., predicting directly at the final step) for generating the synthetic test results.

### Denoising

We first test the denoising performance of the pre-trained GSFM on synthetic data contaminated by random noise. The noisy test data is generated by injecting Gaussian noise with a noise level of 30%, as follows:

```latex
$$y=x+\epsilon \cdot std(x) \cdot rand(0,1),$$
```

where ```latex $\epsilon$ ``` is the noise level, ```latex $std(x)$ ``` represents the standard deviation of the clean data ```latex $x$ ```, and ```latex $rand(0,1)$ ``` is the standard normal distribution.

[IMAGE: Denoising performance comparison between pre-trained GSFM and two benchmarks on synthetic data]

Visually, the denoised results from our GSFM and two benchmarks appear very similar, with each method successfully suppressing the random noise and preserving the main seismic reflection events.

**Table 2: MSE comparison of denoising performance at different noise levels**

| Noise level | GSFM | Benchmark 1 | Benchmark 2 |
|-------------|------|-------------|-------------|
| 10% | **3.09e-07** | 3.44e-07 | 3.21e-07 |
| 20% | 9.20e-07 | 9.50e-07 | **9.02e-07** |
| 30% | 1.67e-06 | 1.73e-06 | **1.65e-06** |
| 40% | 2.61e-06 | 2.68e-06 | **2.59e-06** |
| 50% | 3.79e-06 | 3.91e-06 | **3.77e-06** |
| 60% | **4.60e-06** | 4.81e-06 | 4.61e-06 |

The results reveal that GSFM consistently outperforms Benchmark 1 across all noise levels. Benchmark 2 shows slightly better performance than GSFM at intermediate noise levels (20% to 50%), which is likely attributed to one more round of task-specific fine-tuning on labeled data. However, the performance gap is marginal, and GSFM demonstrates superior robustness at the highest noise level.

### Backscattered noise attenuation

[IMAGE: Backscattered noise attenuation performance comparison on synthetic data]

Similar to the denoising case, the visual differences among the results produced by the three methods are minimal. All methods successfully suppress the backscattered noise and preserve the primary seismic reflections. GSFM achieves the lowest MSE of ```latex $9.59e\text{-}07$ ```, outperforming Benchmark 1 (```latex $1.10e\text{-}06$ ```) and Benchmark 2 (```latex $1.26e\text{-}06$ ```).

### Interpolation

[IMAGE: Interpolation performance comparison on synthetic data with 50% missing traces]

**Table 3: MSE comparison of interpolation performance at different missing levels**

| Missing level | GSFM | Benchmark 1 | Benchmark 2 |
|---------------|------|-------------|-------------|
| 10% | **1.45e-08** | 3.60e-08 | 1.73e-08 |
| 20% | 1.93e-08 | 4.02e-08 | **1.88e-08** |
| 30% | 2.85e-08 | 4.51e-08 | **2.30e-08** |
| 40% | 8.54e-08 | 8.69e-08 | **5.21e-08** |
| 50% | 4.08e-08 | 6.39e-08 | **3.45e-08** |
| 60% | **3.65e-07** | 5.16e-07 | 5.53e-07 |

At the highest missing data level (60%), GSFM significantly outperforms both benchmarks. These results demonstrate that the diffusion model boosts the performance of the networks, enabling GSFM to outperform Benchmark 1 consistently.

### Low-frequency extrapolation

[IMAGE: Low-frequency extrapolation performance comparison on synthetic data]

Unlike the previous tasks, the difference figures here clearly showcase the differences among the three methods. Both our GSFM and Benchmark 2 achieve superior extrapolation quality, with minimal residuals and negligible signal leakage. In contrast, Benchmark 1 exhibits more significant signal leakage. GSFM achieves an MSE of ```latex $6.0e\text{-}07$ ```, while Benchmark 2 slightly outperforms GSFM with an MSE of ```latex $3.11e\text{-}07$ ```. Benchmark 1 performed significantly worse, with an MSE of ```latex $1.80e\text{-}03$ ```.

### Understanding performance differences among the methods

Benchmark 1, based on the conventional NN paradigm, consistently underperforms compared to GSFM and Benchmark 2. While its performance gap is less pronounced in the first three tasks, it becomes significantly evident in the low-frequency extrapolation task. For the first three tasks, the target data (clean, complete seismic data) remains consistent across tasks, enabling the network to learn a more generalized mapping. However, in the low-frequency extrapolation task, the target data shifts to clean, full-band seismic data, including low frequencies that are absent in the input. This change introduces a more specific and challenging relationship to learn.

In contrast, GSFM, despite sharing the similar architecture as Benchmark 1, leverages the GDMs to capture and learn a more unified distribution. By modeling the joint distribution of clean, complete, and full-band seismic data, GSFM is able to bridge the gap between the input and target data more effectively.

Benchmark 2 consistently demonstrates strong performance across tasks even slightly outperforms GSFM in terms of MSE for certain tasks. However, this slight advantage is achieved through task-specific fine-tuning, which relies heavily on labeled datasets and requires additional computational resources. Since Benchmark 2 conducts fine-tuning using synthetic data, it still faces generalization challenges. In contrast, our GSFM undergoes fine-tuning directly on field data in an SSL manner, enabling it to address the generalization challenges faced by Benchmark 2.

## Field data examples

In this section, we fine-tune the pre-trained GSFM on real data and evaluate the performance of our fine-tuned GSFM on field data across denoising, interpolation, and low-frequency extrapolation tasks.

### Field data and fine-tuning configuration

We use a marine field dataset acquired using a streamer survey in North West Australia. The original dataset consists of 1824 shot gathers activated with air gun sources, with an approximate horizontal spacing of 18.75 m and a sampling rate of 1 ms. Each shot gather contains 648 receivers, spaced 12.5 m apart. For testing purposes, we select every third shot gather starting from the left, resulting in a total of 200 shot gathers. To reduce the computational burden during training, the number of receivers in the field data was reduced to 324, and the time samples are downsampled from 6016 to 376 [harsuko2024optimizing].

During fine-tuning on field data using the three different strategies, we ensure a fair comparison by using the same total number of iterations, set to 30000. However, in Strategy 3, these 30000 iterations are divided into 10 stages (```latex $S = 10$ ```, ```latex $N_{\text{stage}} = 3000$ ```). The learning rate is fixed at ```latex $5e\text{-}5$ ```, the batch size is set to 4, and the EMA rate is configured to 0.999. Only a single sampling step is used for field data predictions.

Since the field data does not include random noise, we do not fine-tune for the denoising task. Instead, we independently optimize the pre-trained GSFM for the backscattered noise attenuation, interpolation, and low-frequency extrapolation tasks in a sequential workflow:

1. **Backscattered noise attenuation**: Fine-tuning is first applied to address inherent noise in the field data. The noise added to pseudo labels is extracted from the area outside the first arrival.
2. **Interpolation**: After obtaining denoised results, 50% of seismic traces are artificially removed from the data to construct incomplete seismic data for fine-tuning.
3. **Low-frequency extrapolation**: The denoised data is used as initial training data for fine-tuning the GSFM to perform low-frequency extrapolation.

### Backscattered noise attenuation

[IMAGE: Backscattered noise attenuation comparison between fine-tuned GSFM and benchmarks on field data]

Both benchmarks fail to suppress the backscattered noise effectively, leading to severe signal leakage in their outputs. In contrast, our fine-tuned GSFM demonstrates superior performance, successfully preserving the true signal while significantly reducing the noise.

[IMAGE: Backscattered noise attenuation comparison between different fine-tuning strategies]

Strategy 1 shows minimal improvement over the pre-trained GSFM. Strategy 2 provides moderate enhancements in noise reduction, yet residuals persist. Strategy 3 achieves substantial gains, effectively reducing noise and preserving the signal structure.

[IMAGE: Denoised products at different fine-tuning stages (stages 1, 5, and 10)]

The stepwise refinement process is clearly evident. Early in the fine-tuning phase (stage 1), signal leakage remains prominent. By stage 5, the model demonstrates reduced signal leakage. At stage 10, the processed output is of high quality, with excellent noise attenuation and signal preservation.

### Interpolation

[IMAGE: Interpolation performance comparison between fine-tuning strategies]

Strategy 1 produces results that closely resemble the pre-trained GSFM's output, leaving significant interpolation gaps unaddressed. Strategy 2 improves interpolation performance by reducing gaps, yet some residual inaccuracies remain. Strategy 3 delivers the most refined results, reconstructing the missing traces with superior accuracy.

[IMAGE: Interpolated products at different fine-tuning stages]

[IMAGE: MSE metric of interpolation results across 200 shot gathers at different fine-tuning stages]

The MSE trends reveal a clear improvement as fine-tuning progresses. Stage 10 yields the lowest MSE across all shot gathers.

### Low-frequency extrapolation

[IMAGE: Low-frequency extrapolation comparison between fine-tuning strategies]

Strategy 1 offers only negligible improvements over the pre-trained GSFM. Strategy 2 delivers better results but struggles to effectively recover frequencies below 2 Hz. Strategy 3 achieves the most substantial enhancement, recovering nearly complete low-frequency components both below 4 Hz and 2 Hz.

[IMAGE: Low-frequency extrapolation at different fine-tuning stages]

At stage 10, the model delivers its best performance, with nearly complete recovery of low-frequency components, including those below 2 Hz.

### Uncertainty quantification

Traditional NN-based seismic processing paradigms lack the ability to effectively quantify the uncertainty of their processing products. Due to the inherent probabilistic nature of GDMs, they naturally lend themselves to estimating uncertainty. Although our GSFM employs a deterministic sampling process by targeting ```latex $x_0$ ``` and setting ```latex $\sigma_t=0$ ```, the sampling process still originates from random noise. Therefore, when the seed for sampling is randomized, the initial random noise varies, leading to slight differences in the generated processing results.

The method for uncertainty quantification is straightforward. The input condition is replicated ```latex $B$ ``` times along the batch dimension. Then, ```latex $B$ ``` different random noise are sampled, and the GSFM processes these inputs to generate ```latex $B$ ``` corresponding predictions. The mean provides the final predicted result, while the standard deviation provides an indication of uncertainty.

[IMAGE: Mean of multiple interpolation results and corresponding uncertainty for field data]

The regions with significant signal leakage exhibit higher uncertainty values. This correlation demonstrates that the calculated uncertainty effectively identifies regions in the processed results with higher errors.

[IMAGE: Uncertainty dynamics at fine-tuning stages for field data interpolation]

Key observations:
- **Prediction consistency**: Across all stages, the mean prediction results appear visually similar.
- **Reduction in signal leakage**: As fine-tuning progresses (from stage 1 to stage 10), signal leakage consistently decreases.
- **Uncertainty dynamics**: Standard deviation figures reveal a steady reduction in uncertainty as fine-tuning progresses, aligning with reduced signal leakage.

These findings highlight the potential of using uncertainty as a guiding metric during fine-tuning. If the uncertainty stabilizes and shows no significant reduction across successive iterations, it may indicate convergence.

## Discussion

### Comparison of predicting target and noise

We compare the performance of GSFM trained to predict target ```latex $x_0$ ``` with a GSFM variant trained to predict noise ```latex $\epsilon$ ```. The noise-targeted GSFM produces suboptimal results when using lower sampling step sizes (```latex $T=1$ ```, 100, and 500), with significant residual noise. Even with maximum sampling steps (```latex $T=1000$ ```), noticeable signal leakage persists.

**Table 4: Comparison of denoising MSE between x0-targeted and noise-targeted GSFM**

| Noise Level | GSFM (x0) | Noise T=1 | Noise T=100 | Noise T=500 | Noise T=1000 |
|-------------|-----------|-----------|-------------|-------------|--------------|
| 10% | **3.09e-07** | 1.57e-01 | 2.84e-02 | 1.01e-05 | 2.50e-01 |
| 20% | **9.20e-07** | 1.63e-01 | 8.3e-03 | 1.28e-05 | 2.40e-01 |
| 30% | **1.67e-06** | 1.64e-01 | 7.1e-03 | 1.58e-05 | 4.46e-06 |
| 40% | **2.61e-06** | 1.61e-01 | 3.43e-05 | 1.46e-05 | 7.63e-06 |
| 50% | **3.79e-06** | 1.61e-01 | 3.32e-05 | 1.73e-05 | 1.30e-01 |
| 60% | **4.60e-06** | 1.60e-01 | 6.40e-03 | 4.50e-03 | 1.53e-01 |

**Table 5: Comparison of interpolation MSE between x0-targeted and noise-targeted GSFM**

| Missing Level | GSFM (x0) | Noise T=1 | Noise T=100 | Noise T=500 | Noise T=1000 |
|---------------|-----------|-----------|-------------|-------------|--------------|
| 10% | **1.45e-08** | 1.58e-01 | 9.20e-03 | 1.28e-05 | 3.88e-06 |
| 20% | **1.93e-08** | 1.66e-01 | 9.63e-04 | 1.27e-05 | 2.50e-01 |
| 30% | **2.85e-08** | 1.66e-01 | 1.90e-05 | 4.93e-05 | 3.03e-06 |
| 40% | **8.54e-08** | 1.62e-01 | 3.20e-05 | 1.13e-05 | 2.39e-06 |
| 50% | **4.08e-08** | 1.65e-01 | 1.10e-03 | 2.17e-01 | 1.80e-06 |
| 60% | **3.65e-07** | 1.60e-01 | 4.40e-03 | 1.32e-05 | 3.65e-06 |

The ```latex $x_0$ ```-targeted GSFM consistently achieves lower MSE values irrespective of the sampling step size. The noise-targeted model requires high step sizes (```latex $T=500$ ``` and 1000) to deliver results comparable to the ```latex $x_0$ ```-targeted GSFM using step size ```latex $T=1$ ```.

### Comparison of different sampling steps

**Table 6: MSE metric of x0-targeted GSFM using different sampling steps**

| Task | T=1 | T=10 | T=50 | T=100 | T=500 | T=1000 |
|------|-----|------|------|-------|-------|--------|
| Denoising | 1.67e-06 | 1.61e-06 | 1.61e-06 | 1.61e-06 | 1.61e-06 | 1.61e-06 |
| Backscattered noise attenuation | 9.59e-07 | 9.59e-07 | 9.59e-07 | 9.59e-07 | 9.59e-07 | 9.58e-07 |
| Interpolation | 4.08e-08 | 4.08e-08 | 4.07e-08 | 4.08e-08 | 4.07e-08 | 4.07e-08 |
| Low-frequency extrapolation | 6.0e-07 | 5.91e-07 | 6.08e-07 | 6.16e-07 | 6.05e-07 | 6.05e-07 |

The performance of our ```latex $x_0$ ```-targeted GSFM remains highly stable across varying sampling steps. This finding suggests that our ```latex $x_0$ ```-targeted GSFM is highly efficient and robust, achieving optimal performance with minimal sampling steps -- an important advantage for practical applications.

### Computation and memory consumption

Our GSFM employs a U-Net-based network architecture shared with the two benchmarks. The computational time and memory consumption for pre-training are nearly identical. By adopting the ```latex $x_0$ ```-target-based GDM, our GSFM achieves comparable results with a single sampling step, significantly reducing the computational burden during inference. The iterative fine-tuning strategy ensures that additional time cost remains reasonable.

### Limitations and Future work

The core idea behind our GSFM is to leverage GDMs to capture and learn the joint distribution of seismic data. However, certain SPTs require transformations across fundamentally different domains. For example, tasks like velocity analysis require transforming seismic shot gathers into a smooth background velocity model, effectively converting data from the time domain to the depth domain. The distribution of background velocity models is inherently disjoint from the distribution of seismic shot gathers, posing significant challenges for optimization.

This limitation highlights a current constraint of our GSFM: it struggles to handle tasks where the target data is not of the same domain as the input data. For pure seismic preprocessing, this is fine. However, for additional tasks like velocity model building, this is a limitation. Future work aims to extend GSFM by developing strategies to handle non-overlapping distributions, possibly through modular designs or hybrid frameworks.

## Conclusions

We introduced the generative seismic foundation model (GSFM), a novel framework built upon generative diffusion models (GDMs) to address multi-task seismic processing. GSFM leverages a generative approach to learn and capture the underlying joint distribution of seismic data, aiming to represent clean, complete, and broadband characteristics. By encoding tasks with class labels and integrating synthetic pre-training with iterative fine-tuning on field data, GSFM achieves a unified framework for seismic denoising, backscattered noise attenuation, interpolation, and low-frequency extrapolation.

On synthetic data, our pre-trained GSFM achieved performance comparable to traditional pre-training strategies followed by extensive fine-tuning, while significantly outperforming a benchmark model with the same architecture across all tasks. On field data, the iterative fine-tuning strategy effectively addressed the generalization challenges inherent in traditional pre-training and fine-tuning paradigms. The fine-tuned GSFM consistently outperformed both benchmarks.

Through comparative experiments, we demonstrated that our iterative fine-tuning strategy is optimal for refining GSFM on field data. Furthermore, the uncertainty quantification capability of our GSFM highlighted its potential for evaluating the reliability of processing results, adding a layer of interpretability critical for decision-making in seismic workflows.

In summary, GSFM represents a significant step forward in seismic processing by unifying multiple tasks under a single generative framework. Its ability to generalize across synthetic and field data, coupled with its efficiency and versatility, demonstrates the value of incorporating GDMs into geophysical applications.

## Code and Data Availability

The data and accompanying codes that support the findings of this study are available at: https://github.com/DeepWave-KAUST/GSFM.
