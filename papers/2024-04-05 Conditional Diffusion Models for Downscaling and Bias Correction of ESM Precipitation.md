Accurately simulating rainfall is essential to understand the impacts of climate change, especially extreme events such as floods and droughts. Climate models simulate the atmosphere at a coarse resolution and often misrepresent precipitation, leading to biased and overly smooth fields. We improve climate model precipitation using a generative machine learning model that is data-efficient, preserves key climate signals such as trends and variability, and significantly improves the representation of extreme events.

# Introduction

With global warming, we anticipate more intense rainfall events and associated natural hazards, e.g., in terms of floods and landslides, in many regions of the world [IPCC_AR6_SYR_SPM_2023]. Understanding and accurately simulating precipitation is particularly important for adaptation planning and, hence, for mitigating damages and reducing risks associated with climate change. Earth System Models (ESMs) play a crucial role in simulating precipitation patterns for both historical and future scenarios. However, these simulations are computationally extremely demanding, primarily because they require solving complex partial differential equations. To manage the computational load, ESMs resort to approximate solutions on discretized grids with coarse spatial resolution (typically around 100 km). The consequence is that these models do not resolve small-scale dynamics, such as many of the processes relevant to precipitation generation. This leads to considerable biases in the ESM fields compared to observations. Moreover, the coarse spatial resolution prevents accurate projections of localized precipitation extremes. Therefore, precipitation fields simulated by ESMs cannot be used directly for impact assessments [zelinka2020causes] and especially tasks such as water resource and flood management, which require precise spatial data at high resolution [gutmann2014intercomparison].

Statistical bias correction methods can be used as a post-processing to adjust statistical biases. Quantile mapping (QM) is the most common method for improving the statistics of ESM precipitation fields [tong2021bias; gudmundsson2012downscaling; cannon2015bias; miao2019improving]. QM reduces the bias using a mapping that, locally at each grid cell, aligns the estimated cumulative distribution of the model output with the observed precipitation patterns over a reference time period. Although QM is effective in correcting distributions of single grid cells, it falls short in improving the spatial structure and patterns of precipitation simulations [hess2022physically]. A visual inspection shows that ESM precipitation remains too smooth compared to the observational data after applying quantile mapping.

To address these problems, deep learning methods have recently been introduced [pan2019improving; li2022convolutional; hess2023deep; pan2021learning; franccois2021adjusting; hess2022physically]. In these approaches, the statistical relationships between model simulations and observational data are learned implicitly. A general constraint when using machine learning methods for bias correction is that individual samples of observational and Earth System Model data are always unpaired. In this context, a sample is a specific weather situation at a specific point in time. The reason for this lack of pairs is that simulations, even with very similar initial conditions, diverge already after a short period of time due to the chaotic nature of the underlying atmospheric dynamics. Currently, one can, therefore, not utilize the wide range of supervised machine learning (ML) techniques that have shown great success in various disciplines in recent years and the available options are consequently restricted to self- and unsupervised machine learning methods. Recent studies [hess2023deep; pan2021learning; franccois2021adjusting; hess2022physically] applied generative adversarial networks (GANs [goodfellow2020generative]) and specifically cycleGANs [Zhu_2017_ICCV] to improve upon existing bias correction techniques. A major limitation of GAN-based approaches is that the stability and convergence of the training process are difficult to control and that it is challenging to find metrics that indicate training convergence. In addition, GANs often suffer from mode collapse, where only a part of the target probability distribution is approximated by the GAN.

As noted above, the low spatial resolution of ESM fields prevents local risk and impact assessment, necessitating the additional use of downscaling methods. In line with the climate literature, we refer to increasing the spatial resolution as downscaling throughout our manuscript, although we are aware that, especially in the machine learning literature, the term upsampling is more prevalent. We use the term downscaling only when we want to increase the information content in an image as well as the number of pixels. When we refer to upsampling (downsampling), we only mean an increase (decrease) in the number of pixels. Statistical downscaling aims to learn a transformation from the low-resolution ESM fields to high-resolution observations. Recent developments lean towards using machine learning methods for this task [rampal2022high; hobeichi2023using; rampal2024downscaling]. The potential for machine learning-based downscaling methods was already shown in [vandal2017deepsd; van2023deep; doury2023regional; doury2024suitability; rampal2025reliable].

Recently, Hess et al. [hess2025fast] used an unconditional consistency model (CM) for downscaling 3° ```latex $\times$ ``` 3.75° precipitation data to 0.75°```latex $\times$ ``` 0.9375°. Our work addresses the more challenging task of downscaling from 1°```latex $\times$ ``` 1.25° to 0.25°```latex $\times$ ``` 0.25° resolution, a scale essential for regional impact assessments. We show that the consistency model applied to our higher resolution setting with limited amounts of training data struggles to approximate the distribution, highlighting an advantage of our conditional training approach. The analysis is further extended to out-of-distribution scenarios, particularly those involving extreme precipitation and future emission projections.

Diffusion models (DMs) have recently emerged as the state-of-the-art ML approach for conditional image generation [saharia2022photorealistic; rombach2022high; saharia2022image] and image-to-image translation [saharia2022palette], mostly outperforming GANs across different tasks. Diffusion models (Fig. 1 and fig. 7) avoid the common issues present with GANs in exchange for slower inference speed. A diffusion model consists of a forward and a backward process. During the forward process, noise is added to an image in subsequent steps to gradually remove its content. The amount of noise added follows a predefined equation. During the backward process, a neural network is trained to reverse each of these individual noising steps to recover the original image. The trained diffusion model can generate an image of the training data distribution, given a noise image as input. Recent work [wan2024debias] introduced a framework for downscaling and bias correction, combining a diffusion model that is responsible for downscaling and a model based on optimal transport responsible for bias correction. Optimal transport [cuturi2013sinkhorn] learns a map between two data distributions in an unsupervised setting. However, this framework is computationally expensive and has so far only been demonstrated on synthetic datasets, without evaluation on real-world observational or ESM fields. In contrast, our approach is computationally efficient by combining computationally efficient QM for large-scale bias correction with a conditional diffusion model that performs both small-scale bias correction and downscaling by generating matching small-scale patterns. We demonstrate its effectiveness for precipitation data, highlighting its ability to correct biases, downscale accurately, and capture extremes, uncertainties, and trends. A major advantage is that our conditional training allows us to use a relatively small dataset for training and still capture the distribution accurately. In contrast, unconditional models often need considerably more data to capture the full data variability, as we also show in our comparison to [hess2025fast] (see fig. 8).

Existing work leveraging state-of-the-art ML methods for bias correction and downscaling does not systematically investigate out-of-distribution scenarios like future emission scenarios and especially the representation of extreme events of the generative models in detail. Understanding the generalization performance of the models under these conditions is, however, crucial for impact modelers who rely on these outputs for risk assessments under future climate conditions. We will therefore present a detailed analysis of the generalization capability of our approach, both in terms of its performance in preserving climate change trends, as well as in capturing extreme events and their trends.

A major challenge in bias correction and downscaling of ESMs is that the whole class of state-of-the-art supervised machine learning methods is not applicable in this setting. This is due to two fundamental issues. First, due to the chaotic nature of atmosphere and ocean dynamics, ESM simulations and observational data are inherently unpaired. This means that the weather on a specific day in an ESM simulation does not correspond to the observed weather on the same day, which prevents directly training a supervised ML method on the task. Second, training a ML model on observational data and applying it to ESM data is unreliable due to the substantial distribution shift between both datasets caused by systematic biases in the ESM. This violation of the assumption of independently and identically distributed (i.i.d.) data leads to poor generalization. Our proposed framework directly addresses both challenges. We reformulate the problem in a novel way, which allows us to train arbitrary ML models in a conditional setup without the need for explicit ESM-observation pairs, while at the same time resolving the distribution shift.

We present a novel framework based on state-of-the-art conditional diffusion models that allows us to perform both bias correction and downscaling with one single neural network, which only takes precipitation as input and output. We use a conditional diffusion model (Fig. 1 and fig.  7) to correct low-resolution (LR) ESM fields toward high-resolution (HR) observational data (OBS). The supervised formulation of the task allows us to train a conditional diffusion model that is more data efficient (requiring less training data) than its unconditional counterpart because it is trained to only learn the small-scale precipitation patterns, given the large-scale patterns. The model then learns to copy the correct large-scale information from the condition channel. An unconditional model that learns to approximate the full distribution of precipitation at all scales is unnecessarily complex for the task. In general, our task of bias correction and downscaling can be seen as taking a field from a distribution ```latex $p(\text{ESM})$ ``` and transforming it into a field from a conditional distribution ```latex $p(\text{OBS}|\text{ESM})$ ```.

A key idea of our framework is to reformulate the problem in a way that yields a clear training objective. A key part of it is a statistical mapping to an embedding space, which ensures that training and inference data are identically distributed. We achieve this by introducing transformations ```latex $f$ ``` and ```latex $g$ ``` that map observational (OBS) and ESM data to a shared embedding space (see Methods and Fig. 1A). This space is explicitly designed to solve the two fundamental issues mentioned above: it creates a valid supervised objective by providing paired samples of observational data and their perturbed embeddings (OBS, f(OBS)), and it ensures the training and inference distributions match by making the distributions of the embeddings ```latex $f(OBS)$ ``` and ```latex $g(ESM)$ ``` similar. On this shared embedding space, we can train a conditional diffusion model to approximate the inverse of ```latex $f$ ``` (Fig. 1B and Fig. 1C). The neural network is trained to predict the clean OBS data given the embedded OBS data, thereby only relying on pairs between OBS and ```latex $f(OBS)$ ```. For inference, the ESM data is mapped into the same embedding space using the transformation ```latex $g$ ```. The statistical similarity of the resulting embeddings ```latex $f(OBS)$ ``` and ```latex $g(ESM)$ ``` enables the diffusion model, which was trained exclusively on observational data, to generalize effectively to downscale and bias-correct the ESM fields. The diffusion model will map the embedded ESM data towards the distribution of observational data, resulting in bias-corrected and downscaled ESM fields.

This framework offers great flexibility as it can be applied to any ESM, with minimal adjustments in the embedding pipeline. The embedding transformation for the ESM has two key components. First, we use quantile mapping (QM) as a fast and effective method to correct large-scale biases in the ESM. Second, we introduce noise to remove small-scale information in the precipitation fields. We define large scales as those spatial scales that are effectively corrected using QM alone, while smaller spatial scales, which require additional correction, are referred to as small scales (Fig. 2). This noise selectively targets small-scale patterns, leaving intact large-scale patterns. In our approach, quantile mapping addresses large-scale biases, while the small-scale biases and downscaling are handled by our diffusion model. The task of our model is then to perform downscaling and bias correction by regenerating these small-scale features, in a way that ensures consistency with the preserved large-scale patterns. When applying our framework on a different region or ESM, it is computationally inexpensive to recompute the quantile mapping (QM) for the embedding transformation.

[IMAGE: Schematic overview of our approach]

# Results

[IMAGE: Power spectral densities (PSDs) for different choices noising scale of the diffusion model]

The ability of the diffusion model ```latex $DM$ ``` to approximate ```latex $f^{-1}$ ``` and the effectiveness of the transformations ```latex $f,g$ ``` will determine the overall performance of the downscaling and bias correction model ```latex $\omega = DM \circ g$ ```. Therefore, we first investigate the effectiveness of the embedding transformations ```latex $f$ ``` and ```latex $g$ ```, followed by an analysis of the downscaling and bias correction performance of the diffusion model ```latex $DM$ ```, on the observational dataset. Once we have shown that both work as expected, we investigate the performance of the diffusion model in bias correction and downscaling of the ESM precipitation fields. Without loss of generality, we chose the 0.25° ERA5 reanalysis [hersbach_era5_2020] data as observational data and the state-of-the-art GFDL-ESM4 [dunne_gfdl_2020] at 1° as our ESM.

## Embedding evaluation

Transformations ```latex $f,g$ ``` are chosen so that they map observational (OBS) and model (ESM) data to a common embedding space ```latex $\mathbf{V^{emb}}$ ```, where all samples are identically distributed. For constructing ```latex $f$ ``` and ```latex $g$ ``` we need ```latex $f(ERA5)$ ``` and ```latex $g(GFDL)$ ``` to be unbiased with respect to each other. The transformations need to be chosen such that the embedded data share the same distribution and the same power spectral density (PSD). We assess if they are statistically unbiased towards each other by analyzing their histograms and latitude / longitude profiles, as well as their spatial PSDs (after applying pre-processing transformations). Figure 9 shows that ```latex $f(ERA5)$ ``` and ```latex $g(GFDL)$ ``` have the same spatial distribution (fig. 9A) with minor differences in temporal statistics shown by the histogram (fig. 9B) and latitudinal/ longitudinal profiles (fig. 9C and fig. 9D).

The individual operations that make up the transformations ```latex $f$ ``` and ```latex $g$ ``` do not change the large-scale patterns of their respective inputs, as desired for a valid bias correction. The goal of downscaling and bias correction ```latex $\omega$ ``` (Fig. 1) is to rely on the unbiased large-scale patterns of the ESM and correct statistics, as well as small-scale patterns. The transformation ```latex $g$ ``` preserves the unbiased information from the ESM by construction. Therefore, we want the diffusion model, approximating ```latex $f^{-1}$ ```, to also preserve unbiased information.

The extreme case of erasing all detail with large amounts of noise (Fig. 2A) leads to learning the unconditional distribution ```latex $p(ERA5)$ ```, which is then not a correction of ```latex $GFDL$ ``` but a generative emulation of the ERA5 reanalysis data. We tested this by adding the same amount of noise to the output of our diffusion model that was added to create ```latex $g(GFDL)$ ```. This ensures that both the downscaled and bias-corrected fields, as well as the original GFDL fields, lack the small-scale details up to the same point.

To verify that large-scale patterns are preserved by the diffusion model, we compute image similarity metrics between the low pass filtered version of the input of the diffusion model (embedded ERA5 data ```latex $f(ERA5)$ ```) and the low pass filtered output of the diffusion model ```latex $DM(f(ERA5))$ ```. The output of the low pass filter leaves the large-scale features unchanged. The comparison yields an average structural similarity index (SSIM [SSIM]) value of 0.85 and a Pearson correlation coefficient of 0.95 for the validation dataset. This verifies that large-scale patterns are well preserved by the diffusion model.

Our diffusion model is able to reconstruct high-resolution fields following the ERA5 distribution from embedded ERA5 fields ```latex $f(ERA5)$ ```, with only minor discrepancies in small-scale patterns (fig.  10A). A comparison between the mean absolute spatial-temporal difference between the first downsampled and then bilinearly upsampled ERA5 and the ground truth ERA5 fields at 0.25° yields a mean bias of 0.27 mm d```latex $^{-1}$ ```. The downscaling of our diffusion model reduces this bias to 0.21 mm d```latex $^{-1}$ ``` (at 0.25°). Our diffusion model thus approximates ```latex $f^{-1}$ ``` well, and we successfully created a shared embedding space in which ```latex $f(ERA5)$ ``` and ```latex $g(GFDL)$ ``` are identically distributed.

## Evaluation of downscaling and bias correction performance

[IMAGE: Comparative visualization of individual randomly selected samples]

[IMAGE: Comparison of climatologies and model biases]

[IMAGE: Evaluation of our diffusion model's performance for downscaling and bias correction]

We investigate the inference performance of our diffusion model on embedded GFDL data ```latex $g(GFDL)$ ```. We compare the downscaling and bias correction performance of our diffusion model to a benchmark consisting of first applying bilinear upsampling followed by QM for bias correction.

Figure 3 presents a qualitative comparison between the different individual precipitation fields. The upsampled ```latex $GFDL$ ``` fields, as well as our benchmark are visually too smooth. They therefore appear blurry compared to the ERA5 precipitation fields despite having the same spatial resolution of 0.25°. Our diffusion model produces high-resolution detailed outputs that are visually indistinguishable from the ```latex $ERA5$ ``` reanalysis that we treat as the ground truth. We also compared our diffusion model to a different state-of-the-art diffusion model implementation, EDM [karras2022elucidating]. The EDM model was trained for the same number of epochs, while taking twice as long for one. The EDM almost perfectly corrects the spectrum (fig. 11A). However in both the histogram (fig. 11B) as well as in latitudinal and longitudinal profiles (fig. 11C and fig. 11D) the EDM model is inferior to our proposed diffusion model. We also compared our method against a VQ-VAE-based generative model, finding that our model outperforms it across these metrics (for details, see SI Sec. S7 and fig. 12).

To further validate our choice of architecture, we also compare the diffusion model's performance against two other state-of-the-art deep learning models, a UNet and a Transformer, using the same experimental setup. The results (fig. 13) show a significant advantage for the DM in reproducing small-scale spatial patterns, by aligning better with the ERA5 reference spectrum (fig. 13A). In contrast, all three models perform comparably well in correcting the overall precipitation distribution and the latitudinal/longitudinal mean profiles (fig. 13B-D). The generative process of the diffusion model is particularly well-suited for correcting the high-frequency spatial details. Another advantage over both deterministic models is the DM's stochasticity, which allows for the generation of ensembles to quantify uncertainty.

The analysis of temporally averaged precipitation fields shows that the climatology of the diffusion model-corrected GFDL data (Fig. 4A) and the high-resolution ERA5 data (Fig. 4C) is more accurate and less smooth than the climatology of the GFDL data (Fig. 4B). A comparison between the absolute temporally and absolute spatial-temporally averaged diffusion model corrected GFDL and ERA5 fields (Fig. 4D) yields a bias of 0.32 mm d```latex $^{-1}$ ```. This is a substantial improvement over the original GFDL dataset, which yields a bias of 0.69 mm d```latex $^{-1}$ ``` (Fig. 4E). Our diffusion model performs comparably with the state-of-the-art bias correction performance of our benchmark, which is by design optimal for this task, at 0.26 mm d```latex $^{-1}$ ``` (Fig. 4F). For a quantitative comparison including Root Mean Square Error (RMSE) and Pearson correlation for these climatologies, see Table  1.

There are large differences between the GFDL and ERA5 data in small-scale patterns (Fig. 5A). The histogram of precipitation intensities (Fig. 5B) also confirms that the ESM is only really accurate for precipitation events up to 40 mm d```latex $^{-1}$ ```, after which the respective frequencies diverge. The latitudinal and longitudinal mean profiles (Fig. 5C and Fig. 5D) indicate the presence of regional biases.

Our framework demonstrates comparable skill to the QM-based benchmark in correcting the latitude and longitude profiles, for which QM is near optimal by construction (Fig. 5C and Fig. 5D). Comparing the histograms (Fig. 5B and fig. 14) shows that our diffusion model is superior compared to the benchmark, strongly outperforming it for extreme values, in particular.

For the spatial patterns and especially the small-scale spatial features, the QM benchmark shows only slight improvements over the original GFDL data (Fig. 5A). The diffusion model is vastly superior in correcting these small-scale spatial patterns (Fig. 5A and Fig. 3) and almost completely removes the small-scale biases, as seen in the spatial PSD.

To verify that large-scale patterns are preserved by the diffusion model, we compute image similarity metrics between the low-pass-filtered embedded GFDL data and the low-pass filtered output of the diffusion model. The comparison yields an average structural similarity index value (SSIM [SSIM]) of 0.77 and a Pearson correlation coefficient of 0.90, verifying that large-scale patterns are well preserved by the diffusion model.

We also assess our model's performance on extreme precipitation events. For this, we use the R95p metric, which is defined as the total annual precipitation from wet days (PR ```latex $>$ ``` 1 mm d```latex $^{-1}$ ```) that exceed the 95th percentile of our reference period. The difference between the R95p values for the ERA5 and DM corrected GFDL (fig. 15A), the ERA5 and QM corrected GFDL (fig. 15B) and ERA5 and GFDL (fig. 15C), demonstrate that the diffusion model effectively corrects the bias in extreme precipitation events, performing at least as well as the quantile mapping correction. To further test the model's performance on correcting characteristics of rainfall events in the tail of the distribution, we conduct a return-level analysis for extreme rainfall events (fig. 16). We calculated the average return periods for both moderately extreme (```latex $>$ ```50 mm d```latex $^{-1}$ ```) and very extreme (```latex $>$ ```80 mm d```latex $^{-1}$ ```) events. The raw GFDL model has a significant wet bias, substantially underestimating the return periods (3.33 years and 4.60 years) compared to the ERA5 reference (4.11 years and 7.38 years). Our DM successfully mitigates this bias, yielding more realistic return periods of 4.18 and 7.98 years.

We show that the spatial correlation between the climatologies is improved through our method by computing the Pearson correlation between the temporally averaged fields. The Pearson correlation between ERA5 and GFDL climatology is 0.83, while the correlation between ERA5 and DM-corrected GFDL is 0.98, which is the same as that for the QM-corrected GFDL data. We also investigate how our DM captures the statistics of consecutive dry days (CDD) and consecutive wet days (CWD) compared to the QM benchmark and the raw GFDL (fig. 17). Our diffusion model produces superior CDD (fig. 18A and fig. 18B) and CWD (fig. 18D and fig. 18E) statistics compared to our QM benchmark and GFDL, as shown in the difference plots of CDD / CWD.

Our method therefore accurately preserves the large-scale precipitation content, while successfully correcting small-scale structure of the precipitation fields, as well as statistical biases in histograms and latitude / longitude profiles (Fig. 5). Finally, we confirmed the temporal consistency of our model by analyzing autocorrelation (fig. 19) and seasonal spell duration (fig. 20). We further validated the robustness of our metrics over an extended validation period (1995-2014) (fig. 21).

We also test our framework on a different region of similar size over South Asia. We choose the same GFDL dataset and keep the experimental setup and evaluation identical to the South American region. The setup for quantile mapping the South Asia GFDL data and creating the benchmark data is also the same. We retrained our DM on mapping embedded ERA5 data (over South Asia) to the original ERA5 data. The noising scale in this experiment is the same as for South America, as the PSDs for both regions diverge around the same spatial scale. The evaluation (fig. 22) confirms that our DM successfully corrects precipitation biases in this new region and most notably outperforms the QM baseline in representing small-scale spatial features.

To further assess our framework's robustness, we conducted an additional experiment using a different ESM. We replaced the GFDL dataset with the MPI-ESM-HR model while keeping the experimental setup and evaluation protocol identical. The MPI and GFDL data diverge at a similar spatial scale in the PSD over the South American domain, allowing us to use the same noising scale hyperparameter s. Quantile delta mapping was applied in the same way as for the GFDL data. Consequently, our diffusion model did not require retraining and could be applied directly to the embedded MPI data at inference. Evaluation on our main metrics (fig. 23) demonstrates our framework's ability to generalize to different ESMs. Our DM not only restores spatial variability across all scales significantly better than the QM benchmark (fig. 23A), but also shows superior ability to reproduce the frequency of extreme precipitation events (fig. 23B).

## Evaluation of ensemble spread

One of the key strengths of our method lies in its capability to generate a diverse ensemble of downscaled and bias-corrected fields from a single condition. We therefore evaluate the ability of our diffusion model to represent and produce accurate estimates of uncertainty, a critical aspect for robust climate modeling and decision-making. We generate a 50-member DM ensemble by running the model 50 times, each conditioned on the same low-resolution ERA5 year, producing one-year trajectories. The corresponding high-resolution year serves as the ground truth. Our results demonstrate that the DM ensemble effectively reproduces the correct precipitation patterns, as shown by the close alignment between the ensemble mean and the high resolution ground truth of ERA5 over the annual cycle (fig. 24). Probabilistic performance, evaluated using CRPS, highlights that the DM significantly outperforms a bilinear baseline, with lower mean CRPS values (0.76 mm d```latex $^{-1}$ ``` vs 0.90 mm d```latex $^{-1}$ ```), as well as better temporally and spatially averaged CRPS (fig. 25). Furthermore, we confirm that the DM ensemble produces well-calibrated uncertainty estimates with a spread-skill plot. Our model achieves near-perfect alignment with the 1:1 line, indicating an accurate representation of uncertainty (fig. 26). For more details see SI Sec. S[ensembleSI\].

## Evaluation on future climate scenarios

Evaluating the performance of downscaling models is crucial for their application in climate impact studies under future climate scenarios. We assess our diffusion model's ability to preserve climate change signals in the underlying ESM simulations by applying it to a high-emission future scenario (SSP5-8.5). Figure 6 compares the relative climate change signal between the late 21st century (2081-2100) and the historical period (1995--2014) for annual mean and annual extreme precipitation. We find that our downscaled 0.25° fields successfully capture the mean precipitation change, closely matching the pattern and magnitude shown in the original 1° GFDL data (Fig. 6A and Fig. 6B). The diffusion model also robustly preserves the climate change signal for extreme precipitation indices, including Rx1Day (wettest day for each year) and R95p (Fig. 6C - Fig. 6F). The spatial patterns of change for the extremes are well-reproduced in the DM-corrected output compared to the original model data. Notably, slight differences are observed in the northwestern domain (Fig. 6C and Fig. 6E), where the DM-correction projects a slightly stronger increase in extreme events under SSP5-8.5. A slight increase in extremes aligns with the diffusion model's bias correction capabilities, reflecting its role in addressing the known under-representation of extreme precipitation in the original GFDL simulations.

Furthermore, we demonstrate that our conditionally trained diffusion model generalizes robustly to unseen future emission scenarios by accurately preserving regional precipitation trends without requiring retraining. We analyze the full annual mean precipitation timeseries from 2015 to 2100 over two representative regions, one exhibiting a strong negative trend and one with a pronounced positive trend (fig. 27). For each region, we compare the annual mean precipitation from the original GFDL SSP5-8.5 data at 1° with the DM-corrected output at 0.25° resolution. The diffusion model consistently preserves the direction and magnitude of the trends found in the original GFDL data across the entire timeseries, for both the negative (fig. 27 blue) and positive trend (fig. 27 red) regions. This demonstrates the model's ability to maintain physically meaningful long-term changes in precipitation, further supporting its generalization capability to future scenarios. Note that the absolute values do not have to coincide, as our model corrects the bias and hence the numerical values. Our model can generalize to unseen climates, preserving the trends, since there is no decrease in performance during inference on GFDL SSP5-8.5 data. Note that our set-up generalized to unseen climate scenarios without any external constraints. The reason why our model preserves trends well is likely given by the fact that the trend is dominated by the large-scale patterns and our model learned to rely on the large-scale patterns of the condition and only generates small-scale patterns.

[IMAGE: Comparison of relative climate change signals]

# Discussion

We introduced a framework based on generative machine learning that allows both bias correction and downscaling of Earth system model fields with a single diffusion model. We achieve this by first mapping observational fields and ESM data to a shared embedding space and then applying the learned inverse of the observation embedding transformation to the embedded ESM fields. We learn the inverse transformation with a conditional diffusion model. Although the underlying observational and ESM fields are unpaired, our framework allows for training on paired data (between observations and embedded observations, see above) and therefore any supervised machine learning method can be adopted to the task, which allows for more flexibility. Supervised methods are often superior in performance and more natural for the downscaling application. The diffusion model is trained on individual samples and has successfully learned to reproduce the statistics of observational data. For the observational ground truth, we chose the ERA5 reanalysis, and for the ESM data to be corrected and downscaled, we chose fields from GFDL-ESM4.

We demonstrated our framework's robustness and generalizability in two additional experiments (Sec. 2.2). When applying the model to a new geographical region in South Asia with the same ESM, the DM requires retraining to adapt to the new regional characteristics. In contrast, when applying the framework to an entirely different ESM (MPI-ESM) over the South American region, the core DM did not need to be retrained since the same noising scale hyperparameter could be used. For different ESMs a new noising scale hyperparameter could be necessary, requiring retraining of our DM with a different noising scale; however this depends on the choice of the spatial scale below which bias correction is desired, and for comparable outputs, we recommend to keep the noising scale ```latex $s$ ``` fixed for different ESMs. For example, to correct multiple ESMs at once, one can use the most heavily biased model to select the noising scale. A single diffusion model can then be trained to correct all ESMs at once, saving significant computational resources during inference. In general, we expect that many ESMs (like the MPI and GFDL model we use) will have similar spatial scales up to which they can capture realistic spatial precipitation features, because they have a similar resolution and have similar limitations from parameterization schemes. In all cases, readjusting the computationally inexpensive Quantile Delta Mapping (QDM) is a required step in the embedding process. The results will also depend on the specific quantile mapping scheme, QDM is chosen to preserve trends.

Our diffusion model corrects small-scale biases of the ESM fields, while completely preserving the large-scale structures, which is key for impact assessments, especially with regard to extremes and local impacts in terms of floods or landslides. The diffusion model performs particularly well for extreme events where traditional methods struggle. The method improves the temporal precipitation distribution at the grid cell level and surpasses the state-of-the-art approach (quantile mapping) in correcting spatial patterns. The downscaling performance has also been shown to be excellent. The diffusion model manages to generate small-scale details for the low resolution ESM data, that match those of high resolution observations. Our model preserves relevant information from the large scales, such as trends and extremes, and generates bias corrected and downscaled precipitation fields with adequate uncertainties.

We show that our method is robust in the out-of-distribution setting of downscaling and bias-correcting the SSP5-8.5 future emission scenario. It is critical for impact assessments that our model is able to accurately preserve the climate change signal of the original SSP5-8.5 data.

A key innovation of our approach is the embedding strategy, which makes the training process independent of the source ESM (apart from a single data-dependent hyperparameter setting the spatial scale below which the fields are corrected), which not only allows the framework to be flexibly applied to downscale and bias-correct a wide range of ESMs but also allows it to be used with different state-of-the-art machine learning backbone models. Another key advantage of our framework is its data efficiency. In our conditional approach the model only needs to learn how to generate small-scale features given the large-scale ones. The task is considerably less demanding than that of unconditional models (e.g., Hess et al. [hess2025fast]), which must learn the entire data distribution from scratch during training. This data efficiency makes our method applicable to datasets with shorter record lengths than ERA5, such as newer observational products.

Indeed, comparing results for generated climatologies between our conditional DM and the unconditional consistency model (CM) by Hess et al. [hess2025fast], it becomes apparent that the CM struggles to learn the target distribution accurately, leading to blurring (fig. 8) that would hinder applications for impact assessments.

Our method is not specific to ERA5 and GFDL because the training of the diffusion model does not directly depend on the ESM choice. A specific ESM choice will only modify a hyperparameter in the embedding transformations ```latex $f$ ``` and ```latex $g$ ```. This, however, requires almost no fine-tuning, as the temporal frequencies can always be matched with quantile mapping. The only parameter that might change for different datasets and use cases is the amount of noise that is added to the observational and ESM datasets. We choose the amount of noise such that the PSDs of the observational ground truth and the ESM fields align beyond a certain scale. This means that we have complete flexibility in deciding which patterns we want to preserve and which we want to correct. This is a major advantage over existing GAN based approaches.

We can decrease the level of detail that is preserved by the diffusion model through increasing the amount of noise added in the transformations ```latex $f$ ``` and ```latex $g$ ```. The amount of noise added is directly proportional to the freedom the diffusion model has in generating diverse outputs and inversely proportional to the model's ability to preserve large-scale patterns.

The downscaled and bias corrected fields will automatically inherit time consistency between different samples up to the noising scale. This means that ESM fields showing two successive days will still look like two successive days after the correction. Future work could build a video diffusion model that inputs and outputs full time series instead of single frames, in order to guarantee time consistency across all scales.

We focused on precipitation data over the South American continent, because of its heavily tailed distribution and the pronounced spatial intermittency. Especially at small scales, precipitation data is extremely challenging to model and therefore serves as a reasonable choice to show the framework's capabilities in a particularly difficult setting. Regional data is chosen due to computational constraints, yet the diverse terrain of our study region, encompassing land, sea, and a wide range of altitudes, enables robust testing of the downscaling and bias correction performance, also given the substantial biases of the GFDL model in this region. We also conducted additional experiments for another region over South Asia, and using another ESM, namely the MPI-ESM-HR, in order to confirm the generality of our approach. The extension to global scales is straightforward and requires no major changes in the architecture. We intend to include more variables in a consistent manner on a global scale in future research. Optimizing the inference strategy, with speedup techniques such as distillation [luhman_knowledge_2021], to decrease the sampling time will prove helpful in this context.

As for any ML model, the ability to generate the rarest extremes is limited by their frequency in the training data. Our conditional approach helps mitigate this to some extent by inheriting the large-scale patterns for these events directly from the ESM.

It is straightforward to extend our methodology to downscaling and bias correction of numerical or data-driven weather predictions on short- to medium-range or even seasonal temporal scales. This would not require any fundamental changes to the architecture. This would, however, require a target dataset with sufficiently high resolution. The ability of the diffusion model to not disturb the temporal consistency between samples can be useful in this scenario. Future work could then focus on extending this model to a multivariate setting, which would be essential for weather prediction and for assessing physical consistency between variables.

# Materials and Methods

## Data

For the study region, we focus on the South American continent and the surrounding oceans. Specifically, the targeted area spans from latitude 0°N to 63°S and from longitude -90°W to -27°E. For the ablation study of the South Asian region, we selected an area from 0.75°N to 64.5°N latitude and from 42°E to 105.75°E longitude. The training period comprises ERA5 data from 1992-01-01 to 2011-01-01. The range of years included for the evaluation on ERA5 and GFDL spans from 2011-01-02 to 2014-12-01. Additionally, an extended 20-year window (1995--2014) is used for analyses requiring greater statistical robustness.

### ERA5

ERA5 [hersbach_era5_2020] is a state-of-the-art atmospheric reanalysis dataset provided by the European Center for Medium-Range Weather Forecasting (ECMWF). Reanalysis refers to the process of combining observations from various sources, such as weather stations, satellites, and other instruments, with a numerical weather model to create a continuous and comprehensive representation of the Earth's atmosphere. We use the daily total precipitation data at 0.25° horizontal resolution as the target for the diffusion model.

### GFDL

The climate model output is taken from a state-of-the-art ESM from Phase 6 of the Coupled Model Intercomparison Project (CMIP6), namely GFDL-ESM4 [dunne_gfdl_2020]. We abbreviate the model with GFDL throughout the paper. The dataset contains daily precipitation data of the first ensemble member (r1i1p1f1) of the historical simulation (esm-hist). The data is available from 1850 to 2014, at 1° latitudinal and 1.25° longitudinal resolution and a daily temporal resolution.

GFDL-ESM4 [dunne_gfdl_2020] SSP5-8.5 represents a high-emission future pathway. We use daily-resolution data from the CMIP6 archive, provided at 1° latitude and 1.25° longitude spatial resolution, covering the period from 2015 to 2100.

### MPI

For our ablation study, we repeat our experiments for the MPI-ESM HR model [gutjahr2019max]. We abbreviate MPI-ESM-HR with MPI in the paper. The data has 0.9375°×0.9375° spatial resolution. We use daily data from 1992 to 2014 using data from 1992-2011 for training and 2011 to 2014 for inference.

## Benchmark dataset

In order to benchmark our method, we first apply bilinear interpolation to increase the resolution of the GFDL fields from 1° to 0.25°. After that, we apply quantile delta mapping [cannon2015bias] to fit the upsampled GFDL data to the original 0.25° ERA5 data. QM is fitted on past observations and can then be used to correct the statistics of any (past/present) ESM field towards that reference period. We use quantile delta mapping (QDM) and chose the ERA5 training period from 1992-01-01 to 2011-01-01 as the reference period to fit the GFDL to ERA5. The benchmark dataset to evaluate our approach is then constructed by applying QM to the GFDL validation period (2011-01-02 to 2014-12-01). Some analyses required a longer evaluation period (1995-2014). To create a fair benchmark for these specific cases QDM was also recalibrated, it was both fitted and applied using data exclusively from this 1995-2014 window. For the SSP5-8.5 data, we use the 1995 to 2014 period of ERA5 as reference data and the historical GFDL data as the model input to fit the QDM. We then apply this mapping to the full time period of the GFDL SSP5-8.5 data (2015--2100).

## Data pre-processing

The units of the GFDL data and MPI data are ```latex $\text{kg m}^{-2} \text{s}^{-1}$ ```, and for ERA5 ```latex $\text{m} \text{h}^{-1}$ ```. For consistency, both are transformed to ```latex $\text{mm d$ ```^{-1}```latex $}$ ```.

Our pre-processing pipeline consists of:

- Only GFDL: rescaling the original 1°```latex $\times$ ``` 1.25° GFDL data to 1```latex $\times$ ```1° (64```latex $\times$ ```64 pixel).

- Only MPI: rescaling the original 0.9375°```latex $\times$ ``` 0.9375° GFDL data to 1```latex $\times$ ```1° (64```latex $\times$ ```64 pixel).

- Add +1 mm d```latex $^{-1}$ ``` precipitation to each value in order to be able to apply a log-transformation to the data.

- Apply the logarithm with base 10 in order to compress the range of values.

- Standardize the data, i.e. subtract the mean and divide by the standard deviation to facilitate training convergence.

- Transform the data to the range \[-1,1\] to facilitate the convergence of the training.

An ablation study (fig. 28) confirms the choice of our precipitation pre-processing pipeline, showing that omitting the log-transformation or the final range scaling leads to spectral discrepancies or distributional biases. As part of the transformation ```latex $g$ ```, the 1° GFDL data is bilinearly upsampled. This and the downsampling and upsampling of ERA5 data, which is part of ```latex $f$ ```, are already done during pre-processing. The downsampling of 0.25° ERA5 data (256```latex $\times$ ```256 pixel) to 1° (64```latex $\times$ ```64 pixel) is done by only keeping every fourth pixel in each field. For the just mentioned upsampling, we apply bilinear interpolation to increase the resolution from 1° to 0.25°. Note that bilinear interpolation to 0.25° does not increase the amount of information in the images compared to the 1° fields. After preprocessing the data as described, the embedding transformation ```latex $f$ ``` is applied. The diffusion model is trained with the preprocessed ```latex $f(ERA5)$ ``` as a condition and the original 0.25° ERA5 data as a target. Before we apply the embedding transformation ```latex $g$ ``` we first pre-process the 1° GFDL data by applying quantile delta mapping (QDM [cannon2015bias]) with 500 quantiles. The bilinear upsampling is then used to increase the resolution to 0.25```latex $\times$ ```0.25° (256```latex $\times$ ```256 pixels). The preprocessed data are used as input to the embedding transformation ```latex $g$ ```. The corresponding output serves as the condition during the inference process of the diffusion model

## Embedding framework

Our framework introduces transformations ```latex $f$ ``` & ```latex $g$ ``` that map OBS and ESM data to a shared embedding space ```latex $f: \mathbf{V^{obs}} \rightarrow \mathbf{V^{emb}}$ ``` and ```latex $g: \mathbf{V^{esm}} \rightarrow \mathbf{V^{emb}}$ ```. The goal is to do bias correction and downscaling of ESM fields, i.e., to obtain samples from the conditional distribution ```latex $\omega = p(OBS|ESM)$ ```. Training a conditional model to approximate this distribution directly is not possible because OBS and ESM are unpaired. Therefore, we will train the model without the ESM data, only using OBS data and utilize a trick to enable transfer learning and inference on the ESM data. We apply transformations on ESM and OBS such that the resulting datasets are similarly distributed and therefore allow for generalization. The arrows in the diagram of Figure 1 show that we can represent the mapping that achieves the bias correction and downscaling as ```latex $\omega = f^{-1} \circ g$ ```. Our idea is to approximate ```latex $f^{-1}$ ``` with a neural network ```latex $f^{-1} \approx \epsilon$ ```. We chose a conditional diffusion model (DM), denoted by the conditional distribution ```latex $p(OBS|f(OBS))$ ```, to approximate ```latex $f^{-1} = DM: \mathbf{V^{emb}} \rightarrow \mathbf{V^{obs}}$ ```. The diffusion model (Fig. 1C) is only trained on pairs ```latex $(OBS,f(OBS))$ ```. The shared embedding space allows us to evaluate the trained model on ESM embeddings ```latex $p(OBS|g(ESM))$ ```, as all embeddings are identically distributed.

### Constructing the embedding space

The goal of ```latex $f$ ``` and ```latex $g$ ``` is to map OBS and ESM to a shared embedding space, where ```latex $f(OBS)$ ``` and ```latex $g(ESM)$ ``` are identically distributed (Fig. 1). To achieve this, both embedded datasets need to be unbiased towards each other. OBS and ESM are biased towards each other in terms of statistical biases between distributions and biases between small-scale patterns visible in the spatial power spectral density (PSD) (fig. 10A).

As mentioned earlier, the input for the embedding transformation ```latex $f$ ``` is 0.25° ERA5 data, which is first preprocessed, then downsampled and upsampled. The input to the embedding transformation ```latex $g$ ``` is the preprocessed and upsampled 0.25° GFDL data. By first downsampling ERA5 to 1° and then upsampling it to 0.25° we ensure that the fields match the information content of the original 1° GFDL fields.

To remove small-scale pattern bias, we apply a noising procedure analogous to the forward diffusion process as part of ```latex $f$ ``` and ```latex $g$ ```. Gaussian noise contains all frequencies in equal measure and the Fourier transform of Gaussian noise is itself Gaussian noise, so its power must be equal across all frequencies in expectation. The power spectrum of pure Gaussian noise corresponds to a horizontal line in the spectrum of Fig. 2A, reflecting the fact that it contains all frequencies in equal amounts. Adding noise to an image results in a hinge shape in the PSD of the noisy images (Fig. 2B, 2C and 2D). Increasing the variance of the noise increases its power and, as a result, its PSD will shift upward. Adding noise hence acts as a low-pass filter, while the variance of the added noise determines the cut-off frequency. Increasing variance leads to higher cut-off points as the power of the noisy frequencies increases. Both ERA5 and GFDL data are noised up to the cutoff frequency, denoted by ```latex $s$ ```. The scale ```latex $s$ ``` is determined by the point where ERA5 and the ESM data (in our case GFDL) start to disagree in their spatial PSDs (Fig. 2), i.e., the intersection between the two. Adding noise in this way ensures that ```latex $f(ERA5)$ ``` is unbiased compared to ```latex $g(GFDL)$ ``` in the PSD by erasing all information beneath ```latex $s$ ```. In our implementation, the transformations ```latex $f$ ``` and ```latex $g$ ``` utilize the same cosine scheduler as the forward diffusion process to add Gaussian noise to the data. ERA5 data undergoes 50 noise steps within ```latex $f$ ```, while ```latex $g$ ``` applies the same 50 noise steps to the GFDL data. We ensure that the observational and ESM data have aligned distributions by incorporating Quantile Mapping (QM) directly into the transformation ```latex $g$ ```. It only needs to be included in ```latex $g$ ```. The quantile-mapped and bilinearly downscaled data is then noised as described above, as part of the embedding transformation. It is important to clarify that QM is not included because the diffusion model is unable to do bias correction. QM is only used as a tool in our framework to ensure that in the embedding space ```latex $f(ERA5)$ ``` and ```latex $g(GFDL)$ ``` are identically distributed, such that ```latex $g(GFDL)$ ``` can be used for the inference of the diffusion model.

### Determining the noising scale

The choice of the spatial scale ```latex $s$ ``` influences up to which scale we correct the spatial PSD. We note that the PSD shows spectral distributions normalized to 1; therefore, we can still observe slight changes above ```latex $s$ ``` when small-scale patterns are corrected. The point ```latex $s$ ``` is a hyperparameter chosen before training and purely depends on the datasets ESM and OBS and can be adjusted to the specific needs in a given context and task.

In the extreme case, where ```latex $s$ ``` is maximal, the conditional images will contain pure noise (Fig. 2A). In this case, the diffusion model is equivalent to an unconditional model. As an unconditional model, the diffusion model will correct all biases at all spatial scales, however, at the expense of completely losing any paring between the condition and the output. We chose ```latex $s$ ``` to be at the intersection of the ERA5 and GFDL spectrum around 512 km (Fig. 2B). Thereby, we trust in the ESM's ability to model large-scale structures above the point ```latex $s$ ```, which we do not want to correct with the diffusion model.

## Network architecture and training

The general architecture of our diffusion model ```latex $DM$ ``` consists of a Denoising Diffusion Probabilistic Model (DDPM) architecture [ho2020denoising] conditioned on low resolution images. For details about diffusion models and conditional diffusion models, see SI Sec. S5.1 and SI Sec. S5.2. We employ current state-of-the-art techniques to facilitate faster convergence and find the following to be important for convergence and sample quality [saharia2022photorealistic]: The memory efficient architecture, "Efficient U-Net", in combination with dynamic clipping and noise conditioning augmentation [ho2022cascaded] turned out to be effective for our relatively small dataset. We adopt the Min-SNR [hang2023efficient] formulation to weight the loss terms of different timesteps based on the clamped signal-to-noise ratios. The diffusion model architecture utilizes a cosine schedule for noising the target data and a linear schedule for the condition during noise condition augmentation with 100 steps each. The diffusion model is trained to do v-prediction. The U-Net follows the ```latex $64 \times 64 \rightarrow 256 \times 256$ ``` Efficient U-Net architecture [saharia2022photorealistic]. The diffusion model has approximately 730 million trainable parameters and is trained for 100 epochs using the ADAM optimizer [kingma2014adam] with a batch size of 2 and a learning rate of ```latex $1e^{-4}$ ```. Note that in the case of fig. 10, where the inference data is also embedded OBS data and there is no ESM data present, the model performs better when being trained and evaluated with 1000 denoising steps, instead of the 100 steps that we used in all our experiments that include ESM data. The model with 100 steps is superior in training and inference speed and also in correcting the histograms, when correcting ESM data. We also compared the effect of not adding noise (SI Sec. S6.1) and the effect of not applying QM (SI Sec. S6.3) as shown in Figures 29, 30, 31, 32, 33, as well as different noise choices (SI Sec. S6.2, fig. 34) during both training and inference.

# Supplementary Text

## Unconditional Diffusion Models

Diffusion Models can be separated into two parts, a forward and a backward diffusion process. The forward diffusion process is a probabilistic model ```latex $q(\mathbf{x}_t|\mathbf{x}_{t-1})$ ``` that produces a noisy version of a given image ```latex $\mathbf{x}_t$ ``` in ```latex $t$ ``` noising steps. The model is chosen to be a Gaussian model: ```latex $q(\mathbf{x}_t|\mathbf{x}_{t-1}) = \mathcal{N}(\mu(\mathbf{x}_{t-1}),\beta_{t} \mathbf{I})$ ```, where ```latex $\beta_t$ ``` controls the amount of noise that is added in each step. In other studies, the model is often chosen to be of the form ```latex $q(\mathbf{x}_t|\mathbf{x}_{t-1}) = \mathcal{N}(\sqrt{1-\beta_{t} } \mathbf{x}_{t-1},\beta_{t} \mathbf{I})$ ``` [ho2020denoising]. In practice, we use the reparametrization trick to sample from a Gaussian distribution by ```latex $\mathcal{N}(\mu, \sigma) = \mu + \sigma \epsilon$ ``` where ```latex $\epsilon\sim \mathcal{N}(0, 1)$ ```. Thus, a noisy version of ```latex $\mathbf{x}_0$ ``` can be obtained as ```latex $\mathbf{x}_t = \sqrt{1-\beta_{t} } \mathbf{x}_{t-1} + \sqrt{\beta_{t}} \epsilon$ ``` after the ```latex $t$ ``` noising steps.The noise scheduler ```latex $\beta_{t}$ ``` is chosen to add small amounts of noise in the beginning and larger amounts later, to preserve a reasonable amount of information throughout the process.

The backward process ```latex $q(\mathbf{x}_{t-1}|\mathbf{x}_t)$ ``` models how to restore the previous version ```latex $q(\mathbf{x}_{t-1})$ ``` of a given image ```latex $\mathbf{x}_t$ ``` at a certain noise step ```latex $t$ ```. This process is also modelled by a Gaussian ```latex $q(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}(\mu(\mathbf{x}_{t}),\sigma{(\mathbf{x}_t)} )$ ```. The problem is that ```latex $\mu(\mathbf{x}_{t})$ ``` is not known.

Using Bayes' theorem, the model can be rewritten as a product of Gaussians: ```latex $q(\mathbf{x}_{t-1}|\mathbf{x}_t) = \frac{q(\mathbf{x}_t|\mathbf{x}_{t-1}) q(\mathbf{x}_{t-1})}{q(\mathbf{x}_t)}$ ```. Each term is a Gaussian distribution, and their product is also a Gaussian distribution. Computing the product and taking the mean of that expression is a valid way to model the backward process. However, in practice, the distribution ```latex $q(\mathbf{x}_{t-1})$ ``` is unknown, so we cannot explicitly compute ```latex $q(\mathbf{x}_{t-1}|\mathbf{x}_t)$ ```.

Predicting the state before a noising operation ```latex $\mathbf{x}_{t-1}$ ``` can be done by conditioning on the noisy image ```latex $\mathbf{x}_t$ ``` and the noise free image ```latex $\mathbf{x}_0$ ``` 
```latex
$$q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0) = \frac{q(\mathbf{x}_t|\mathbf{x}_{t-1},\mathbf{x}_0) q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_t|\mathbf{x}_0)}$$
```

The terms on the right-hand side are Gaussian and can be explicitly computed. The resulting Gaussian has a mean term that depends on ```latex $\mathbf{x}_t$ ``` and ```latex $x_0$ ```, while the variance is a constant depending on the time step ```latex $t$ ```.

```latex
$$\begin{aligned}
q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) & = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0), \sigma_t^2 \mathbf{I}) \label{eq:raw_bw}\
{\sigma}_t^2 & = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t \
\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) & = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t \label{eq:mu_t}
\end{aligned}$$
```

with ```latex $\alpha_t=1-\beta_t, \quad \bar{\alpha}_t=\prod_{s=0}^t \alpha_s$ ```.

The following equation describes how ```latex $\mathbf{x}_0$ ``` is connected to ```latex $\mathbf{x}_t$ ```, when applying the forward diffusion model ```latex $T$ ``` times:

```latex
$$\label{eq:fw_noise}
\begin{aligned}
\mathbf{x}_t & = \sqrt{1-\beta_t} \mathbf{x}_{t-1} + \sqrt{\beta_t} \epsilon_{t-1} \
& = \sqrt{\alpha_t} \mathbf{x}_{t-2} + \sqrt{1-\alpha_t} \epsilon_{t-2} \
& = \ldots \
& = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}
\end{aligned}$$
```

where ```latex $\epsilon, \ldots, \epsilon_{t-2}, \epsilon_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ ```. Solving for ```latex $x_0$ ``` yields:

```latex
$$\label{eq:x_0}
\mathbf{x}_0=\frac{1}{\sqrt{\bar{\alpha}_t}}\left(\mathbf{x}_t-\sqrt{1-\bar{\alpha}_t} \epsilon\right).$$
```

Combining Eq. [eq:x_0\] with Eq. [eq:mu_t\] leads to: 
```latex
$$\tilde{\boldsymbol{\mu}}_t\left(\mathbf{x}_t\right)=\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}\right).$$
```

The backward process (Eq. [eq:raw_bw\]) is then modelled as: 
```latex
$$q(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}\left(\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{\bar\alpha_t}} \epsilon\right), {\sigma}_t^2 \mathbf{I} \right),$$
```

so given a noisy image in step t, this model will predict a less noisy version of that image ```latex $\mathbf{x}_{t-1}$ ```. The only unknown in this equation is ```latex $\epsilon$ ```. The idea is to parameterize ```latex $\epsilon$ ``` with a neural network ```latex ${\epsilon}_\theta$ ```. The objective of the network is then to estimate the noise that was added to a (noisy) image ```latex $\mathbf{x}_{t-1}$ ``` at each time step ```latex $t$ ```:

```latex
$$\label{eq:final_mu}
\tilde{\boldsymbol{\mu}_\theta}\left(\mathbf{x}_t, t\right)=\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t\right)\right)$$
```

Using the reparametrization trick and inserting Eq. [eq:final_mu\] into Eq. [eq:raw_bw\], the backward diffusion process (also called de-noising process) denotes as:

```latex
$$q(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}\left(\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t\right)\right), {\sigma}_t^2 \mathbf{I} \right).$$
```

Following the reparametrization trick, every iteration of the backward process takes the form:

```latex
$$\mathbf{x}_{t-1} \leftarrow \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} {\epsilon}_\theta\left(\mathbf{x}_t, t\right)\right)+\sigma_t\boldsymbol{\epsilon}_t$$
```

with ```latex $\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ ```.

The neural network in the backward diffusion process can be learned through the following algorithm proposed by ho2020denoising: 
```latex
$$\begin{array}{l}
\hline
\textbf{Algorithm 1: }\text{Training} \
\hline
1: \textbf{repeat} \
\begin{array}{l}
2: \quad \mathbf{x}_0 \sim q(\mathbf{x}_0) \
3: \quad t \sim \text{Uniform}(\{1, \ldots, T\}) \
4: \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \
5: \quad \text{Take gradient descent step on} \
       \quad \quad \nabla_\theta\left\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\left(\sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}, t\right)\right\|^2 \
6: \quad \textbf{until } \text{converged} \
\end{array} \
\hline
\end{array}$$
```

Model inference can be achieved with the following algorithm ho2020denoising: 
```latex
$$\begin{array}{l}
\hline
\textbf{Algorithm 2: }\text{Sampling} \
\hline
\begin{array}{l}
1: \mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \
2: \textbf{for } t=T, \ldots, 1 \textbf{ do} \
3: \quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \text{ if } t>1, \text{ else } \mathbf{z}=\mathbf{0} \
4: \quad \mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right) + \sigma_t \mathbf{z} \
5: \textbf{end for} \
6: \textbf{return } \mathbf{x}_0
\end{array} \
\hline
\end{array}$$
```

The inference corresponds to generating a noise-free image, given a noisy input image.

## Conditional Diffusion Models

The goal of a conditional diffusion model is to learn an approximation of the distribution ```latex $p(\mathbf{x}|c)$ ```, where ```latex $c$ ``` is some conditional information. The method learns to model the reverse diffusion process as ```latex $p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_t, c)$ ```. Starting with a pure noisy image, it gets denoised in ```latex $T$ ``` steps. The forward diffusion process is identical to the unconditional case. The difference to the unconditional case is that the model has knowledge about the condition during the backward process. Theoretically, the model could also be conditioned during the forward process. The backward process looks as follows:

```latex
$$\mathbf{x}_{t-1} \leftarrow \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} {\epsilon}_\theta\left(\mathbf{x}_t, c, t\right)\right)+\sigma_t\boldsymbol{\epsilon}_t.$$
```

The condition is integrated by concatenating the condition and the noisy image in the color channel of the images. The network takes a two-channel image as input and produces a one-channel image as output in each backward step.

# Comparative Analysis of Diffusion Model Variants

## No noise

In order to show the importance of noising the biased small scales of the ESM condition, we conducted an experiment where no noise was applied to the condition during both training and inference. The climatology (fig. 29D) of the noiseless model is very similar to our proposed model (fig. 29E), with a mean absolute bias of 0.29 mm d```latex $^{-1}$ ```. The difference between the two methods becomes evident when examining the spatial PSD (fig. 30A), where our method is far superior in correcting the small-scale biases. The histogram (fig. 30B) further highlights the superiority of our approach (noise level n=50), particularly in the range from 50-150 mm d```latex $^{-1}$ ```. Furthermore, the latitudinal and longitudinal profiles (fig. 30C and fig. 30D) reveal that the output of the noise-free model is overall less accurate and exhibits too much variability compared to our proposed approach. We conclude that the model's poorer performance stems from its overreliance on biased information from the GFDL conditions during inference. During training, the model was conditioned on low-resolution ERA5 fields, which are nearly unbiased compared to high-resolution ERA5. As a result, the model only had to make minimal corrections during training and thus will do the same at inference. The good performance of our method is not limited to daily precipitation statistics; we also find that it performs better in representing the ERA5 statistics of consecutive dry and consecutive wet days (fig. 31C and fig. 31F). For the precipitation extremes, the R95p for not noising the data appears slightly worse than with our model (fig. 32A and fig. 32C).

## Different noise levels

We tested the effect of different noise levels (added to the conditions) by analyzing their impact on the downscaling results, maintaining the same noise level during both training and inference. The results for the noiseless case (n=0) were discussed in the previous paragraph, while our originally proposed approach uses a noise level of n=50. The dependence of climatology on the noise level appears small for the smaller noise levels n=0 (fig. 29D and fig. 33D) and n=50 (fig. 29E and fig. 33E), with a mean absolute bias of 0.32 mm d```latex $^{-1}$ ```. However, at a higher noise level (n=80), the climatology deviates slightly (fig. 29F) and the bias becomes more pronounced (fig. 33F), with the mean absolute bias increasing to 0.36 mm d```latex $^{-1}$ ```. When looking at the PSD (fig. 34A) we find that our model (n=50) performs best. Also the histogram (fig. 34B) and latitudinal as well as longitudinal profiles (fig. 34C and fig. 34D) show that the choice of noise level n=50 is optimal. One could argue that, since high noise levels (n=80) remove more information, the model's reconstruction task is harder, because it needs to learn more dependencies and cannot rely on the condition as much. This shows that the conditional information is indeed valuable for the generation. Hence, as we train all models for the same amount of time, we would expect the diffusion model to initially perform worse when removing more information under the same amount of training time.

## No quantile mapping

Fig. 30 (orange) shows that the diffusion model without quantile mapping (QM) struggles to fully correct the characteristic biases in the GFDL data. The climatology and mean average bias presented in fig. 29C and fig. 33C highlight the critical role of QM as part of the embedding transformation. Without QM, the model reduces the mean absolute bias from 0.69 mm d```latex $^{-1}$ ``` to 0.59 mm d```latex $^{-1}$ ```, but falls short of the 0.32 mm d```latex $^{-1}$ ``` achieved when QM is applied. The spatial Pearson correlation between the temporally averaged fields of the DM bias-corrected GFDL data without quantile mapping is 0.89, which is still an improvement over 0.83 of the raw GFDL data. The PSD (fig. 30A) shows that the diffusion model is still perfectly able to correct the spectrum of the embedded GFDL data without applying quantile mapping. The reason why the method without QM struggles with the histogram (fig. 30B) and latitudinal and longitudinal profiles (fig. 30C and fig. 30D) is that without QM, the training and inference distribution of the condition are very different from the distribution of the training condition. The transformation learned by the model during training does not generalize to the out-of-distribution condition when QM is excluded. The PSD is almost the same as before, as applying QM has almost no effect on the spatial variability of individual fields. The application of QM also helps to represent the ERA5 statistics of consecutive dry and consecutive wet days (fig. 31B and fig. 31E). Extreme precipitation, as represented by the R95p metric in fig. 32A with QM and fig. 32B without QM, also benefits from the application of QM.

Overall, we find that QM is an essential part of our method, by ensuring that the embedded GFDL data is distributed like the embedded ERA5 data. In other words, it is important because of our unpaired data setting. Alone, however, it cannot correct the spatial PSD (Fig. 5A) and is therefore not useful for downscaling. The histogram (Fig. 5B) is also corrected a lot better with our diffusion model. Applying our DM also improves the CDD (fig. 31A and fig. 31B) and CWD (fig. 31D and fig. 31E) statistics compared to the benchmark without QM.

# Comparative Analysis to VQ-VAE model

We have included an additional comparison with an established generative model, the VQ-VAE. This model is primarily designed for representation learning and compression, with its training objective focused on mapping data to a latent space and reconstructing it. To adapt the VQ-VAE for our downscaling task, we employ a two-step training process inspired by the original VQ-VAE work [van2017neural].

Step 1: We train the VQ-VAE to compress and reconstruct high-resolution (HR) ERA5 images. Step 2: Using the encoder of the trained model, we construct a dataset of latent representations for each HR field. A conditional PixelCNN is then trained to autoregressively model the prior distribution ```latex $p(z \mid c)$ ``` where ```latex $c$ ``` are samples from the embedded ERA5 distribution. This allows us to sample latents ```latex $z$ ``` conditioned on the embedded ERA5 fields, which are subsequently decoded by the VQ-VAE decoder to generate HR ERA5 fields.

A notable limitation of the VQ-VAE is its inability to generate high-frequency information, as shown by the power spectral density compared to the HR ERA5 ground truth (fig. 12A). Furthermore, the histogram reveals that the VQ-VAE performs significantly worse in capturing the target distribution for high precipitation values (fig. 12B). The model also struggles to consistently match the latitudinal and longitudinal means with the ground truth (fig. 12C, fig. 12D).

Overall, the VQ-VAE lacks the fine detail in small-scale variability and falls short in overall accuracy compared to our diffusion model approach.

# Ensemble uncertainty evaluation

For the following evaluations, we generate a 50 member DM ensemble by conditioning our model 50 times on the same low-resolution ERA5 year. This results in a 50-member ensemble of one-year trajectories. The resulting ground truth will be the corresponding high-resolution ERA5 year.

## Ensemble mean evaluation

To evaluate the accuracy of the DM ensemble in reproducing precipitation patterns, we compare the spatially averaged daily precipitation of a 50-member ensemble of downscaled high-resolution fields, obtained by conditioning our DM on low-resolution (i.e. upscaled) ERA5 fields, to the corresponding ERA5 high-resolution ground truth, at 0.25° resolution over one year. Figure 24 illustrates that the ensemble mean generated by the DM closely aligns with the ERA5 high-resolution ground truth throughout the annual cycle. This demonstrates the ability of the diffusion model ensemble to -- on average -- capture the temporal variability of precipitation while maintaining well-calibrated ensemble members.

## Continuous Ranked Probability Score

We evaluate the probabilistic downscaling performance of our model using the Continuous Ranked Probability Score (CRPS), which extends the concept of the mean absolute error (MAE) to probabilistic forecasts (for details and definition, see [gneiting2007strictly]). In our case, each downscaling realization corresponds to a forecast realization in tasks like weather forecasting. To compute the CRPS, we compare 50 downscaled outputs from our diffusion model to the corresponding high-resolution ERA5 ground truth. The 50-member ensemble is generated by running the diffusion model 50 times, always conditioned on the same low-resolution ERA5 year. As a baseline, we use a deterministic, bilinearly upsampled version of the low-resolution ERA5 reference year and a 50-member VQ-VAE ensemble generated the same way as the DM. The corresponding high-resolution ERA5 year serves as the ground-truth reference. This results in a CRPS for every day of the year and every spatial location. Comparing the spatially and temporally averaged CRPS values (0.90 mm d```latex $^{-1}$ ``` vs 0.76 mm d```latex $^{-1}$ ```) and the maximum CRPS values (14.31 mm d```latex $^{-1}$ ``` vs 26.66 mm d```latex $^{-1}$ ```), our DM significantly outperforms the baseline of bilinear upsampling, while also slightly surpassing the VQ-VAE in terms of both the mean CRPS value (0.80 mm d```latex $^{-1}$ ```) and the maximum CRPS value (15.79 mm d```latex $^{-1}$ ```) (lower CRPS is better).

In fig. 25A we compare the CRPS time series over 1 year, where we averaged the spatial dimensions. Our model is consistently below the bilinearly upsampled baseline and performs on par with the VQ-VAE. Next, the annual mean CRPS is computed, and then the difference is taken between the bilinearly upsampled ERA5 and the diffusion model ensemble (fig. 25B), as well as between the bilinearly upsampled ERA5 and the VQ-VAE ensemble (fig. 25C). The results show that both models outperform the bilinear baseline over the continent. Positive differences (blue) indicate a higher (worse) CRPS value for the upsampling baseline. The performance of the DM and VQ-VAE is comparable, except in the southern part of the Andes, where our DM consistently outperforms the baseline.

## Spread-skill plot

We evaluated the statistical consistency of our model using a spread-skill plot, which evaluates the relationship between the predicted root mean squared spread (RMSS) and the root-mean-squared error (RMSE) of the ensemble mean. The spread-skill plot relates the predicted model spread to the actual model error. We follow the implementation of haynes2023creating, for more details see their work. The x-axis represents the average standard deviation of the DM ensemble distribution, while the y-axis shows the RMSE of the model's mean prediction. Each point on the plot corresponds to a bin of predicted spread values. The model uncertainty estimate is biased if spread values are above the 1:1 line (under-dispersive, over-confident) and if they are under the 1:1 line (over-dispersive / under-confident). A perfect calibration is a spread--skill of 1:1, along the diagonal line.

In our comparison, we use 50 ensemble members generated by conditioning once the DM and once our VQ-VAE model on one year of low-resolution ERA5 data. Both our DM and also our VQ-VAE show very good calibration overall. However, the diffusion model is superior for very large spreads, where the VQ-VAE is overconfident (fig. 26). As desired, our DM matches the 1:1 spread-skill line, indicating that its uncertainty is well calibrated.

## Temporal consistency

Our framework preserves the temporal structure inherited from the ESM and, to some extent, even corrects existing biases. We validated this over a longer validation period between 1995 and 2014. The raw GFDL data shows excessively high temporal autocorrelation, our DM corrects this, closely matching the ERA5 reference (fig. 19). Similarly, the DM produces more realistic statistics for consecutive dry and wet days (CDD/CWD). This improvement is consistent across seasons, with the model substantially reducing GFDL's biases in spell durations (fig. 20). Finally, we confirmed that all primary evaluation metrics are robust over the longer validation period, where the DM yields consistent results (fig. 21) compared to Figure 5. Our method is in general not suited to produce fields that are time consistent on the small spatial scales, as it was trained on single snapshots during training. Achieving this would require training a video model on sequences of frames.

# Quantitative Performance Summary

To provide a coarse quantitative comparison of the model performance, we calculate the Mean Absolute Bias, Pearson Correlation (```latex $r$ ```), and Root Mean Square Error (RMSE) between the temporally averaged climatologies of the reference data (ERA5) and the various model outputs. While these global metrics do not capture fine-scale spatial realism, they serve as a sanity check for overall model fidelity.

::: 
  **Model**                 **Mean Absolute Bias \[mm d```latex $^{-1}$ ```\]**   **Pearson ```latex $r$ ```**   **RMSE \[mm d```latex $^{-1}$ ```\]**
  ------------------------ ---------------------------------------- ----------------- --------------------------
  GFDL                                       0.69                         0.83                   1.19
  QM-Corrected GFDL                          0.26                         0.98                   0.37
  Diffusion Model (Ours)                     0.32                         0.98                   0.44

[IMAGE: Detailed visualisation of the training (blue) and inference process (green) of the conditional diffusion model introduced in our study]

[IMAGE: Comparison of climatology to Consistency Model]

[IMAGE: We demonstrate the effect of the embedding transformations]

[IMAGE: Evaluation of the diffusion model’s performance to reconstruct ERA5 at 0.25° resolution]

[IMAGE: Comparing the downscaling and bias correction performance of our DM to an EDM-diffusion model]

[IMAGE: Comparing the diffusion model’s downscaling and bias correction performance to a VQ-VAE]

[IMAGE: Comparison of different machine learning backbones.]

[IMAGE: Comparison of the absolute histogram errors]

[IMAGE: Performance evaluation of the diffusion model in representing extreme rainfall events]

[IMAGE: Empirical return periods of extreme precipitation events.]

[IMAGE: Duration of consecutive dry and wet periods in the different datasets]

[IMAGE: Performance of different methods regarding consecutive dry and wet period statistics]

[IMAGE: Temporal autocorrelation of daily precipitation over the extended 1995-2014 period.]

[IMAGE: Seasonal evaluation of consecutive dry and wet day spell durations.]

[IMAGE: Model performance evaluation over the extended 1995-2014 period.]

[IMAGE: Evaluation of the diffusion model’s downscaling and bias correction performance over South Asia.]

[IMAGE: Evaluation of the diffusion model’s downscaling and bias correction performance for a different ESM.]

[IMAGE: Spatially averaged daily precipitation at 0.25° resolution over one year]

[IMAGE: Evaluating our DM using the Continuous Ranked Probability Score (CRPS)]

[IMAGE: Spread-skill plot for 50 downscaling ensemble members generated by conditioning on one year of low-resolution ERA5 data]

[IMAGE: Comparison of the trends for the SSP5-8.5 scenario over different sub-regions]

[IMAGE: Ablation study of data pre-processing strategies.]

[IMAGE: Time-averaged precipitation for different diffusion model training and inference variants]

[IMAGE: Evaluating the role of QM and noising in the embedding transformation]

[IMAGE: Comparing consecutive dry day (CDD) and consecutive wet days (CWD) differences]

[IMAGE: We use the R95p metric to investigate the performance of different diffusion model variants on extreme events]

[IMAGE: Time-averaged precipitation differences for various diffusion model training and inference variants]

[IMAGE: Evaluating the role of different noising strengths in the embedding transformation]
