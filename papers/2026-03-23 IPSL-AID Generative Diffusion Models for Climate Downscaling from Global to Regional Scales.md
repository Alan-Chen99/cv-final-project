# IPSL-AID: Generative Diffusion Models for Climate Downscaling from Global to Regional Scales

- **arXiv**: [2604.03275](https://arxiv.org/abs/2604.03275)
- **Submitted**: 2026-03-23
- **Authors**: Kishanthan Kingston, Olivier Boucher, Freddy Bouchet, Pierre Chapel, Rosemary Eade, Jean-Francois Lamarque, Redouane Lguensat, Kazem Ardaneh
- **Affiliations**: Sorbonne University / CNRS / IPSL (Paris); ENS/PSL; IRD; Climate Modeling and Analysis LLC (USA)
- **Journal**: Environmental Data Science, 2026
- **Code**: [github.com/kardaneh/IPSL-AID](https://github.com/kardaneh/IPSL-AID)
- **Keywords**: Climate downscaling, Generative models, Diffusion models, Uncertainty quantification, Machine learning

## Abstract

Effective adaptation and mitigation strategies for climate change require high-resolution projections to inform strategic decision-making. Conventional global climate models, which typically operate at resolutions of 150 to 200 kilometers, lack the capacity to represent essential regional processes. IPSL-AID is a global to regional downscaling tool based on a denoising diffusion probabilistic model designed to address this limitation. Trained on ERA5 reanalysis data, it generates 0.25 degree resolution fields for temperature, wind, and precipitation using coarse inputs and their spatiotemporal context. It also models probability distributions of fine-scale features to produce plausible scenarios for uncertainty quantification. The model accurately reconstructs statistical distributions, including extreme events, power spectra, and spatial structures. This work highlights the potential of generative diffusion models for efficient climate downscaling with uncertainty estimation.

## 1. Introduction

Anthropogenic climate change poses substantial risks to critical socio-economic sectors, such as agriculture, forestry, energy, and water supply. The development of effective adaptation and mitigation strategies requires accessible, high-resolution climate projections to anticipate impacts at both local and regional scales. General circulation models (GCMs), which are widely used in climate research, typically operate at spatial resolutions of approximately 150--200 km (IPCC, 2023). This coarse resolution is insufficient to capture essential fine-scale processes, especially those shaped by topography, land-sea contrast, and surface heterogeneity, all of which are vital to regional climate dynamics and extreme weather events. Therefore, downscaling climate model outputs to finer resolutions is necessary to provide relevant climate information at the local level.

Weather and climate downscaling methods are typically classified into two categories (Hewitson & Crane, 1996; Wilby & Wigley, 1997). (i) Dynamic downscaling uses nested regional climate models (RCMs) to explicitly simulate physical processes at finer spatial scales (Giorgi & Mearns, 1991; Vrac et al., 2012; Giorgi & Gutowski, 2015). However, the high computational demands of this approach limit its feasibility for large ensembles or long simulation periods. (ii) Statistical downscaling establishes empirical relationships between large-scale climate predictors, such as pressure fields and circulation indices, and local-scale variables (Wilby et al., 2004; Vrac et al., 2012; Maraun & Widmann, 2018). However, this approach has limited capacity to capture complex nonlinear interactions and to maintain multivariate physical consistency within the generated fields.

Recent studies have shown that RCMs can be emulated with relatively low computational cost (Doury et al., 2023; Rampal et al., 2024). Artificial intelligence-based methods for weather prediction, including Pangu-Weather (Bi et al., 2023), GraphCast (Lam et al., 2023), and FourCastNet (Pathak et al., 2022), have demonstrated significant potential (Koldunov et al., 2024). These data-driven models facilitate efficient, automated downscaling by learning to incorporate fine-scale details from reanalysis datasets such as ERA5. However, these approaches are generally deterministic and may introduce systematic biases. Critically, they often lack a robust framework for quantifying the uncertainty inherent in the downscaling process, which is essential for comprehensive risk assessment. Recent works in generative downscaling have highlighted the potential of diffusion models (Rampal et al., 2024). CorrDiff, introduced by Mardani et al. (2025), is a residual corrective diffusion model that downscales 25 km ERA5 inputs to 2 km resolution over Taiwan, demonstrating strong performance in reconstructing spatial spectra and synthesizing radar reflectivity. Similarly, Watt & Mansfield (2024) used a diffusion model to downscale 2 degree ERA5 fields to 0.25 degree over the continental United States, exhibiting superior performance compared to U-Net and linear interpolation. While these studies confirm the promise, they are constrained to fixed regional domains. Moreover, neither study addressed precipitation downscaling, a critical variable for impact assessment, and both relied on region-specific training that cannot be generalized to new geographic areas without retraining.

This paper introduces IPSL AI Downscaling (IPSL-AID), a tool for global to regional downscaling that uses denoising diffusion probabilistic models (Ho et al., 2020; Song et al., 2020a,b; Karras et al., 2022) with several key innovations. First, we propose a *random block sampling strategy* that enables training the model on global data with limited GPU resources, thereby delivering a model that can be used for any arbitrary region without retraining, representing a step toward foundation models for downscaling. Second, the model architecture processes *multiple time steps simultaneously* (with a batch size of $B=70$), enabling temporally coherent evaluations. Third, the model jointly downscales *multiple atmospheric variables* (temperature, winds, and precipitation in the current work) using a flexible approach to include more variables. Fourth, it provides a *comprehensive evaluation*, including deterministic metrics, distributional fidelity, spectral recovery, and probabilistic skills. Our results demonstrate that the model accurately reconstructs fine-scale variability and extreme events, while its generative nature enables uncertainty quantification---essential for risk assessment. This study represents an initiative toward the global to regional downscaling of the CMIP6 climate projections.

## 2. Methodology Overview

IPSL-AID is based on denoising diffusion probabilistic models (DDPMs). These models are trained to iteratively add and remove noise from data, thereby learning to reconstruct underlying fine-scale fields. For downscaling, we condition the model on coarse fields and their spatiotemporal context to generate multiple plausible realizations. While IPSL-AID encompasses the full design space of diffusion models introduced by Karras et al. (2022), including variance-preserving, variance-expanding, and improved DDPM formulations, this work focuses on the Elucidated Diffusion Model (EDM). The core architecture is a U-Net implemented within the EDM framework. Appendix A provides a technical description of the model architecture, training procedure, loss function, sampling algorithm, and overall workflow.

## 3. Data, Sampling Strategy, and Metrics

The model is trained using four ERA5 variables (Hersbach et al., 2020): 2-m temperature (T2m), 10-m zonal wind (10U), 10-m meridional wind (10V), and precipitation (TP), at 0.25 degree resolution. To balance temporal variability and computational cost, we select four random time steps per day.

Training high-resolution (HR) global downscaling models is computationally intensive and requires substantial memory and GPU resources. To address this challenge, a random spatial block sampling strategy was developed. During each training epoch, $s$ spatial blocks of size $144 \times 360$ are generated, with block centers placed randomly. The longitude of each block is treated as periodic, while the latitude is constrained within valid global boundaries. Several values for the number of spatial blocks per epoch ($s=6$, 9, and 12) were tested, using $s=12$ was identified as an effective balance between computational efficiency and spatial diversity. For evaluation and inference, 20 fixed blocks corresponding to the ERA5 resolution of $1440 \times 721$ were used for global predictions. This framework is applicable to regional downscaling by specifying the center and spatial extent of the region.

A coarse-up procedure based on bilinear interpolation is used to separate large-scale and fine-scale components of the ERA5 fields. Let $\mathbf{y}^{\mathrm{HR}}$ be a high-resolution (HR) field. The HR field is first reduced to a coarse resolution of $16 \times 32$ (approximately 2.25 degree grid, comparable to typical GCMs resolution) by bilinear interpolation, which eliminates small-scale variability while preserving large-scale structure. The resulting coarse field is then scaled back to the original HR grid using the same interpolation method, yielding a smooth coarse-up (CU) approximation, $\mathbf{y}^{\mathrm{CU}}$, unlike Mardani et al. (2025), who typically rely on mean regression models. The residual field, $\mathbf{R} = \mathbf{y}^{\mathrm{HR}} - \mathbf{y}^{\mathrm{CU}}$, represents the fine-scale information lost during the upscaling process and serves as the training target for the diffusion model.

A conditional diffusion model is trained to generate fine-scale residual fields given CU fields and auxiliary spatiotemporal conditioning. The CU fields are provided as spatial conditioning inputs and are augmented with geographical variables, including latitude, longitude, topography ($z$), and land-sea mask (LSM). Temporal information is incorporated using cosine-sine representations of the day of year and hour of day (time embedding). The trained model generates residual fields ($\mathbf{R}'$) conditioned on the CU inputs and spatiotemporal context. The final HR prediction is obtained by adding the generated residual to the CU field ($\mathbf{y}^{\mathrm{CU}} + \mathbf{R}'$), which preserves large-scale consistency and enhances fine-scale details. We trained a U-Net model as a baseline using the L2 loss function to predict the HR fields. The diffusion model has 92,140,548 trainable parameters, compared with 92,135,940 for the U-Net baseline. The models were trained on data from 2015 to 2019, validated on 2020 data, and tested on 2021 data. Training ran for 100 epochs with a batch size of 70, using four NVIDIA A100 GPUs (64 GB each) on the LEONARDO supercomputer. Training, evaluation, and inference for the diffusion model took approximately 6 days.

### 3.1 Metrics

The model is evaluated using a set of metrics and spatial statistical diagnostics to assess both the pointwise accuracy and the distributional fidelity of the predictions. The primary metrics include:

- **Mean Absolute Error**: $\mathrm{MAE} = \frac{1}{N} \sum_{n=1}^{N} |y_n - \hat{y}_n|$
- **Normalized MAE**: $\mathrm{NMAE} = \frac{\sum_{n=1}^{N} |y_n - \hat{y}_n|}{\sum_{n=1}^{N} |y_n|}$
- **Root Mean Square Error**: $\mathrm{RMSE} = \sqrt{\frac{1}{N} \sum_{n=1}^{N} (y_n - \hat{y}_n)^2}$
- **Coefficient of determination**: $R^2 = 1 - \frac{\sum_{n=1}^{N} (y_n - \hat{y}_n)^2}{\sum_{n=1}^{N} (y_n - \bar{y})^2}$
- **Continuous Ranked Probability Score**: $\mathrm{CRPS}(F_n, y_n) = \int_{-\infty}^{\infty} d\hat{y} \, (F_n(\hat{y}) - \mathbf{1}(\hat{y} - y_n))^2$

The CRPS is calculated using 100 time steps, for which the EDM sampler is run 10 times. Here, $y_n$ and $\hat{y}_n$ represent the HR fields from the ERA5 and predicted values, $\bar{y}$ is the mean of the truth, $N$ is the number of samples, and $F$ is the predictive cumulative distribution function.

**Power Spectral Density (PSD)** analysis is used to evaluate the model's ability to reconstruct spatial variability across scales, from large-scale structures to smaller-scale features. **Probability Density Functions (PDFs)** on logarithmic scales enable comparisons of statistical distributions and the evaluation of extreme-value representation. Distributional consistency is further quantified using the **Kullback-Leibler (KL) divergence** $\mathrm{KL}(p \parallel q) = \int dy\, p(y) \log \frac{p(y)}{q(y)}$, where $p$ and $q$ denote the reference and predicted PDFs, respectively. Additionally, the **Pearson correlation coefficient** $\rho_{p,q} = \frac{\mathrm{cov}(p,q)}{\sigma_p \sigma_q}$ is computed between the predicted and true fields to assess the similarity of their spatial patterns and overall structure. The $q$th quantile of a sample $a$ is $Q(q) = (1-g)\,y[j] + g\,y[j+1]$, with $y$ sorted, $j = \lfloor (n-1)q \rfloor$, and $g = (n-1)q - j$.

For the diffusion model, we also evaluated **rank histograms**, which assess how well the predicted ensemble represents the truth, and the **spread-skill ratio (SSR)**, which quantifies the consistency between ensemble spread and prediction error. Given an ensemble of size $M$ (we used $M=10$), the sorted predictions are $\hat{y}_{(1)} \le \hat{y}_{(2)} \le \cdots \le \hat{y}_{(M)}$. The rank $r$ of reference $y_n$ relative to the ensemble is given by $r = \sum_{m=1}^{M} \mathbf{1}(\hat{y}_m \le y_n)$, so that $r \in \{0, \dots, M\}$. The rank histogram is obtained by plotting the histogram of $r$ over all prediction-truth pairs.

The SSR is defined as $\mathrm{SSR} = \frac{\sigma_{\mathrm{ens}}}{\mathrm{RMSE}}$, where the ensemble spread is given by $\sigma_{\mathrm{ens}} = \sqrt{\frac{M+1}{M}} \sqrt{\frac{1}{N} \sum_{n=1}^{N} \frac{1}{M} \sum_{m=1}^{M} (\hat{y}_n^{(m)} - \bar{\hat{y}}_n )^2}$, and $\mathrm{RMSE} = \sqrt{\frac{1}{N} \sum_{n=1}^{N} (\bar{\hat{y}}_n - y_n)^2}$. Here, $\hat{y}_n^{(m)}$ denotes the $m$-th ensemble member (with $m=1,\dots,M$), $\bar{\hat{y}}_n = \frac{1}{M} \sum_{m=1}^{M} \hat{y}_n^{(m)}$ is the ensemble mean, $y_n$ is the reference value, and $N$ is the number of samples. The corrective factor $\sqrt{(M+1)/M}$ in the definition of $\sigma_{\mathrm{ens}}$ ensures the optimal value of the SSR is 1, as explained in Fortin et al. (2014).

## 4. Results and Discussion

### 4.1 Deterministic Performance

The model successfully reconstructs the fine-scale spatial details of the HR fields. For T2m, the prediction achieves an MAE of $0.44 \pm 0.02$ K, an RMSE of $0.71 \pm 0.04$ K, and an $R^2$ of 1.00. The corresponding ensemble mean improves these scores, with an MAE of $0.32 \pm 0.01$ K, an RMSE of $0.52 \pm 0.01$ K, and an $R^2$ of 1.00. The wind components are predicted with similar accuracy. For 10U, the prediction yields an MAE of $0.52 \pm 0.02$ m/s, an RMSE of $0.79 \pm 0.03$ m/s, and an $R^2$ of 0.98, while the ensemble mean achieves an MAE of $0.40 \pm 0.01$ m/s, an RMSE of $0.62 \pm 0.03$ m/s, and an $R^2$ of $0.988 \pm 0.002$. Similarly, for 10V, the prediction has an MAE of $0.50 \pm 0.01$ m/s, an RMSE of $0.74 \pm 0.03$ m/s, and an $R^2$ of 0.98, while the ensemble mean yields an MAE of $0.38 \pm 0.01$ m/s, an RMSE of $0.59 \pm 0.03$ m/s, and an $R^2$ of $0.985 \pm 0.002$. For TP, the prediction shows an average MAE of $0.10 \pm 0.00$ mm/hr, which appears very low; however, the NMAE of $0.67 \pm 0.02$ indicates a larger relative error. The ensemble mean slightly improves the absolute error, with an MAE of $0.055 \pm 0.001$ mm/hr and an RMSE of $0.234 \pm 0.001$ mm/hr, while the NMAE remains relatively high at $0.53 \pm 0.01$ and $R^2$ reaches $0.68 \pm 0.01$. The discrepancy between MAE and NMAE arises from the variable's skewed distribution, characterized by a high frequency of zero values, which makes absolute error metrics less informative.

The CRPS metric is $0.25 \pm 0.01$ K for T2m, $0.29 \pm 0.01$ m/s for 10U, $0.28 \pm 0.01$ m/s for 10V, and 0.04 mm/hr for TP. Watt & Mansfield (2024) reported CRPS values of 0.254 K (T2m), 0.224 m/s (10U), and 0.232 m/s (10V) over the United States, whereas Mardani et al. (2025), for the Taiwan domain, report higher values for extremes: 0.55 K, 0.86 m/s, and 0.95 m/s, respectively. The low CRPS for precipitation suggests that the model's predicted distribution is confident, particularly in correctly capturing the prevalence of dry (zero-precipitation) conditions. Overall, the diffusion model yields physically consistent and statistically accurate downscaling, with the ensemble mean improving deterministic performance.

### 4.2 Statistical Fidelity

The statistical fidelity of the downscaled fields was evaluated by jointly analyzing density scatter plots, PDFs, and PSDs of the model predictions, the HR ERA5 reference, and the CU inputs. For the T2m and (10U, 10V) components, the predictions show a strong linear relationship with the reference fields, as indicated by points tightly clustered along the diagonal and $R^2$ values near 0.99. Small vertical lines in the density plot for 10U and 10V indicate that some true values correspond to predicted values with the opposite sign. These discrepancies likely stem from challenges in downscaling coastal winds influenced by land-sea breezes and affect only a small fraction of datapoints. For precipitation, although the scatter indicates increased spread, particularly at higher intensities, the model substantially sharpens near the center, reduces underestimation, and more closely aligns with the one-to-one relationship, producing the same zero-precipitation points.

The model exhibits a strong capacity to replicate the full statistical distribution of T2m. The predicted mean ($\mu = 278.89$ K) and standard deviation ($\sigma = 21.33$ K) closely match those of the ground truth ($\mu = 278.91$ K, $\sigma = 21.32$ K), whereas the CU input displays a slightly reduced variance ($\sigma = 21.18$ K). This high level of distributional agreement is quantitatively supported by a KL divergence of 0.00 and a Pearson correlation of 1.00. The log-PDF curves of the prediction and reference are nearly indistinguishable across the T2m range, demonstrating accurate reproduction of both central tendencies and distribution tails. The model's performance for the (10U, 10V) components is comparably robust. For 10U, both the mean ($\mu_{\mathrm{pred}} = -0.04$ m/s, $\mu_{\mathrm{truth}} = -0.04$ m/s) and variance ($\sigma_{\mathrm{pred}} = 5.63$ m/s, $\sigma_{\mathrm{truth}} = 5.64$ m/s) are accurately captured, while the CU input underestimates variability ($\sigma = 5.26$ m/s). KL divergence of 0.00 and Pearson correlation of 0.99 for both wind components. Precipitation prediction presents significant challenges due to its intermittency, non-Gaussian distribution, and localized extremes. Nevertheless, the model demonstrates substantial improvement over the CU input, with a KL divergence of 0.001 and a Pearson correlation of 0.73.

The U-Net baseline reproduces statistical distributions effectively (KL divergence and Pearson correlations of approximately $10^{-4}$ and 0.99, respectively). However, its performance declines for precipitation with a KL divergence of 0.3. Analysis of residual fields shows that the U-Net generates PDFs that are more peaked and narrower than those from the diffusion model across variables, indicating underestimation of fine-scale variability and extremes.

### 4.3 Power Spectral Density

The model's ability to recover spatial variability across scales is evaluated by analyzing the PSDs of the generated fields and comparing them with the HR reference and the CU input. For all variables, the diffusion model prediction closely matches the reference spectrum throughout most of the resolved wavenumber range, extending into smaller-scale regimes. The CU input shows significant depletion at high wavenumbers, reflecting the loss of small-scale variability due to spatial averaging. In contrast, the predicted spectra maintain both the spectral slope and amplitude, indicating the model's capacity to reproduce physically consistent small-scale variability. U-Net spectra generally follow the large-scale truth data but exhibit earlier attenuation at higher wavenumbers, indicating a reduced ability to capture small-scale variability compared to the diffusion model. In the case of precipitation, the recovery of high-wavenumber energy is especially pronounced, signifying the restoration of fine-scale structures associated with convective and localized rainfall processes, which remain parameterized at the 25 km resolution.

### 4.4 Extreme Value Analysis

| Quantile (q) | T2m Truth (K) | T2m Diff (K) | T2m U-Net (K) | T2m CU (K) | 10U Truth (m/s) | 10U Diff (m/s) | 10U U-Net (m/s) | 10U CU (m/s) | TP Truth (mm/hr) | TP Diff (mm/hr) | TP U-Net (mm/hr) | TP CU (mm/hr) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.90 | 300.09 | 300.07 | 300.08 | 300.01 | 7.61 | 7.60 | 7.56 | 7.09 | 0.20 | 0.21 | 0.21 | 0.27 |
| 0.95 | 301.20 | 301.17 | 301.14 | 300.97 | 10.24 | 10.24 | 10.19 | 9.71 | 0.47 | 0.49 | 0.49 | 0.49 |
| 0.975 | 302.57 | 302.57 | 302.52 | 302.13 | 12.33 | 12.33 | 12.27 | 11.77 | 0.92 | 0.94 | 0.89 | 0.75 |
| 0.99 | 306.40 | 306.41 | 306.35 | 305.55 | 14.48 | 14.47 | 14.40 | 13.85 | 1.71 | 1.73 | 1.56 | 1.13 |
| 0.995 | 309.21 | 309.21 | 309.16 | 308.30 | 15.81 | 15.80 | 15.72 | 15.11 | 2.43 | 2.45 | 2.15 | 1.45 |

For T2m, both diffusion model and the U-Net reconstruction closely match observed quantiles from 90th to 99.5th percentile. The diffusion model shows perfect agreement with truth (309.21 versus 309.21 K at $q=0.995$), while U-Net slightly underestimates the highest quantiles (309.16 versus 309.21 K). CU input underestimates high temperatures, with discrepancies reaching 0.91 K at the 99.5th percentile. For wind components, both models slightly underestimate extreme wind speeds. The diffusion model remains close to truth (15.80 versus 15.81 m/s at $q=0.995$ for 10U), while U-Net shows larger underestimation (15.72 m/s). For precipitation, the diffusion model slightly underestimates moderate extremes but overestimates most extreme events (2.45 versus 2.43 mm/hr at $q=0.995$). The U-Net underestimates the upper tail more remarkably (2.15 versus 2.43 mm/hr at $q=0.995$). CU input overestimates lower precipitation quantiles but significantly underestimates the highest quantiles, with values below two-thirds of ground truth at $q=0.995$.

### 4.5 Spatial Error Distribution

Spatial error distributions indicate regional variation in model performance. The MAE for T2m is higher at high latitudes, particularly near the poles, and in areas with complex topography, such as the Himalayas and the Andes. The (10U, 10V) components also show increased errors in coastal regions and along steep topographic gradients, likely attributable to the modulation of surface winds by land-sea contrasts and orographic influences. In contrast, precipitation errors are most pronounced in the tropics, especially within the Intertropical Convergence Zone and over tropical rainforests, where intense and localized rainfall events pose significant challenges for downscaling.

### 4.6 Rank Histograms

The rank histograms are generally close to uniform, indicating that the diffusion model ensemble is reasonably well calibrated overall. However, the temperature histogram exhibits a slight upward slope with a negative bias, suggesting that the truth often lies above the ensemble members, indicating a tendency toward underestimation. The 10U histogram is fairly flat but shows a mild positive bias, pointing to a slight overestimation. In contrast, the 10V histogram is nearly uniform with negligible bias, indicating very good calibration. The precipitation histogram displays a noticeable spike at the lowest rank and a slight skew, suggesting that the truth frequently falls below the ensemble range, which implies overestimation and potential under-dispersion.

### 4.7 Spread-Skill Analysis

Density plots of ensemble spread versus RMSE show the ideal 1:1 relationship (SSR=1), corresponding to a perfectly calibrated ensemble. All variables exhibit SSR values slightly smaller than 1, indicating a weak overall under-dispersion. The highest density of samples is concentrated at low RMSE and low spread across all variables, reflecting good predictive skill. However, deviations from the 1:1 line become more apparent at larger RMSE values. For 10U, 10V, and TP, the spread tends to remain relatively small as RMSE increases, indicating under-dispersion in higher-error regimes. A similar, though less pronounced, behavior is observed for T2m.

## 5. Conclusions and Limitations

We have introduced IPSL-AID, a diffusion-based generative model designed for downscaling climate data from global-to-regional scales. This model effectively reconstructs high-resolution fields of temperature, winds, and precipitation using coarse input data and their spatiotemporal context, trained on ERA5 reanalysis with a novel random block sampling strategy that enables global applicability without retraining for new regions. A primary advantage of this approach is its probabilistic nature. By learning the underlying distribution of residuals at fine scales, the model generates multiple plausible high-resolution realizations from a single coarse input. This enables uncertainty quantification through ensemble spread analysis, rank histograms, and spread-skill ratios diagnostic.

Quantitative metrics demonstrate that IPSL-AID achieves low deterministic errors (MAE of 0.44 K for T2m, 0.52 m/s for 10U, 0.50 m/s for 10V) and accurately represents the statistical distribution. The model successfully reproduces spatial variability across scales (PSD analysis), and captures extreme events with quantile estimates closely matching observations up to the 99.5th percentile. The model demonstrates robust performance across diverse atmospheric variables and regions while preserving physical consistency, as evidenced by inter-variable correlation analysis showing realistic multivariate dependencies (e.g., temperature-wind patterns over oceans and continents). The ensemble mean further improves deterministic scores, with MAE reduced to 0.32 K for T2m and 0.40 m/s for 10U.

### Limitations

- Performance degrades in regions with complex topography (e.g., Himalayas, Andes), coastal zones, and areas characterized by intense convective precipitation, where errors remain elevated.
- For precipitation, the normalized MAE of 0.53 indicates substantial relative error despite low absolute error, reflecting the inherent challenge of downscaling non-Gaussian fields.
- The rank histograms reveal systematic tendencies: slight underestimation for temperature, mild overestimation for 10U winds, and under-dispersion for precipitation extremes.
- Spread-skill analysis indicates under-dispersion in higher-error regimes, suggesting the ensemble may not fully capture uncertainty in challenging conditions.
- Block boundary artifacts remain visible during global inference.
- The five-year training period is short compared to the multi-decadal timescales typically required for climate studies.
- The computational demands of global training---6 days on four A100 GPUs---may limit accessibility for some research groups.

### Future Work

- Extend the model to additional atmospheric variables and apply the framework to downscale future projections from CMIP6 models. This will require developing and integrating appropriate bias correction or debiasing strategies (Wan et al., 2023, 2025).
- Distributed multi-GPU training strategies to accommodate longer training periods (e.g., 40 years) while maintaining global coverage.
- Block boundary artifact mitigation: (i) overlap-block method with cosine-weighted blending during inference, (ii) explicit boundary regularization in the loss function, (iii) frequency-domain spectral loss, and (iv) adaptive Fourier-domain filtering as post-processing.
- Incorporate additional conditioning variables (e.g., soil moisture, surface fluxes) and explore alternative diffusion formulations to improve performance for precipitation extremes.
- The apparent contradiction between low CRPS and high NMAE for precipitation reflects the different aspects they measure: CRPS captures probabilistic calibration, whereas NMAE reflects the difficulty of deterministic prediction for a non-Gaussian variable.

## Appendix A: Generative Diffusion Model

Karras et al. (2022) introduced a unified design space that categorizes the architectural and procedural choices of diffusion-based generative models. While IPSL-AID encompasses the full design space, including elucidated diffusion model (EDM), variance-preserving, variance-expanding, and improved DDPM, this work focuses on the EDM and reports its results.

### A.1 Preconditioning

The preconditioned denoising model stabilizes training by standardizing the scales of inputs, outputs, and targets across varying noise levels. The EDM preconditioned model $D_\theta(\mathbf{x}; \sigma)$ is defined as:

$$D_\theta(\mathbf{x}; \sigma) = c_{\mathrm{skip}}(\sigma) \mathbf{x} + c_{\mathrm{out}}(\sigma) F_\theta(c_{\mathrm{in}}(\sigma) \mathbf{x}; c_{\mathrm{noise}}(\sigma))$$

The input $\mathbf{x}$ is defined as $\mathbf{x}=\mathbf{y}+\sigma\mathbf{n}$, where $\mathbf{y}$ is the clean signal and $\mathbf{n} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ is standard Gaussian noise. The coefficients are:

- $c_{\mathrm{in}}(\sigma) = 1 / (\sigma_{\mathrm{data}}^2 + \sigma^2)^{1/2}$ -- ensures the network sees inputs with roughly unit variance across all noise levels
- $c_{\mathrm{skip}}(\sigma) = \sigma_{\mathrm{data}}^2 / (\sigma_{\mathrm{data}}^2 + \sigma^2)$ -- passes part of the input directly to the output to reduce amplification of network errors
- $c_{\mathrm{out}}(\sigma) = \sigma \, \sigma_{\mathrm{data}} / (\sigma_{\mathrm{data}}^2 + \sigma^2)^{1/2}$ -- ensures the network output has the correct magnitude to match the target
- $c_{\mathrm{noise}}(\sigma) = \frac{1}{4} \log \sigma$ -- gives the network explicit information about the noise level

with $\sigma_{\mathrm{data}} = 1$. The noise scale $\sigma$ is sampled from a log-normal distribution, $\log \sigma \sim \mathcal{N}(P_{\text{mean}}, P_{\text{std}}^2)$. The target is:

$$F_{\text{target}}(\mathbf{y},\mathbf{n};\sigma) = \frac{\mathbf{y} - c_{\text{skip}}(\sigma) \mathbf{x}}{c_{\text{out}}(\sigma)}$$

For conditional image generation, the preconditioning framework was extended by concatenating the conditioning image with the noised input along the channel dimension before passing it to the network.

### A.2 Architecture

The EDM preconditioning architecture employs a U-Net-based denoising network, adapted from Dhariwal & Nichol (2021). It utilizes an encoder-decoder structure with symmetric downsampling and upsampling paths, incorporating multi-head self-attention blocks at selected spatial resolutions. The input consists of a noisy image $\mathbf{x} \in \mathbb{R}^{C \times H \times W}$, a scalar noise level or timestep embedding $\sigma$, and optional class or augmentation embeddings. The output is a denoised tensor $\mathbf{y}'$.

**Default Configuration**: Base channel count $C_{\mathrm{base}} = 128$, channel multipliers per resolution $[1, 2, 3, 4]$, three residual blocks per resolution. Self-attention at resolutions $[32, 16, 8]$ with dropout $p = 0.10$. Embedding dimension $C_{\mathrm{emb}} = 4 \times C_{\mathrm{base}}$. All convolutional and linear layers initialized with Kaiming uniform initialization.

**Mapping and Embedding Layers**: Noise levels $\sigma$ are represented using sinusoidal positional embedding: $\mathbf{e}_\sigma = \text{PE}(\sigma) \in \mathbb{R}^{C_{\mathrm{base}}}$, processed by two fully connected layers with SiLU activations. Optional class and augmentation embeddings are projected into the same space and added to $\mathbf{e}$.

**Encoder, Decoder, and Attention**: The encoder consists of convolutional and residual (U-NetBlock) layers, which progressively downsample. Each U-NetBlock incorporates group normalization, two $3 \times 3$ convolutions, adaptive scaling with $\mathbf{e}$, and optional self-attention. Skip connections are retained for use in the decoder, which mirrors the encoder with upsampling. Self-attention is applied at spatial resolutions $[32, 16, 8]$, with 64 channels per attention head: $\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}(\mathbf{Q}^\top \mathbf{K}/\sqrt{d_k})\mathbf{V}$.

### A.3 Loss Function

The weighted denoising loss reads:

$$\mathcal{L} = \mathbb{E}_{\sigma, \mathbf{y}, \mathbf{n}} \left[ \lambda(\sigma) \left\| D_{\theta}(\mathbf{y} + \sigma\mathbf{n}, \sigma) - \mathbf{y} \right\|_2^2 \right]$$

where $\lambda(\sigma) = (\sigma^2 + \sigma_{\mathrm{data}}^2) / (\sigma \, \sigma_{\mathrm{data}})^2$ balances the contributions of different noise levels. The loss is augmented with a conditional image fed into the network.

### A.4 Sampler

Samples are generated by numerically integrating the reverse-time stochastic differential equation. The sequence of time steps is:

$$t_i = \left(\sigma_{\max}^{1/\rho} + \frac{i}{N-1} (\sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho}) \right)^\rho$$

for $i = 0, \dots, N-1$, with $t_N = 0$, where $\rho$ controls step distribution. The sample is initialized with $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, t_0^2 \mathbf{I})$. At each step:

1. A temporary noise increment $\gamma_i$ may be added if $t_i \in [S_{\min}, S_{\max}]$: $\gamma_i = \min(S_{\mathrm{churn}}/N, \sqrt{2}-1)$, yielding $\hat{t}_i = t_i + \gamma_i t_i$.
2. Add Gaussian noise: $\hat{\mathbf{x}}_i = \mathbf{x}_i + \sqrt{\hat{t}_i^2 - t_i^2} \, \boldsymbol{\epsilon}_i$ where $\boldsymbol{\epsilon}_i \sim \mathcal{N}(\mathbf{0}, S_{\mathrm{noise}} \mathbf{I})$.
3. Compute first-order derivative: $\mathbf{d}_i = (\hat{\mathbf{x}}_i - D_\theta(\hat{\mathbf{x}}_i, \hat{t}_i)) / \hat{t}_i$, and Euler step: $\mathbf{x}_{i+1} = \hat{\mathbf{x}}_i + (t_{i+1} - \hat{t}_i) \mathbf{d}_i$.
4. If $t_{i+1} \neq 0$, apply 2nd-order Heun correction: $\mathbf{d}_i' = (\mathbf{x}_{i+1} - D_\theta(\mathbf{x}_{i+1}, t_{i+1})) / t_{i+1}$, then $\mathbf{x}_{i+1} = \hat{\mathbf{x}}_i + (t_{i+1} - \hat{t}_i) \frac{1}{2}(\mathbf{d}_i + \mathbf{d}_i')$.
5. Return final denoised sample $\mathbf{x}_N$.

### A.5 Workflow

The ERA5 data at 0.25 degree resolution ($\mathbf{y}^{\mathrm{HR}}$) are first sampled into sequences of 70 time steps with 12 blocks. Geospatial variables and temporal embeddings are extracted and combined with the coarse-upsampled $\mathbf{y}^{\mathrm{CU}}$ and the residual $\mathbf{R} = \mathbf{y}^{\mathrm{HR}} - \mathbf{y}^{\mathrm{CU}}$ as inputs to the model. The model predicts the residual $\mathbf{R}'$, which is added back to the CU to produce the final HR prediction $\hat{\mathbf{y}}^{\mathrm{HR}}$.

## Appendix B: Extended Diagnostics

### B.1 Ablation Study

An ablation study was conducted to evaluate the impact of EDM sampler parameters on reconstruction accuracy and computational cost, performed for T2m over 10 training epochs. Baseline: $N=10$, $\sigma_{\min}=0.002$, $\sigma_{\max}=80$, $\rho=7$.

| Parameter | Value | MAE (K) | RMSE (K) | $R^2$ | Time |
|---|---|---|---|---|---|
| $N$ | 10 | 0.63 | 0.99 | 0.998 | 2h38 |
| $N$ | 20 | 0.55 | 0.89 | 0.998 | 3h43 |
| $N$ | 40 | 0.53 | 0.85 | 0.998 | 6h05 |
| $\sigma_{\min}$ | 0.002 | 0.63 | 0.99 | 0.998 | 2h38 |
| $\sigma_{\min}$ | 0.02 | 0.58 | 0.93 | 0.998 | 2h33 |
| $\sigma_{\min}$ | 0.2 | 0.46 | 0.76 | 0.999 | 2h35 |
| $\sigma_{\max}$ | 8 | 0.55 | 0.87 | 0.998 | 2h33 |
| $\sigma_{\max}$ | 80 | 0.63 | 0.99 | 0.998 | 2h38 |
| $\sigma_{\max}$ | 100 | 0.64 | 1.00 | 0.998 | 2h32 |
| $\rho$ | -10 | 0.64 | 1.04 | 0.998 | 2h32 |
| $\rho$ | 7 | 0.63 | 0.99 | 0.998 | 2h38 |
| $\rho$ | 10 | 0.61 | 0.96 | 0.998 | 2h38 |

Key findings: Increasing $N$ from 10 to 40 improves performance but more than doubles inference time. The largest improvement comes from increasing $\sigma_{\min}$ from 0.002 to 0.2, yielding the best MAE (0.46 K) and RMSE (0.76 K) with similar runtime. A higher minimum noise level helps prevent overfitting to high-frequency details. $\sigma_{\max}$ and $\rho$ have smaller effects.

### B.2 Ensemble Members

A 10-member ensemble is used to quantify plausible HR realizations and assess uncertainty. Three representative ensemble members illustrate the variability among predictions. The ensemble mean smooths small-scale variability and yields a more deterministic pattern. The ensemble standard deviation highlights regions of higher uncertainty.

### B.3 Zoomed-in Example (South America)

A zoomed-in comparison over South America shows that whereas the U-Net accurately reproduces large-scale patterns, it clearly over-smooths fine spatial details for precipitation and winds. By contrast, the diffusion model distinctly reproduces high-frequency structures. Residual errors are most pronounced over the Andes and in regions of intense convective activity, such as the Amazon.

### B.4 Residual Fields

The residual fields (difference between HR prediction and CU input) represent the fine-scale structures restored by the models. The $R^2$ scores of the residuals appear lower than those of the full fields due to reduced variance, not degraded performance. The diffusion model generates residuals centered near zero with a dispersion corresponding to the truth variance, while the U-Net produces slightly narrower and sharper PDFs, underestimating fine-scale variability. For precipitation, the diffusion model produces broader tails reflecting extreme events. The PSDs confirm that the diffusion model injects energy at high wavenumbers while the U-Net exhibits earlier attenuation.

### B.5 Inter-variable Correlation Analysis

To assess whether the diffusion model preserves physical consistency, temporal co-variability between variable pairs was evaluated using the Pearson correlation coefficient (Vrac & Friederichs, 2015). Capturing realistic multivariate dependencies is particularly important because many environmental and climate impacts depend on compound conditions, such as hot-dry days or wind-precipitation interactions (Coppola et al., 2021). For each pair, correlations are computed over time at each grid cell, yielding a spatial map of local coupling strength:

$$\text{corr}(h,w) = \frac{\sum_{b=1}^{B} (y_1(b,h,w) - \bar{y}_1(h,w))(y_2(b,h,w) - \bar{y}_2(h,w))}{\sqrt{\sum_{b=1}^{B} (y_1(b,h,w) - \bar{y}_1(h,w))^2} \sqrt{\sum_{b=1}^{B} (y_2(b,h,w) - \bar{y}_2(h,w))^2}}$$

The predicted correlations closely match the truth, capturing large-scale positive and negative structures and preserving dominant physical couplings. Agreement is strongest for temperature-wind pairs, while precipitation correlations are more heterogeneous but still retain key patterns absent in the coarse inputs.

## References

- Bi, K. et al. (2023). Accurate medium-range global weather forecasting with 3D neural networks. *Nature*, 619(7970), 533-538.
- Coppola, E. et al. (2021). Climate hazard indices projections based on CORDEX-CORE, CMIP5 and CMIP6 ensemble. *Climate Dynamics*, 57(5), 1293-1383.
- Dhariwal, P. & Nichol, A. (2021). Diffusion models beat GANs on image synthesis. *NeurIPS*, 34, 8780-8794.
- Doury, A. et al. (2023). Regional climate model emulator based on deep learning. *Climate Dynamics*, 60(5), 1751-1779.
- Fortin, V. et al. (2014). Why Should Ensemble Spread Match the RMSE of the Ensemble Mean? *J. Hydrometeorology*, 60(4), 1708-13.
- Giorgi, F. & Gutowski, W.J. (2015). Regional dynamical downscaling and the CORDEX initiative. *Ann. Rev. Environ. Res.*, 40(1), 467-490.
- Giorgi, F. & Mearns, L.O. (1991). Approaches to the simulation of regional climate change: a review. *Rev. Geophys.*, 29(2), 191-216.
- Hersbach, H. et al. (2020). The ERA5 global reanalysis. *QJRMS*, 146(730), 1999-2049.
- Hewitson, B.C. & Crane, R.G. (1996). Climate downscaling: techniques and application. *Climate Research*, 7(2), 85-95.
- Ho, J. et al. (2020). Denoising diffusion probabilistic models. *NeurIPS*, 33, 6840-6851.
- IPCC (2023). Linking global-to-regional Climate Change. In *Climate Change 2021 -- The Physical Science Basis*, Cambridge Univ. Press, pp. 1363-1512.
- Karras, T. et al. (2022). Elucidating the design space of diffusion-based generative models. *NeurIPS*, 35, 26565-26577.
- Koldunov, N. et al. (2024). Emerging AI-based weather prediction models as downscaling tools. arXiv:2406.17977.
- Lam, R. et al. (2023). Learning skillful medium-range global weather forecasting. *Science*, 382(6677), 1416-1421.
- Maraun, D. & Widmann, M. (2018). *Statistical downscaling and bias correction for climate research*. Cambridge Univ. Press.
- Mardani, M. et al. (2025). Residual corrective diffusion modeling for km-scale atmospheric downscaling. *Commun. Earth Environ.*, 6(1), 124.
- Pathak, J. et al. (2022). FourCastNet: A global data-driven high-resolution weather model using adaptive Fourier neural operators. arXiv:2202.11214.
- Rampal, N. et al. (2024). Enhancing regional climate downscaling through advances in machine learning. *AI Earth Syst.*, 3(2), 230066.
- Song, J. et al. (2020a). Denoising diffusion implicit models. arXiv:2010.02502.
- Song, Y. et al. (2020b). Score-based generative modeling through stochastic differential equations. arXiv:2011.13456.
- Vrac, M. et al. (2012). Dynamical and statistical downscaling of the French Mediterranean climate: uncertainty assessment. *Nat. Hazards Earth Syst. Sci.*, 12(9), 2769-2784.
- Vrac, M. & Friederichs, P. (2015). Multivariate--intervariable, spatial, and temporal--bias correction. *J. Climate*, 28(1), 218-237.
- Wan, Z.Y. et al. (2023). Debias coarsely, sample conditionally: statistical downscaling through optimal transport and probabilistic diffusion models. arXiv:2305.15618.
- Wan, Z.Y. et al. (2025). Regional climate risk assessment from climate models using probabilistic machine learning. arXiv:2412.08079.
- Watt, R.A. & Mansfield, L.A. (2024). Generative diffusion-based downscaling for climate. arXiv:2404.17752.
- Wilby, R.L. & Wigley, T.M.L. (1997). Downscaling general circulation model output: a review of methods and limitations. *Prog. Phys. Geogr.*, 21(4), 530-548.
- Wilby, R.L. et al. (2004). Guidelines for use of climate scenarios developed from statistical downscaling methods. *IPCC TGCIA*, 27.
