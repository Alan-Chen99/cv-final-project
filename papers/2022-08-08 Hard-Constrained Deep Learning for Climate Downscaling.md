# Hard-Constrained Deep Learning for Climate Downscaling

**Authors:** Paula Harder, Alex Hernandez-Garcia, Venkatesh Ramesh, Qidong Yang, Prasanna Sattegeri, Daniela Szwarcman, Campbell D. Watson, David Rolnick

**Published:** 2022-08-08 (arXiv: 2208.05424)

**Journal:** JMLR, Volume 24, 2023

---

## Abstract

The availability of reliable, high-resolution climate and weather data is important to inform long-term decisions on climate adaptation and mitigation and to guide rapid responses to extreme events. Forecasting models are limited by computational costs and, therefore, often generate coarse-resolution predictions. Statistical downscaling, including super-resolution methods from deep learning, can provide an efficient method of upsampling low-resolution data. However, despite achieving visually compelling results in some cases, such models frequently violate conservation laws when predicting physical variables. In order to conserve physical quantities, here we introduce methods that guarantee statistical constraints are satisfied by a deep learning downscaling model, while also improving their performance according to traditional metrics. We compare different constraining approaches and demonstrate their applicability across different neural architectures as well as a variety of climate and weather data sets. Besides enabling faster and more accurate climate predictions through downscaling, we also show that our novel methodologies can improve super-resolution for satellite data and natural images data sets.

## 1 Introduction

Accurate modeling of weather and climate is critical for taking effective action to combat climate change. In addition to shaping global understanding of climate change, local and regional predictions guide adaptation decisions and provide the impetus for action to reduce greenhouse gas emissions (Gutowski et al., 2020). Predicted and observed quantities such as precipitation, wind speed, and temperature impact decisions in sectors such as agriculture, energy, and transportation. While these quantities are often required at a fine geographical and temporal scale to ensure informed decision-making, most climate and weather models are extremely computationally expensive to run (sometimes taking months even on super-computers), resulting in coarse-resolution predictions. Thus, there is a need for fast methods that can generate high-resolution data based on the low-resolution models that are commonly available.

The terms *downscaling* in climate science and *super-resolution* (SR) in machine learning (ML) refer to a map from low-resolution (LR) input data to high-resolution (HR) versions of that same data; the high-resolution output is referred to as the super-resolved (SR) data. Downscaling via established statistical methods---*statistical downscaling*---has been long used by the climate science community to increase the resolution of climate data (Maraun and Widmann, 2018). In statistical downscaling, there are two subfields, *perfect prognosis* and *model output statistics* (Maraun and Widmann, 2018). Whereas perfect prognosis learns the relationship between LR and HR observations, model output statistics learns directly the function from model output to observations, including a form of bias correction.

In perfect prognosis, predictands and predictors usually include different variables. If both inputs and outputs consist of the same variables, this is referred to as super-resolution, even in a climate context. In parallel, computer vision SR has evolved rapidly using various deep learning architectures, with such methods now including super-resolution convolutional neural networks (CNNs) (Dong et al., 2016), generative adversarial models (GANs) (Wang et al., 2018), vision transformers (Yang et al., 2020), and normalizing flows (Lugmayr et al., 2020). Increasing the temporal resolution via frame interpolation is also an active area of research for video enhancement (Liu et al., 2017) that can be transferred to spatiotemporal climate data. Recently, deep learning approaches have been applied to a variety of climate and weather data sets, covering both model output data and observations. In addition to using neural networks to learn parametrization, replace model parts in a hybrid setup, or run full forecasts, downscaling is a field for deep learning to improve and accelerate Earth system simulations (Reichstein et al., 2019). Climate super-resolution has mostly focused on CNNs (Vandal et al., 2017), recently shifting toward GANs (Stengel et al., 2020; Wang et al., 2021).

Most statistical downscaling tools are applied offline as a tool for post-processing. In that case, machine learning methods can be directly employed on the output data, following data reformatting. However, downscaling tools could be applied online within a global climate model too (e.g. Quiquet et al., 2018), where a lower resolution output of a climate model part is downscaled, and its high-resolution version is fed back into the climate model.

There are certain tasks that are more suited for hard-constraining than others. One important point is that there exists a relationship between low-resolution and high-resolution samples for downscaling or between input and output for other tasks, given by an equation. This can be the case when modeling physical quantities, with, for example, mass or energy conservation that exists between LR and HR pairs. On the one hand, if we consider compressed or blurry images and the task is to remove the effects of compression or blur, there may be no known constraint between low and high resolution, so constraining methodologies would not be applicable. On the other hand, for some data from e.g. satellites or telescopes, images are created by summing photons across a given field of view, so the value at a given pixel can be interpreted as the sum of values at unobserved subpixels; in such cases, hard constraints could potentially be useful.

In this work, we introduce novel methods to strictly enforce physics-inspired consistency constraints between low-resolution (input) and high-resolution (output) images. We do this via a constraint layer at the end of a neural architecture, which renormalizes the prediction either additively, multiplicatively, or with an adaptation of the softmax layer. We use climate and weather data sets based on European Center for Medium-Range Weather Forecasts (ECMWF) reanalysis data version 5 (ERA5) (Hersbach et al., 2020), Weather Research and Forecast Model (WRF) data (Auger et al., 2021), and the Norwegian Earth System Model (NorESM) (Seland et al., 2020) data, spanning different quantities such as water content, temperature, water vapor, and liquid water content. For the ERA5 data, we increase the resolution by different factors, we create data sets with an enhancement of factors ranging from 2 over 4 and 8 to 16. We show the utility of our methods across architectures including CNNs, GANs, CNN-RNNs, and a novel architecture that we introduce to apply super-resolution in both spatial and temporal dimensions. Besides climate data sets, we show that our methods are able to improve predictive accuracies for lunar satellite imagery super-resolution as well as on standard image super-resolution benchmark data sets, like Set5, Set14, Urban100 and BSD100. Our code is available at https://github.com/RolnickLab/constrained-downscaling and our main data set can be found at https://drive.google.com/file/d/1IENhP1-aTYyqOkRcnmCIvxXkvUW2Qbdx/view.

### Contributions

Our main contributions can be summarized as follows:
- We introduce a novel constraining methodology for deep learning-based downscaling methods, which guarantees that physical consistency constraints such as mass and energy conservation between low-resolution and high-resolution are satisfied.
- We show that our method improves predictive performance across different deep learning architectures on a variety of climate data sets.
- Additionally, we show that our method increases the accuracy of super-resolution in other domains, such as natural images and satellite imagery.
- Finally, we introduce a new deep learning architecture for downscaling along both spatial and temporal dimensions.

## 2 Related Work

### Deep Learning for Climate Downscaling

There exists extensive work on ML methods for climate and weather observation and prediction downscaling, from CNN architectures (Vandal et al., 2017) to GANs (Stengel et al., 2020) and normalizing flows (Groenke et al., 2020). Recently, GANs have become a very popular architecture choice, including many works on precipitation model downscaling (Wang et al., 2021; Watson et al., 2020; Chaudhuri and Robertson, 2020) as well as other quantities such as wind and solar data (Stengel et al., 2020). Unified frameworks comparing methods and benchmarks were introduced by Bano-Medina et al. (2020) to assess different SR-CNN setups and by Kurinchi-Vendhan et al. (2021) with the introduction of a new data set for wind and solar SR. To date, there has been limited work on spatiotemporal SR with climate data. Some authors have looked at super-resolving multiple time steps at once without increasing the temporal resolution (Harilal et al., 2021; Leinonen et al., 2021). Serifi et al. (2021) did increase the temporal resolution by simply treating the time steps as different channels and using a standard SR-CNN.

### Constrained Learning for Climate

Various works on ML for climate science have attempted to enforce certain physical constraints via soft penalties in the loss (Beucler et al., 2019), linearly constrained neural networks for convection (Beucler et al., 2021), or aerosol microphysics emulation (Harder et al., 2022) using completion or correction methods. Zanna and Bolton (2020) and Zanna and Bolton (2021) use a final fixed convolutional layer to achieve momentum and vorticity conservation in an ML ocean model. A different line of work incorporates constraints into machine learning based on flux balances (Sturm and Wexler, 2020, 2022; Yuval et al., 2021). These strategies use domain knowledge of how properties flow to ensure conservation of different quantities. Instead of predicting tendencies directly, fluxes are predicted. Hess et al. (2022) introduces one global constraint to be applied to bias-correct the precipitation prediction generated by a GAN. Outside of climate science, recent work has emerged on enforcing hard constraints on the output of neural networks (e.g. Donti et al., 2021).

### Constrained Learning for Downscaling

In super-resolution for turbulent flows, MeshfreeFlowNet (Jiang et al., 2020) employs a physics-informed model which adds PDEs as regularization terms to the loss function. In parallel to our work, the first approaches employing hard constraints for climate-related downscaling were introduced: Geiss and Hardin (2023) introduced an enforcement operator applied to multiple CNN architectures for scientific data sets. A CNN with a multiplicative renormalization layer is used for atmospheric chemistry model downscaling in Geiss et al. (2022). We are the first to compare a variety of different hard-constraining approaches and also apply them to multiple deep learning architectures.

## 3 Enforcing Constraints

When modeling physical quantities such as precipitation or water mass, principled relationships such as mass conservation can naturally be established between low-resolution and high-resolution samples. Here, we introduce a new methodology to incorporate these constraints within a neural network architecture. We choose hard constraints enforced through the architecture over soft constraints that use an additional loss term. Hard constraints guarantee certain constraints even at inference time, whereas soft constraining encourages the network to output values that are close to satisfying constraints, by minimizing a penalty during training, but do not provide any guarantees. Additionally, for our case hard constraining increases the predictive ability, and soft constraining can lead to unstable training and an accuracy-constraints trade-off (Harder et al., 2022). Adding hard constraints restricts the hypothesis space to a smaller subspace that satisfies the constraints. With that, we reformulate the learning problem to an easier problem and achieve better results including prior knowledge.

### 3.1 Setup

Consider the case of downscaling low-resolution pixels x by a factor of N in each linear dimension, and let n := N^2. Let y_i, i = 1, ..., n be the values in the predicted high-resolution patch that correspond to x. The set {y_i} for i = 1, ..., n is also referred to as a super-pixel. Then, a conservation law takes the form of the following constraint:

$$\frac{1}{n}\sum_{i=1}^{n}y_i = x \quad \text{(Eq. 1)}$$

Depending on the predicted quantity, there may additionally be an inequality constraint associated with the data. In our work, there was only one example, concerning the positivity of several physical quantities (e.g. water mass). The inequality for this case would be:

$$\forall i \in [[1,n]], \quad y_i \geq 0$$

We note that the methodologies we suggest in this work only deal with this special case.

### 3.2 Constraint Layer

We introduce three different alternatives as constraint layers: additive constraining, multiplicative constraining, and softmax-based constraining. These are all added at the end of any neural architecture and all satisfy Eq. 1 by construction. The constraints are applied for each pair of input pixel x and the corresponding SR N x N patch.

We will use y_tilde_i, i = 1, ..., n to denote the intermediate outputs of the neural network before the constraint layer and y_i, i = 1, ..., n to be the final outputs after applying the constraints.

**Additive constraining**

For our Additive Constraint Layer (AddCL), we take the intermediate outputs and reset them using the following operation:

$$y_j = \tilde{y}_j + x - \frac{1}{n}\sum_{i=1}^{n}\tilde{y}_i \quad \text{(Eq. 3)}$$

We also consider a more complex additive approach, the Scaled Additive Constraint Layer (ScAddCL), which was introduced in parallel work to ours by Geiss and Hardin (2023):

$$y_j = \tilde{y}_j + \left(x - \frac{1}{n}\sum_{i=1}^{n}\tilde{y}_i\right) \cdot \frac{\sigma + \tilde{y}_i}{\sigma + \frac{1}{n}\sum_{i=1}^{n}\tilde{y}_i}$$

with sigma := sign(1/n * sum(y_tilde_i) - x), so sigma in {-1, 1}. The pixel values are assumed to be in [-1, 1]. For more details see Geiss and Hardin (2023).

**Multiplicative constraining**

For the Multiplicative Constraint Layer (MultCL) approach, we rescale the intermediate output using the corresponding input value x:

$$y_j = \tilde{y}_j \cdot \frac{x}{\frac{1}{n}\sum_{i=1}^{n}\tilde{y}_i} \quad \text{(Eq. 4)}$$

A similar approach is used in Geiss et al. (2022). Note that this approach can violate non-negativity constraints (e.g. 18 pixels per 128x128 patch for 8x upsampling), so it is sometimes detrimental. Multiplicative constraining can however be generalized by introducing any function g:

$$y_j = g(\tilde{y}_j) \cdot \frac{x}{\frac{1}{n}\sum_{i=1}^{n}g(\tilde{y}_i)} \quad \text{(Eq. 5)}$$

If g is positive, the output is guaranteed to be positive too.

**Softmax constraining**

For predicting quantities like atmospheric water content, we want to enforce the output to be non-negative for it to be physically valid. Here, we use a softmax multiplied by the corresponding input pixel value x:

$$y_j = \exp(\tilde{y}_j) \cdot \frac{x}{\frac{1}{n}\sum_{i=1}^{n}\exp(\tilde{y}_i)}$$

This Softmax Constraint Layer (SmCL) is a special case of Eq. 5 with g = exp and enforces y_i >= 0, i = 1, ..., n.

**Differences of Constraint Layers**

The four different constraint layers have in common that they all enforce Eq. 1 by construction and we will see in Section 5 that the differences in performance are rather small. To point out and summarize the differences: Whereas ScAddCL ([-1,1]) and MultCL (non-zero) are restricted in the range of input values they can handle, AddCL and SmCL work with any inputs. SmCL gives only positive outputs, which can be either beneficial by serving as an additional physical constraint or too restrictive if the output domain includes negative values. MultCL might get unstable for values close to zero. Additionally, the choice of constraint layer influences the variance among super-pixels, with SmCL having the highest variance.

### 3.3 Generalization of Our Constraining Methodologies

The focus of this work is on a consistency constraint for downscaling, but the methodology is not limited to this and can be applied to different setups. It can be slightly adapted to e.g. enforce a weighted formulation of Eq. 1, global constraint, or mass conservation constraints for emulation. Here we show how our constraint layers can be employed for different cases, starting with a more general setup and then formulating special relevant cases.

#### 3.3.1 Generalization Setup

We consider the learning task (supervised or unsupervised), where X in R^{n_in} is our input and y in R^{n_out} the final output. Let (I_j)_{j=1,...,n_p} be a partition of {1,...,n_out} into n_p subsets (n_p determines how many different constraints are imposed, e.g. n_in for our downscaling setup), g_{ij}: D subset R -> R, i in I_j an invertible function and h_j: R^{n_out} -> R an arbitrary function. The set of constraints is given by:

$$\sum_{i \in I_j} g_{ij}(y_i) = h_j(X) \quad \text{(Eq. 6)}$$

for each j = 1, ..., n_p.

These constraints can then be enforced with the above-introduced layers restated as follows:

$$y_i^{\text{AddCL}} = g_{ij}^{-1}\left(\tilde{y}_i + \frac{1}{n}h_j(X) - \frac{1}{n}\sum_{k \in I_j}\tilde{y}_k\right)$$

$$y_i^{\text{MultCL}} = g_{ij}^{-1}\left(\tilde{y}_i \cdot \frac{h_j(X)}{\sum_{k \in I_j}\tilde{y}_k}\right)$$

$$y_i^{\text{SmCL}} = g_{ij}^{-1}\left(\exp(\tilde{y}_i) \cdot \frac{h_j(X)}{\sum_{k \in I_j}\exp(\tilde{y}_k)}\right)$$

for i in I_j and j = 1, ..., n_p.

The main case considered in this work (Eq. 1) is a special case with h_j(X) = nX_j for j indexing all super-pixels and g being the identity function. Note that MultCL and SmCL cannot be directly applied if h_j = 0 for any j, leading to a constant prediction.

#### 3.3.2 Weighted Formulation

In an Earth system modeling context data often originates from a latitude-longitude grid. This implies that the areas in each field are not exactly the same. The downscaling consistency constraint (Eq. 1) is then changed to a weighted formulation:

$$\frac{1}{n}\sum_{i=1}^{n}\alpha_i y_i = x \quad \text{(Eq. 7)}$$

Analogously, the AddCL, MultCL, and SmCL are reformulated as:

$$y_i^{\text{AddCL}} = \frac{1}{\alpha_i}\left(\tilde{y}_i + x - \frac{1}{n}\sum_{k=1}^{n}\tilde{y}_k\right)$$

$$y_i^{\text{MultCL}} = \tilde{y}_i \cdot \frac{x}{\alpha_i \frac{1}{n}\sum_{k=1}^{n}\tilde{y}_k}$$

$$y_i^{\text{SmCL}} = \exp(\tilde{y}_i) \cdot \frac{x}{\alpha_i \frac{1}{n}\sum_{k=1}^{n}\exp(\tilde{y}_k)}$$

We note that in our case we do not use a weighted formulation, since the ERA5 LR data is created by average pooling without weighting and the WRF data covers a small area, so there the lat-lon cells have about the same area.

#### 3.3.3 Relaxing Constraints and Global Constraining

The constraint layers can be relaxed by increasing the constraint window size; this can then impose soft constraints. In the extreme case, this would reduce the number of constraints to one and gives the possibility of adding global constraint. The constraints would be the same as in Eq. 1, but with n being the number of total pixels.

#### 3.3.4 Application in Emulation

Our constraining methodology is not limited to downscaling and can enforce mass conservation e.g. in emulation tasks. An example could be aerosol microphysics emulation (Harder et al., 2022), where different aerosol masses need to be conserved within each time step. The predicted aerosol masses among different size bins y_i, i in I_dust for a specific aerosol type, e.g. dust, have to add up to the sum of the input aerosol masses X_i, i in I_dust of the same species:

$$\sum_{i \in I_{\text{dust}}} y_i = \sum_{i \in I_{\text{dust}}} X_i$$

This conservation of mass can be enforced with the AddCL, MultCL, or SmCL:

$$y_i^{\text{AddCL}} = \tilde{y}_i + \sum_{k \in I_{\text{dust}}} X_k - \sum_{k \in I_{\text{dust}}}\tilde{y}_k$$

$$y_i^{\text{MultCL}} = \tilde{y}_i \cdot \frac{\sum_{k \in I_{\text{dust}}} X_k}{\sum_{k \in I_{\text{dust}}}\tilde{y}_k}$$

$$y_i^{\text{SmCL}} = \exp(\tilde{y}_i) \cdot \frac{\sum_{k \in I_{\text{dust}}} X_k}{\sum_{k \in I_{\text{dust}}}\exp(\tilde{y}_k)}$$

Here, SmCL again would additionally guarantee positive masses.

## 4 Data

To test and evaluate our proposed method, we create a variety of data sets as well as use existing and established ones. We generate multiple data sets based on the ERA5 data using average pooling to create the LR inputs, which has been the standard methodology in climate downscaling studies. We also use data sets based on the outputs of models such as the Weather and Research Forecasting (WRF) Model and the Norwegian Earth System Model (NorESM) that contain real low-resolution simulation data matched to high-resolution data. Finally, we test our methods on non-climate data sets: lunar satellite imagery and natural images. An overview of all the different data sets used can be found in Table 1.

**Table 1:** The different data sets used to test constraint layers.

| Name | Source | Type | Dim. LR/HR | Size train/val/test |
|------|--------|------|------------|-------------------|
| TCW2 | ERA5 | water cont. | (1,64,64)/(1,128,128) | 40k/10k/10k |
| **TCW4** | ERA5 | water cont. | (1,32,32)/(1,128,128) | 40k/10k/10k |
| TCW8 | ERA5 | water cont. | (1,16,16)/(1,128,128) | 40k/10k/10k |
| TCW16 | ERA5 | water cont. | (1,8,8)/(1,128,128) | 40k/10k/10k |
| TCW OOD | ERA5 | water cont. | (1,32,32)/(1,128,128) | 40k/10k/10k |
| **TCW T1** | ERA5 | water cont. | (3,32,32)/(3,128,128) | 40k/10k/10k |
| **TCW T2** | ERA5 | water cont. | (2,32,32)/(3,128,128) | 40k/10k/10k |
| MEn | ERA5 | water vapor, liq. water, temp. | (3,32,32)/(3,128,128) | 40k/10k/10k |
| **WRF** | WRF | temp. | (1,45,45)/(1,135,135) | 20k/4k/4k |
| NorESM | NorESM | temp. | (1,32,32)/(1,64,64) | 24k/12k/12k |
| Lunar | satell. | photons | (1,32,32)/(1,128,128) | 132k/16k/16k |
| Nat | Nat. images | RGB | (3,128,128)/(3,512,512) | var. |

### 4.1 ERA5 Data Set

The ERA5 data set (Hersbach et al., 2020) is a so-called *reanalysis* product from the ECMWF that combines model data with worldwide observations. The optimal physical model state that best fits the observations is found through the process of data assimilation. ERA5 is available as global, hourly data with a 0.25 x 0.25 degree resolution, which is roughly 25 km per pixel in the mid-latitudes. It covers all years starting from 1950.

**Total water content data set.** For this work, the quantity we focus on is the total column water (tcw) that is given in kg/m^2 and describes the vertical integral of the total amount of atmospheric water content, including water vapour, cloud water, and cloud ice but not precipitation.

**Spatial SR data.** To obtain our high-resolution data points we extract a random 128x128 pixel image from each available time step (each time step is 721x1440 and there are roughly 60,000 time steps available). We randomly sample 40,000 data points for training and 10,000 for each validation and testing. The low-resolution counterparts are created by taking the mean over NxN patches, where N is our upsampling factor. This operation is physically sound, considering that conservation of water content means that the water content (density per squared meter) described in an LR pixel should be equal to the average of the corresponding HR pixels. We can also observe in LR-modeled data such as WRF data (see below) that the modeled quantities in a low-resolution run are approximately the mean of a high-resolution run, which further justifies our coarsening strategy.

**Spatio-Temporal data sets.** Including the temporal evolution of our data, we create two additional data sets. For the first data set, one sample consists of 3 successive time steps, the same time steps for both input and target, but at different resolutions. This is done to perform spatial SR for multiple time steps simultaneously. We select three random 128x128 pixel areas per global image, resulting in the same number of examples as the procedure described above. We split the data randomly as before, and each time step is downsampled by taking the spatial mean. We then create a second data set, that is built for the learning task of increasing both spatial and temporal dimensions. We again crop three images out of a series of three successive time steps to obtain our high-resolution target. To create the low-resolution input, we decrease both temporal and spatial dimensions. To decrease the temporal resolution, we remove the intermediate (the second) time step in each sample, i.e. perform sub-sampling. To decrease the spatial resolution we apply the same operation as before, i.e. compute the mean spatially. These results result in two LR inputs. Temporally coarse-graining by subsampling not by averaging is done to avoid leakage of future information into previous time steps.

**OOD data set.** For the data sets described above, the train-val-test split is done randomly. To understand how our constraining influences out-of-distribution generalization, we create a data set with a split in time. Here, we expect patterns to appear in the later time steps that are out-of-distribution of what was previously observed. We train on older data and then test on more recent years: for training, we use the years 1950-2000, for validation 2001-2010, and for final testing 2011-2020.

**Energy data set.** Also originating from the ERA5 data, we create a second data set including different physical variables coming with different constraints as well. This data set is constructed to preserve moist static energy and water masses while predicting water vapor, liquid water content, and air temperature. The variables are taken from the pressure level at 850hPa.

### 4.2 WRF Data

In Watson et al. (2020), a data set using the Advanced Research version of the WRF Model is introduced. It comprises hourly operational weather forecast data for Lake George in New York, USA from 2017-01-01 to 2020-03-20. The variable we consider for this work is the temperature at 2m above the ground. Unlike the previous data sets, this one does not involve synthetic downsampling but includes two forecasts run at different resolutions with different physics-based parameterizations: one at 9 km horizontal resolution and one at 3 km. Our goal is to predict the 3 km resolution temperature field given the 9 km one and builds on work by Auger et al. (2021), which used the same data set.

### 4.3 Constraints in Our Data Sets

In predicting distinct physical quantities, there are different constraints we need to consider. Most of our data sets include the downscaling constraints given by Eq. 1, which are satisfied by the LR-HR pairs either approximately (for simulations that are run at LR and HR with quantities respecting physical conservation laws) or exactly (in the case of average pooling for creating the LR version).

**Water content conservation.** For predicting the total column-integrated water content, we are given the low-resolution water content Q^(LR) and must obtain the super-resolved version Q^(SR). The downscaling constraint or mass conservation constraint (Eq. 1) for each LR pixel q^(LR) and the corresponding super-pixel (q_i^(SR))_{i=1,...,n} is then given by:

$$\frac{1}{n}\sum_{i=1}^{n}q_i^{(SR)} = q^{(LR)} \quad \text{(Eq. 8)}$$

**Moist static energy conservation.** One of our tasks includes predicting column-integrated water vapor, liquid water, and temperature while conserving both water mass and moist static energy. The (column-integrated) moist static energy S is approximated by:

$$S \approx ((1 - Q_v) \cdot c_{pd} + Q_L \cdot c_l) \cdot T + L_v \cdot Q_v \quad \text{(Eq. 9)}$$

where L_v ~ 2.5008 * 10^6 + (c_pw - c_L) * (T - 273.16) is the latent heat of vaporization in J/kg. The water vapor Q_v [kg/kg], the liquid water Q_L [kg/kg], and the temperature T [K] are being predicted, whereas c_pd, c_pv and c_L [J/(K*kg)] are heat capacity constants.

We use the following procedure to predict these quantities while conserving moist static energy:
1. Given LR T^LR, Q_V^LR, Q_L^LR
2. Calculate LR S^LR with Eq. 9
3. Predict SR S^SR, Q_v^SR, Q_L^SR while enforcing Eq. 1 using one of our constraint layers
4. Calculate SR T^SR using Eq. 9 and SR S^SR, Q_v^SR, Q_L^SR

This means we predict T^SR not directly, but by predicting S^SR. We are then able to predict the temperature T while ensuring (approximate) energy conservation by applying our constraint layer to the prediction of S^SR.

**Different simulations.** If the LR-HR pairs are not created by taking the local mean of the HR but by using two simulations run at different resolutions, the downscaling constraint is not automatically satisfied in the data. This is the case for our WRF and NorESM data sets. Even though the downscaling constraint is not exactly obeyed, it is approximately, and we can still apply our constraining in the same way as before. If the real low-resolution data and the downsampled high-resolution data are not significantly dissimilar, constraining can still benefit the predictive ability.

## 5 Experimental Setup

We conduct two sets of experiments:
1. Show the applicability of our constraining method to different neural network architectures.
2. Show the applicability of our constraining method to different data sets and different constraint types.

In most of our experiments, we use synthetic low-resolution data created by applying average pooling to the original high-res samples, as is usually done to test perfect prognosis downscaling setups. Additionally, we consider cases with pairs of real low-res and high-res simulations to show that our methods work in the intended final application.

### 5.1 Architectures

We test our constraint methods throughout a variety of standard deep learning SR architectures including an SR CNN, conditional GAN, a combination of an RNN and CNN for spatio-temporal SR, and a new architecture combining optical flow with CNNs/RNNs to increase the resolution of the temporal dimension. The original, unconstrained versions of these architectures then also serves as a comparison for our constraining methodologies.

**SR-CNNs.** Our SR CNN network, similar to Lim et al. (2017), consists of convolutional layers using 3x3 kernels and ReLU activations. The upsampling is performed by a transpose convolution followed by residual blocks (convolution, ReLU, convolution, adding the input, ReLU).

**SR-GAN.** A conditional GAN architecture (Mirza and Osindero, 2014) is a common choice for super-resolution (Ledig et al., 2016). Our version uses the above-introduced CNN architecture as the generator network. The discriminator consists of convolutional layers with a stride of 2 to decrease the dimensionality in each step, with ReLU activation. It is trained as a classifier to distinguish SR images from real HR images using a binary cross-entropy loss. The generator takes as input both Gaussian noise as well as the LR data and then generates an SR output. It is trained with a combination of an MSE loss, helping reconstruction, and the adversarial loss given by the discriminator.

**SR-ConvGRU.** We apply an SR architecture based on the GAN presented by Leinonen et al. (2021), which uses ConvGRU layers to address the spatio-temporal nature of super-resolving a time series of climate data. Here, we use the generator on its own, both during inference and training time without the discriminator, providing a deterministic approach.

**SR-FlowConvGRU.** To increase the temporal resolution of our data we employ the Deep Flow method (Liu et al., 2017), a deep learning architecture for video frame interpolation combining optical flow methods with neural networks. We introduce a new architecture combining the Deep Flow model and the ConvGRU network (FlowConvGRU): First, we increase the temporal resolution resulting in a higher-frequency time-series of LR images on which we then apply the ConvGRU architecture to increase the spatial resolution. The combined neural networks are then trained end-to-end.

### 5.2 Training

Our models were trained with the Adam optimizer, a learning rate of 0.001, and a batch size of 256. We trained for 200 epochs, which took about 3-6 hours on a single NVIDIA A100 Tensor Core GPU, depending on the architecture. All models use the MSE as their criterion, the GAN additionally uses its discriminator loss term. All the data are normalized between 0 and 1 for training, except for the cases where the ScAddCL is applied. In the case of this constraint layer we scale the data between -1 and 1 as proposed in Geiss and Hardin (2023). For our time-dependent models though, ConvGRU and FlowConvGRU, we are scaling between 0 and 1, because the original scaling led to NaN-values during training.

### 5.3 Baselines

**Pixel enlargement.** This baseline consists of scaling the LR input to the same size as the HR by duplicating the pixels. We include this to have reference metrics that reflect how close the LR is to the HR data. This baseline conserves mass by construction.

**Bicubic upsampling.** As a simple non-ML baseline, we use bicubic interpolation for spatial SR and take the mean of two frames for temporal SR.

**Soft constraining.** Soft-constraining has been successfully applied before to a variety of physics-informed deep-learning tasks. Here we use it to see how it compares to hard constraints. Soft-constraining is done by adding a regularization term to the loss function. Our MSE loss is then changed to the following:

$$\text{Loss} = (1 - \alpha) \cdot \text{MSE} + \alpha \cdot \text{Constraint violation}$$

where the constraint violation is the mean overall constraint violations between an input pixel x and the corresponding super-pixel y_i, i = 1, ..., n:

$$\text{Constraint violation} = \text{MSE}\left(\frac{1}{n}\sum_{i=1}^{n}y_i, \; x\right)$$

We conducted an experiment to investigate the impact of alpha values on final model performance; the results are reported in the appendix. For our main paper we choose alpha = 0.99.

**Unconstrained counterparts.** Furthermore, we always compare against an unconstrained version of the above-introduced standard SR NN architectures (SR-CNN, SR-GAN, SR-ConvGRU, SR-FlowConvGRU).

**Clipping.** We also run the standard CNN, but with clipping applied at inference. This is a common practice to remove negative values. This method does not guarantee mass conservation nor significantly improves performance.

## 6 Results and Discussion

For evaluating our results, we use typical metrics for weather and climate super-resolution: root-mean-square error (RMSE), mean absolute error (MAE) and mean bias as well as typical metrics for super-resolution: peak signal-to-noise ratio (PSNR), structural similarity index measure (SSIM), multi-scale SSIM (MS-SSIM), Pearson correlation and Fractional Skill Score (FSS). We show RMSE and MS-SSIM in the main paper, while the others can be found in the appendix. Most metrics are highly correlated in our case. For the GAN giving a probabilistic prediction, we also use continuous ranked probability score (CRPS). Because we are interested in the violation of conservation laws and predicting non-physical values, we also look at the average constraint violation, the number of (unwanted) negative pixels, and the average magnitude of negative values. We additionally look at the variance among the pixels within a predicted super-pixel and investigate the difference for constraining methods.

### 6.1 Different Constraining Methods

Whereas hard-constraining shows exact conservation and appears to enhance performance, the application of soft-constraining on the other hand does decrease constraint violation, but still maintains a significant magnitude of it. Also, soft-constraining seems to suffer from an accuracy-constraints trade-off, where depending on the regularization factor alpha, either the constraint violation is reduced, or the accuracy increases, but it struggles to do both simultaneously. Among the hard-constraining methodologies, the multiplicative renormalization layer, MultCL, performs the weakest in terms of predictive skills, which could be due to instability when inputs get close to zero. The three other methods, ScAddCL, AddCL, and SmCL, often have very similar measurements. SmCL shows the advantage of also enforcing positivity when necessary. ScAddCL divides the number of violations by more than 2 compared to the AddCL and MultCL gets close to zero violation in many cases.

### 6.2 Different Architectures

For all architectures (CNN, GAN, ConvGRU, FlowConvGRU), adding the constraint layers enforces the constraint and improves the evaluation metrics compared to the unconstrained case. Constraining the GAN leads to less of a performance boost, but AddCL and SmCL still enhance the predictions compared to the unconstrained GAN. Including the temporal dimensions, the constraining improves the prediction quality much more significantly than in the case with just a single time step.

### 6.3 Different Data Sets and Constraints

The success of our constraining methodology does not depend on the upsampling factor: the constraining methods work well and improve all metrics for upsampling factors of 2, 4, 8, and 16. When applied to our out-of-distribution data set, the improvement achieved by adding constraints is even more pronounced than for the randomly split data. The constraints can help architectures with their generalization ability.

Not only mass can be conserved, but other quantities such as moist static energy. Moving on to different quantities of the ERA5 data set---temperature, water vapor, and liquid water---one can observe similar results for liquid water Q_L and water vapor Q_v as for the total water content: ScAddCL, AddCL, and SmCL significantly improve results in all measures over the unconstrained CNN, while enforcing energy and mass conservation. For temperature, on the other hand, MultCL performs the strongest, followed by SmCL, whereas AddCL and ScAddCL achieve smaller improvements in the scores.

**WRF temperature data.** Our WRF temperature data set includes low-resolution data points drawn from a separate simulation, rather than downsampling, and therefore it results in a much harder task. The scores are improved slightly with our constraint layer, this might be counterintuitive given there is a violation in the training data, but this violation is relatively small, it appears like random noise, so no bias is introduced. This way the constraints again lead to a simpler learning problem and are able to improve performance. The fact that the constraints are slightly violated in the original data set could motivate soft-constraining, but nevertheless, we can observe that soft-constraining harms the predictive performance, while hard-constraining is surprisingly beneficial. The constraint violation in the original data has an RMSE of 0.6838 on average.

**Table 2:** Metrics for different constraining methods applied to the SR CNN on WRF temperature data, calculated over 10,000 test samples (mean over 3 runs).

| Data | Model | Constraint | RMSE | MAE | MS-SSIM | Constr. viol. |
|------|-------|-----------|------|-----|---------|--------------|
| WRF | Enlarge | none | 1.015 | 0.648 | 94.51 | **0.000** |
| WRF | CNN | none | 0.952 | 0.618 | 94.92 | 0.181 |
| WRF | CNN | soft | 1.020 | 0.660 | 94.57 | 0.032 |
| WRF | CNN | SmCL | **0.950** | **0.592** | **95.25** | **0.000** |

Finally, applying our constraint methodology can improve results in other domains, even in cases where there is no physics involved. Both for the lunar satellite imagery and the natural images benchmark data sets, the application of SmCL improves the traditional metrics.

### 6.4 Perceptual Quality of Predictions

Additionally to an enhancement quantitatively, we can see an improved visual quality for some examples for the water content data. For the WRF temperature forecast data, we see a very significant improvement in the perceptual quality of the prediction---much more detail is added to the prediction when adding our constraining. For the lunar satellite imagery, applying constraints can make the image slightly less blurry.

### 6.5 Development of Error During Training

Observing how the MSE develops during training, we can see that the curve of the constrained network is generally lower than the unconstrained one. Additionally, it can be seen that constraining helps smooth both the training and validation curves.

### 6.6 Spatial Distribution of Errors

A known issue in downscaling methods is the so-called coastal effect, where errors of predictions tend to be more pronounced in coastal regions. Besides coastal region areas, mountain ridges can also be critical. Both predictions show more errors in coastal and mountainous regions. However, if we analyze the difference in errors between the unconstrained and constrained versions, we can see that constraining leads to lower errors in those areas.

### 6.7 Limitations

In the case of our WRF data set, we have seen that the constraining methodology can improve predictive performance even if the underlying constraints are slightly violated by the original data. In cases where low-resolution and its high-resolution counterpart are too far apart, our model is not always able to increase the predictive skill. We built a data set from two different resolutions of the Norwegian Earth System Model (NorESM) (Seland et al., 2020), and applying our constraining methods improved the visual similarity of the predictions, but decreased the predictive ability. In the case of other sampling strategies such as subsampling spatially, our methods are not applicable in their current form and they depend on having constraints that can be formulated with Eq. 6.

## 7 Conclusion and Future Work

This work presents a novel methodology to incorporate physics-inspired downscaling hard constraints into neural network architectures for climate super-resolution. We show that this method performs well across different deep learning architectures, upsampling factors, predicted quantities, and data sets. We demonstrate its effectiveness both on standard downscaling data sets and on data created by independent simulations. Our constrained models are not only guaranteed to satisfy consistency such as mass conservation between LR and HR, but also increase predictive performance across metrics and use cases. Compared to soft-constraining through the loss function, our methodology does not suffer from the common accuracy-constraints enforcement trade-off. Our hard-constraining performance enhancement is not only limited to climate super-resolution but also noticeable in satellite imagery of the lunar surface as well as standard benchmark data sets of natural images. Within the climate context, our constraint layer can help with common issues connected to deep learning applied to downscaling: it dampens the coastal effect, errors get lower in critical regions, out-of-distribution generalization is improved and training can be more stable. Hard-constraining can weaken performance if the enforced relationships are strongly violated in the true data (see NorESM data). If a bias exists in the LR (or other input) it can be propagated to the HR prediction by constraining on the LR.

Future work could extend the application of our constraint layer to other climate-related tasks beyond downscaling. Climate model emulation for example could strongly benefit from a reliable and performance-enhancing method to enforce physical laws. For post-processing purposes, the offline application of our method, our code is readily available. To deploy these constrained super-resolution methods online, the next step is to use Fortran-Python bridges (e.g. Ott et al., 2020) to include them in global climate model runs.

## Appendix A: Tuning Soft-Constraining

Investigation of the influence of the factor alpha on the soft-constraining method shows that increasing alpha improves the mass conservation but only up to a value between 0.014 and 0.017. At the same time, the predictive skill decreases with the increase of alpha significantly.

**Table 3:** Metrics for different alpha values calculated over 10,000 validation samples.

| Data | Alpha | RMSE | MAE | MS-SSIM | Mass viol. | #Neg per mil. |
|------|-------|------|-----|---------|-----------|--------------|
| TCW4 | 0.0001 | 0.241 | 0.102 | 99.95 | 0.021 | 1.21 |
| TCW4 | 0.001 | **0.237** | **0.100** | **99.96** | 0.022 | **0.12** |
| TCW4 | 0.01 | 0.247 | 0.103 | 99.96 | 0.022 | 1.39 |
| TCW4 | 0.1 | 0.252 | 0.104 | 99.95 | 0.023 | 0.41 |
| TCW4 | 0.9 | 0.268 | 0.110 | 99.95 | 0.020 | 16.83 |
| TCW4 | 0.99 | 0.297 | 0.133 | 99.94 | **0.014** | 31.01 |
| TCW4 | 0.999 | 0.477 | 0.261 | 99.84 | 0.016 | 600.96 |
| TCW4 | 0.9999 | 0.706 | 0.433 | 99.71 | 0.017 | 3867.90 |
| TCW4 | 1 | 2.618 | 1.814 | 94.22 | 0.017 | 960.42 |

## Appendix B: Clipping for Nonnegativity

Clipping gives a very small increase in performance, but still performs significantly worse than SmCL, which achieves also zero negative values. A combination of a constraint layer such as MultCL and clipping would lead to the clipping layer to destroy the enforced consistency given by the constraint layer if applied afterwards.

**Table 4:** Metrics for SR CNN + clipping on water content data (mean over 3 runs).

| Data | Model | Constraint | RMSE | MAE | MS-SSIM | Mass viol. | #Neg per mil. |
|------|-------|-----------|------|-----|---------|-----------|--------------|
| TCW4 | CNN | none | 0.661 | 0.327 | 99.39 | 0.059 | 2.41 |
| TCW4 | CNN | clip | 0.657 | 0.326 | 99.44 | 0.058 | **0** |
| TCW4 | CNN | SmCL | **0.582** | **0.291** | **99.49** | **0.000** | **0** |

## Appendix C: Score Tables

### SR CNN Results Across Upsampling Factors

**Table 5:** Metrics for different constraining methods applied to an SR CNN across upsampling factors, calculated over 10,000 test samples of the water content data (mean over 3 runs).

| Data | Factor | Model | Constraint | RMSE | MAE | MS-SSIM | Mass viol. | #Neg per mil. |
|------|--------|-------|-----------|------|-----|---------|-----------|--------------|
| TCW2 | 2x | Enlarge | none | 0.422 | 0.361 | 99.61 | 0.000 | 0 |
| TCW2 | 2x | Bicubic | none | 0.322 | 0.137 | 99.90 | 0.066 | 0.25 |
| TCW2 | 2x | CNN | none | 0.251 | 0.105 | 99.95 | 0.026 | 1.40 |
| TCW2 | 2x | CNN | soft | 0.301 | 0.137 | 99.23 | 0.016 | 104.65 |
| TCW2 | 2x | CNN | AddCL | 0.216 | 0.092 | **99.96** | **0.000** | 1.31 |
| TCW2 | 2x | CNN | ScAddCL | **0.199** | **0.088** | **99.96** | **0.000** | 0.02 |
| TCW2 | 2x | CNN | MultCL | 0.223 | 0.094 | **99.96** | **0.000** | **0** |
| TCW2 | 2x | CNN | SmCL | 0.215 | 0.094 | **99.96** | **0.000** | **0** |
| TCW4 | 4x | Enlarge | none | 1.286 | 0.717 | 97.60 | 0.000 | 0 |
| TCW4 | 4x | Bicubic | none | 0.800 | 0.401 | 99.12 | 0.169 | 0.53 |
| TCW4 | 4x | CNN | none | 0.657 | 0.326 | 99.40 | 0.058 | 2.41 |
| TCW4 | 4x | CNN | soft | 0.801 | 0.410 | 99.15 | 0.023 | 581.54 |
| TCW4 | 4x | CNN | AddCL | 0.580 | 0.290 | **99.50** | **0.000** | 1.42 |
| TCW4 | 4x | CNN | ScAddCL | **0.575** | **0.289** | **99.50** | **0.000** | 0.07 |
| TCW4 | 4x | CNN | MultCL | 0.606 | 0.300 | 99.47 | **0.000** | **0** |
| TCW4 | 4x | CNN | SmCL | 0.582 | 0.291 | 99.49 | **0.000** | **0** |
| TCW8 | 8x | Enlarge | none | 2.181 | 1.294 | 92.39 | 0.000 | 0 |
| TCW8 | 8x | Bicubic | none | 1.557 | 0.900 | 96.49 | 0.318 | 6.56 |
| TCW8 | 8x | CNN | none | 1.358 | 0.782 | 97.15 | 0.109 | 15.48 |
| TCW8 | 8x | CNN | soft | 1.640 | 0.965 | 96.06 | 0.029 | 103,702 |
| TCW8 | 8x | CNN | AddCL | 1.267 | **0.733** | **97.41** | **0.000** | 632.32 |
| TCW8 | 8x | CNN | ScAddCL | **1.264** | 0.734 | **97.41** | **0.000** | 0.15 |
| TCW8 | 8x | CNN | MultCL | 1.331 | **0.733** | 97.22 | **0.000** | 0.10 |
| TCW8 | 8x | CNN | SmCL | 1.268 | 0.734 | 97.40 | **0.000** | **0** |
| TCW16 | 16x | Enlarge | none | 3.425 | 2.159 | 85.55 | 0.000 | 0 |
| TCW16 | 16x | Bicubic | none | 2.723 | 1.730 | 91.72 | 0.510 | 53.67 |
| TCW16 | 16x | CNN | none | 2.450 | 1.545 | 92.68 | 0.203 | 4.15 |
| TCW16 | 16x | CNN | soft | 2.794 | 1.776 | 90.74 | 0.036 | 2250.77 |
| TCW16 | 16x | CNN | AddCL | **2.364** | **1.491** | **92.96** | **0.000** | 457.34 |
| TCW16 | 16x | CNN | ScAddCL | 2.368 | 1.495 | 92.94 | **0.000** | 2.12 |
| TCW16 | 16x | CNN | MultCL | 2.409 | 1.518 | 92.77 | **0.000** | 0.17 |
| TCW16 | 16x | CNN | SmCL | 2.368 | 1.492 | 92.95 | **0.000** | **0** |

### SR GAN Results

**Table 6:** Metrics for different constraining methods applied to an SR GAN, calculated over 10,000 test samples of the 4x upsampling water content data (mean over 3 runs).

| Data | Model | Constraint | RMSE | MAE | CRPS | MS-SSIM | Mass viol. | #Neg per mil. |
|------|-------|-----------|------|-----|------|---------|-----------|--------------|
| TCW4 | GAN | none | 0.628 | 0.313 | 0.1522 | 99.44 | 0.0453 | 3.46 |
| TCW4 | GAN | AddCL | **0.602** | 0.306 | 0.1519 | **99.46** | **0.000** | 7.38 |
| TCW4 | GAN | ScAddCL | 0.604 | **0.305** | **0.1508** | **99.46** | **0.000** | 0.05 |
| TCW4 | GAN | MultCL | 0.732 | 0.406 | 0.1978 | 99.13 | **0.000** | **0** |
| TCW4 | GAN | SmCL | 0.603 | 0.310 | 0.1520 | **99.46** | **0.000** | **0** |

### SR ConvGRU Results

**Table 7:** Metrics for different constraining methods applied to an SR ConvGRU, calculated over 10,000 test samples.

| Data | Model | Constraint | RMSE | MAE | MS-SSIM | Mass viol. | #Neg per mil. |
|------|-------|-----------|------|-----|---------|-----------|--------------|
| TCW T1 | Enlarge | none | 1.292 | 0.718 | 97.72 | 0.000 | 0 |
| TCW T1 | Bicubic | none | 0.807 | 0.402 | 99.16 | 0.169 | 2.16 |
| TCW T1 | ConvGRU | none | 0.672 | 0.340 | 99.42 | 0.102 | 55.45 |
| TCW T1 | ConvGRU | AddCL | **0.499** | **0.260** | **99.64** | **0.000** | 1358.49 |
| TCW T1 | ConvGRU | ScAddCL | **0.499** | **0.260** | **99.64** | **0.000** | 10.58 |
| TCW T1 | ConvGRU | MultCL | 0.903 | 0.472 | 98.98 | **0.000** | 0.25 |
| TCW T1 | ConvGRU | SmCL | 0.500 | **0.260** | **99.64** | **0.000** | **0** |

### FlowConvGRU Results

**Table 8:** Metrics for different constraining methods applied to FlowConvGRU, calculated over 10,000 test samples.

| Data | Model | Constraint | RMSE | MAE | MS-SSIM | Mass viol. | #Neg per mil. |
|------|-------|-----------|------|-----|---------|-----------|--------------|
| TCW T2 | Interpolation | none | 0.834 | 0.428 | 99.10 | 0.169 | 2.14 |
| TCW T2 | FlowConvGRU | none | 0.673 | 0.352 | 99.40 | 0.072 | 18.27 |
| TCW T2 | FlowConvGRU | AddCL | **0.509** | 0.275 | **99.63** | **0.000** | 37.10 |
| TCW T2 | FlowConvGRU | ScAddCL | **0.509** | **0.274** | **99.63** | **0.000** | 13.40 |
| TCW T2 | FlowConvGRU | MultCL | 0.719 | 0.383 | 99.27 | **0.000** | **0** |
| TCW T2 | FlowConvGRU | SmCL | 0.514 | 0.276 | 99.62 | **0.000** | **0** |

### OOD Results

**Table 9:** Metrics for the SR CNN on OOD water content data (mean over 3 runs).

| Data | Model | Constraint | RMSE | MAE | MS-SSIM | Mass viol. | #Neg per mil. |
|------|-------|-----------|------|-----|---------|-----------|--------------|
| TCW OOD | Enlarge | none | 1.274 | 0.711 | 97.60 | 0.000 | 0 |
| TCW OOD | Bicubic | none | 0.792 | 0.397 | 98.63 | 0.167 | 0.55 |
| TCW OOD | CNN | none | 0.661 | 0.327 | 99.39 | 0.059 | 4.93 |
| TCW OOD | CNN | AddCL | 0.575 | **0.287** | **99.50** | **0.000** | 1.65 |
| TCW OOD | CNN | ScAddCL | **0.573** | 0.288 | **99.50** | **0.000** | 0.21 |
| TCW OOD | CNN | MultCL | 0.591 | 0.294 | 99.47 | **0.000** | **0** |
| TCW OOD | CNN | SmCL | 0.579 | 0.289 | 99.49 | **0.000** | **0** |

### Energy Data Set Results

**Table 10:** Metrics for different constraining methods applied to the SR CNN for water vapor, liquid water, and temperature (mean over 3 runs).

| Data | Var. | Model | Constraint | RMSE | MAE | MS-SSIM | Constr. viol. |
|------|------|-------|-----------|------|-----|---------|--------------|
| MEn | Q_v | Enlarge | none | 0.474 | 0.262 | 94.74 | 0.000 |
| MEn | Q_v | Bicubic | none | 0.326 | 0.182 | 97.12 | 0.07 |
| MEn | Q_v | CNN | none | 0.260 | 0.141 | 98.14 | 0.02 |
| MEn | Q_v | CNN | AddCL | 0.250 | 0.133 | 98.28 | 0.00 |
| MEn | Q_v | CNN | ScAddCL | 0.250 | 0.133 | 98.28 | 0.00 |
| MEn | Q_v | CNN | MultCL | 0.250 | 0.133 | 98.28 | 0.00 |
| MEn | Q_v | CNN | SmCL | **0.248** | **0.132** | **98.30** | **0.00** |
| MEn | Q_L | Enlarge | none | 0.0217 | 0.00862 | 98.34 | 0.00000 |
| MEn | Q_L | Bicubic | none | 0.0186 | 0.00765 | 98.96 | 0.00236 |
| MEn | Q_L | CNN | none | 0.0157 | 0.00617 | 99.15 | 0.00067 |
| MEn | Q_L | CNN | AddCL | **0.0155** | **0.00588** | **99.18** | **0.00000** |
| MEn | Q_L | CNN | ScAddCL | **0.0155** | **0.00588** | 99.17 | **0.00000** |
| MEn | Q_L | CNN | MultCL | 0.0166 | 0.00647 | 99.06 | **0.00000** |
| MEn | Q_L | CNN | SmCL | **0.0155** | 0.00585 | 99.17 | **0.00000** |
| MEn | T | Enlarge | none | 0.470 | 0.288 | 99.03 | 0.0 |
| MEn | T | Bicubic | none | 0.281 | 0.156 | 99.67 | 159.1 |
| MEn | T | CNN | none | 0.459 | 0.287 | 99.03 | 139.7 |
| MEn | T | CNN | AddCL | 0.276 | 0.160 | 99.67 | 0.0 |
| MEn | T | CNN | ScAddCL | 0.280 | 0.163 | 99.67 | 0.0 |
| MEn | T | CNN | MultCL | **0.270** | **0.155** | **99.69** | **0.0** |
| MEn | T | CNN | SmCL | 0.272 | **0.155** | 99.68 | **0.0** |

## Appendix D: Additional Scores

Additional scores for the water content data set including mean bias, PSNR, SSIM, Pearson correlation, and negative mean show a similar trend to the metrics shown in the main paper: all of them are improved by adding constraints in the architecture. Without or with soft constraining there are small biases appearing in the predictions, but hard constraining removes those biases. Soft-constraining leads to the most significantly negative predictions, which would cause issues in the context of climate models and predictions.

### Fractional Skill Score

**Table 11:** FSS for different constraining methods and SR CNN on ERA5 water content data (TCW4).

| Data | Model | Constraint | 95perc. w=2 | 95perc. w=4 | 95perc. w=8 | 99perc. w=2 | 99perc. w=4 | 99perc. w=8 |
|------|-------|-----------|-------------|-------------|-------------|-------------|-------------|-------------|
| TCW4 | Enlarge | none | 0.970 | 0.989 | 0.997 | 0.935 | 0.974 | 0.991 |
| TCW4 | Bicubic | none | 0.971 | 0.987 | 0.994 | 0.935 | 0.969 | 0.986 |
| TCW4 | CNN | none | 0.978 | 0.992 | 0.997 | 0.950 | 0.979 | 0.993 |
| TCW4 | CNN | soft | 0.971 | 0.989 | 0.997 | 0.935 | 0.974 | 0.991 |
| TCW4 | CNN | ScAddCL | **0.981** | **0.993** | **0.998** | **0.956** | **0.983** | **0.994** |
| TCW4 | CNN | AddCL | **0.981** | **0.993** | **0.998** | **0.956** | **0.983** | **0.994** |
| TCW4 | CNN | MultCL | 0.979 | 0.992 | 0.998 | 0.951 | 0.980 | 0.993 |
| TCW4 | CNN | SmCL | **0.981** | **0.993** | **0.998** | 0.955 | **0.983** | **0.994** |

### Super-Pixel Variance

**Table 12:** Variance among super-pixels for different constraining methods.

| Data | Model | Constraint | Variance |
|------|-------|-----------|----------|
| TCW4 | Enlarge | none | 0.00 |
| TCW4 | Bicubic | none | 0.85 |
| TCW4 | CNN | none | 1.22 |
| TCW4 | CNN | soft | 0.96 |
| TCW4 | CNN | ScAddCL | 1.33 |
| TCW4 | CNN | AddCL | 1.32 |
| TCW4 | CNN | MultCL | 1.24 |
| TCW4 | CNN | SmCL | 1.34 |
| TCW4 | HR | none | 1.65 |

## Appendix F: NorESM Data

Our NorESM data set is based on the second version of the Norwegian Earth System Model (NorESM2). We build our data set on two different runs: NorESM-MM which has a 1-degree resolution for model components and NorESM2-LM which has a 2-degree resolution for atmosphere and land components. We use the temperature at the surface (tas) and a time period from 2015 to 2100. The scenarios ssp126 and ssp585 are used for training, ssp370 for validation and ssp245 for testing.

The results show that the best scores are in all cases achieved by the unconstrained CNN. This is probably due to the stronger violation of the downscaling constraints between low-resolution and high-resolution samples. The violation of the constraints here is 2.48 (RMSE), which is much higher than for the WRF case (0.68). The visual quality of the prediction, on the other hand, seems to be improved by constraining.

**Table 13:** Metrics for NorESM data (mean over 3 runs).

| Data | Model | Constraint | RMSE | MAE | MS-SSIM | Constr. viol. |
|------|-------|-----------|------|-----|---------|--------------|
| Tas NorESM | Enlarge | none | 2.987 | 1.915 | 95.96 | 0.000 |
| Tas NorESM | Bicubic | none | 2.910 | 1.864 | 96.36 | 0.073 |
| Tas NorESM | CNN | none | **2.348** | **1.559** | **96.93** | 1.034 |
| Tas NorESM | CNN | soft | 2.928 | 1.874 | 96.28 | 0.041 |
| Tas NorESM | CNN | AddCL | 2.885 | 1.847 | 96.45 | 0.000 |
| Tas NorESM | CNN | ScAddCL | 2.884 | 1.846 | 96.46 | 0.000 |
| Tas NorESM | CNN | MultCL | 2.888 | 1.859 | 96.43 | 0.000 |
| Tas NorESM | CNN | SmCL | 2.885 | 1.847 | 96.45 | 0.000 |

## Appendix G: Non-Climate Data

### Lunar Data

Recent work on super-resolution for lunar satellite imagery has shown how deep learning can be used to enhance the captured data to help future missions to the moon. To increase the resolution of images from regions like the south pole, where there is no high-resolution data available, a machine learning-ready data set has been created. It consists of 220,000 images cropped out of the Narrow-Angle Camera (NAC) imagery from NASA's Lunar Reconnaissance Orbiter (LRO). The average sampling is justified in this case, because the real LR images would be created with summing photon counts in low-light regions.

**Table 14:** Metrics for lunar data (mean over 3 runs).

| Data | Model | Constraint | RMSE | MAE | SSIM | PSNR |
|------|-------|-----------|------|-----|------|------|
| Lunar | CNN | none | 0.00217 | 0.00146 | 90.08 | 37.57 |
| Lunar | CNN | SmCL | **0.00213** | **0.00144** | **90.40** | **37.74** |

### Natural Images

The standard benchmark data sets for super-resolution deep learning architectures applied to natural images include the OutdoorSceneTraining (OST), DIV2K, and Flickr2k data sets for training and Set5, Set14, Urban100, and BSD100 for testing. Our constraints depend on the downsample technique used and can not directly be applied to other downsample techniques such as sub-sampling or bicubic interpolation.

**Table 15:** Metrics of the SR-GAN with and without SmCL on standard benchmark data sets.

| Data | Model | Constraint | RMSE | MAE | SSIM | PSNR |
|------|-------|-----------|------|-----|------|------|
| Set5 | SR-GAN | none | 8.57 | 4.80 | 92.48 | 29.47 |
| Set5 | SR-GAN | SmCL | **6.61** | **4.01** | **93.95** | **31.73** |
| Set14 | SR-GAN | none | 15.75 | 8.82 | 86.06 | 24.28 |
| Set14 | SR-GAN | SmCL | **14.07** | **8.12** | **87.37** | **25.16** |
| Urban100 | SR-GAN | none | 25.00 | 14.57 | 81.40 | 20.17 |
| Urban100 | SR-GAN | SmCL | **23.25** | **13.60** | **83.19** | **20.80** |
| BSD100 | SR-GAN | none | 14.38 | 8.28 | 85.95 | 24.97 |
| BSD100 | SR-GAN | SmCL | **13.52** | **7.82** | **87.09** | **25.50** |

