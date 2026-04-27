[IMAGE: ClimaX is built as a foundation model for any weather and climate modeling task. On the weather front, these tasks include standard forecasting tasks for various lead-time horizons at various resoluti]

# Introduction

Modeling weather and climate is an omnipresent challenge for science and society. With rising concerns around extreme weather events and climate change, there is a growing need for both improved weather forecasts for disaster mitigation and climate projections for long-term policy making and adaptation efforts [masson2021climate]. Currently, numerical methods for global modeling of weather and climate are parameterized via various general circulation models (GCM) [lynch2008origins]. GCMs represent system of differential equations relating the flow of energy and matter in the atmosphere, land, and ocean that can be integrated over time to obtain forecasts for relevant atmospheric variables [lynch2008origins; bauer2015quiet]. While extremely useful in practice, GCMs also suffer from many challenges, such as accurately representing physical processes and initial conditions at fine resolutions, as well as technological challenges in large-scale data assimilation and computational simulations [bauer2020ecmwf]. These factors limit their use in many scenarios, especially in simulating atmospheric variables quickly at very short time scales (e.g., a few hours) or accurately at long time scales (e.g., beyond 5-7 days) [zhang2019predictability].

In contrast, there has been a steady rise in data-driven approaches for forecasting of atmospheric variables, especially for meteorological applications [grover2015deep; dueben2018challenges; weber2020deep; scher2019weather; scher2018toward; kashinath2021physics; schultz2021can; reichstein2019deep; huntingford2019machine; schneider2017earth]. The key idea here is to train deep neural networks to predict the target atmospheric variables using decades of historical global datasets, such as the ERA-5 reanalysis dataset [hersbach2020era5]. Unlike GCMs, these networks are not explicitly grounded in physics, and lack general-purpose utility for Earth system sciences as they are trained for a specific predictive modeling task. Yet, with growing compute and datasets, there is emerging evidence that these models can achieve accuracies competitive with state-of-the-art numerical models in many scenarios, such as nowcasting of precipitation [ravuri2021skilful; sonderby2020metnet] and medium-range forecasting of variables like temperature, wind and humidity [weyn2020improving; rasp2021data; keisler2022forecasting; pathak2022fourcastnet; bi2022pangu; lam2022graphcast]. While these trends are encouraging, there remain concerns regarding the generality of such data-driven methods to diverse real-world scenarios, such as forecasting of extreme weather events and longer-term climate projections, especially under limited spatiotemporal supervision and computational budgets.

Variants of the aforementioned challenges apply broadly throughout machine learning (ML). In disciplines such as natural language processing and computer vision, it is well acknowledged that ML models trained to solve a single task using supervised learning are label-hungry during training and brittle when deployed outside their training distribution [taori2020measuring]. Recent works have shown that it is possible to mitigate the supervision bottleneck by *pretraining* [devlin2018bert; he2022masked] large unsupervised "foundation" models [Bommasani2021FoundationModels] on huge passive datasets, such as text and images scraped from the internet [ramesh2022hierarchical; brown2020language; liu2021swin; reed2022gato]. Post pretraining, there are many ways to *finetune* the same model on arbitrary target task(s) with little to none (i.e., zero-shot) additional supervision. Besides low target supervision, these models also generalize better to shifts outside their training distribution [hendrycks2020pretrained; zhang2022delving], improving their reliability.

Inspired by the above successes, this work studies the question: how do we design and train a foundation model for weather and climate that can be efficiently adapted for general-purpose tasks concerning the Earth's atmosphere? We propose ClimaX, a foundation model for weather and climate. For pretraining any foundation model, the key recipe is to train a deep architecture on a large dataset using an unsupervised objective. For example, many foundation models for language and vision train large transformers on Internet-scale datasets using generative modeling. While conceptually simple, this scaling recipe is riddled with challenges for weather and climate domains, that we discuss below and propose to resolve with ClimaX.

First, it is unclear what constitutes an Internet-scale passive dataset for pretraining ClimaX. The size of historical weather and climate datasets at any given time is fixed and increases at an almost constant rate everyday, as it corresponds to processed sensor measurements of naturally occurring phenomena. Our first key proposal is to go beyond these datasets to explicitly utilize physics-informed climate simulation models. Many such models are in use today, for example, the CMIP6 collection [eyring2016overview] of climate modeling simulations consists of runs of $\sim$100 distinct climate models from 49 different climate modeling groups. We show that the heterogeneity in these simulation datasets serves as a source of rich and plentiful data for pretraining ClimaX.

Second, we need a model architecture that can aptly embrace the heterogeneity of the above climate datasets. Climate data is highly multimodal, as observations typically correspond to many different, unbounded variables with varying datatypes (e.g., pressure, temperature, humidity). Moreover, many observational datasets are irregular in the sense that they differ in their spatiotemporal coverage and might correspond to different subsets of atmospheric variables. We resolve the above challenges in ClimaX by repurposing the vision transformer [dosovitskiy2020image; vaswani2017attention]. In contrast to earlier work where the input data is represented as an image with different atmospheric variables treated as the channels thereof [pathak2022fourcastnet; bi2022pangu], we treat them as separate modalities to enable more flexible training even with irregular datasets. This has the side-effect of drastically increasing the sequence length, which we propose to resolve via a cross-attention style channel aggregation scheme prior to the self-attention layers.

Third and last, we need a pretraining objective that can learn complex relationships between the atmospheric variables and permit effective finetuning for downstream tasks. Given the spatiotemporal nature of climate data, we propose a randomized forecasting objective for pretraining ClimaX. Here, the goal of the model is to forecast an arbitrary set of input variables at an arbitrary time into the future. While simple and intuitive, we show that such a pretraining objective aids finetuning to novel tasks and timescales even beyond the pretraining window, such as sub-seasonal to seasonal cumulative predictions, climate projections, and downscaling of climate models. See Figure 1 for a list of tasks considered in this work.

Empirically, we demonstrate that a single pretrained model can be finetuned for many tasks (e.g., multi-scale weather forecasting, climate projections, downscaling) under a range of operating conditions involving different spatiotemporal resolutions, geographical regions, and target prediction variables, including those unseen during training. Notably, our benchmark results are state-of-the-art on ClimateBench [watson2022climatebench] and competitive with the operational Integrated Forecasting System (IFS) [wedi2015modelling] on WeatherBench [rasp2020weatherbench], even when our model is trained on moderate resolutions using only a maximum of 80 NVIDIA V100 GPUs.

Finally, we show promising scaling laws of ClimaX with natural axes of performance improvements for larger number of pre-training datasets, larger models, and scaling to higher resolution gridded datasets. While especially the last is in line with recent and concurrent works on data-driven weather forecasting [pathak2022fourcastnet; bi2022pangu; lam2022graphcast], to the best of our knowledge, ClimaX is the first of its kind data-driven model that can effectively scale using heterogeneous climate datasets during pretraining, and generalize to diverse downstream tasks during finteuning, paving the way for a new generation of data-driven models for Earth systems science.

# Background and Related Work {#sec:background}

Current weather and climate models in use today rely extensively on numerical methods and computational simulations to predict and understand the Earth's weather and climate systems. These tasks include various *numerical weather prediction* (NWP) systems which use computer simulations to make short-term forecasts of weather conditions as well as climate models which use similar techniques to simulate and predict the long-term changes in the Earth's climate. Most notably, at the core of both weather and climate models lie the same set of primitive equations.

For climate modeling, earth system models (ESM) [hurrell2013community], or "coupled models", that couple together simulations which govern the atmosphere, cryosphere, land, and ocean processes are considered the state-of-the-art. Primarily these simulations are based on general circulation models (GCMs) [satoh2004atmospheric; lynch2008origins; adopted2014climate; masson2021climate] which date back to the works of phillips1956general [lorenz1967nature] solving Navier-Stokes equations on a rotation sphere to model fluid circulation. These models are often used to perform various *factor sensitivity* studies to examine how the changes in certain forcing factors like greenhouse gas concentrations can affect the global or regional climate and help in *climate projections* to help understand future conditions.

Numerical Weather Prediction (NWP) models share many components of GCMs, especially the atmospheric components [bauer2015quiet; lynch2008origins; kalnay2003atmospheric]. However, incorporating *data assimilation* [law2015data; grover2022rethinking] which involves combining observations and various measurements of the atmosphere and oceans together with these numerical models is important for accurate forecasts and simulations. Another significant distinction between weather and climate models is the framing of the solution for underlying equations: *initial value problem* for weather, while *boundary value problem* for climate [bauer2015quiet]. Different difficulty levels of these solution approaches results in the fact where climate models tend to be global often at coarser spatio-temporal resolutions while weather models can range from global to local and regional models of very high spatio-temporal resolutions [warner2010numerical].

Despite their noted success, including the recent 2021 Nobel Prize in Physics [ravishankara2022complex], there is considerable debate around the limitations of general circulation models (GCMs), particularly structural errors across models and the fact that current GCMs are designed to reproduce observed climate [balaji2022general]. The climate science community has been aware of these challenges which resulted in the creation of Coupled Model Intercomparison Project (CMIP) as a standardized protocol for evaluating and comparing the performance of different climate models [meehl2000coupled]. As we will see in the following sections, not only has CMIP been playing a crucial role in the advancement of our understanding of climate change and its potential impacts, its evaluation procedure has resulted in enormous quantity of data making modern deep learning based approaches quite attractive for many tasks. Notably, encoding this knowledge into a "foundation" machine learning model with much faster inference and data assimilation capabilities can pave the way for a much wider impact.

## Data sources {#sec:data}

Unlike data in computer vision or natural language processing, weather and climate data is not solely based on sensed data, instead incorporates information from a diverse range of sources. For example, *reanalysis* weather data blends meteorological observations with past short-range weather forecasts via data assimilation [bauer2015quiet]. The data measurements themselves are highly heterogeneous, representing various physical variables with different data types (e.g. pressure, temperature, humidity) that are recorded at different, relatively sparse, spatial locations at different temporal frequencies. These measurements can be integrated together with known physics inform the design of climate simulations, which again produce data with different variables at different scales. From a machine learning perspective, the plethora of available data thus spans multiple axes: from direct weather measurements at land, sea, or atmosphere, over multiple decades of re-analyzed weather data at different spatial scales, to physics-informed climate projections for various scenarios. Most notably, the data shares the same set of primitive equations, but with fairly different characteristics. Below we describe two of the most commonly used data sources for weather and climate modeling.

### CMIP6 {#subsec:data-cmip}

The Coupled Model Intercomparison Project (CMIP) [meehl2000coupled] is an international effort across different individual climate modeling groups to come together to compare and evaluate their global climate models. While the main goal of CMIP is to improve the understanding of Earth's climate system and improve the accuracy of its simulations, the recent data from their experimental runs is easily accessible on the CMIP6 [eyring2016overview] archive. In CMIP6, where "6" refers to the most recent phase of the project, $49$ groups are involved with their experiments covering wide range of climate variables including temperature, precipitation, sea level and others from hundreds of models. This results in global projections of various climate scenarios from as early as 1850 onwards, all following similar governing equations, but with different *forcings*, e.g., greenhouse gas emissions that affect the climate.

### ERA5 {#subsec:data-era5}

The ERA5 reanalysis archive [hersbach2018era5; hersbach2020era5] of the European Center for Medium-Range Weather Forecasting (ECMWF) is the predominant data source for learning and benchmarking weather forecasting systems. Once completed, the ERA5 reanalysis is set to embody a detailed record of the global atmosphere, land surface and ocean waves from 1950 onwards. The currently available ERA5 reanalysis data combines the state of the art forecasting model called Integrated Forecasting System (IFS) [wedi2015modelling] of ECMWF with available observations to provide the best guess of the state of the atmosphere, ocean-wave and land-surface quantities at any point in time. In its raw form, the available reanalyzed data is huge: 40 years, from 1979 to 2018, on a $0.25° \times 0.25°$ global latitude-longitude grid of the Earth's sphere, at hourly intervals with different climate variables at 37 different altitude levels plus the Earth's surface. The grid overall contains $721 \times 1440$ grid points for latitude and longitude, respectively. The altitude levels are presented as pressure levels.

## Tasks {#sec:tasks}

Given the scale of data availability, increasing compute requirements of current numerical methods despite it being difficult to incorporate real observational data into them, machine learning is increasingly finding applications in many of the tasks related to weather and climate modeling. When it comes to **weather**, the main task of interest is *forecasting* the future values of key weather variables. These tasks can take the following forms depending on temporal and spatial horizons of interest:

- **Global forecasting** tasks that range from a few hours (i.e., nowcasting) to days and weeks in lead time (i.e., short and medium range forecasting). Often these tasks are evaluated on the ERA5 reanalysis dataset (see 2.1.2) with Operational IFS [wedi2015modelling] of the European Center for Medium-Range Weather Forecasting (ECMWF) being the current state-of-the-art NWP baselines.

- **Regional forecasting** tasks which could range from weather forecasting in continental North America or Europe to individual state, county or city.

- **Sub-seasonal to seasonal prediction (S2S)** [vitart2018sub; Vitart2022] which is the task of forecasting the weather with lead times between 2 weeks and 2 months. S2S bridges the gap between weather forecasting and seasonal climate prediction, and is critical to disaster mitigation. Often at such long horizons, predicting instantaneous values of key weather variables can be a difficult task and therefore the focus is often on averaged value of key weather variables over a certain time horizon, e.g. weekly average precipitation.

Whereas deep learning approaches for regional or S2S tasks are scarce, most of the recent and concurrent work focuses on global forecasting tasks. rasp2021data were the first to use pretraining on climate simulations to achieve good data-driven medium-range weather prediction with a ResNet [he2016deep], weyn2020improving used CNNs on a cubed sphere for global weather prediction, weyn2021sub forecast weather sub-seasonally with a large ensemble of deep-learning weather prediction models, keisler2022forecasting applied a graph neural network based approach to weather forecasting, ravuri2021skilful use deep generative models of radar for precipitation nowcasting, arcomano2020machine build a reservoir computing-based, low-resolution, global prediction model, and MetNet [sonderby2020metnet] takes as input radar and satellite data to forecast probabilistic precipitation maps. These approaches are complemented by general machine learning models for fluid dynamics [li2020fourier; kochkov2021machine; lu2021learning; brandstetter2022clifford; brandstetter2022message]. Finally, recent state-of-the-art neural weather models such as FourCastNet [pathak2022fourcastnet], Pangu-weather [bi2022pangu], or GraphCast [lam2022graphcast], which also perform global forecasting tasks, use the highest resolution $0.25°$ ERA5 data, and are optimized on the respective hardware resources.

On the other hand, **climate** tasks have to deal with much longer time horizons. Possible categories of tasks where machine learning can help include climate projection and climate model downscaling:

- **Climate projection** is the task of generating estimates of climate change under different future socio-economic scenarios. Usually, this takes the form of figuring out the response of the climate system to different forcing factors such as greenhouse gases and aerosol emissions. Climate projection is a crucial task in understanding and preparing for the potential impacts of climate change.

  While the application of machine learning in this field is still in its early stages, recent efforts have been made to standardize evaluation in this domain. One example of this is ClimateBench [watson2022climatebench], which is a benchmark dataset drawing on CMIP6 to provide an evaluation framework for machine learning models that aim to improve the accuracy of climate projections. This benchmark aims to provide a consistent and reliable evaluation method for various machine learning models that are applied to climate projections.

- A more popular application of ideas in machine learning is towards **downscaling** of climate model. Global climate models typically have a coarse spatial resolution, which means that they can only provide a rough estimate of climate conditions at a local or regional scale. Moreover, the simulations often reflect systematic biases that deviate from trends in the observation data. The aim of climate model downscaling is to create locally accurate climate information from global climate projections by relating those to observed local climatological conditions. This process improves the spatial and temporal resolution of the data, making it more suitable for use in local and regional analyses. Downscaling methods can be divided into *dynamic* approaches that relate outputs of global climate models with those of regional climate models, and *statistical* approaches that infer the desired transformations using data-driven approaches [wilby1997downscaling]. Dynamic approaches are physically consistent, but can be slow and have large biases, whereas statistical approaches need large amounts of data to learn expressive mappings that hold for target output scenarios.

Similar to weather forecasting, deep learning has emerged as appealing alternative in climate science as well. Recent approaches comprise surrogate models to emulate climate projections  [weber2020deep; scher2019weather; scher2018toward; beusch2020emulating; mansfield2020predicting], extract contextual cues from existing datasets or simulations [reichstein2019deep; huntingford2019machine; schneider2017earth], and perform climate model downscaling [sachindra2018statistical; vandal2017deepsd; bano2020configuration]. Climate model downscaling usually inputs low-resolution reanalysis data and local orographic information to obtain high-resolution local information. Many recent approaches are based on convolutional architectures [hohlein2020comparative; vaughan2021convolutional; markou2022practical].

## Foundation models {#subsec:foundmodels}

Bommasani2021FoundationModels gave the term "foundation models" to the emerging paradigm of training scalable deep learning models on broad data via self-supervision which could then be adapted (often via finetuning) to a wide range of downstream tasks. Current notable examples include BERT [devlin2018bert], GPT [brown2020language] and PaLM [chowdhery2022palm], in language, CLIP [radford2021learning], Florence [yuan2021florence], BEiT [wang2022image] for vision-language. Outside applications on data crawled from web, this paradigm has also started finding success in various scientific domains like protein design [verkuil2022language]. Key significance of such models has been identified as *emergence* with respect to model capabilities and *homogenization* with respect to methodologies for different tasks, domains, and modalities, enabled by the principles of transfer learning [thrun2012learning] at scale. While a foundation model itself should be considered incomplete, it can provide a common basis from which various task-specific models can be derived. Current research at the intersection of weather and climate science and ML has largely focused on designing separate models for every task of interest despite potential availability of fairly diverse large scale data with shared underlying physics and geology across these tasks. A few recent works have proposed pretraining techniques for satellite imagery and remote sensing [yuan2020self; cong2022satmae; reed2022scale] but they have so far not been applied to multi-sensory data and variables in weather and climate.

# Approach

Given the availability of large scale data sources, together with shared physics and geology between various weather and climate tasks, we aim to build a generalizable deep learning foundation model. The model needs to be able to input heterogeneous datasets of different variables, and provide spatio-temporal coverage based on physical groundings. We, therefore, first take a closer look at input representations, and next design a model to cope with their heterogeneity - local, global, and across variables.

## Input representation {#sec:input-repr}

We are interested in gridded prediction tasks, in which the model takes an input of shape $V \times H \times W$ and predicts an output of shape $V' \times H' \times W'$. $V$ refers to the number of input variables, which can be weather conditions such as geopotential and temperature, or climate forcing factors such as CO$_2$ and SO$_2$. $H$ and $W$ refer to the spatial resolution of the input data, which depends on how densely we grid the globe. This general representation captures a broad variety of downstream tasks in Earth systems science. Similarly, $V', H', W'$ refer to the variables and spatial resolution of the predicted outputs. We mainly work with two spatial resolutions: $5.625°$ ($32 \times 64$ grid points) and $1.40625°$ ($128 \times 256$ grid points). Semantically, a $H \times W$ map can represent the entire globe or a specific region such as North America.

## Model architecture {#sec:model_arc}

We aim to design a foundation model that we can pretrain on heterogeneous data sources and then finetune to solve various downstream weather and climate tasks. From 3.1, one could think of the tasks as image-to-image translation problems with $V$ input channels and $V'$ output channels. This makes any image architecture a natural fit, such as UNet [ronneberger2015u], ResNet [he2016deep], or Vision Transformers (ViT) [dosovitskiy2020image]. However, the settings of climate and weather tasks are much broader, where we may want to make predictions for regional or even spatially incomplete data, forecast unseen climate variables, or finetune the model on data at different resolutions from pretraining. Current CNN-based architectures are not applicable in these scenarios, as they require the input to be perfectly gridded, contain a fixed set of variables, and have a fixed spatial resolution. Transformers-based architectures, on the other hand, provide much better flexibility by treating the image-like data as a set of tokens. Therefore, we build ClimaX architecture upon Vision Transformers (ViT) [dosovitskiy2020image; vaswani2017attention], and propose two major architectural changes, namely *variable tokenization* and *variable aggregation* to further improve the flexibility and generality, which we will describe next.

[IMAGE: Pretraining phase of ClimaX. Variables are encoded using variable-separate tokenization, and subsequently aggregated using variable aggregation. Together with position embedding and lead time embeddin...]

### Variable tokenization

Given an input of shape $V \times H \times W$, ViT tokenizes the input into a sequence of $(H/p) \times (W/p) = h \times w$ patches, with each patch having a size of $V \times p^2$, where $p$ is the patch size. This tokenization scheme works well for image data, as $V$ is always the RGB channels, which is the same for all datasets. However, this is not true for climate and weather data, where the number of physical variables can vary between different datasets. For example, in the CMIP6 project [eyring2016overview], each dataset contains simulated data of a different climate model, and thus has a different set of underlying variables. Therefore, we propose *variable tokenization*, a novel tokenization scheme that tokenizes each variable in the input separately. Specifically, each input variable as a spatial map of shape $H \times W$ is tokenized into a sequence of $h \times w$ patches, which results in $V \times h \times w$ patches in total. Finally, each input patch of size $p^2$ is linearly embedded to a vector of dimension $D$, where $D$ is the chosen embedding size. The output of the variable tokenization module therefore has a dimension of $V \times h \times w \times D$. Figure 3 illustrates our proposed tokenization scheme.

[IMAGE: Variable tokenization. Each variable is independently tokenized.]

### Variable aggregation

While variable tokenization allows ClimaX to learn from datasets with varying numbers of input variables, it has two inherent problems. First, it results in a sequence of length $V \times h \times w$ which increases linearly with the number of variables. Since we use attention to model the sequence, the memory complexity scales quadratically with the number of variables. This is computationally expensive, as we can have up to $48$ input variables in our experiments. Moreover, because we tokenize each variable separately, the input sequence will contain tokens of different variables with very different physical groundings, which can create difficulties for the attention layers to learn from. We therefore propose *variable aggregation* to solve the two mentioned challenges. For each spatial position in the $h \times w$ map, we perform a cross-attention operation, in which the query is a learnable vector, and the keys and values are the $V$ embedding vectors of $V$ variables at that position. The cross-attention module outputs a single vector for each spatial position, thus reducing the sequence length to $h \times w$, significantly lowering the computational cost. Moreover, the sequence now contains unified tokens with universal semantics, creating an easier task for the attention layers. Figure 4 shows our proposed variable aggregation.

[IMAGE: Position-based variable aggregation reduces a sequence of length $V \times h \times w$ to $h \times w$.]

### Transformer

Post variable aggregation, we need a sequence model for generating the output tokens. While in principle, one could use any general sequence model, we propose to extend a standard Vision Transformer (ViT). Moreover, since the standard ViT treats image modeling as pure sequence-to-sequence problems, it can perform tasks that some other variations cannot [liu2021swin; liu2021swinv2], such as learning from spatially incomplete data, where the input does not necessarily form a complete grid. This is useful in the regional forecasting task we consider in 4.2.2. In the experiments, we report results with $8$ attention layers, an embedding size of $1024$, and a hidden dimension of $1024 \times 4$. After the attention layers, we employ a prediction head that takes a token and outputs a vector of size $V' \times p^2$. The prediction head is a 2-layer MLP with a hidden dimension of $1024$. We provide more details in 6.

## Datasets

### Pretraining

We believe that CMIP6's diversity and scale presents an attractive opportunity for pretraining large-scale foundation models. However, handling the inconsistent set of variables across different data sources can be a challenge. In this work we only use a subset of variables from five different data sources (MPI-ESM, TaiESM, AWI-ESM, HAMMOZ, CMCC) containing global projections of climate scenarios from 1850 to 2015 with the time delta of $6$ hours as described in 1. Due to variable original resolution, we choose to simplify our data-loading by regridding them to commonly used resolutions [rasp2020weatherbench; rasp2021data] of $5.625°$ ($32 \times 64$ grid points) and $1.40625°$ ($128 \times 256$ grid points)[^2].

### Finetuning and evaluation

We use the ERA5 reanalysis data as described in 8.2, as the source of datasets for finetuning and evaluation for various weather related downstream tasks. Due to its large size, it is common to regrid [rasp2020weatherbench; rasp2021data] the high-resolution data to lower resolutions like $5.625°$ ($32 \times 64$ grid points) and $1.40625°$ ($128 \times 256$ grid points) to fit within the available computational constraints[^3]. We follow the evaluation procedure by rasp2021data and use this data to assess the forecasting performance of our ML models at different lead time horizons. More details about the individual datasets are in their appropriate experiment sections.

## Training

### Pretraining

We pretrain ClimaX on CMIP6 data to predict future weather conditions given the current conditions. That is, given the weather snapshot $X_t$ of shape $V \times H \times W$ at a particular time $t$, ClimaX learns to predict the future weather scenario $X_{t + \Delta t}$ of the same shape at lead time $\Delta t$. To obtain a pretrained model that is generally applicable to various temporal forecasting tasks, we randomize the lead time from $6$ hours to $168$ hours (i.e., 1 week) during pretraining. We add the lead time embedding to the tokens to inform the model of how long it is forecasting into the future. The lead time embedding module is a single-layer MLP that maps a scalar to a vector of the embedding size $D$. 2 depicts the forward pass of ClimaX in pretraining. For an input $X_t$, we sample a lead time $\Delta t \sim \mathcal{U}[6, 168]$ and get the corresponding ground truth $X_{t + \Delta t}$. Input variables are tokenized separately using variable tokenization, and are subsequently aggregated at each spatial location, resulting in a sequence of $h \times w$ unified tokens. We add the tokens with the lead time embedding and positional embedding before feeding the sequence to the ViT backbone. The output of the last attention layer is fed to a prediction head, which transforms the sequence back to the original shape of $V \times H \times W$.

We employ the latitude-weighted mean squared error [rasp2020weatherbench] as our objective function. Given the prediction $\Tilde{X}_{t + \Delta t}$ and the ground truth $X_{t + \Delta t}$, the loss is computed as: $$\mathcal{L} = \frac{1}{V \times H \times W} \sum_{v=1}^V \sum_{i=1}^H \sum_{j=1}^W L(i)(\Tilde{X}_{t + \Delta t}^{v,i,j} - X_{t + \Delta t}^{v,i,j})^2, \label{eq:lat_mse}$$ in which $L(i)$ is the latitude weighting factor: $$L(i) = \frac{\cos(\text{lat}(i))}{\frac{1}{H} \sum_{i'=1}^H \cos(\text{lat}(i'))},$$ where $\text{lat}(i)$ is the latitude of the corresponding $i\text{th}$ row of the grid. The latitude weighting term accounts for the non-uniformity in areas when we grid the round globe. Grid cells toward the equator have larger areas than the cells near the pole, and thus should be assigned more weights.

### Finetuning

ClimaX has four learnable components, including the token embedding layers, the variable aggregation module, the attention blocks, and the prediction head. We evaluate the performance of ClimaX on various downstream tasks, which we categorize into two finetuning scenarios: one in which the downstream variables belong to the set of pretraining variables, and the other with variables unseen during pretraining. In the first case, we finetune the entire model, and in the latter, we replace the embedding layers and the prediction head with newly initialized networks, and either finetune or freeze the other two components. We present more details of each downstream task in 4.

# Experiments {#sec:exps}

We finetune ClimaX on a diverse set of downstream tasks to evaluate its performance and generality. We categorize the tasks into forecasting, climate projection, and climate downscaling. The experiments aim to answer the following questions:

- How does ClimaX perform on global forecasting compared to the current state-of-the-art NWP system?

- Can we finetune ClimaX to make forecasts for a specific region or at different temporal horizons from pretraining?

- How well does ClimaX perform on climate tasks that are completely different from pretraining?

In addition to the main experiments, we analyze the scaling property of ClimaX, i.e., how the performance of ClimaX improves with increasing data size, model capacity, and data resolution. Finally, we perform comprehensive ablation studies to understand the trade-off between computation and performance when finetuning ClimaX.

## Neural baselines

In global forecasting, we compare ClimaX with IFS [wedi2015modelling], the current gold standard in weather forecasting. In tasks we do not have a baseline, we compare with UNet [ronneberger2015u; gupta2022towards] and ResNet [he2016deep], two CNN baselines commonly used in vision tasks. We borrow the ResNet architecture from Weatherbench [rasp2020weatherbench]. The exact architectural details of these baselines are in 6.2.

## Forecasting

### Global forecasting {#sec:global-forecast}

Given global weather conditions $X_t$ at a particular time $t$, we want to forecast the weather at a future time $X_{t + \Delta t}$, in which $\Delta t$ is the lead time. The input variables include $6$ atmospheric variables at $7$ vertical levels, $3$ surface variables, and $3$ constant fields, resulting in $48$ input variables in total. The details of the variables are in 2. We evaluate ClimaX on predicting four target variables: geopotential at $500$hPa (Z500), the temperature at $850$hPa (T850), the temperature at $2$ meters from the ground (T2m), and zonal wind speed at $10$ meters from the ground (U10). Z500 and T850 are the two standard verification variables for most medium-range NWP models and are often used for benchmarking in previous deep learning works, while the two surface variables, T2m and U10, are relevant to human activities. We consider seven lead times: $6$ hours, $\{1, 3, 5, 7\}$ days, 2 weeks, and 1 month, which range from nowcasting to short and medium-range forecasting and beyond. We consider predicting each target variable at each lead time a separate task, and finetune a separate model for each task. We discuss alternative finetuning protocols in Section 4.6.

We compare ClimaX with IFS and the two CNN baselines on the ERA5 dataset at both $5.625°$ and $1.40625°$ resolutions. Following [rasp2020weatherbench], we split the data into three sets, in which the training data is from $1979$ to $2015$, the validation data is in $2016$, and the test data is in $2017$ and $2018$. We finetune ClimaX and train the other deep learning baselines using the latitude-weighted MSE loss in [\[eq:lat_mse\]](#eq:lat_mse){reference-type="ref+Label" reference="eq:lat_mse"}. We perform early stopping on the validation loss for all deep learning models, and evaluate the best checkpoint on the test set. For IFS, we download the predictions from the TIGGE archive [bougeault2010thorpex] for the year $2018$[^4]. We compare all methods on latitude-weighted root mean squared error (RMSE) and latitude-weighted anomaly correlation coefficient (ACC), two commonly used metrics in previous works. The formulations of the two metrics are in 9.1. Lower RMSE and higher ACC indicates better performance.

[\[fig:global-forecasting-lowres,fig:global-forecasting\]](#fig:global-forecasting-lowres,fig:global-forecasting){reference-type="ref+Label" reference="fig:global-forecasting-lowres,fig:global-forecasting"} show the performance of ClimaX and the baselines at $5.625°$ and $1.40625°$, respectively. At low resolution, IFS outperforms ClimaX on 6-hour to 5-day prediction tasks. On longer horizons, however, ClimaX performs comparably to or slightly better than IFS, especially on 14-day prediction. At higher resolution, the performance of ClimaX closely matches that of IFS even for short horizons, and is superior in forecasting at $7$ days and beyond. The trends are similar for both RMSE and ACC. The two CNN baselines perform similarly and achieve reasonable performance, but lag behind ClimaX and IFS on all tasks. We include other additional task-specific baselines [pathak2022fourcastnet; bi2022pangu; lam2022graphcast] in 9.2. These baselines are trained on higher-resolution ERA5 ($0.25°$) so are not directly comparable.

[IMAGE: Performance on global forecasting on ERA5 at $5.625°$.]

[IMAGE: Performance on global forecasting on ERA5 at $1.40625°$.]

### Regional forecasting {#sec:regional-forecast}

It is not always possible to make global predictions, especially when we only have access to regional data In this section, we evaluate ClimaX on *regional forecasting* of the relevant variables in North America, where the task is to forecast the future weather in North America given the current weather condition in the same region. We create a new dataset from the ERA5 data at $1.40625°$ that has the same set of variables but just focuses on the North America region. We call this dataset ERA5-NA and present details of how to construct it in 8.2. Training, validation, and test splits are done similarly to 4.2.1. 7 illustrates the finetuning process of ClimaX on this task, where the only difference from global forecasting is the input now only contains tokens that belong to North America.

Since the task has not been considered in previous works, we compare ClimaX with the two CNN baselines ResNet and UNet, and the scratch-trained version of ClimaX, which we refer to as Cli-ViT. In addition, we finetune two ClimaX models, in which one was pretrained on CMIP6 at $1.40625°$, and the other was pretrained on $5.625°$ data. To finetune the low-resolution model on higher-resolution data, we follow the common practice of interpolating the positional embedding [dosovitskiy2020image; touvron2021training]. We denote this model as ClimaX-pos-interp. We evaluate all methods on predicting Z500, T2m, and T850 at lead times of $3$, $5$, and $7$ days. Latitude-weighted RMSE is used as the evaluation metric.

8 compares the performance of ClimaX and the baselines. ClimaX is the best performing method among different target variables and lead times. Interestingly, even though pretrained on data at a lower resolution, ClimaX-pos-interp achieves the second best performance in predicting Z500 and T850, and only underperforms ResNet in predicting T2m at 3-day lead time. This result shows that ClimaX can gain strong performance on tasks that have different spatial coverage or even different spatial resolution from pretraining.

[IMAGE: Finetuning setup for Regional Forecasting in North America.]

[IMAGE: Performance on Regional (North America) forecasting for key variables.]

### Sub-seasonal to seasonal cumulative prediction

Sub-seasonal to seasonal (S2S) prediction is the task of forecasting at a time range between $2$ weeks and $2$ months [vitart2018sub], which bridges the gap between weather forecasting and climate projection. Compared to the other two well-established tasks, S2S prediction has received much less attention, despite having a significant socioeconomic value in disaster mitigation. Recent works have proposed data-driven approaches based on both traditional machine learning [hwang2019improving; prokhorenkova2018catboost; taylor2018forecasting] and deep learning [weyn2021sub; zhou2021informer; oreshkin2019n], but their performances often lag behind adaptive bias correction methods [mouatadid2023adaptive] on standard benchmarks [mouatadid2023subseasonalclimateusa]. Here, following the S2S competition (<https://s2s-ai-challenge.github.io/>), we aim to predict the biweekly average statistics of weeks 3-4 and weeks 5-6, which correspond to lead times of $2$ weeks and $4$ weeks, respectively. We construct ERA5-S2S, a new dataset from $5.625°$ ERA5 that has the same input variables, but the output variables are averaged from the lead time to $2$ weeks ahead into the future.

We compare ClimaX with ResNet, UNet, and Cli-ViT on the S2S prediction of four target variables: T850, T2m, U10, and V10. [\[tab:s2s_prediction\]](#tab:s2s_prediction){reference-type="ref+Label" reference="tab:s2s_prediction"} compares the RMSE of ClimaX and the baselines. ClimaX achieves the lowest error for all variables, and the performance gap with the best baseline UNet is larger at increasing lead times. ClimaX also has significant performance gains over its scratch-trained counterpart Cli-ViT, showing the effectiveness of our pretraining procedure in capturing features that are generally useful for various temporal prediction tasks.

## Climate projection

To further test the generality of ClimaX, we evaluate the model on ClimateBench [watson2022climatebench], a recent benchmark designed for testing machine learning models for climate projections. The goal of ClimateBench is to predict the annual mean global distributions of surface temperature, diurnal temperature range, precipitation, and the $90$th percentile of precipitation, given the four anthropogenic forcing factors: carbon dioxide (CO$_2$), sulfur dioxide (SO$_2$), black carbon (BC), and methane (CH$_4$). We note that this is not a temporal modeling task, as we do not predict the future given the past. Instead, we answer questions like *what will be the annual mean temperature for a specified CO$_2$ level?* In particular, note that the input variables and the task itself are completely different from pretraining.

9 illustrates the finetuning pipeline of ClimaX for ClimateBench. As the input and output variables are unseen during pretraining, we replace the pretrained embedding layers and prediction heads with newly initialized networks, while keeping the attention layers and the variable aggregation module. We consider two finetuning protocols, in which we either freeze[^5] (ClimaX$_{\text{frozen}}$) or finetune (ClimaX) the attention layers. In addition, we introduce two components to the pipeline in 2. We use a history of the preceding ten years of the forcing factors to make predictions for a particular year, creating an input of shape $T \times V \times H \times W$. Each time slice of the input goes through variable tokenization, variable aggregation, and the attention layers as usual, which output a feature tensor of shape $T \times h \times w \times D$, where $D$ is the embedding size. The feature tensor then goes through a global average pooling layer, reducing the dimension to $T \times D$. Finally, the $10$-year history is aggregated using a cross-attention layer before being fed to the prediction head, which linearly transforms the $D$-dimensional feature vector to a $H \times W$ map. The history aggregation and the global pooling modules are the two additions to the original ClimaX architecture. These architectural designs are inspired by the neural network baseline in [watson2022climatebench].

We compare ClimaX with ClimaX$_\text{frozen}$, Cli-ViT, and the best baseline from ClimateBench. Following [watson2022climatebench], we use the standard mean squared error ([\[eq:lat_mse\]](#eq:lat_mse){reference-type="ref+Label" reference="eq:lat_mse"} without the weighting term) as the loss function. We evaluate all methods on RMSE, NRMSE$_s$ (Spatial), NRMSE$_g$ (Global), and Total = NRMSE$_s$ + 5 $\times$ NRMSE$_g$ [watson2022climatebench]. Details of the metrics are in 9.1. [\[tab:climate_bench\]](#tab:climate_bench){reference-type="ref+Label" reference="tab:climate_bench"} shows the results. ClimaX$_\text{frozen}$ performs the best in predicting two temperature-related variables, followed by ClimaX. This shows that the pretrained attention layers can serve as a strong feature extractor in seemingly unrelated tasks. Where downstream data is scarce (ClimateBench has only $754$ data points), further finetuning the attention layer can lead to overfitting and thus slightly hurt the performance. In two precipitation-related tasks, ClimaX$_\text{frozen}$ slightly underperforms ClimateBench baseline in terms of NRMSE$_s$ and NRMSE$_g$ but outperforms on RMSE. We hypothesize that this was because ClimaX did not observe the precipitation variable during pretraining, which has very different behaviors from other variables.

[IMAGE: Finetuning pipeline for ClimateBench. A different set of input and output variables requires different embedding layers and prediction heads. Attention layers can be frozen or finetuned.]

## Climate model downscaling

Climate models are often run at coarse grids due to their high computational cost. Although these predictions are useful in understanding large-scale climate trends, they do not provide sufficient detail to analyze regional and local phenomena. Downscaling aims to obtain higher-resolution projections and reduce biases from the outputs of these models. To evaluate the applicability of ClimaX to the task of climate model downscaling, we construct a new dataset based on CMIP6 and ERA5 data sources for coarse inputs and higher resolution targets. Specifically, we use all MPI-ESM, a dataset from CMIP6, and its variables listed in 1 at $5.625°$ as input, and train separate models to downscale to each ERA5 target variable at $1.40625°$. We compare ClimaX with Cli-ViT and the two CNN baselines, UNet and ResNet, as most recent deep downscaling methods [vandal2017deepsd; rodrigues2018deepdownscale; hohlein2020comparative; vandal2019intercomparison; liu2020climate] are based on convolution. We were not able to compare with YNet [liu2020climate], the current best method on deep downscaling as we did not have access to high-resolution auxiliary data such as elevation and topographical information. For all methods, we first bilinearly interpolate the input to match the resolution of the desired output before feeding it to the model. We evaluate all methods on RMSE, Pearson correlation, and Mean bias, which were commonly used in existing deep downscaling works [vandal2017deepsd; liu2020climate]. Details of the metrics are in 9.1.

[\[tab:downscaling\]](#tab:downscaling){reference-type="ref+Label" reference="tab:downscaling"} compares ClimaX and the baselines quantitatively. ClimaX achieves the lowest RMSE and a mean bias closest to $0$ for all three target variables, and performs similarly to the baselines in terms of Pearson correlation. While pretrained to perform forecasting, ClimaX has successfully captured the spatial structure of weather data, which helps in downstream tasks like downscaling. 10 visualizes the downscaled predictions of ClimaX for the three target variables. The input is at a much lower resolution and contains a lot of bias compared to the ground truth. While the prediction is missing some fine details, it has successfully captured the general structure of the ERA5 data and removed input biases.

[IMAGE: Example visualizations of downscaled prediction of key variables by ClimaX.]

## Scaling laws analysis {#sec:scaling}

Transformers have shown favorable scaling properties for language [kaplan2020scaling; hoffmann2022training], vision [zhai2022scaling], or even multi-modal tasks [henighan2020scaling; hendricks2021decoupling; reed2022gato]. That is, their performance improves with respect to data size and model capacity given sufficient compute. In this section, we study the scaling laws of ClimaX in weather forecasting. 11 presents the performance of ClimaX as a function of data size and model capacity. The $x$-axis is the pretraining data size measured in Gigabytes, which corresponds to $1$ to $5$ CMIP6 datasets, and the $y$-axis shows the RMSE of ClimaX on the 3-day forecasting task. We compare four ClimaX models with different capacities by varying the embedding dimension from $128$ to $1024$. All experiments are conducted on the $5.625°$ data. The error rate of the two biggest models decreases consistently as we increase the data and model size. This highlights the unique ability of ClimaX in learning from diverse and heterogeneous data sources, which allows us to further improve the performance by simply pretraining on more data. However, the two smaller models do not scale as well as the bigger ones, where increasing data size does not gain much improvement or can sometimes hurt performance. This result shows that larger models not only perform better but are also more data efficient.

In addition to data size and model capacity, data resolution is another important scaling dimension in the context of weather and climate. In many vision tasks such as classification, understanding the general, high-level structure of the image is sufficient to make accurate predictions. To model the underlying complex physical processes that govern weather and climate, however, it is important for a model to look at fine-grained details of the input in order to understand the spatial and temporal structure of data as well as the interactions between different variables. High-resolution data contains finer details and local processes of weather conditions that are not present in the low-resolution data, and thus provides stronger signals for training deep learning models. 12 compares the performance of ClimaX pretrained and finetuned on $5.625°$ and $1.40625°$ data on global forecasting. Except for T2m at $1$ day and $3$ days lead times, ClimaX ($1.40625°$) consistently achieves lower RMSE and higher ACC than the low-resolution model. We note that for the high-resolution data we have to use a larger patch size ($4$ compared to $2$ for low-resolution data) due to lack of memory issue. We can further improve the performance of ClimaX on the $1.40625°$ data by reducing the patch size, as the model is able to capture better details.

[IMAGE: Error on ERA5 3-day forecasting for different variables with respect to CMIP6 5.625$°$ data seen during pre-training. Bigger models are more sample efficient.]

[IMAGE]
[IMAGE]
<figcaption>Scaling performance with respect to data resolution. Despite a larger patch size, ClimaX (<span class="math inline">$1.40625°$</span>) achieves consistently better performance than the low-resolution model on almost all tasks, except for T2m forecast at 1 day and 3 days lead times. </figcaption>
</figure>

## Ablation studies {#sec:ablation}

In the main forecasting results, we finetune a separate ClimaX model for each target variable at each lead time, as we found this protocol led to the best performance. However, this can be computationally expensive, as finetuning cost scales linearly with respect to the number of target variables and lead times. In this section, we consider different finetuning alternatives to investigate the trade-off between computation and performance.

[IMAGE: Performance of ClimaX and its variations on weather forecasting. ClimaX-cont is a lead-time-conditioned model that we finetune to make predictions at 6 hours to 7 days. ClimaX-iter forecasts at a $6$-...]

### Should we finetune ClimaX for each variable separately or all at once?

Instead of finetuning ClimaX for each target variable separately, we could alternatively finetune once to predict all variables in the input simultaneously, which we denote as ClimaX-all-vars. 13 shows that ClimaX-all-vars achieves comparable performance to ClimaX in most of the tasks and only underperforms for forecasting T2m. This suggests that with a limited budget, one can finetune ClimaX to predict all target variables at the same time without losing much performance.

### Should we do iterative forecast or direct forecast?

To avoid finetuning a different model for each lead time, we can finetune ClimaX to make predictions at a short horizon such as $6$ hours, and roll out the predictions during inference to make forecasts at longer horizons. We call this model ClimaX-iter, where *iter* stands for iterative prediction [rasp2020weatherbench]. We note that in order to roll out more than one step, ClimaX-iter must predict for all input variables, or in other words. This provides the benefit of finetuning a single model that can predict for any target variable at any lead time. 13 shows that ClimaX-iter works reasonably well up to 1-day prediction, but the performance degrades significantly at longer lead times. This is not surprising, because ClimaX-iter is not finetuned to predict multiple steps into the future, leading to quick error accumulation. One can employ a multi-step objective for finetuning as in pathak2022fourcastnet to achieve better results.

### Can we finetune ClimaX to work for all lead times?

Another way to avoid finetuning for each lead time separately is to finetune a lead-time-conditioned model. Specifically, during finetuning, we randomize the lead time from $6$ hours to $7$ days, resembling the pretraining setting. Note that unlike ClimaX-iter, we still have to finetune a separate model for each target variable. We call this model ClimaX-cont, wherein *cont* stands for *continuous*, a standard term used in previous works [rasp2020weatherbench]. 13 shows that ClimaX-cont performs competitively on $6$-hour to $7$-day forecasting, but fails to extrapolate to $2$ weeks and $1$ month lead times that are unseen during training. One can also randomize the lead time from $6$ hours to $1$ month, but that means the model sees much fewer data points for each target lead time, potentially hurting the performance.

The cost for finetuning each set of weights is a constant $C$, which is about $15$ hours on an $8 \times \text{V}100s$. Among different finetuning protocols, ClimaX is the most expensive, whose total cost is $C \times \#variables \times \#lead\_times$, scaling linearly with the number of target variables and lead times. Following ClimaX are ClimaX-all-vars and ClimaX-cont, whose total costs are $C \times \#lead\_times$ and $C \times \#variables$, respectively. Finally, ClimaX-iter is the cheapest finetuning protocol, where we only have to finetune a single model that works for all target variables and at all lead times. The performance is proportional to the computational cost, as ClimaX is the best performing model, while ClimaX-iter is the worst.

# Discussion and Future Work {#sec:discuss}

The scaling of datasets, model architectures, and computation has resulted in a transformative impact in various subdisciplines of artificial intelligence, from natural language and speech processing to computer vision, as well as scientific applications in biology and chemistry. In particular, it has led to the emergence of general-purpose foundation models that are trained on large datasets and compute clusters, and can be easily adapted to a variety of downstream tasks efficiently, both in terms of compute and data supervision. Our work represents a pioneering effort to enable such broad scaling and generality in data-driven models for weather and climate. This approach goes beyond the limitations of both traditional numerical modeling and existing data-driven forecasting methods. Unlike ClimaX, numerical models scale only in terms of computation and not in terms of dataset size, whereas existing data-driven models are typically limited to specific tasks and lack general-purpose applicability across a wide range of tasks.

In addition to traditional considerations in language and vision, foundation models like ClimaX open up new opportunities for scaling through the use of simulation datasets and grid resolutions. To simplify our approach, we chose to use pretraining datasets that include standard variables that have been benchmarked in previous research on data-driven forecasting [rasp2020weatherbench; pathak2022fourcastnet]. Additionally, we avoided datasets that simulate future scenarios under different forcings to prevent any potential leakage for the climate projection task. Future research could explore incorporating both observational and simulated datasets that include a wider range of climate variables, higher spatiotemporal resolutions, and even extend into future scenarios. Further, we showed that resolution plays a crucial role in scaling of ClimaX. Due to our compute restrictions, we trained ClimaX on low to moderate resolutions. Nevertheless, our empirical trends suggest that scaling to higher resolutions ($0.25°$) is likely to lead to even better results.

Scaling efforts in the future can benefit from better sequence modeling architectures, especially those designed for multimodal spatiotemporal inputs. As we saw in ClimaX, the number of channels for climate datasets is much greater than those handled for standard multimodal settings (e.g., audio-video, vision-language models). Moreover, in practice, there is also a significant range of resolutions across different climate datasets. This heterogeneity drastically increases the raw length of input sequences for standard architectures such as ViT. In the future, we believe that investigating single multi-scale architectures (e.g., [fan2021multiscale]) can potentially aid in scaling to such diverse multi-resolution and multi-modal datasets by learning to infer features relevant to atmospheric phenomena at increasing spatial resolutions.

In conclusion, we believe that the generality of our approach has potential applications beyond the tasks considered in this work. It would be interesting to explore the generalization of a pretrained ClimaX backbone to other Earth systems science tasks, such as predicting extreme weather events [miralles2019land; sillmann2017understanding] and assessing anthropogenic contributions to climate change [rosenzweig2008attributing; hook2013depletion], as well as broader domains that are closely tied to weather and climate conditions, such as agriculture, demography, and actuarial sciences.

# Model {#sec:app:arch}

This section presents the implementation details and hyperparameters of ClimaX and the two CNN baselines UNet and ResNet.

## ClimaX

### Implementation details

ClimaX receives a tensor of shape $V \times H \times W$ and outputs a tensor of shape $V' \times H \times W$, where the number of input and output variables $V$ and $V'$ can vary between different datasets[^6]. To do that, we assume a set $\mathcal{V}$ that contains all possible variables we could encounter during pretraining and finetuning. Each variable in $\mathcal{V}$ has a separate token embedding layer.

The variable tokenization module tokenizes the input to a sequence of $V \times h \times w$ tokens, with each token being a vector of size $p^2$. After that, for each token, we extract the corresponding embedding layer that transforms the token to a vector of dimension $D$. Each embedding layer is a single convolution layer with $in\_channels=1, out\_channels=D, kernel\_size=p, stride=p$. This results in a tensor of shape $V \times h \times w \times D$.

To differentiate between tokens of different input variables, we add the sequence with a *variable positional embedding*, which is a tensor of shape $|\mathcal{V}| \times D$. For each input variable, we extract the corresponding variable positional embedding to add to its tokens. After that, all tokens go through the variable aggregation module, which outputs a tensor of shape $h \times w \times D$.

The tokens are then fed to the attention layers, which output a tensor of the same shape $h \times w \times D$. The prediction head takes each token of dimension $D$ and maps it to a vector of dimension $|\mathcal{V}| \times p^2$, and the output is reshaped to $|\mathcal{V}| \times H \times W$. Finally, we extract predictions of $V'$ target variables and compute the loss.

### Hyperparameters

+------------------+-----------------------------------------+-------------------------------------------------------------------------------------------------+
| Hyperparameter   | Meaning                                 | Value                                                                                           |
+:=================+:========================================+:================================================================================================+
| $\mathcal{V}$    | Default variables                       | All ERA5 variables in Table 2 |
+------------------+-----------------------------------------+-------------------------------------------------------------------------------------------------+
| $|\mathcal{V}|$  | Number of default variables             | 48                                                                                              |
+------------------+-----------------------------------------+-------------------------------------------------------------------------------------------------+
| $p$              | Patch size                              |   --------------------------                                                                    |
|                  |                                         |   $2$ for $5.625°$                                                                        |
|                  |                                         |   $4$ for $1.40625°$                                                                      |
|                  |                                         |   --------------------------                                                                    |
|                  |                                         |                                                                                                 |
|                  |                                         |   : Default hyperparameters of ClimaX                                                           |
+------------------+-----------------------------------------+-------------------------------------------------------------------------------------------------+
| $D$              | Embedding dimension                     | $1024$                                                                                          |
+------------------+-----------------------------------------+-------------------------------------------------------------------------------------------------+
| Depth            | Number of ViT blocks                    | $8$                                                                                             |
+------------------+-----------------------------------------+-------------------------------------------------------------------------------------------------+
| \# heads         | Number of attention heads               | $16$                                                                                            |
+------------------+-----------------------------------------+-------------------------------------------------------------------------------------------------+
| MLP ratio        |   -----------------------------------   | $4$                                                                                             |
|                  |   Determine the hidden dimension of     |                                                                                                 |
|                  |   the MLP layer in a ViT block          |                                                                                                 |
|                  |   -----------------------------------   |                                                                                                 |
|                  |                                         |                                                                                                 |
|                  |   : Default hyperparameters of ClimaX   |                                                                                                 |
+------------------+-----------------------------------------+-------------------------------------------------------------------------------------------------+
| Prediction depth | Number of layers of the prediction head | $2$                                                                                             |
+------------------+-----------------------------------------+-------------------------------------------------------------------------------------------------+
| Hidden dimension | Hidden dimension of the prediction head | $1024$                                                                                          |
+------------------+-----------------------------------------+-------------------------------------------------------------------------------------------------+
| Drop path        | For stochastic depth [huang2016deep]   | $0.1$                                                                                           |
+------------------+-----------------------------------------+-------------------------------------------------------------------------------------------------+
| Dropout          | Dropout rate                            | $0.1$                                                                                           |
+------------------+-----------------------------------------+-------------------------------------------------------------------------------------------------+

: Default hyperparameters of ClimaX

## CNN Baselines {#sec:app:cnn_arc}

### ResNet Hyperparameters

We use the following hyperparameters for ResNet in all of our experiments.

  Hyperparameter     Meaning                                            Value
  ------------------ -------------------------------------------------- -------
  Padding size       Padding size of each convolution layer             $1$
  Kernel size        Kernel size of each convolution layer              $3$
  Stride             Stride of each convolution layer                   $1$
  Hidden dimension   Number of output channels of each residual block   $128$
  Residual blocks    Number of residual blocks                          $28$
  Dropout            Dropout rate                                       $0.1$

  : Default hyperparameters of ResNet

### UNet Hyperparameters

We borrow our UNet implementation from [PDEArena](https://github.com/microsoft/pdearena/blob/main/pdearena/modules/twod_unet.py) [gupta2022towards]. We use the following hyperparameters for UNet in all of our experiments.

+--------------------------+---------------------------------------------+----------------+
| Hyperparameter           | Meaning                                     | Value          |
+:=========================+:============================================+:===============+
| Padding size             | Padding size of each convolution layer      | $1$            |
+--------------------------+---------------------------------------------+----------------+
| Kernel size              | Kernel size of each convolution layer       | $3$            |
+--------------------------+---------------------------------------------+----------------+
| Stride                   | Stride of each convolution layer            | $1$            |
+--------------------------+---------------------------------------------+----------------+
| Channel multiplications  |   ----------------------------------------- | $[1, 2, 2, 4]$ |
|                          |   Determine the number of output channels   |                |
|                          |   for Down and Up blocks                    |                |
|                          |   ----------------------------------------- |                |
|                          |                                             |                |
|                          |   : Default hyperparameters of UNet         |                |
+--------------------------+---------------------------------------------+----------------+
| Blocks                   | Number of blocks                            | $2$            |
+--------------------------+---------------------------------------------+----------------+
| Use attention            | If use attention in Down and Up blocks      | False          |
+--------------------------+---------------------------------------------+----------------+
| Dropout                  | Dropout rate                                | $0.1$          |
+--------------------------+---------------------------------------------+----------------+

: Default hyperparameters of UNet

### Other implementation details

Following the implementation of ResNet in rasp2020weatherbench [rasp2021data; ernst2021structured], we found the following details important for the performance of both CNN baselines:

- Use Batch normalization

- Use Leakyrelu with a slope of $0.3$ as the activation function

- Postnorm instead of Prenorm

- Use periodic convolutions in the longitude direction but not the latitude direction.

- Use a kernel size of $7$ in the first CNN layer.

# Training details

#### Data normalization

We normalized all inputs during pre-training as well as fine-tuning. For each variable, at each pressure level (for atmospheric variables), we compute the mean and standard deviation to normalize them to zero mean and unit variance. We de-normalize the predictions to get back to the original range before computing evaluation metrics.

#### Software and hardware stack

We use PyTorch [pytorch2019], `timm` [rw2019timm], `numpy` [harris2020array] and `xarray` [Hoyer_2017] to manage our data and model training. We used 32GB NVIDIA V100 devices for training. For pretraining we distribute the batch across 80 V100s on AzureML. We leverage `fp16` floating point precision in our model.

## Pretraining

### Objective

We use the loss function in [\[eq:lat_mse\]](#eq:lat_mse){reference-type="ref+Label" reference="eq:lat_mse"} for pretraining.

### Optimization

We used the AdamW optimizer [kingma2014adam; loshchilov2017decoupled] with parameters ($\beta_1 = 0.9, \beta_2 = 0.95$). We used weight decay of $1e-5$ for all parameters except for the positional embedding. We used a learning rate of $5e-4$, with a linear warmup schedule for $10000$ steps ($5$ epochs), followed by a cosine-annealing schedule for $190000$ steps ($95$ epochs).

## Finetuning

### Objective

We use lat-weighted MSE in [\[eq:lat_mse\]](#eq:lat_mse){reference-type="ref+Label" reference="eq:lat_mse"} for finetuning ClimaX in temporal forecasting and downscaling tasks. In ClimateBench, we finetune using standard MSE without the weighting term, as this led to better results and was suggested by [watson2022climatebench].

### Optimization

For all tasks, we used AdamW with parameters ($\beta_1 = 0.9, \beta_2 = 0.999$). We used weight decay of $1e-5$ for all parameters except for the positional embedding. We used a linear warmup schedule for $10000$ steps ($5$ epochs), followed by a cosine-annealing schedule for $90000$ steps ($45$ epochs). The learning rate for each task is as follows:

  Task                  Learning rate
  --------------------- ---------------
  Weather forecasting   $5e-7$
  Climate projection    $5e-4$
  Climate downscaling   $5e-5$

  : Learning rate for finetuning ClimaX in different downstream tasks

We used a small learning rate for weather forecasting as the task resembles pretraining. For downscaling, we used a larger learning rate, as the nature of the task is different from pretraining, even though the input variables are similar. In climate projection, we needed to initialize new weights for the embedding layers and prediction heads, and thus used a similar learning rate to training from scratch.

# Datasets {#sec:app:data}

## CMIP6-ClimaX {#sec:app:cmip6_details}

We created CMIP6-ClimaX for pretraining ClimaX, which consists of $5$ datasets from the CMIP6 project. We downloaded the datasets from the official CMIP6 search interface at <https://esgf-data.dkrz.de/search/cmip6-dkrz/>. These datasets share the following attributes:

- Experiment ID: historical

- Table ID: 6hrPlevPt, i.e., 6-hourly data on pressure levels.

- Variant label: r1i1p1f1. The variant label distinguishes among closely related simulations by a single model, in which "r" specifies the initial condition, "i" specifies the observational dataset and initialization method used for determining the initial condition, "p" specifies the perturbed physics version of the model, and "f" specifies the forcing index.

All datasets have a temporal coverage from $1850$ to $2015$ and a temporal resolution of $6$ hours. We chose these datasets as they contain similar climate variables at similar vertical levels to ERA5. We note that there are more than $5$ datasets from CMIP6 that suit our selection criteria, but we were not able to download others due to some issues on the data servers. We regridded these datasets to $5.625°$ and $1.40625°$ using the xesmf Python package [zhuang2018xesmf] using bilinear interpolation. We provide a detailed description of these $5$ data sources and the available variables we used to construct CMIP6-ClimaX in 1.

::: {#tab:cmip6_data}
+-------------+---------------------+----------------------------------------------------------+
| Data Source | Original resolution | Variables                                                |
+:============+:====================+:============+:========+:=================================+
| 3-5         |                     | Type        | Abbrev. | Levels                           |
+-------------+---------------------+-------------+---------+----------------------------------+
| MPI         | 100km               | Single      | t2m     |                                  |
|             |                     +-------------+---------+----------------------------------+
|             |                     | Single      | u10     |                                  |
|             |                     +-------------+---------+----------------------------------+
|             |                     | Single      | v10     |                                  |
|             |                     +-------------+---------+----------------------------------+
|             |                     | Atmospheric | z       | 50, 250, 500, 600, 700, 850, 925 |
|             |                     +-------------+---------+----------------------------------+
|             |                     | Atmospheric | u       | 50, 250, 500, 600, 700, 850, 925 |
|             |                     +-------------+---------+----------------------------------+
|             |                     | Atmospheric | v       | 50, 250, 500, 600, 700, 850, 925 |
|             |                     +-------------+---------+----------------------------------+
|             |                     | Atmospheric | t       | 50, 250, 500, 600, 700, 850, 925 |
|             |                     +-------------+---------+----------------------------------+
|             |                     | Atmospheric | q       | 50, 250, 500, 600, 700, 850, 925 |
+-------------+---------------------+-------------+---------+----------------------------------+
| Tai         | 100km               | Single      | t2m     |                                  |
|             |                     +-------------+---------+----------------------------------+
|             |                     | Atmospheric | z       | 250, 500, 600, 700, 850, 925     |
|             |                     +-------------+---------+----------------------------------+
|             |                     | Atmospheric | u       | 250, 500, 850                    |
|             |                     +-------------+---------+----------------------------------+
|             |                     | Atmospheric | v       | 250, 500, 850                    |
|             |                     +-------------+---------+----------------------------------+
|             |                     | Atmospheric | t       | 250, 500, 850                    |
|             |                     +-------------+---------+----------------------------------+
|             |                     | Atmopheric  | q       | 250, 500, 600, 700, 850, 925     |
+-------------+---------------------+-------------+---------+----------------------------------+
| AWI         | 250km               | Single      | t2m     |                                  |
|             |                     +-------------+---------+----------------------------------+
|             |                     | Single      | u10     |                                  |
|             |                     +-------------+---------+----------------------------------+
|             |                     | Single      | v10     |                                  |
|             |                     +-------------+---------+----------------------------------+
|             |                     | Atmospheric | z       | 50, 250, 500, 600, 700, 850, 925 |
|             |                     +-------------+---------+----------------------------------+
|             |                     | Atmospheric | u       | 50, 250, 500, 600, 700, 850, 925 |
|             |                     +-------------+---------+----------------------------------+
|             |                     | Atmospheric | v       | 50, 250, 500, 600, 700, 850, 925 |
|             |                     +-------------+---------+----------------------------------+
|             |                     | Atmospheric | t       | 50, 250, 500, 600, 700, 850, 925 |
|             |                     +-------------+---------+----------------------------------+
|             |                     | Atmospheric | q       | 50, 250, 500, 600, 700, 850, 925 |
+-------------+---------------------+-------------+---------+----------------------------------+
| HAMMOZ      | 250km               | Single      | t2m     |                                  |
|             |                     +-------------+---------+----------------------------------+
|             |                     | Single      | u10     |                                  |
|             |                     +-------------+---------+----------------------------------+
|             |                     | Single      | v10     |                                  |
|             |                     +-------------+---------+----------------------------------+
|             |                     | Atmospheric | z       | 50, 250, 500, 600, 700, 850, 925 |
|             |                     +-------------+---------+----------------------------------+
|             |                     | Atmospheric | u       | 50, 250, 500, 600, 700, 850, 925 |
|             |                     +-------------+---------+----------------------------------+
|             |                     | Atmospheric | v       | 50, 250, 500, 600, 700, 850, 925 |
|             |                     +-------------+---------+----------------------------------+
|             |                     | Atmospheric | t       | 50, 250, 500, 600, 700, 850, 925 |
|             |                     +-------------+---------+----------------------------------+
|             |                     | Atmospheric | q       | 50, 250, 500, 600, 700, 850, 925 |
+-------------+---------------------+-------------+---------+----------------------------------+
| CMCC        | 100km               | Atmospheric | z       | 50, 250, 500, 600, 700, 850, 925 |
|             |                     +-------------+---------+----------------------------------+
|             |                     | Atmospheric | u       | 50, 250, 500, 600, 700, 850, 925 |
|             |                     +-------------+---------+----------------------------------+
|             |                     | Atmospheric | v       | 50, 250, 500, 600, 700, 850, 925 |
|             |                     +-------------+---------+----------------------------------+
|             |                     | Atmospheric | t       | 250, 500, 850                    |
+-------------+---------------------+-------------+---------+----------------------------------+

: Resolution and variables of CMIP6-ClimaX dataset used for pretraining. *Static* represents variables don't depend on time, *Single* represents surface variables, and *Atmospheric* represents time-varying atmospheric properties at the chosen altitudes.
:::

We note that AWI and HAMMOZ are not the best data sources for higher resolution $1.40625°$ training, because their original resolution at $250$ km is lower than $1.40625°$, which is about $156$ km. We wanted to use other higher-resolution datasets but were not able to download them. We believe pretraining on other high-resolution datasets would lead to better performance.

## ERA5 {#sec:app:era5_details}

We use the preprocessed version of ERA5 from WeatherBench [rasp2020weatherbench] for finetuning ClimaX. WeatherBench was created as a standard benchmark data and evaluation framework for comparing data-driven weather forecasting models. WeatherBench regridded the original ERA5 at $0.25°$ to three lower resolutions: $5.625°$, $2.8125°$, and $1.40625°$. See <https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation> for more details of the raw ERA5 data. Table 2 summarizes the variables we use for finetuning ClimaX.

::: {#tab:era5_data}
           Type Variable name               Abbrev.   ECMWF ID   Levels
  ------------- --------------------------- --------- ---------- ----------------------------------
         Static Land-sea mask               LSM       172        
         Static Orography                                        
         Single 2 metre temperature         T2m       167        
         Single 10 metre U wind component   U10       165        
         Single 10 metre V wind component   V10       166        
    Atmospheric Geopotential                Z         129        50, 250, 500, 600, 700, 850, 925
    Atmospheric U wind component            U         131        50, 250, 500, 600, 700, 850, 925
    Atmospheric V wind component            V         132        50, 250, 500, 600, 700, 850, 925
    Atmospheric Temperature                 T         130        50, 250, 500, 600, 700, 850, 925
    Atmospheric Specific humidity           Q         133        50, 250, 500, 600, 700, 850, 925
    Atmospheric Relative humidity           R         157        50, 250, 500, 600, 700, 850, 925

  : ECMWF variables used in our ERA5 dataset. *Static* represents variables don't depend on time, *Single* represents surface variables, and *Atmospheric* represents time-varying atmospheric properties at the chosen altitudes.
:::

### ERA5-NA

We constructed ERA5-NA from ERA5 to evaluate ClimaX and the baselines on regional forecasting. ERA-NA has the same set of variables as in Table 2, but only contains data that belongs to the North America region. To do this, we first identified the latitude and longitude range to form a rectangular area that encapsulates North America, using the standard CORDEX domains <https://cordex.org/wp-content/uploads/2012/11/CORDEX-domain-description_231015.pdf>. For each data sample, we then extracted the spatial positions that fall into this range, forming in ERA5-NA.

### ERA-S2S

We built ERA5-S2S from ERA5 to serve as a benchmark dataset for sub-seasonal to seasonal prediction. ERA5-S2S consists of two sub-datasets, whose the goals are to predict the biweekly average statistics of target variables in weeks $3$ and $4$, and weeks $5$ and $6$, respectively. The input includes all variables in Table 2, while the output variables are are averaged over two weeks, starting from the start of week 3 (5) and to the end of week 4 (6).

## ClimateBench

We refer to watson2022climatebench for complete details of ClimateBench.

# Quantitative evaluation

## Metrics {#sec:app:metrics}

This section presents all evaluation metrics we use in 4. For all metrics, we denote $\Tilde{X}$ and $X$ as the prediction and ground truth, which have a shape of $N \times H \times W$, where $N$ is the number of forecasts, or the number of test samples, $H \times W$ is the spatial resolution. $L(i)$ is the latitude weighting term to account for the non-uniformity in areas of the grid cells. We have removed the time notation for simplicity.

### Weather forecasting metrics

#### Root mean square error (RMSE)

$$\text{RMSE} = \frac{1}{N} \sum_{k=1}^{N} \sqrt{\frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W L(i)(\Tilde{X}_{k,i,j} - X_{k,i,j})^2}. \label{eq:lat_rmse}$$

#### Anomaly correlation coefficient (ACC)

Anomaly correlation coefficient (ACC) is the spatial correlation between prediction anomalies $\Tilde{X}^{'}$ relative to climatology and ground truth anomalies $X^{'}$ relative to climatology: $$\begin{gathered}
    \text{ACC} = \frac{\sum_{k,i,j} L(i) \Tilde{X}^{'}_{k,i,j} X^{'}_{k,i,j}}{\sqrt{\sum_{k,i,j} L(i) \Tilde{X}^{'2}_{k,i,j} \sum_{k,i,j} L(i) X^{'2}_{k,i,j}}}, \\
    \Tilde{X}^{'} = \Tilde{X}^{'} - C, X^{'} = X^{'} - C,
\end{gathered}$$ in which climatology $C$ is the temporal mean of the ground truth data over the entire test set $C = \frac{1}{N}\sum_k X$.

### Climate projection metrics

#### Normalized spatial root mean square error (NRMSE$_s$)

Normalized spatial root mean square error (NRMSE$_s$) measures the spatial discrepancy between the temporal mean of the prediction and the temporal mean of the ground truth: $$\text{NRMSE}_s = \sqrt{\left\langle \left(\frac{1}{N} \sum_{k=1}^N \Tilde{X} - \frac{1}{N} \sum_{k=1}^N X \right)^2 \right\rangle} \bigg/ \frac{1}{N} \sum_{k=1}^N \left\langle X \right\rangle,$$ in which $\langle A \rangle$ is the global mean of $A$: $$\langle A \rangle = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W L(i) A_{i,j}$$

#### Normalized global root mean square error (NRMSE$_g$) 

Normalized global root mean square error (NRMSE$_g$) measures the discrepancy between the global mean of the prediction and the global mean of the ground truth: $$\text{NRMSE}_g = \sqrt{\frac{1}{N} \sum_{k=1}^N \left(\langle \Tilde{X} \rangle - \langle X \rangle\right)^2}  \bigg/ \frac{1}{N} \sum_{k=1}^N \left\langle X \right\rangle.$$

#### Total normalized root mean square error (TRMSE)

Total normalized root mean square error (TRMSE) is the weighted sum of NRMSE$_s$ and NRMSE$_g$: $$\text{TRMSE} = \text{NRMSE}_s + \alpha \cdot \text{NRMSE}_g,$$ where $\alpha$ is chosen to be $5$ as suggested by watson2022climatebench.

### Climate downscaling metrics

#### Root mean square error (RMSE)

This is the same as [\[eq:lat_rmse\]](#eq:lat_rmse){reference-type="ref+Label" reference="eq:lat_rmse"}.

#### Mean bias

Mean bias measures the difference between the spatial mean of the prediction and the spatial mean of the ground truth. A positive mean bias shows an overestimation, while a negative mean bias shows an underestimation of the mean value. $$\text{Mean bias} = \frac{1}{N \times H \times W} \sum_{k=1}^N \sum_{i=1}^H \sum_{j=1}^W \Tilde{X} - \frac{1}{N \times H \times W} \sum_{k=1}^N \sum_{i=1}^H \sum_{j=1}^W X$$

#### Pearson coefficient

Pearson coefficient measures the correlation between the prediction and the ground truth. We first flatten the prediction and ground truth, and compute the metric as follows: $$\rho_{\Tilde{X}, X}=\frac{\operatorname{cov}(\Tilde{X}, X)}{\sigma_{\Tilde{X}} \sigma_X}$$

## Results summary {#app:results_summary}

Table [\[tab:rmse-compare\]](#tab:rmse-compare){reference-type="ref" reference="tab:rmse-compare"} and [\[tab:acc-compare\]](#tab:acc-compare){reference-type="ref" reference="tab:acc-compare"} summarize the global forecasting results of ClimaX and the baselines for all target variables and at all lead times. In addition to IFS and the two CNN-based baselines in the main text, we include FourCastNet [pathak2022fourcastnet], PanguWeather [bi2022pangu], and GraphCast [lam2022graphcast] for comprehensiveness. We want to emphasize that the results obtained by these methods are not comparable with ClimaX, as they were trained on ERA5 at $0.25°$, a much higher resolution compared to $5.625°$ and $1.40625°$ data used to train ClimaX. In 4.5, we had a discussion on how the performance of ClimaX scales favorably with respect to data resolution. We hope this summary will provide future works with an easier comparison with existing baselines.

In spite of being trained on much lower resolutions, ClimaX outperforms FourCastNet in forecasting Z500, T850, and U10 at lead times from 3 days and beyond, in terms of both RMSE and ACC. For T2m, ClimaX achieves better results at horizons longer than 3 days. PanguWeather performs better than ClimaX on most of the tasks, but the gap between the two methods shrinks and becomes negligible as the lead time increases. ClimaX even outperforms PanguWeather in predicting U10 at 7 days lead times. This is because ClimaX is finetuned to perform direct prediction, which mitigates error accumulation for long horizon prediction. GraphCast achieves the lowest RMSE among all methods, but performs worse in terms of ACC compared to ClimaX and PanguWeather.

