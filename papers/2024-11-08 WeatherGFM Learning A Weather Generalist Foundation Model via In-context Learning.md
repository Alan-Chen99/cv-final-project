# Abstract

The Earth's weather system involves intricate weather data modalities and diverse weather understanding tasks, which hold significant value to human life. Existing data-driven models focus on single weather understanding tasks (e.g., weather forecasting). While these models have achieved promising results, they fail to tackle various complex tasks within a single and unified model. Moreover, the paradigm that relies on limited real observations for a single scenario hinders the model's performance upper bound. Inspired by the in-context learning paradigm from visual foundation models and large language models, in this paper, we introduce the first generalist weather generalist foundation model (WeatherGFM) to address weather understanding tasks in a unified manner. Specifically, we first unify the representation and definition for diverse weather understanding tasks. Subsequently, we design weather prompt formats to handle different weather data modalities, including single, multiple, and temporal modalities. Finally, we adopt a visual prompting question-answering paradigm for the training of unified weather understanding tasks. Extensive experiments indicate that our WeatherGFM can effectively handle up to ten weather understanding tasks, including weather forecasting, super-resolution, weather image translation, and post-processing. Our method also showcases generalization ability on unseen tasks.

# Introduction

Modeling Earth weather systems involves a series of complex subprocesses that are intended to transform intricate Earth observation data into applications like weather forecasting [chen2023fengwu; bi2023accurate], downscaling [chen2022rainnet], assimilation [huang2024diffda], retrieval [liu2011satellite], and bias correction [gong2024cascast]. During the past decade, many data-driven machine learning methods have been investigated for various weather understanding tasks and delivering desirable performance on specific tasks. For example, recent studies using large-scale training data (e.g., ERA5 reanalysis data [hersbach2020era5]) have exceeded the accuracy of conventional numerical weather forecasts. However, current weather foundational models face challenges regarding generalizability and data scale limitations. On the one hand, the Earth observation system consists of a variety of observation devices, such as satellites, radar, and weather stations, which produce diverse modalities of data. Consequently, designing a specific model for a single-task scenario is highly complex, time-consuming, and labor-intensive. On the other hand, large-scale data in fields such as computer vision can be obtained at a low cost, whereas weather understanding tasks face an intrinsic bottleneck in data scale due to restrictions on individual scenes and single observation devices. For instance, local short-term precipitation forecasting models can only utilize a finite range of observational data.

A significant trend in AI research is the development of foundation models, shifting towards large-scale pre-training and in-context learning. This paradigm enables unified processing of a multitude of complex tasks and generalization to unseen tasks. For example, large language models (LLMs) can perform a variety of language-centric tasks (e.g., sentiment analysis, question answering and machine translation) by combining language input-output examples with new query inputs (prompts) without optimizing model parameters [brown2020language]. Similarly, vision foundation models [wang2023images; liu2023unifying; chen2024learning] employ visual prompts with query inputs to carry out diverse image-centric tasks, such as semantic segmentation, depth estimation, and image restoration. These studies highlight the significant potential of generalist foundational models.

The study of foundation models remains largely limited in weather understanding, with the majority focused on Computer Vision and Natural Language Processing. While there has been some progress with large foundation models in weather and climate, the focus is mainly on weather forecasting and downscaling tasks. For example, Climax [nguyen2023climax] uses a pre-training-finetuning paradigm for weather forecasting and downscaling. Aurora [bodnar2024aurora] employs LoRA to unify weather forecasting and quick prediction of atmospheric chemistry. However, these studies do not take into account the modeling of multi-modalities and multi-tasks. This poses a challenge: *Is it possible to design a universal foundation model capable of handling the variety of complex weather understanding tasks and data modalities?*

In this paper, we first propose a weather generalist foundation model, WeatherGFM, to uniformly address a variety of complex weather understanding tasks and data modalities. Unlike prior studies that focused on weather forecasting, our proposed method can expand the task scope to weather forecasting, weather super-resolution (i.e., weather downscaling) [veillette2020sevir], weather image translation (similar to retrieval in weather) [veillette2020sevir], and post-processing [gong2024cascast]. These tasks all belong to the domain of weather understanding, but their modalities are distinct. Specifically, Sequence modal data can be utilized for weather forecasting, such as short-term predictions based on radar data. Multi-modal data can be employed for weather image translation, such as converting multi-modal satellite data to generate radar data. Single-modal data can be applied to various common scenarios, such as radar image super-resolution and post-processing. To unify the diverse weather data modalities into a general representation, we introduce a weather prompt format that assigns different prompt phrases to various modalities. By leveraging in-context learning, our WeatherGFM achieves a promising in-context ability on both various seen tasks and unseen tasks. The significance of our work can be summarized as:

- We propose the first weather generalist foundation model (i.e., WeatherGFM), which can handle more than ten weather understanding tasks.
- Our weather prompt design supports a diversity of weather data modalities, including time-series, multi-modal, and single-modal data.
- Our WeatherGFM with in-context learning first demonstrates the generalization ability to unseen weather understanding tasks.

# Related Work

**Weather understanding and beyond.** Over the past decade, machine learning techniques have consistently attracted attention in the field of weather and climate. Numerous data-driven machine learning models have been proposed to address classical tasks in weather understanding [veillette2020sevir], such as forecasting, super-resolution, image translation, and post-processing. Weather forecasting [bi2023accurate] aims to predict future observations from past data. Weather super-resolution tasks, i.e., weather downscaling, [chen2022rainnet] focus on recovering high-resolution data from low-resolution observations. Weather image translation tasks [stock2024srvit] involves converting existing observational data into desired target modalities, such as transforming satellite observations into ground-based weather radar data. Post-processing tasks seek to enhance existing model results, such as bias correction and deblurring [gong2024cascast]. Despite significant advancements, current methods often rely on specialized datasets and customized single-task models for certain scenarios. Consequently, single-task models struggle to exhibit strong generalization abilities and fail to capture the interconnections between diverse tasks, which hinders the establishment of simulations for the Earth system.

**Weather foundation model.** The rise of foundation models [liu2024visual; zhao2024easygen; zhao2024unifashion] in Natural Language Processing and computer vision has sparked interest in their application for weather and climate. Large foundation models, enhanced through pre-training, improve the generalization of AI climate models and can be fine-tuned for specific tasks. [pathak2022fourcastnet] proposed FourCastNet, a climate pre-trained model using Vision Transformer for high-resolution predictions and rapid inference through self-supervised pre-training and autoregressive fine-tuning. Pangu-Weather [bi2023accurate] utilizes a 3D Earth-specific Transformer for accurate global predictions. ClimaX [nguyen2023climax] introduces supervised pre-training to weather prediction, offering flexibility for diverse forecasting tasks. A pre-training foundation model usually requires mask modeling for pre-training and then undergoes fine-tuning on specific tasks, such as fine-tuning the pre-trained model on weather forecasting, remote sensing classification and segmentation tasks [bodnar2024aurora; satmae; satmaeplus; s2mae].

**Visual in-context learning.** In recent advancements, visual in-context learning has emerged as a promising research area, inspired by the success of language models like GPT-3 [brown2020language]. These models adapt to various NLP tasks using prompts or in-context examples without extensive retraining. Similarly, in the vision domain, models such as MAE-VQGAN [hojel2024finding] and Painter [wang2023images] have begun exploring in-context learning. However, challenges persist, especially in low-level tasks requiring detailed pixel manipulation. To address this, PromptGI [liu2023unifying] and GenLV have incorporated in-context learning concepts into their designs to unify low-level vision tasks with diverse input and output modalities, aiming to develop generalist models. Vision-language models like Unified-IO [lu2022unified] and Unified-IO 2 [lu2024unified] have made significant progress in integrating multiple tasks, highlighting the potential for unified approaches across modalities. Additionally, compositional visual reasoning, exemplified by Visual Programming [gupta2023visual], aligns with in-context learning goals by emphasizing visual task synthesis. ViperGPT [suris2023vipergpt] further demonstrates foundational models for visual reasoning, employing computational techniques similar to our objectives, though without relying on programmatic inputs. These collective efforts pave the way for more sophisticated and versatile visual in-context learning frameworks.

# Method

## Unified representation of weather understanding tasks

[IMAGE: Illustration of the unified representation for weather understanding tasks (figure1.pdf)]

Weather understanding tasks involve processing multi-source observational data [veillette2020sevir], such as geostationary satellites (GEOS), polar-orbiting satellites (POES), weather radars, and ground observation stations. Each task (e.g., weather forecasting, spatial and temporal super-resolution, weather image translation, and post-processing) utilizes different types of input and output data. To address this challenge, we first developed a unified data representation that can standardize these diverse tasks. Unlike traditional methods that rely on task-specific models for each distinct task, we introduce a universal foundational model capable of addressing various weather understanding tasks through a single and general solution.

Several key weather understanding tasks can be framed using different types of input and output data. For instance, the weather spatial super-resolution (SR) task generates a high-resolution image ```latex $x_{HR}$ ``` from a low-resolution image ```latex $x_{LR}$ ```, while weather temporal super-resolution predicts a high-resolution image ```latex $x^{t}_{HR}$ ``` based on two consecutive observed input images ```latex $x^{t-1}_{LR}$ ``` and ```latex $x^{t+1}_{LR}$ ```, where ```latex $t$ ``` represents a particular moment in time. The weather temporal super-resolution task aims to restore the missing observed data in time ```latex $t$ ```. Weather forecasting relies on a sequence of observed data points ```latex $\{x^1, x^2, \ldots, x^t\}$ ``` that are gathered over the past ```latex $t$ ``` time steps. These observed data points serve as condition, enabling the prediction of future data points such as points ```latex $\{x^{t+1}, x^{t+2}, \ldots \}$ ```. The image translation task focuses on converting an input image from one modality (e.g., satellite image) to another modality (e.g., radar image). Formally, we can represent these tasks as projections from the source input data ```latex $X_S$ ``` to the target output data ```latex $X_T$ ```:

```latex
$$\tau: X_S  \xrightarrow{} X_T.$$
```

When ```latex $X_S = x_{LR}$ ``` and ```latex $X_T = x_{HR}$ ```, the task corresponds to spatial SR. Similarly, when ```latex $X_S = \{x^1, x^2, \ldots, x^t \}$ ``` and ```latex $X_T = \{x^{t+1}, x^{t+2}, \ldots \}$ ```, the task represents weather forecasting. As these tasks differ in their input and output formats, as well as sequence lengths, the key challenge lies in unifying them within one coherent data representation.

## WeatherGFM: Weather Generalist Foundation Model

We present the Weather Generalist Foundation Model (WeatherGFM) to tackle the challenges inherent in a range of weather understanding tasks. Through in-context learning, our WeatherGFM can uniformly handle various weather understanding tasks involving multiple data modalities.

**Weather prompt designing.** In large language models and vision foundation models, task prompts commonly provide specific task-related input-output pairs. In machine translation [machine_translation], the model is given English to French text pairs as prompts. The model can perform machine translation tasks based on these sample prompts for a given input. In visual tasks [painter], the visual prompt image1 may be a natural image, and image2 is the corresponding segmented image. The model will conduct the segmentation task for a new input image3 to obtain the segmented image.

[IMAGE: Comparison of weather prompts with text and visual prompts design (promptdesign.pdf)]

Following this paradigm, we designed weather prompts for weather understanding tasks. Since the input for weather understanding tasks involves multiple modalities, such as a single weather observation variable, multiple different weather variables, and time-series weather variables, we proposed three prompts to handle different modalities of input. Weather prompt1 is similar to visual prompts, converting a single modality image into a target image. In weather prompt2, the input modality can be two different channel satellite observation images (e.g., IR069 and IR107 data), and the output can be weather radar observation data for image translation tasks. In weather prompt3, time-series prompts can be input to perform weather forecasting-related tasks. With these forms of prompt design, our method can handle most weather understanding tasks.

**Weather in-context learning.** Inspired by the success of in-context learning in large language models [in-context_larning] and vision foundation models [painter], we propose to unify the weather understanding problem as the visual prompting question-answer paradigm. Specifically, given a visual question-answer prompt pair ```latex $(P_{in}, P_{target})$ ``` as a task-guided prompt and a query input ```latex $X_{in}$ ```, the model is expected to perceive the context of the prompt (i.e., what task it represents). Consequently, the model can perform the corresponding operations on the query with the prompt. This process can be formulated as follows:

```latex
$$X_{target} = F_{\tau} (P_{in}, P_{target}, X_{in}; \theta)$$
```

where ```latex $F_{\tau}$ ``` represents a universal foundation model parameterized by ```latex $\theta$ ```. ```latex $P_{in}$ ``` and ```latex $P_{target}$ ``` denotes the input and target of task prompts. We can determine what task will be performed on the input ```latex $X_{in}$ ``` by selecting the task-specific prompt ```latex $P_{in}$ ``` and ```latex $P_{target}$ ```, and then obtain the target ```latex $X_{target}$ ``` for the corresponding task through the model ```latex $F_{\tau}$ ```.

[IMAGE: Overall approach of our weather generalist foundation model WeatherGFM (Methods.pdf)]

**Mixed-modal mask modeling.** Upon redefining the output spaces of the aforementioned representative vision tasks, it is observed that both the input and output of these tasks are in the form of images as transformers-based architectures could provide flexibility by treating the image-like data as a set of tokens. Therefore, we build the WeatherGFM architecture on Vision Transformers (ViT) and propose a mixed-modal masked image modeling (MMIM) pipeline to train multiple weather understanding tasks. Inspired by the concept of Visual Question Answering [painter; liu2023unifying; GenLV], we introduce mixed-modality masking on various weather modalities for visual question-and-answer modeling in weather understanding tasks. This process can be formulated as follows:

```latex
$$P^{'}_{target}, X^{'}_{target} = F_{\tau} (P_{in}, M(P_{target}), X_{in}, M(X_{target}); \theta)$$
```

where we randomly conduct mask operation ```latex $M$ ``` on the prompt target ```latex $P_{target}$ ``` as well as the ground truth ```latex $X_{target}$ ``` according to the mask ratio. Meanwhile, the prompt input ```latex $P_{in}$ ``` and the input query ```latex $X_{in}$ ``` will be retained entirely. ```latex $P^{'}_{target}$ ``` and ```latex $X^{'}_{target}$ ``` represent the predicted target output of model ```latex $F_{\tau}$ ```. The optimization objectives are as follows:

```latex
$$\mathrm{L}^{total}_{\theta} = \mathrm{L}_2(P^{'}_{target},P_{target}) + \mathrm{L}_2(X^{'}_{target},X_{target})$$
```

where we use MSE (mean square error) loss ```latex $\mathrm{L_2}$ ``` to train the weather generalist foundation model. In the inference stage, we keep the ```latex $P_{in}$ ```, ```latex $P_{target}$ ```, and ```latex $X_{in}$ ``` intact while the target image is fully masked. This target full masking strategy allows generalist foundation models to generate the corresponding target through a visual question-and-answer format. Our WeatherGFM comprises two main elements: the format for input data and the architectural design.

**Input format:** Given an input of shape ```latex $(C,H,W)$ ```, ViT predicts an output of shape ```latex $(C^{'},H^{'},W^{'})$ ```, where ```latex $C$ ``` represents the input channels and ```latex $C^{'}$ ``` represents the output channels. Different tasks have different channels. The model tokenizes the input into a sequence of patches, with each patch having a size of ```latex $C\times p^{2}$ ```, where ```latex $p$ ``` is the patch size. Unlike RGB-based image data, where the channels are fixed, the number of physical variables in climate and weather data can vary between different datasets and tasks. To adapt the ViT to different weather-related downstream tasks, we designed task-specific patch embedding layers within the architecture. After the patch embedding layer, we use an MLP layer to align the embeddings of different tasks to the same space:

```latex
$$\begin{aligned}
\mathrm{z_{C}} &= \mathrm{PatchEmbed_{C}(x)}, \mathrm{x}\in \mathbb{R}^{C\times H \times W}, &\mathrm{z_{C}} \in \mathbb{R}^{N\times D}, \\
\mathrm{z_{0}} &= \mathrm{MLP_{C}(LN(\mathrm{z_{C}}))}, &\mathrm{z_{0}} \in \mathbb{R}^{N\times D}
\end{aligned}$$
```

where ```latex $\mathrm{N, D}$ ``` denotes the number of input tokens and the transformer dimension, respectively. For the masked area, we follow previous works [GIP] to use a learnable token vector to replace each masked patch. We adopt the block-wise masking strategy, taking the masking ratio as 75%.

**Architecture:** A vanilla vision Transformer (ViT) is adopted as the backbone architecture. It consists of task-specific patch-embedded layers and several alternating layers made of Multi-Head Self-Attention (MHSA) and MLP blocks. Layer Normalization (LN) is applied before every block, and residual connections are applied after every block. This process can be formulated as follows:

```latex
$$\begin{aligned}
\mathrm{z_{\ell}^{'}} &= \mathrm{MHSA(LN(\mathrm{z_{\ell-1}}))}+\mathrm{z_{\ell-1}}, \ell = 1...L, \\
\mathrm{z_{\ell}} &= \mathrm{MLP(LN(\mathrm{z_{\ell}^{'}}))}+\mathrm{z_{\ell}^{'}}, \ell = 1...L,
\end{aligned}$$
```

where L denotes the number of layers. After the attention layers, we employ a prediction head and then unpatchify the output of the prediction head. The prediction head is a one-layer MLP with a hidden dimension of 1024.

# Experiments

## Weather Understanding Tasks

We incorporate up to 10 tasks including diverse weather forecasting, weather super-resolution, weather image translation and weather post-processing tasks into our experiments.

**SEVIR.** The Storm EVent ImageRy dataset (SEVIR) [veillette2020sevir] is a spatiotemporally aligned dataset that contains over 10,000 weather events represented by five spatially and temporally aligned sensors. These sensors consist of three channels (C02, C09, C13) from the GOES-16 satellite, one NEXRAD derived vertically integrated liquid (VIL) mosaic variable, and lighting detections from the GOES GLM sensor. Each SEVIR event spans 4 hours with 5-minute intervals, sampled randomly (with oversampling of events with moderate and high precipitation) using the NOAA Storm Event Database. In our task, we uniformly resize the resolution of images from different modalities to 256x256. Moreover, we filter the events within the SEVIR dataset and pick out those events that include both the three channels of the GOES-16 satellite and the one variable derived from weather radar. Ultimately, the dataset we utilize comprises 11,508 events with four distinct sensing modalities. Among them, 11,308 events are selected as the training set, while 100 events are designated as the validation set and 100 events are designated as the test set. Consequently, the training set contains a total of 2.2M images, while the validate/test set has a total of 19.6K images.

**POMINO-TROPOMI, GEOS-CF.** In addition, we add a weather image translation task for environment monitoring: Translate geostationary NO2 data to polar-orbiting satellites NO2 data (GEOS2POES-NO2) based on POMINO-TROPOMI product [NO2translation] and GEOS-CF dataset [keller2021description]. In this task, the input images are sourced from GEMS as well as the GEOS-CF datasets, while the output images are obtained from the TROPOMI dataset. The original image has a resolution of 1400x800. We also divide it into grids of 256x256 with a sliding step size of 128. Each original image can thus be segmented into 45 pieces of 256x256 pictures. We utilize the observational data from January 2021 to April 2022. After processing, each modality has 20,000 images with a resolution of 256x256. Among them, we allocate 18,000 images as the training set, 1,000 images as the validation set, and 1,000 images as the test set.

## Implementation and Evaluation

**Training details.** During training, we resize the weather images of different resolutions to a resolution of 256x256 and input them into the model in accordance with the combination mode of ```latex $P_{in}, P_{out}, X_{in}, X_{out}$ ``` in the task-specific prompt format, resulting in a N x 256 x 256 total input resolution. The L1 loss is employed as the loss function. For optimization, the AdamW optimizer with a cosine learning rate scheduler is utilized. The base learning rate is 1e-4. The batch size is 20.

**Evaluation metrics.** Besides RMSE, we also include the Critical Success Index (CSI), which is commonly used in weather understanding tasks (e.g., precipitation nowcasting) and is defined as

```latex
$$\text{CSI} = \frac{\text{Hits}}{\text{Hits} + \text{Misses} + \text{F.Alarms}}$$
```

To count the Hits (truth=1, pred=1), Misses (truth=1, pred=0) and F.Alarms (truth=0, pred=1), the prediction and the ground-truth are normalized using mean-variance normalization and binarized at different thresholds. Following SEVIR [veillette2020sevir], for radar output tasks, we have established thresholds at [16, 74, 133, 160, 181, 219]. GEOS-visible output tasks are assigned thresholds of [2000, 3200, 4400, 5600, 6800]. The GEOS-IR107 output tasks operate with thresholds set to [-6000, -4000, 0, 2000]. Lastly, the GEOS-IR069 output task employs thresholds of [-4000, -5000, -6000, -7000].

## Experimental results

Currently, there is no general weather foundation model that can comprehensively handle all the discussed weather understanding tasks simultaneously. Although many machine learning methods have been investigated for single tasks, they generally adopt different backbone networks and design strategies tailored to them. For a fair comparison, we have trained a series of baselines (i.e., single-task model) for each weather understanding task under a consistent training setup, including commonly used UNet [weatherUNet] and ViT [nguyen2023climax] networks. Notably, the purpose of this paper is not to achieve state-of-the-art performance on every task. We focus on examining whether a generalist foundation model can handle multiple complex weather understanding tasks and weather data modalities. Beyond quantitative performance results, we are more concerned with the prompt learning capabilities of the generalist foundation model and the generalization ability it brings.

**Main quantitative results.** WeatherGFM is compared with single-task UNet and ViT baselines across 10 weather understanding tasks using RMSE and CSI metrics. Key results:

| Task Category | Task | Best Model | WeatherGFM vs ViT |
|---|---|---|---|
| Weather SR | Satellite Spatial SR | WeatherGFM | RMSE 0.042 vs 0.047 |
| Weather SR | Radar Temporal SR | WeatherGFM | RMSE 0.327 vs 0.333 |
| Weather SR | Radar Spatial SR | ViT (RMSE) / WeatherGFM (CSI) | RMSE 0.121 vs 0.120, CSI/74 0.831 vs 0.830 |
| Forecasting | Satellite extrapolation | WeatherGFM | RMSE 0.347 vs 0.408 |
| Forecasting | Radar extrapolation | WeatherGFM | RMSE 0.467 vs 0.490 |
| Post-processing | Deblur | ViT (RMSE) / WeatherGFM (CSI/74) | RMSE 0.264 vs 0.163 |
| Translation | GOES2Radar | WeatherGFM | RMSE 0.436 vs 0.445 |
| Translation | GOES-IR2GOES-IR | ViT (RMSE) / WeatherGFM (CSI) | Mixed results |
| Translation | GOES-IR2GOES-Visible | WeatherGFM | RMSE 0.439 vs 0.448 |
| Translation | GOES2POES-NO2 | WeatherGFM | RMSE 0.302 vs 0.549 |

**Prompt standard deviation across 20 prompts:**

| | GOES2Radar | Radar extrapolation | GOES-IR2GOES-IR | Radar Spatial SR | Radar Temporal SR | Deblur |
|---|---|---|---|---|---|---|
| Avg. RMSE | 0.0087 | 0.0012 | 0.0481 | 0.0001 | 0.0002 | 0.0016 |
| Avg. CSI | 0.0187 | 0.0201 | 0.0284 | 0.0006 | 0.0010 | 0.0047 |

[IMAGE: Case studies of WeatherGFM with different prompts in the radar extrapolation task (prompt_effect2.pdf)]

**Weather Generalist foundation model can achieve strong universal capabilities.** Our WeatherGFM, equipped with a straightforward ViT backbone, shows impressive performance and adaptability in ten weather understanding tasks. It is not only capable of conducting weather forecasting and super-resolution tasks but is also proficient in dealing with weather image translation and post-processing tasks. Overall, our WeatherGFM achieves promising performance on a diversity of weather understanding tasks.

**Weather Generalist foundation model outperforms the performance of the single-task model.** Our WeatherGFM achieves results that outperform the baseline in weather forecasting, weather super-resolution, and image translation tasks. For instance, in radar extrapolation tasks, our WeatherGFM with universal ViT-based model outperforms the single-task ViT model. This indicates that a unified approach to weather understanding tasks can potentially break the performance upperbound of single-task models.

**In-context learning can generate correct outputs across a variety of data modalities and tasks.** Our WeatherGFM effectively carries out a wide array of weather understanding tasks on multi-modal weather data. In practical scenarios, weather forecasting and weather image transformation represent two substantially different tasks due to differences in temporal modalities. Despite their intricacies, our WeatherGFM with in-context learning can successfully recognize distinct task types, highlighting its significant generalization capacity.

[IMAGE: Visual results of the weather understanding tasks by WeatherGFM (VS_v2.pdf)]

## Ablation Studies and Explorations

**Exploration of different task prompts.** To investigate the impact of various visual prompts on quantitative performance, we randomly select 20 meteorological prompts for each task and calculate their quantitative metrics on the test set. We note that weather super-resolution tasks are minimally affected by the randomness of weather prompts, whereas weather forecasting tasks and image transformation tasks exhibit more significant variability, reaching approximately 0.02 in CSI. For certain weather events, employing different prompts yields more precise outputs. This indicates that our method can comprehend specific weather cases based on weather prompts rather than being a black box model incapable of interactive operations.

**Exploration on out-of-distribution tasks.** To evaluate the generalization ability of our WeatherGFM, we have devised a variety of out-of-distribution (OOD) tasks that were not encountered during the training phase, including GEOS-IR107 extrapolation, weather image translation GEOS-IR107 to GEOS-IR069, weather temporal SR at 15 minutes and GEOS-visible satellite extrapolation. Our WeatherGFM generates correct outputs for the first three tasks, which are similar to the training distribution. However, the model encounters difficulties with the more challenging task of multiple-modal satellite spatial SR, where its outputs fail to provide effective meteorological information. These OOD tests demonstrate the model's ability to identify tasks outside the training distribution from new prompts, showcasing a degree of generalization.

[IMAGE: Visual results of WeatherGFM on OOD tasks (oodtest.pdf)]

**Scaling law of weather foundation model.** To evaluate the impact of data and model scale on performance, we compared single-task models, the base version of our WeatherGFM, and its large version. We established a baseline using a 30M parameter ViT under a single-task with 0.5 million samples. Subsequently, in a multi-task setting with 4 million samples, our model was configured with a base version of 110M and a large version of 330M parameters. Improvements in performance on various tasks are achieved with the increase of model and data scale. In specific tasks like radar super-resolution, we observe that scaling up both the data and the model is essential for performance gains.

[IMAGE: The effect of model sizes on CSI performance (scaling_law_csi.pdf)]

# Conclusion

We introduce the first weather generalist foundation model, WeatherGFM. By employing a unified representation for multiple weather understanding tasks and a multi-modal prompt design, our WeatherGFM skillfully addresses various tasks, such as weather forecasting, super-resolution, image translation, and post-processing through in-context learning. We conduct comprehensive explorations of the model's adaptability for various tasks and its generalization capabilities to unseen tasks, and its scaling law at the data and model size. This study will facilitate the development of future large-scale generalist weather and climate foundation models.

# Appendix

## Details of Weather Understanding Tasks

**Weather forecasting.** Radar echo extrapolation aims to forecast data for the subsequent 1-2 hours utilizing observations from past moments [gao2024prediff]. This task, similar to precipitation nowcasting, plays a significant role in predicting local weather conditions. It can directly impact traffic plans, disaster warnings, and energy management. Likewise, meteorological satellite image extrapolation is crucial for monitoring and analyzing meteorological conditions. Based on the SEVIR dataset, we consider two weather forecasting tasks: radar echo extrapolation [gong2024cascast] and satellite image extrapolation [satellite_extrapolation]. Our weather prediction tasks incorporate observations from the hour before (0, 30, 60, and 90 minutes past) and the hour ahead (120 and 180 minutes into the future) for both radar and satellite IR-069 extrapolation. Consequently, for this task, the SEVIR data was extracted and processed to generate 135,696 sequences for training, along with an independent set of 1,200 sequences to validate/test the fitted model.

**Weather super-resolution (SR).** Weather spatial super-resolution task [veillette2020sevir] generates a high-resolution image from a low-resolution (LR) image, while temporal super-resolution predicts a high-resolution (HR) image based on two consecutive observed input images. We take into consideration three weather super-resolution tasks: spatial SR for satellite IR-069, spatial SR for radar VIL, and temporal SR for radar VIL with a one-hour interval. We utilize the SEVIR dataset as the source of the HR image. To obtain the LR image, we employ the "Bicubic" interpolation approach, which is commonly used in vision image SR. In the context of meteorology, this is analogous to statistical downscaling [weatherSR_deepsd]. Specifically, for the VIL image, given that its original resolution is 384x384, we resize it to 256x256 to serve as the HR image and resize it to 64x64 to function as the LR image, thereby implementing a 4x super-resolution task. For the IR-069 image, since its original image has a resolution of 196x196, we resize it to 256x256 to be the HR image and resize it to 64x64 to be the LR image, thus carrying out a 3x super-resolution task. For each spatial SR for satellite tasks, the SEVIR data was extracted and processed to yield 542,784 images for training, along with an independent set of 4,800 images for validating/testing. For the weather temporal SR, we use the radar VIL image at 1 hour (0 and 60 minutes) as the input to predict the radar VIL image at 30 minutes. For the temporal SR task, the SEVIR data was further extracted and processed to generate 407,088 sequences for training, along with an independent set of 3,600 sequences to validate/test the fitted model.

**Weather image translation.** Weather image translation involves converting observation data (e.g., satellite data) to a desired weather image [veillette2020sevir]. For example, depictions of storms obtained from weather radar are extremely important. However, most areas of the world do not have access to ground-based radar. It is useful for generating weather radar images of storm depictions from satellite observation [veillette2020sevir]. We consider three weather image translation tasks based on SEVIR dataset: translate geostationary IR-069 to geostationary IR-107 data (GEOS-IR2GEOS-IR), geostationary IR-069 to geostationary Visible data (GEOS-IR2GEOS-Vis), translate geostationary IR-069 and IR-107 to radar VIL data (GEOS-IR2Radar). In addition, we add a weather image translation task for environment monitoring: Translate geostationary NO2 data to polar-orbiting satellites NO2 data (GEOS2POES-NO2) based on POMINO-TROPOMI product [NO2translation]. For the image translation tasks based on SEVIR dataset, we split SEVIR into 542,784 training samples, 4,800 validation samples and 4,800 test samples.

**Weather post-processing.** Post-processing (e.g., bias correction) aims to minimize or eliminate systematic biases in model outputs and observational data, which emerge due to uncertainties in weather models and measurement errors. Various methods, including statistical, machine learning, and deep learning techniques, can be employed for post-processing, tailoring the approach based on the specific application and data characteristics. By minimizing or eliminating systematic biases, post-processing improves the quality and reliability of weather and climate data. In our experiment, we consider a classic post-processing task: Deblurring for radar VIL nowcasting. We employ the output of Earthformer and the corresponding high-quality image as a training sample. Deblurring aims to learn how to map from the output of Earthformer to the corresponding high-quality image.

## Implementation details and hyperparameters

The L1 loss is employed as the loss function. For optimization, the AdamW optimizer with a cosine learning rate scheduler is utilized. The base learning rate is 1e-4. The batch size is 20 and the accumulation gradient iterations are 4. We use 16 Nvidia A100 GPUs for training. A total of 50 epochs are executed. We leverage fp16 floating point precision in our model.

**Default hyperparameters of WeatherGFM:**

| Hyperparameter | Meaning | Large | Base |
|---|---|---|---|
| p | Patch size | 16 | 16 |
| Encoder dimension | Encoder Embedding dimension | 1024 | 768 |
| Decoder dimension | Decoder Embedding dimension | 512 | 512 |
| Encoder depth | Number of Encoder blocks | 24 | 12 |
| Decoder depth | Number of Decoder blocks | 8 | 8 |
| Encoder Heads | Encoder's attention heads | 16 | 12 |
| Decoder Heads | Decoder's attention heads | 16 | 16 |
| MLP ratio | Hidden dimension of MLP layer in ViT block | 4 | 4 |
| Masked ratio | Percentage of the masked target data | 75% | 75% |

**ViT Hyperparameters** (from [beyer2022better]):

| Hyperparameter | Meaning | Value |
|---|---|---|
| p | Patch size | 16 |
| Dimension | Embedding dimension | 512 |
| Depth | Number of Encoder blocks | 16 |
| Heads | Attention heads | 8 |
| MLP dim | MLP hidden dimension | 1024 |

**UNet Hyperparameters:**

| Hyperparameter | Meaning | Value |
|---|---|---|
| Padding size | Padding size of each convolution layer | 1 |
| Kernel size | Kernel size of each convolution layer | 3 |
| Stride | Stride of each convolution layer | 1 |
| Channel multiplications | Output channels for Down and Up blocks | [1, 2, 4, 8, 8] |
| Blocks | Number of blocks | 3 |
| Use attention | If use attention in Down and Up blocks | False |
| Dropout | Dropout rate | 0 |
| Inner channel | Number of channels in intermediate layers | 64 |

## Extendability

To assess the generalizability of our framework, we also utilize the ERA5 dataset [hersbach2020era5]. With the introduction of new meteorological variables, we add a new patch embedding layer for each of these variables to the original model. Following the approach of ClimaX [nguyen2023climax], we use a cross-attention module to aggregate all variable embeddings into a single vector. This allows us to handle the task using the single-modal mode in WeatherGFM. Specifically, we employ an MLP layer to align the embeddings for this task. In line with ClimaX [nguyen2023climax], we use 48 ECMWF (European Centre for Medium-Range Weather Forecasts) variables as input and evaluate the performance of WeatherGFM using the temperature at 2 meters above ground (T2m). We consider seven lead times: 6 hours and 1, 3, 5, 7 days, covering a range from nowcasting to short- and medium-range forecasting. Instead of training separate models for each target variable, our WeatherGFM is trained once to predict all variables across all lead times simultaneously. During fine-tuning, we randomize the lead time from 6 hours to 7 days.

**Comparison of RMSE and ACC on T2m variable across different lead times:**

| Lead Time (hr) | IFS RMSE | ClimaX RMSE | WeatherGFM RMSE | IFS ACC | ClimaX ACC | WeatherGFM ACC |
|---|---|---|---|---|---|---|
| 6 | **0.97** | 1.11 | 1.08 | **0.99** | 0.98 | 0.98 |
| 24 | **1.02** | 1.19 | 1.23 | **0.99** | 0.97 | 0.97 |
| 72 | **1.30** | 1.47 | 1.56 | **0.98** | 0.96 | 0.96 |
| 120 | 1.71 | 1.83 | **1.68** | **0.96** | 0.94 | 0.95 |
| 168 | 2.23 | 2.17 | **1.76** | 0.93 | 0.91 | **0.94** |

ClimaX views predicting at each lead time as a separate task and fine-tunes a separate model for every individual task. In contrast, WeatherGFM utilizes a single model to deal with all of these tasks. The Climax method trained for 100 epochs using 80 V100 GPUs, while WeatherGFM trained for 20 epochs using 8 A100 GPUs, indicating much faster convergence.

## Effects of Multi-task Training

WeatherGFM-4tasks is trained on four tasks: Radar Temporal SR, GOES2GOES, GOES2Radar, and Radar Spatial SR. WeatherGFM-10tasks is trained on all ten tasks.

| Tasks | Model | RMSE | CSI/74 | CSI/133 | CSI/181 | CSI/219 |
|---|---|---|---|---|---|---|
| GOES2Radar | ViT-ST | 0.445 | 0.424 | 0.242 | 0.134 | 0.045 |
| | WeatherGFM-4Tasks | 0.460 | 0.443 | 0.263 | 0.166 | 0.059 |
| | WeatherGFM-10Tasks | 0.436 | 0.447 | 0.266 | 0.157 | 0.053 |
| Radar Temporal SR | ViT-ST | 0.333 | 0.585 | 0.366 | 0.215 | 0.063 |
| | WeatherGFM-4Tasks | 0.353 | 0.576 | 0.355 | 0.209 | 0.074 |
| | WeatherGFM-10Tasks | 0.327 | 0.597 | 0.376 | 0.217 | 0.073 |

The results indicate that for radar image generation tasks (Radar Temporal SR, GOES2Radar, and Radar Spatial SR), both WeatherGFM-4tasks and WeatherGFM-10tasks outperform ViT-ST. This is likely because most of the selected tasks are related to radar image generation. However, for the satellite image generation task (GOES2GOES), WeatherGFM-4tasks does not perform as well. This suggests that multi-task learning of similar tasks can enhance the model's performance on those tasks.

## More details of scaling law

Increasing the capacity of the model generally leads to better performance when the data size remains constant. In contrast, for smaller models, an increase in training data may result in poorer performance. We hypothesize that this could be due to the specificity of different tasks within the training data, which makes it more challenging for the model to fit effectively.

[IMAGE: RMSE performance comparison across different model configurations (scaling_law_rmse.pdf)]

## OOD Quantitative Results

| Tasks | Model | RMSE | CSI/-4K | CSI/0 | CSI/2K |
|---|---|---|---|---|---|
| IR107 Satellite extrapolation | UNet | 0.991 | 0.695 | 0.642 | 0.074 |
| | ViT | 0.413 | 0.899 | 0.776 | 0.245 |
| | WeatherGFM* | 0.389 | 0.903 | 0.774 | 0.244 |

| Tasks | Model | RMSE | CSI/16 | CSI/74 | CSI/133 | CSI/160 | CSI/181 | CSI/219 |
|---|---|---|---|---|---|---|---|---|
| Temporal SR at 15min | UNet | 0.676 | 0.211 | 0.627 | 0.428 | 0.351 | 0.262 | 0.083 |
| | ViT | 0.218 | 0.838 | 0.761 | 0.598 | 0.525 | 0.445 | 0.190 |
| | WeatherGFM* | 0.272 | 0.814 | 0.703 | 0.507 | 0.419 | 0.336 | 0.117 |

*WeatherGFM has not been trained or fine-tuned for these tasks and conducts generalized inference directly.
