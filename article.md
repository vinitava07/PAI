## PREDICTING AXILLARY LYMPH NODE METASTASIS IN EARLY BREAST CANCER USING DEEP LEARNING ON PRIMARY TUMOR BIOPSY SLIDES

Feng Xu 1 ∗†

Chuang Zhu 2 ∗†

Wenqi Tang 2 ∗

Yu Zhang 2

Jie Li 1

Ying Wang 3 Jun Liu 2

Hongchuan Jiang 1

Zhongyue Shi 3

Mulan Jin 2 †

1 Department of Breast Surgery, Beijing Chao-Yang Hospital, Beijing

2 School of Artificial Intelligence, Beijing University of Posts and Telecommunications, Beijing

3 Department of Pathology, Beijing Chao-Yang Hospital, Beijing

## ABSTRACT

Objectives: To develop and validate a deep learning (DL)-based primary tumor biopsy signature for predicting axillary lymph node (ALN) metastasis preoperatively in early breast cancer (EBC) patients with clinically negative ALN.

Methods: A total of 1,058 EBC patients with pathologically confirmed ALN status were enrolled from May 2010 to August 2020. A DL core-needle biopsy (DL-CNB) model was built on the attention-based multiple instance-learning (AMIL) framework to predict ALN status utilizing the DL features, which were extracted from the cancer areas of digitized whole-slide images (WSIs) of breast CNB specimens annotated by two pathologists. Accuracy, sensitivity, specificity, receiver operating characteristic (ROC) curves, and areas under the ROC curve (AUCs) were analyzed to evaluate our model.

Results: The best-performing DL-CNB model with VGG16\_BN as the feature extractor achieved an AUC of 0.816 (95% confidence interval (CI): 0.758, 0.865) in predicting positive ALN metastasis in the independent test cohort. Furthermore, our model incorporating the clinical data, which was called DL-CNB+C, yielded the best accuracy of 0.831 (95%CI: 0.775, 0.878), especially for patients younger than 50 years (AUC: 0.918, 95%CI: 0.825, 0.971). The interpretation of DL-CNB model showed that the top signatures most predictive of ALN metastasis were characterized by the nucleus features including density ( p = 0.015), circumference ( p = 0.009), circularity ( p = 0.010), and orientation ( p = 0.012).

Conclusion: Our study provides a novel DL-based biomarker on primary tumor CNB slides to predict the metastatic status of ALN preoperatively for patients with EBC. The codes and dataset are available at https://github.com/bupt-ai-cz/BALNMP .

Keywords deep learning · axillary lymph node metastasis · breast cancer · core-needle biopsy · whole-slide images

## 1 Introduction

Breast cancer (BC) has become the greatest threat to women's health worldwide [Siegel et al., 2019]. Clinically, identification of axillary lymph node (ALN) metastasis is important for evaluating the prognosis and guiding the treatment for BC patients [Ahmed et al., 2014]. Sentinel lymph node biopsy (SLNB) has gradually replaced ALN dissection (ALND) to identify ALN status, especially for early BC (EBC) patients with clinically negative lymph nodes. Although SLNB had the advantage of less invasiveness than ALND, SLNB still caused some complications such as

* These authors have contributed equally to this work.

† Correspondence: Feng Xu (drxufeng@mail.ccmu.edu.cn), Chuang Zhu (czhu@bupt.edu.cn), Mulan Jin (kinmokuran@163.com).

lymphedema, axillary seroma, paraesthesia, and impaired shoulder function [Kootstra et al., 2008, Wilke et al., 2006]. Moreover, SLNB has been considered a controversial procedure, owing to the availability of radionuclide tracers and the surgeon's experience [Manca et al., 2016, Hindié et al., 2011]. In fact, SLNB can be avoided if there are some reliable methods of preoperative prediction of ALN status for EBC patients.

Several studies intended to predict the ALN status by clinicopathological data and genetic testing score [Dihge et al., 2019, Shiino et al., 2019]. However, due to the relatively poor predictive values and high genetic testing costs, these methods are often limited. Recently, deep learning (DL) can perform high-throughput feature extraction on medical images and analyze the correlation between primary tumor features and ALN metastasis information. In a previous study, deep features extracted from conventional ultrasound and shear wave elastography (SWE) were used to predict ALN metastasis, presenting an area under the curve (AUC) of 0.796 in the test set [Zheng et al., 2020]. Nevertheless, SWE has not been integrated into routine clinical breast examinations in many hospitals. Another recent study demonstrated that the DL model based on diffusion-weighted imaging-magnetic resonance imaging (DWI-MRI) database of 172 patients achieved an AUC of 0.852 for preoperative prediction of ALN metastasis [Luo et al., 2018], but the small sample size enrolled could not be representative.

Currently, DL has enabled rapid advances in computational pathology [Campanella et al., 2019, Gu et al., 2018]. For example, DL methods have been applied to segment and classify glomeruli with different staining and various pathologic changes, thus achieving the automatic analysis of renal biopsies [Mei et al., 2020, Jiang et al., 2021]; meanwhile, DL-based automatic colonoscopy tissue segmentation and classification have shown promise for colorectal cancer detection [Zhu et al., 2021, Feng et al., 2020]; besides, the analysis of gastric carcinoma and precancerous status can also benefit from DL schemes [Iizuka et al., 2020, Song et al., 2020]. More recently, for the ALN metastasis detection, it is reported that DL algorithms on digital lymph node pathology images achieved better diagnostic efficiency of ALN metastasis than pathologists [Hu et al., 2021, Zhao et al., 2020]. In particular, the assistance of algorithm significantly increases the sensitivity of detection for ALN micro-metastases [Steiner et al., 2018]. In addition to diagnosis, several previous studies indicated that deep features based on whole-slide images (WSIs) of postoperative tumor samples potentially improved the prediction performance of lymph node metastasis in a variety of cancers [Zhao et al., 2020, Harmon et al., 2020]. So far, there is no relevant research on preoperatively predicting ALN metastasis based on WSIs of primary BC samples. In this study, we investigated a clinical data set of EBC patients treated by preoperative core-needle biopsy (CNB) to determine whether DL models based on primary tumor biopsy slides could help to refine the prediction of ALN metastasis.

Figure 1: Patient recruitment workflow.

<!-- image -->

## 2 Patients and Methods

## 2.1 Patients

On approval by the Institutional Ethical Committees of Beijing Chaoyang Hospital affiliated to Capital Medical University, we retrospectively analyzed data from EBC patients with clinically negative ALN from May 2010 to August 2020. Written consent was obtained from all patients and their families.

The detailed inclusion criteria were as follows: 1) patients with CNB pathologically confirmed primary invasive BC; 2) patients who underwent breast surgery with SLNB or ALND; 3) baseline clinicopathological data including age, tumor size, tumor type, ER/PR/HER-2 status, and the number of ALN metastasis were comprehensive; 4) complete concordance of molecular status was found between CNB and excision specimens; 5) no history of preoperative radiotherapy and chemotherapy; and 6) adequate volume of biopsy materials with three or more cores for each patient.

The exclusion criteria included the following: 1) patients with physically positive or imaging-positive ALN; 2) missing postoperative pathology information; 3) missing wax blocks and hematoxylin and eosin (H&amp;E) slices; and 4) low-quality H&amp;E slices or WSIs. The patient recruitment workflow is shown in Figure 1.

## 2.2 Deep Learning Model Development

To avoid the inter-observer heterogeneity, all available tumor regions in each CNB slide were examined and annotated by two independent and experienced pathologists blinded to all patient-related information. A WSI was classified into positive (N(+)) or negative (N0) using the proposed DL CNB (DL-CNB) model. Our DL-CNB model was constructed with the attention-based multiple-instance learning (MIL) approach [Ilse et al., 2018]. In MIL, each training sample was called a bag, which consisted of multiple instances [Das et al., 2018, Sudharshan et al., 2019, Couture et al., 2018] (each instance corresponds to an image patch of size 256 × 256 pixels). Different from the general fully supervised problem where each sample had a label, only the label of bags was available in MIL, and the goal of MIL was to predict the bag label by considering all included instances comprehensively. The whole algorithm pipeline comprised the following five steps:

- (1) Training data preparation (Figure 2a). For each raw WSI, amounts of non-overlapping square patches were first cropped from the selected tumor regions. Then each WSI could be represented as a bag with N randomly selected patches. To increase the training samples, M bags were built for each WSI. All M bags were labeled as positive if the slide is an ALN metastasis case, and vice versa. Note that we could add the clinical information of the slide to all the M constructed bags to involve more useful information for predicting, and in this situation, the developed model was called DL-CNB+C.
- (2) Feature extraction (left part of Figure 2b). N feature vectors were extracted for the N image instances in each bag by using a convolutional neural network (CNN) model. The performances of AlexNet [Krizhevsky et al., 2012], VGG16 [Simonyan and Zisserman, 2015] with batch norm (VGG16\_BN), ResNet50 [He et al., 2016], DenseNet121 [Huang et al., 2017], and Inception-v3 [Szegedy et al., 2016] were compared to find the best feature extractor. At this stage, the clinical data were also preprocessed for feature extraction. Concretely, the numerical properties in clinical data were standardizing by removing the mean and scaling to unit variance, thus eliminating the effect of data range and scale; furthermore, considering that there was no natural ordinal relationship between different values of the category attributes, the categorical properties in clinical data were encoded as the one-hot vectors, which could express different values equally.
- (3) MIL (right part of Figure 2b).The extracted N feature vectors of image instances were first processed by the max-pooling [Feng and Zhou, 2017, Pinheiro and Collobert, 2015, Zhu et al., 2017] and reshaping and then were passed to a two-layer fully connected (FC) layer. The N weight factors for the instances in the bag were thus obtained and then were further multiplied to the original feature vectors [Ilse et al., 2018] to adaptively adjust the effect of instance features. Finally, the weighted image feature vectors and the clinical features were fused by concatenation; due to the large difference of dimensions between image features and clinical features, the clinical features were copied 10 times for expansion. Then, the fused features were fed into the classifier, and the outputs and the ground truth labels were used to calculate the cross-entropy loss.
- (4) Model training and testing. We randomly divided the WSIs into training cohort and independent test cohort with the ratio of 4:1 and randomly selected 25% of the training cohort as the validation cohort. We used Adam optimizer with learning rate 1e-4 to update the model parameters and weight decay 1e-3 for regularization. In the training phase, we used the cosine annealing warm restarts strategy to adjust the learning rate [Loshchilov and Hutter, 2017]. In the testing phase, the ALN status is predicted by aggregating the model outputs of all bags from the same slide (Figure 2c).

The deep learning models are available at https://github.com/bupt-ai-cz/BALNMP .

Figure 2: The overall pipeline of the deep learning core-needle biopsy incorporating the clinical data (DL-CNB+C) model to predict axillary lymph node (ALN) status between N0 and N(+). (a): Multiple training bags were built based on clinical data and the cropped patches from the selected tumor regions of each core-needle biopsy (CNB) whole-slide image (WSI). (b): DL-CNB+C model training process included two phases of feature extraction and multiple-instance learning (MIL), and finally the weighted features fused with clinical features were used to predict classification probabilities and calculate the cross-entropy loss. (c): The predicted probabilities of each bag from a raw CNB WSI were merged to guide the final ALN status classification between N0 and N(+).

<!-- image -->

## 2.3 Visualization of Salient Regions From Deep Learning Core-Needle Biopsy Model

We visualized the important regions that were more associated with metastatic status. After the processing of attentionbased MIL pooling, the weights of different patches can be obtained, and the corresponding feature maps were then weighted together in the following FC layers to conduct ALN status prediction. With the attention weights, we created a heat map to visualize the important salient regions in each WSI.

## 2.4 Interpretability of Deep Learning Core-Needle Biopsy Model With Nucleus Features

Interpretability of DL-CNB model with nucleus features was performed to study the contribution of different nucleus morphological characteristics in the prediction of lymph node metastasis [Mueller et al., 2016, Radhakrishnan et al., 2017]. Multiple specially designed nucleus features were firstly extracted for each WSI, and these features together formed a training bag. With the constructed feature bags, the proposed DL-CNB model was re-trained. The weights of different features (instances) can be obtained based on the attention-based MIL pooling, and thus the contribution of different features was yielded. The specific process is described in Figure 3.

Figure 3: Overview on interpretability methods of deep learning core-needle biopsy (DL-CNB) model based on nucleus morphometric features. (a): The selected tumor regions of each whole-slide image (WSI) was cropped into patches. (b): For each patch, we processed nucleus segmentation (a weakly supervised segmentation framework was applied to obtain the nucleus), defined multiple nucleus morphometric features (such as major axis, minor axis, area, orientation, circumference, density, circularity, and rectangularity, which are denoted as f , f 1 2 , f 3 , ..., f n ), and extracted n feature parameters correspondingly. (c): All n kinds of feature parameters from a WSI were quantized into n distribution histograms and saved to n feature matrices ( m ,m ,m ,..., m 1 2 3 n ). (d) The matrices from a WSI were considered as instances of a bag and served as the input of DL-CNB model; the re-trained DL-CNB model could generate scores of features (instances) in the bag, which represented the weight of each feature in pathological diagnosis.

<!-- image -->

## 2.5 Statistical Analysis

The logistic regression was used to predict ALN status by clinical data only model. The clinical difference of N0 and N(+) was compared by using the Mann-Whitney U test and chi-square test. The AUCs of different methods were compared by using Delong et al. [DeLong et al., 1988]. The other measurements like accuracy (ACC), sensitivity (SENS), specificity (SPEC), positive predictive value (PPV), and negative predictive value (NPV) were also used to estimate the model performance. All the statistics were two-sided, and a p -value less than 0.05 was considered statistically significant. All statistical analyses were performed by MedCalc software (V 19.6.1; 2020 MedCalc Software bvba, Mariakerke, Belgium), Python 3.7, and SPSS 24.0 (IBM, Armonk, NY, USA).

## 3 Results

## 3.1 Clinical Characteristics

A total of 1,058 patients with EBC were enrolled for analysis. Among them, 957 (90.5%) patients had invasive ductal carcinomas, 25 (2.4%) patients had invasive lobular carcinomas, and 76 (7.1%) patients had other types. There were 840 patients in the training cohort and 218 patients in the independent test cohort after all WSIs were randomly divided by using N0 as the negative reference standard and others as the positive. The average patient age was 57.6 years (range, 26-90 years) for the training and validation sets and 56.7 years (range, 22-87 years) for the test set. The mean ultrasound tumor size was 2.23 cm (range, 0.5-4.5 cm). A total of 556 patients (52.6%) had T1 tumors, while 502 patients (47.4%) had T2 tumors. According to the results of SLNB or ALND, positive lymph nodes were found in 403 patients. Among them, 210 patients (52.1%) had one or two positive lymph nodes (N+(1-2)), and 193 patients (47.9%) had three or more positive lymph nodes (N+( ≥ 3)). As shown in Table 1, there was no significant difference between the detailed characteristics of the training and independent test cohorts (all p ≥ 0.05).

## 3.2 Convolutional Neural Network Model Selection

The detailed results are summarized in supplementary Table 1. Based on the overall analysis, VGG16\_BN model pretrained on ImageNet [Deng et al., 2009] provided the best performance in the validation cohort and the independent test cohort (AUC: 0.808, 0.816), compared with AlexNet (AUC: 0.764, 0.780), ResNet50 (AUC: 0.644, 0.607), DenseNet121 (AUC: 0.714, 0.739), and Inception-v3 (AUC: 0.753, 0.762). Furthermore, considering other metrics, VGG16\_BN achieved the best ACC, SPEC, and PPV in the independent test cohort. VGG16\_BN consisted of (convolution layer, batch normalization layer, and Rectified Linear Unit (ReLU)) as the basic block where ReLU played a role of activation function to provide the non-linear capability; and max-pooling layers were inserted between basic blocks for downsampling; besides, there was an adaptive average pooling layer at the end of VGG16\_BN for obtaining features with a fixed size. The details of VGG16\_BN are described in supplementary Table 2.

## 3.3 Predictive Value of Deep Learning Core-Needle Biopsy Incorporating the Clinical Data Model Between N0 and N(+)

In the training cohort, DL-CNB+C achieved an AUC of 0.878, while DL-CNB and classification by clinical data only model achieved AUCs of 0.901 and 0.661, respectively. And in the validation cohort, the DL-CNB+C model achieved an AUC of 0.823, which was higher than an AUC of 0.808 obtained by DL-CNB only and an AUC of 0.709 obtained by classification by clinical data.

In the independent test cohort, the DL-CNB+C model still achieved the highest AUC of 0.831, which was better than the AUC of DL-CNB only (AUC: 0.816, p = 0.453) and classification by clinical data only (AUC: 0.613, p ≤ 0.0001). The ACC, SENS, and NPV of DL-CNB+C were also better than those of other methods. The detailed statistical results are summarized in Table 2, and its corresponding receiver operating characteristics (ROCs) are shown in Figure 4.

We further divided N(+) into low metastatic potential (N + (1-2)) and high metastatic potential (N + ( ≥ 3)) according to the number of ALN metastasis. Adopting N0 as the negative reference standard, the combined model showed better discriminating ability between N0 and N + (1-2) (AUC: 0.878) and between N0 and N + ( ≥ 3) (AUC: 0.838).

The detailed statistical results are summarized in supplementary Table 3 and supplementary Table 4, and the corresponding ROCs are shown in supplementary Figure 1 and supplementary Figure 2.

Table 1: Patient and tumor characteristics.

| Characteristics           |                            | All patients   | Training       | Test           | p     |
|---------------------------|----------------------------|----------------|----------------|----------------|-------|
| Number                    |                            | 1058           | 840 (80%)      | 218 (20%)      |       |
| Age, mean ± SD, years     |                            | 57.58 ± 12.523 | 57.80 ± 12.481 | 56.72 ± 12.674 | 0.344 |
| Tumor size, mean ± SD, cm |                            | 2.234 ± 0.8623 | 2.228 ± 0.8516 | 2.256 ± 0.9040 | 0.898 |
| Number of LNM, mean ± SD  |                            | 1.20 ± 2.081   | 1.20 ± 2.095   | 1.20 ± 2.033   | 0.847 |
| Tumor type                | Invasive ductal carcinoma  | 957            | 760 (90.5%)    | 197 (90.4%)    | 0.812 |
|                           | Invasive lobular carcinoma | 25             | 20 (2.4%)      | 5 (2.3%)       |       |
|                           | Other types                | 76             | 60 (78.9%)     | 16 (21.1%)     |       |
| T stage                   | T1                         | 556            | 435 (51.8%)    | 121 (55.5%)    | 0.327 |
|                           | T2                         | 502            | 405 (48.2%)    | 97 (44.5%)     |       |
| ER                        | Positive                   | 831            | 665 (79.2%)    | 166 (76.1%)    | 0.333 |
|                           | Negative                   | 227            | 175 (20.8%)    | 52 (23.9%)     |       |
| PR                        | Positive                   | 790            | 633 (75.4%)    | 157 (72.0%)    | 0.312 |
|                           | Negative                   | 268            | 207 (24.6%)    | 61 (28.0%)     |       |
| HER2                      | Positive                   | 277            | 217 (25.8%)    | 60 (27.5%)     | 0.613 |
|                           | Negative                   | 781            | 623 (74.2%)    | 158 (72.5%)    |       |
| Molecular subtype         | Luminal A                  | 288            | 223 (26.5%)    | 65 (29.8%)     | 0.556 |
|                           | Luminal B                  | 372            | 304 (36.2%)    | 68 (31.2%)     |       |
|                           | Triple negative            | 125            | 99 (11.8%)     | 26 (11.9%)     |       |
|                           | HER2(+)                    | 273            | 214 (25.5%)    | 59 (27.1%)     |       |
| LNM                       | Yes                        | 403            | 521 (62.0%)    | 134 (61.5%)    | 0.880 |
|                           | No                         | 655            | 319 (38.0%)    | 84 (38.5%)     |       |

Qualitative variables are in n (%), and quantitative variables are in mean ±

SD, when appropriate.

SD, standard deviation; ER, estrogen receptor; PR, progesterone receptor; HER-2, human epidermal growth factor receptor-2; LNM, lymph node metastasis.

## 3.4 Predictive Value of Deep Learning Core-Needle Biopsy Incorporating the Clinical Data Model Among N0, N + (1-2) and N + ( ≥ 3)

The overall AUC of multi-classification in the independent test cohort based on DL-CNB+C model was 0.791; there existed the highest precision and recall of 0.747 and 0.947, respectively, in N0; there existed the precision and recall of 0.556 and 0.400 in N + (1-2); and there existed the precision and recall of 0.375 and 0.162 in N + ( ≥ 3). The confusion matrix under the classification threshold of 0.5 is shown in Figure 5. According to the results, the model performed well in differentiating the N0 group while showing poor diagnostic efficacy in the other two groups.

## 3.5 Subgroup Analysis of Deep Learning Core-Needle Biopsy Incorporating the Clinical Data Model

Furthermore, we analyzed the measurement results of the different subgroups in the independent test cohort of predicting ALN status between N0 and N(+) by the DL-CNB+C model. The detailed statistical results are summarized

Table 2: The performance in prediction of ALN status (N0 vs. N(+)).

| Methods        |     | AUC                                                         | ACC (%)                                          | SENS (%)                                           | SPEC (%)                                         | PPV (%)                                                        | NPV (%)                                                        |
|----------------|-----|-------------------------------------------------------------|--------------------------------------------------|----------------------------------------------------|--------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------|
| Clinical data  | T V | 0.661 [0.622, 0.698] 0.709 [0.643, 0.770] 0.613 a,b [0.545, | 64.13 [60.24, 67.88] 67.62 [60.84, 61.93 [55.12, | 64.58 [58.17, 70.63] [54.29, 76.13] [38.89, 61.11] | 63.85 [58.86, 68.62] 68.70 [60.02, 69.40 [60.86, | 52.36 [48.32, 56.38] 55.91 [48.46, 63.11] 50.60 [42.34, 58.83] | 74.55 [70.85, 77.92] 76.92 [70.62, 82.22] 68.89 [63.49, 73.82] |
| only           |     |                                                             | 73.90]                                           | 65.82                                              | 76.52]                                           |                                                                |                                                                |
| only           | I-T | 0.678]                                                      | 68.40]                                           | 50.00                                              | 77.07]                                           |                                                                |                                                                |
| DL-CNB model   | T   | 0.901 [0.875, 0.923]                                        | 80.32 [76.99, 83.35]                             | 94.17 [90.41, 96.77]                               | 71.79 [67.05, 76.21]                             | 67.26 [63.61, 70.71]                                           | 95.24 [92.30, 97.09]                                           |
| DL-CNB model   | V   | 0.808 [0.748, 0.859]                                        | 72.86 [66.31, 78.75]                             | 77.22 [66.40, 85.90]                               | 70.23 [61.62, 77.90]                             | 61.00 [53.95, 67.62]                                           | 83.64 [77.04, 88.62]                                           |
| DL-CNB model   | I-T | 0.816 c [0.758, 0.865]                                      | 74.77 [68.46, 80.39]                             | 80.95 [70.92, 88.70]                               | 70.90 [62.43, 78.42]                             | 63.55 [56.76, 69.84]                                           | 85.59 [79.04, 90.34]                                           |
| DL-CNB+C model | T   | 0.878 [0.622, 0.698]                                        | 76.51 [73.00, 79.77]                             | 93.33 [89.40, 96.14]                               | 66.15 [61.22, 70.84]                             | 62.92 [59.53, 66.19]                                           | 94.16 [90.90, 96.30]                                           |
|                | V   | 0.823 [0.765, 0.872]                                        | 75.71 [69.34, 81.35]                             | 74.68 [63.64, 83.80]                               | 76.34 [68.12, 83.32]                             | 65.56 [57.69, 72.65]                                           | 83.33 [77.19, 88.08]                                           |
|                | I-T | 0.831 [0.775, 0.878]                                        | 75.69 [69.44, 81.23]                             | 89.29 [80.63, 94.98]                               | 67.16 [58.53, 75.03]                             | 63.03 [56.96, 68.71]                                           | 90.91 [84.21, 94.94]                                           |

- 95% confidence intervals are included in brackets.
- AUC, area under the receiver operating characteristic curve; ACC, accuracy; SENS, sensitivity; SPEC, specificity; PPV, positive predictive value; NPV, negative predictive value.
- T, training cohort ( n = 630); V, validation cohort ( n = 210); I-T, independent test cohort ( n = 218).
- ALN, axillary lymph node; DL-CNB+C, deep learning core-needle biopsy incorporating the clinical data.

a

Indicates

p

&lt; 0.0001, Delong et al. in comparison with DL-CNB model in independent test cohort.

- b Indicates p &lt; 0.0001, Delong et al. in comparison with DL-CNB+C model in independent test cohort.
- c Indicates p = 0.4532, Delong et al. in comparison with DL-CNB+C model in independent test cohort.

## ROC Comparision

Figure 4: Comparison of receiver operating characteristic (ROC) curves between different models for predicting disease-free axilla (N0) and heavy metastatic burden of axillary disease (N(+)). Numbers in parentheses are areas under the receiver operating characteristic curve (AUCs).

<!-- image -->

in supplementary Table 5. In the independent test cohort, compared with an AUC of 0.794 (95%CI: 0.720, 0.855) in the subgroup of age &gt; 50, there existed better performance in the subgroup of age ≤ 50 with an AUC of 0.918 (95%CI: 0.825, 0.971, p = 0.015). There were no significant differences regarding other subgroups of ER(+) vs. ER(-) ( p = 0.125), PR(+) vs. PR(-) ( p = 0.659), HER-2(+) vs. HER-2(-) ( p = 0.524), and T1 vs. T2 stage ( p = 0.743) between N0 and N(+).

## 3.6 Interpretability of Deep Learning Core-Needle Biopsy Model

To investigate the interpretability of the DL-CNB, we conducted two studies for digging the correlation factors of ALN status prediction. In the first study, we adopted the attention-based MIL pooling to find the important regions that contributing to the prediction. The heat map in Figure 6a highlights the red patches as the important regions. Although the obtained important areas can provide some clues to the diagnosis of DL-CNB model, it is not clear that the model makes decisions based on what features of the tumor area.

In the second study, we specially designed and extracted multiple nucleus features for each WSI. The weights of different features were then obtained based on the same attention-based MIL pooling in our DL-CNB. The weights highlighted the nucleus features that were most relevant to the ALN status prediction of each WSI. We found that the WSI of N(+) group had higher nuclear density ( p = 0.015) and orientation ( p = 0.012) but lower circumference

Figure 5: The confusion matrix of predicting axillary lymph node (ALN) status between disease-free axilla (N0), low metastatic burden of axillary disease (N + (1-2)), and heavy metastatic burden of axillary disease (N + ( ≥ 3)).

<!-- image -->

( p = 0.009), circularity ( p = 0.010), and area ( p = 0.024) compared with N0 group (Figure 6b and Figure 6c). There were no significant differences in other nucleus features including major axis ( p = 0.083), minor axis ( p = 0.065), and rectangularity ( p = 0.149) between N0 and N(+).

## 4 Discussion

In most previous studies, DL signatures of ALN metastases were based on medical images such as ultrasound, CT, and MRI [Luo et al., 2018, Zhou et al., 2020, Yang et al., 2020]. However, since many patients had undergone CNB at the time of imaging examination, and the reactive changes such as needle path in the tumor would result in the predictive inaccuracy of imaging information. This study focused on preoperative CNB WSI, which also played an important role in BC management and has been increasingly performed in clinical practice. Preoperative CNB can provide not only the histopathological diagnosis of BC but also the molecular status including ER/PR/HER-2 status, which is associated with ALN metastasis [Calhoun and Anderson, 2014]. Otherwise, the morphological features of tumor cells can be visualized on CNB WSI. Therefore, primary tumor biopsy WSI as a complementary imaging tool has the potential for ALN metastasis prediction. To the best of our knowledge, this is the first study to apply the DL-based histopathological features extracted from primary tumor WSIs for ALN prediction analysis.

Here, the best-performing DL-CNB model yielded satisfactory predictions with an AUC of 0.816, a SENS of 81.0%, and a SPEC of 70.9% on the test set, which had superior predictive capability as compared with clinical data alone. Furthermore, unlike other combined models incorporating clinical data [Dihge et al., 2019, Zheng et al., 2020], the DL-CNB+C model slightly improved the ACC to 0.831, which showed that our results were mainly derived from the contribution of DL-CNB model. In addition, during the subgroup analysis stratified by patient's age, our DL-CNB+C model achieved an AUC of 0.918 for patients younger than 50 years, indicating that age was the critical factor in predicting ALN status. Regarding the number of ALN metastasis, the DL-CNB+C model showed better discriminating ability between N0 an N+(1-2), and between N0 and N+( ≥ 3). However, the unfavorable discriminating ability was found between N+(1-2) and N+( ≥ 3). This was consistent with the study of Zheng et al. [Zheng et al., 2020], who also reported poor efficacy between N+(1-2) and N+( ≥ 3), utilizing the DL radiomics model. In the future, further exploration of ALN staging prediction is needed.

Indeed, computer-assisted histopathological analysis can provide a more practical and objective outputAcs et al. [2020]. For example, different molecular subtypes [Jaber et al., 2020] and Oncotype DX risk score [Whitney et al., 2018] occurring in BC could be directly predicted from the H&amp;E slides. On the one hand, our DL model can provide significant information for risk stratification and axillary staging, thereby avoiding axillary surgery and reducing the complication and hospitalization costs. On the other hand, our results also highlight the development of algorithms based on artificial intelligence, which will reduce the labor intensity of pathologists. Similar approaches may be used to the pathology of other organs.

Figure 6: The interpretability of the deep learning core-needle biopsy (DL-CNB) model of two patients. (a, b) The heat maps and nuclear segmentation from core-needle biopsy (CNB) whole-slide images (WSIs) of the N0 and the N(+) separately, and the red regions show greater contribution to the final classification. (c) The statistical analysis of three nuclear characteristics most relevant to diagnosis of all patients.

<!-- image -->

In our study, we are first to quantitatively assess the role of nuclear disorder in predicting ALN metastasis in BC. Our finding is consistent with several recent studies that demonstrate the powerful predictive effect of nuclear disorder on patient survival [Lu et al., 2018, Lee et al., 2017]. Interestingly, the top predictive signatures that distinguished N0 from N(+) were characterized by the nucleus features including density, circumference, circularity, and orientation. We found that the WSI of N(+) had higher nuclear density and polarity but lower circularity, which was understandable since in the tumors with ALN metastasis, tumor cells became poorly differentiated as a result of rapid cell growth, encouraging the nuclei in these structures to form highly clustered and consistently metastatic patterns. Our results showed that nuanced patterns of nucleus density and orientation of tumor cells are important determinants of ALN metastasis.

There are some limitations in our study. First, the selection of regions of interest within each CNB slide required pathologist guidance. Future studies will explore more advanced methods for automatic segmentation of tumor regions. Second, this is a retrospective study, and prospective validation of our model in a large multicenter cohort of EBC patients is necessary to assess the clinical applicability of the biomarker. Third, recent evidence indicated that a set of features related to tumor-infiltrating lymphocytes (TILs) was found to be associated with positive LNs in bladder cancer [Harmon et al., 2020]. However, due to few TILs on breast CNB slides, we only selected sufficient tumor cells for the identification of salient regions rather than whole slides. Finally, we only chose H&amp;E stained images of CNB samples. The clinical utility of immunochemical stained images remains to be established as an interesting attempt.

## 5 Conclusion

In brief, we demonstrated that a novel DL-based biomarker on primary tumor CNB slides predicted ALN metastasis preoperatively for EBC patients with clinically negative ALN, especially for younger patients. Our methods could help to avoid unnecessary axillary surgery based on the widely collected H&amp;E-stained histopathology slides, thereby contributing to precision oncology treatment.

## Data availability statement

The original contributions presented in the study are included in the supplementary material. Further inquiries can be directed to the corresponding authors.

## Ethics statement

Written informed consent was obtained from the individual(s) for the publication of any potentially identifiable images or data included in this article.

## Author contributions

FX, CZ, JiL, YW, and MJ designed the study. CZ, WT, YZ, and JiL trained the model. FX, YW, ZS, JuL,and HJ collected the data. FX, WT, YZ, CZ, YW, MJ, and JuL analyzed and interpreted the data. FX, CZ, WT, YZ, and MJ prepared the manuscript. All authors contributed to the article and approved the submitted version.

## Funding

The work was supported by National Natural Science Foundation of China [No. 8197101438].

## References

Rebecca L Siegel, Kimberly D Miller, and Ahmedin Jemal. Cancer statistics, 2019. CA Cancer J. Clin. , 69(1):7-34, January 2019.

- Muneer Ahmed, Arnie D Purushotham, and Michael Douek. Novel techniques for sentinel lymph node biopsy in breast cancer: a systematic review. Lancet Oncol. , 15(8):e351-62, July 2014.
- Jan Kootstra, Josette E H M Hoekstra-Weebers, Hans Rietman, Jaap de Vries, Peter Baas, Jan H B Geertzen, and Harald J Hoekstra. Quality of life after sentinel lymph node biopsy or axillary lymph node dissection in stage I/II breast cancer patients: a prospective longitudinal study. Ann. Surg. Oncol. , 15(9):2533-2541, September 2008.

Lee Gravatt Wilke, Linda M McCall, Katherine E Posther, Pat W Whitworth, Douglas S Reintgen, A Marilyn Leitch, Sheryl G A Gabram, Anthony Lucci, Charles E Cox, Kelly K Hunt, James E Herndon, 2nd, and Armando E Giuliano. Surgical complications associated with sentinel lymph node biopsy: results from a prospective international cooperative group trial. Ann. Surg. Oncol. , 13(4):491-500, April 2006.

Gianpiero Manca, Domenico Rubello, Elisa Tardelli, Francesco Giammarile, Sara Mazzarri, Giuseppe Boni, Sotirios Chondrogiannis, Maria Cristina Marzola, Serena Chiacchio, Matteo Ghilli, Manuela Roncella, Duccio Volterrani, and Patrick M Colletti. Sentinel lymph node biopsy in breast cancer: Indications, contraindications, and controversies. Clin. Nucl. Med. , 41(2):126-133, February 2016.

Elif Hindié, David Groheux, Isabelle Brenot-Rossi, Domenico Rubello, Jean-Luc Moretti, and Marc Espié. The sentinel node procedure in breast cancer: nuclear medicine as the starting point. J. Nucl. Med. , 52(3):405-414, March 2011.

- L Dihge, J Vallon-Christersson, C Hegardt, and others. Prediction of lymph node metastasis in breast cancer by gene expression and clinicopathological models: development and validation within a population-based . . . . Clin. Cancer Res. , 2019.
- S Shiino, J Matsuzaki, A Shimomura, J Kawauchi, and others. Serum miRNA-based prediction of axillary lymph node metastasis in breast cancer. Clin. Cancer Res. , 2019.
- Xueyi Zheng, Zhao Yao, Yini Huang, Yanyan Yu, Yun Wang, Yubo Liu, Rushuang Mao, Fei Li, Yang Xiao, Yuanyuan Wang, Yixin Hu, Jinhua Yu, and Jianhua Zhou. Deep learning radiomics can predict axillary lymph node status in early-stage breast cancer. Nat. Commun. , 11(1):1236, March 2020.

| Jiaxiu Luo, Zhenyuan Ning, Shuixing Zhang, Qianjin Feng, and Yu Zhang. Bag of deep features for preoperative prediction of sentinel lymph node metastasis in breast cancer. Phys. Med. Biol. , 63(24):245014, December 2018.                                                                                                                                                                                                                                        |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Gabriele Campanella, Matthew G Hanna, Luke Geneslaw, Allen Miraflor, Vitor Werneck Krauss Silva, Klaus J Busam, Edi Brogi, Victor E Reuter, David S Klimstra, and Thomas J Fuchs. Clinical-grade computational pathology using weakly supervised deep learning on whole slide images. Nat. Med. , 25(8):1301-1309, August 2019.                                                                                                                                     |
| Feng Gu, Nikolay Burlutskiy, Mats Andersson, and Lena Kajland Wilén. Multi-resolution networks for semantic segmentation in whole slide images. In Computational Pathology and Ophthalmic Medical Image Analysis , pages 11-18. Springer International Publishing, 2018.                                                                                                                                                                                            |
| Ke Mei, Chuang Zhu, Lei Jiang, Jun Liu, and Yuanyuan Qiao. Cross-Stained segmentation from renal biopsy images using Multi-Level adversarial learning. In ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , pages 1424-1428. ieeexplore.ieee.org, May 2020.                                                                                                                                                    |
| Lei Jiang, Wenkai Chen, Bao Dong, Ke Mei, Chuang Zhu, Jun Liu, Meishun Cai, Yu Yan, Gongwei Wang, Li Zuo, and Hongxia Shi. A deep Learning-Based approach for glomeruli instance segmentation from multistained renal biopsy pathologic images. Am. J. Pathol. , 191(8):1431-1441, August 2021.                                                                                                                                                                     |
| Chuang Zhu, Ke Mei, Ting Peng, Yihao Luo, Jun Liu, Ying Wang, and Mulan Jin. Multi-level colonoscopy malignant tissue detection with adversarial CAC-UNet. Neurocomputing , 438:165-183, May 2021.                                                                                                                                                                                                                                                                  |
| Ruiwei Feng, Xuechen Liu, Jintai Chen, Danny Z Chen, Honghao Gao, and Jian Wu. A deep learning approach for colonoscopy pathology WSI analysis: Accurate segmentation and classification. IEEE J Biomed Health Inform , PP, November 2020.                                                                                                                                                                                                                          |
| Osamu Iizuka, Fahdi Kanavati, Kei Kato, Michael Rambeau, Koji Arihiro, and Masayuki Tsuneki. Deep learning models for histopathological classification of gastric and colonic epithelial tumours. Sci. Rep. , 10(1):1504, January 2020.                                                                                                                                                                                                                             |
| Zhigang Song, Shuangmei Zou, Weixun Zhou, Yong Huang, Liwei Shao, Jing Yuan, Xiangnan Gou, Wei Jin, Zhanbo Wang, Xin Chen, Xiaohui Ding, Jinhong Liu, Chunkai Yu, Calvin Ku, Cancheng Liu, Zhuo Sun, Gang Xu, Yuefeng Wang, Xiaoqing Zhang, Dandan Wang, Shuhao Wang, Wei Xu, Richard C Davis, and Huaiyin Shi. Clinically applicable histopathological diagnosis system for gastric cancer detection using deep learning. Nat. Commun. , 11(1): 4294, August 2020. |
| Yajie Hu, Feng Su, Kun Dong, Xinyu Wang, Xinya Zhao, Yumeng Jiang, Jianming Li, Jiafu Ji, and Yu Sun. Deep learning system for lymph node quantification and metastatic cancer identification from whole-slide pathology images. Gastric Cancer , 24(4):868-877, July 2021.                                                                                                                                                                                         |
| Yu Zhao, Fan Yang, Yuqi Fang, Hailing Liu, Niyun Zhou, Jun Zhang, Jiarui Sun, Sen Yang, Bjoern Menze, Xinjuan Fan, and Others. Predicting lymph node metastasis using histopathological images based on multiple instance learning with deep graph convolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 4837-4846. openaccess.thecvf.com, 2020.                                                                |
| David F Steiner, Robert MacDonald, Yun Liu, Peter Truszkowski, Jason D Hipp, Christopher Gammage, Florence Thng, Lily Peng, and Martin C Stumpe. Impact of deep learning assistance on the histopathologic review of lymph nodes for metastatic breast cancer. Am. J. Surg. Pathol. , 42(12):1636-1646, December 2018.                                                                                                                                              |
| Stephanie A Harmon, Thomas H Sanford, G Thomas Brown, Chris Yang, Sherif Mehralivand, Joseph M Jacob, Vladimir A Valera, Joanna H Shih, Piyush K Agarwal, Peter L Choyke, and Baris Turkbey. Multiresolution application of artificial intelligence in digital pathology for prediction of positive lymph nodes from primary tumors in bladder cancer. JCO Clin Cancer Inform , 4:367-382, April 2020.                                                              |
| Maximilian Ilse, Jakub Tomczak, and Max Welling. Attention-based deep multiple instance learning. In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning , volume 80 of Proceedings of Machine Learning Research , pages 2127-2136. PMLR, 2018.                                                                                                                                                           |
| Kausik Das, Sailesh Conjeti, Abhijit Guha Roy, Jyotirmoy Chatterjee, and Debdoot Sheet. Multiple instance learning of deep convolutional neural networks for breast histopathology whole slide classification. In 2018 IEEE 15th International Symposium on Biomedical Imaging (ISBI 2018) , pages 578-581. ieeexplore.ieee.org, April 2018.                                                                                                                        |
| P J Sudharshan, Caroline Petitjean, Fabio Spanhol, Luiz Eduardo Oliveira, Laurent Heutte, and Paul Honeine. Multiple instance learning for histopathological breast cancer image classification. Expert Syst. Appl. , 117:103-111, March 2019.                                                                                                                                                                                                                      |
| Heather D Couture, J S Marron, Charles M Perou, Melissa A Troester, and Marc Niethammer. Multiple instance learning for heterogeneous images: Training a CNN for histopathology. In Medical Image Computing and Computer Assisted Intervention - MICCAI 2018 , pages 254-262. Springer International Publishing, 2018.                                                                                                                                              |

| Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. Adv. Neural Inf. Process. Syst. , 25:1097-1105, 2012.                                                                                                                                                                       |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for Large-Scale image recognition. In Yoshua Bengio and Yann LeCun, editors, 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings , 2015.                                                      |
| Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 770-778. openaccess.thecvf.com, 2016.                                                                                                                  |
| Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 4700-4708. openaccess.thecvf.com, 2017.                                                                                                   |
| C Szegedy, V Vanhoucke, S Ioffe, and others. Rethinking the inception architecture for computer vision. Proceedings of the IEEE conference on computer vision and pattern recognition , 2016.                                                                                                                                                        |
| Ji Feng and Zhi-Hua Zhou. Deep MIML network. AAAI , 31(1), February 2017.                                                                                                                                                                                                                                                                            |
| Pedro O Pinheiro and Ronan Collobert. From image-level to pixel-level labeling with convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 1713-1721. openac- cess.thecvf.com, 2015.                                                                                                       |
| Wentao Zhu, Qi Lou, Yeeleng Scott Vang, and Xiaohui Xie. Deep multi-instance networks with sparse label assignment for whole mammogram classification. In Medical Image Computing and Computer Assisted Intervention - MICCAI 2017 , pages 603-611. Springer International Publishing, 2017.                                                         |
| Ilya Loshchilov and Frank Hutter. SGDR: stochastic gradient descent with warm restarts. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings OpenReview.net, 2017.                                                                                                |
| Jenna L Mueller, Jennifer E Gallagher, Rhea Chitalia, Marlee Krieger, Alaattin Erkanli, Rebecca MWillett, Joseph Geradts, and Nimmi Ramanujam. Rapid staining and imaging of subnuclear features to differentiate between malignant and benign breast tissues at a point-of-care setting. J. Cancer Res. Clin. Oncol. , 142(7):1475-1486, July 2016. |
| Adityanarayanan Radhakrishnan, Karthik Damodaran, Ali C Soylemezoglu, Caroline Uhler, and G V Shivashankar. Machine learning for nuclear Mechano-Morphometric biomarkers in cancer diagnosis. Sci. Rep. , 7(1):17946, December 2017.                                                                                                                 |
| E R DeLong, D MDeLong, and D L Clarke-Pearson. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. Biometrics , 44(3):837-845, September 1988.                                                                                                                                      |
| Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition , pages 248-255. ieeexplore.ieee.org, June 2009.                                                                                                      |
| Li-Qiang Zhou, Xing-Long Wu, Shu-Yan Huang, Ge-Ge Wu, Hua-Rong Ye, Qi Wei, Ling-Yun Bao, You-Bin Deng, Xing-Rui Li, Xin-Wu Cui, and Christoph F Dietrich. Lymph node metastasis prediction from primary breast cancer US images using deep learning. Radiology , 294(1):19-28, January 2020.                                                         |
| Xiaojun Yang, Lei Wu, Weitao Ye, Ke Zhao, Yingyi Wang, Weixiao Liu, Jiao Li, Hanxiao Li, Zaiyi Liu, and Changhong Liang. Deep learning signature based on staging CT for preoperative prediction of sentinel lymph node metastasis in breast cancer. Acad. Radiol. , 27(9):1226-1233, September 2020.                                                |
| Kristine E Calhoun and Benjamin O Anderson. Needle biopsy for breast cancer diagnosis: a quality metric for breast surgical practice. J. Clin. Oncol. , 32(21):2191-2192, July 2014.                                                                                                                                                                 |
| B Acs, MRantalainen, and J Hartman. Artificial intelligence as the next step towards precision pathology. J. Intern. Med. , 288(1):62-81, July 2020.                                                                                                                                                                                                 |
| Mustafa I Jaber, Bing Song, Clive Taylor, Charles J Vaske, Stephen C Benz, Shahrooz Rabizadeh, Patrick Soon-Shiong, and Christopher WSzeto. Adeep learning image-based intrinsic molecular subtype classifier of breast tumors reveals tumor heterogeneity that may affect survival. Breast Cancer Res. , 22(1):12, January 2020.                    |
| Jon Whitney, German Corredor, Andrew Janowczyk, Shridar Ganesan, Scott Doyle, John Tomaszewski, Michael Feldman, Hannah Gilmore, and Anant Madabhushi. Quantitative nuclear histomorphometry predicts oncotype DX risk categories for early stage ER+ breast cancer. BMC Cancer , 18(1):610, May 2018.                                               |
| Cheng Lu, David Romo-Bucheli, Xiangxue Wang, Andrew Janowczyk, Shridar Ganesan, Hannah Gilmore, David Rimm, and Anant Madabhushi. Nuclear shape and orientation features from H&E images predict survival in early-stage estrogen receptor-positive breast cancers. Lab. Invest. , 98(11):1438-1448, June 2018.                                      |

George Lee, Robert W Veltri, Guangjing Zhu, Sahirzeeshan Ali, Jonathan I Epstein, and Anant Madabhushi. Nuclear shape and architecture in benign fields predict biochemical recurrence in prostate cancer patients following radical prostatectomy: Preliminary findings. European Urology Focus , 3(4):457-466, October 2017.

## Supplementary material

Table 1: The performance comparison of different base models in prediction of ALN status (N0 vs. N(+)).

| Base models   |     | AUC                  | ACC (%)              | SENS (%)             | SPEC (%)             | PPV (%)              | NPV (%)              |
|---------------|-----|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|
| AlexNet       | T   | 0.909 [0.884, 0.930] | 82.70 [79.51, 85.57] | 88.33 [83.58, 92.11] | 79.23 [74.86, 83.15] | 72.35 [68.20, 76.16] | 91.69 [88.59, 94.01] |
|               | V   | 0.764 [0.700, 0.819] | 65.71 [58.87, 72.11] | 89.87 [81.02, 95.53] | 51.15 [42.26, 59.97] | 52.59 [47.84, 57.30] | 89.33 [80.96, 94.28] |
|               | I-T | 0.780 [0.719, 0.833] | 73.39 [67.01, 79.13] | 83.33 [73.62, 90.58] | 67.16 [58.53, 75.03] | 61.40 [55.08, 67.36] | 86.54 [79.71, 91.32] |
|               | T   | 0.912 [0.887, 0.933] | 85.71 [82.74, 88.35] | 85.83 [80.77, 89.99] | 85.64 [81.76, 88.97] | 78.63 [74.17, 82.50] | 90.76 [87.77, 93.08] |
| ResNet50      | V   | 0.644 [0.575, 0.709] | 59.52 [52.55, 66.22] | 70.89 [59.58, 80.57] | 52.67 [43.77, 61.45] | 47.46 [41.80, 53.19] | 75.00 [67.22, 81.44] |
|               | I-T | 0.607 [0.539, 0.673] | 58.72 [51.87, 65.32] | 66.67 [55.54, 76.58] | 53.73 [44.92, 62.38] | 47.46 [41.61, 53.37] | 72.00 [64.65, 78.33] |
|               | T   | 0.967 [0.949, 0.979] | 89.84 [87.21, 92.09] | 95.83 [92.47, 97.98] | 86.15 [82.32, 89.42] | 80.99 [76.85, 84.53] | 97.11 [94.82, 98.41] |
| DenseNet121   | V   | 0.714 [0.648, 0.774] | 68.57 [61.82, 74.79] | 73.42 [62.28, 82.73] | 65.65 [56.85, 73.72] | 56.31 [49.56, 62.84] | 80.37 [73.56, 85.77] |
|               | I-T | 0.739 [0.675, 0.796] | 69.27 [62.68, 75.32] | 85.71 [76.38, 92.39] | 58.96 [50.13, 67.37] | 56.69 [51.21, 62.02] | 86.81 [79.28, 91.89] |
|               | T   | 0.968 [0.951, 0.980] | 91.75 [89.32, 93.77] | 95.42 [91.95, 97.69] | 89.49 [86.01, 92.35] | 84.81 [80.68, 88.20] | 96.94 [94.68, 98.26] |
| Inception-v3  | V   | 0.753 [0.689, 0.810] | 70.48 [63.81, 76.55] | 67.09 [55.60, 77.25] | 72.52 [64.04, 79.95] | 59.55 [51.71, 66.93] | 78.51 [72.39, 83.59] |
|               | I-T | 0.762 [0.700, 0.817] | 71.10 [64.59, 77.02] | 85.71 [76.38, 92.39] | 61.94 [53.16, 70.18] | 58.54 [52.79, 64.06] | 87.37 [80.12, 92.23] |
|               | T   | 0.901 [0.875, 0.923] | 80.32 [76.99, 83.35] | 94.17 [90.41, 96.77] | 71.79 [67.05, 76.21] | 67.26 [63.61, 70.71] | 95.24 [92.30, 97.09] |
| VGG16_BN      | V   | 0.808 [0.748, 0.859] | 72.86 [66.31, 78.75] | 77.22 [66.40, 85.90] | 70.23 [61.62, 77.90] | 61.00 [53.95, 67.62] | 83.64 [77.04, 88.62] |
|               | I-T | 0.816 [0.758, 0.865] | 74.77 [68.46, 80.39] | 80.95 [70.92, 88.70] | 70.90 [62.43, 78.42] | 63.55 [56.76, 69.84] | 85.59 [79.04, 90.34] |
|               | T   | 0.878 [0.622, 0.698] | 76.51 [73.00, 79.77] | 93.33 [89.40, 96.14] | 66.15 [61.22, 70.84] | 62.92 [59.53, 66.19] | 94.16 [90.90, 96.30] |
| VGG16_BN+C    | V   | 0.823 [0.765, 0.872] | 75.71 [69.34, 81.35] | 74.68 [63.64, 83.80] | 76.34 [68.12, 83.32] | 65.56 [57.69, 72.65] | 83.33 [77.19, 88.08] |
|               | I-T | 0.831 [0.775, 0.878] | 75.69 [69.44, 81.23] | 89.29 [80.63, 94.98] | 67.16 [58.53, 75.03] | 63.03 [56.96, 68.71] | 90.91 [84.21, 94.94] |

95% confidence intervals are included in brackets.

- AUC area under the receiver operating characteristic curve, ACC accuracy, SENS sensitivity, SPEC specificity, PPV positive predict value, NPV negative predict value.

T training cohort ( n = 630), V validation cohort ( n = 210), I-T independent test cohort ( n = 218).

Table 2: The detailed parameters of VGG16\_BN.

| Layer name                     | Input channels   | Output channels   | Kernel size   | Stride   | Padding   | Output size     |
|--------------------------------|------------------|-------------------|---------------|----------|-----------|-----------------|
| basic block × 2                | 3                | 64                | 3             | 1        | 1         | [64, 256, 256]  |
| max-pooling layer              |                  |                   | 2             | 2        | 0         | [64, 128, 128]  |
| basic block × 2                | 64               | 128               | 3             | 1        | 1         | [128, 128, 128] |
| max-pooling layer              |                  |                   | 2             | 2        | 0         | [128, 64, 64]   |
| basic block × 3                | 128              | 256               | 3             | 1        | 1         | [256, 64, 64]   |
| max-pooling layer              |                  |                   | 2             | 2        | 0         | [256, 32, 32]   |
| basic block × 3                | 256              | 512               | 3             | 1        | 1         | [512, 32, 32]   |
| max-pooling layer              |                  |                   | 2             | 2        | 0         | [512, 16, 16]   |
| basic block × 3                | 512              | 512               | 3             | 1        | 1         | [512, 16, 16]   |
| max-pooling layer              |                  |                   | 2             | 2        | 0         | [512, 8, 8]     |
| adaptive average pooling layer |                  |                   |               |          |           | [512, 7, 7]     |

The basic block was cascade by convolution layer, batch normalization layer, and Rectified Linear Unit (ReLU).

The input size of the model was [3, 256, 256], which followed the format of [channel, height, width].

Table 3: The performance in prediction of ALN status (N0 vs. N + (1-2)).

| Methods            |     | AUC                                                         | ACC (%)                                                 | SENS (%)                                           | SPEC (%)                                         | PPV (%)                                                        | NPV (%)                                                        |
|--------------------|-----|-------------------------------------------------------------|---------------------------------------------------------|----------------------------------------------------|--------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------|
| Clinical data only | T V | 0.638 [0.595, 0.679] 0.677 [0.602, 0.745] 0.627 a,b [0.551, | 61.00 [56.65, 65.23] 74.29 [67.15, 80.58] 72.67 [65.37, | 65.62 [56.72, 73.79] [30.39, 61.15] [28.62, 61.70] | 59.49 [54.43, 64.40] 83.97 [76.55, 80.60 [72.88, | 34.71 [30.88, 38.75] 48.78 [36.42, 61.29] 39.53 [28.52, 51.73] | 84.06 [80.37, 87.16] 82.09 [77.60, 85.84] 83.72 [79.24, 87.39] |
| Clinical data only |     |                                                             |                                                         | 45.45                                              | 89.79]                                           |                                                                |                                                                |
| Clinical data only | I-T | 0.700]                                                      | 79.18]                                                  | 44.74                                              | 86.92]                                           |                                                                |                                                                |
| DL-CNB model       | T   | 0.912 [0.884, 0.935]                                        | 82.24 [78.67, 85.44]                                    | 97.66 [93.30, 99.51]                               | 77.18 [72.69, 81.25]                             | 58.41 [53.87, 62.81]                                           | 99.01 [97.04, 99.68]                                           |
| DL-CNB model       | V   | 0.756 [0.685, 0.817]                                        | 59.43 [51.76, 66.77]                                    | 97.73 [87.98, 99.94]                               | 46.56 [37.81, 55.48]                             | 38.05 [34.22, 42.04]                                           | 98.39 [89.70, 99.77]                                           |
| DL-CNB model       | I-T | 0.845 c [0.782, 0.895]                                      | 80.23 [73.49, 85.90]                                    | 73.68 [56.90, 86.60]                               | 82.09 [74.53, 88.17]                             | 53.85 [43.66, 63.72]                                           | 91.67 [86.53, 94.96]                                           |
| DL-CNB+C model     | T   | 0.936 [0.911, 0.955]                                        | 84.17 [80.74, 87.21]                                    | 95.31 [90.08, 98.26]                               | 80.51 [76.23, 84.33]                             | 61.62 [56.66, 66.34]                                           | 98.12 [95.99, 99.13]                                           |
|                    | V   | 0.789 [0.721, 0.847]                                        | 66.29 [58.76, 73.24]                                    | 84.09 [69.93, 93.36]                               | 60.31 [51.39, 68.74]                             | 41.57 [35.72, 47.67]                                           | 91.86 [84.94, 95.76]                                           |
|                    | I-T | 0.878 [0.819, 0.923]                                        | 84.30 [77.99, 89.39]                                    | 71.05 [54.10, 84.58]                               | 88.06 [81.33, 93.02]                             | 62.79 [50.52, 73.61]                                           | 91.47 [86.65, 94.66]                                           |

95% confidence intervals are included in brackets.

- AUC, area under the receiver operating characteristic curve; ACC, accuracy; SENS, sensitivity; SPEC, specificity; PPV, positive predictive value; NPV, negative predictive value.
- T, training cohort ( n = 518); V, validation cohort ( n = 175); I-T, independent test cohort ( n = 172).
- ALN, axillary lymph node; DL-CNB+C, deep learning core-needle biopsy incorporating the clinical data.
- a Indicates p = 0.0004, Delong et al. in comparison with DL-CNB model in independent test cohort.
- b Indicates p &lt; 0.0001, Delong et al. in comparison with DL-CNB+C model in independent test cohort.
- c Indicates p = 0.1148, Delong et al. in comparison with DL-CNB+C model in independent test cohort.
- 95% confidence intervals are included in brackets.
- AUC, area under the receiver operating characteristic curve; ACC, accuracy; SENS, sensitivity; SPEC, specificity; PPV, positive predictive value; NPV, negative predictive value.
- T, training cohort ( n = 510); V, validation cohort ( n = 165); I-T, independent test cohort ( n = 173).
- ALN, axillary lymph node; DL-CNB+C, deep learning core-needle biopsy incorporating the clinical data.
- a Indicates p = 0.0005, Delong et al. in comparison with DL-CNB model in independent test cohort.
- b Indicates p &lt; 0.0001, Delong et al. in comparison with DL-CNB+C model in independent test cohort.
- c Indicates p = 0.9689, Delong et al. in comparison with DL-CNB+C model in independent test cohort.
- 95% confidence intervals are included in brackets.
- AUC, area under the receiver operating characteristic curve; ACC, accuracy; SENS, sensitivity; SPEC, specificity; PPV, positive predictive value; NPV, negative predictive value.
- I-T independent test group, ER estrogen receptor, PR progesterone receptor, HER-2 human epidermal growth factor receptor-2, LNM lymph node metastasis.

Table 4: The performance in prediction of ALN status (N0 vs. N + ( ≥ 3)).

| Methods            |     | AUC                                                         | ACC (%)                                                 | SENS (%)                                           | SPEC (%)                                         | PPV (%)                                                        | NPV (%)                                                        |
|--------------------|-----|-------------------------------------------------------------|---------------------------------------------------------|----------------------------------------------------|--------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------|
| Clinical data only | T V | 0.680 [0.638, 0.721] 0.748 [0.675, 0.813] 0.629 a,b [0.553, | 66.67 [62.39, 70.75] 71.52 [63.98, 78.26] 69.36 [61.92, | 65.83 [56.62, 74.24] [58.83, 89.25] [37.18, 69.91] | 66.92 [62.01, 71.58] 70.23 [61.62, 73.88 [65.59, | 37.98 [33.59, 42.58] 40.00 [32.57, 47.92] 37.50 [28.54, 47.40] | 86.42 [83.10, 89.18] 92.00 [86.13, 95.51] 84.62 [79.43, 88.68] |
| Clinical data only |     |                                                             |                                                         | 76.47                                              | 77.90]                                           |                                                                |                                                                |
| Clinical data only | I-T | 0.701]                                                      | 76.14]                                                  | 53.85                                              | 81.08]                                           |                                                                |                                                                |
| DL-CNB model       | T   | 0.906 [0.877, 0.930]                                        | 81.57 [77.93, 84.84]                                    | 93.33 [87.29, 97.08]                               | 77.95 [73.50, 81.97]                             | 56.57 [51.79, 61.23]                                           | 97.44 [95.10, 98.67]                                           |
| DL-CNB model       | V   | 0.755 [0.682, 0.819]                                        | 64.24 [56.42, 71.54]                                    | 91.18 [76.32, 98.14]                               | 57.25 [48.32, 65.85]                             | 35.63 [30.67, 40.92]                                           | 96.15 [89.36, 98.67]                                           |
| DL-CNB model       | I-T | 0.837 c [0.773, 0.888]                                      | 69.94 [62.52, 76.67]                                    | 92.31 [79.13, 98.38]                               | 63.43 [54.68, 71.58]                             | 42.35 [36.61, 48.31]                                           | 96.59 [90.46, 98.83]                                           |
| DL-CNB+C model     | T   | 0.918 [0.891, 0.940]                                        | 82.16 [78.55, 85.38]                                    | 91.67 [85.21, 95.93]                               | 79.23 [74.86, 83.15]                             | 57.59 [52.62, 62.42]                                           | 96.87 [94.45, 98.25]                                           |
|                    | V   | 0.761 [0.689, 0.824]                                        | 66.06 [58.29, 73.24]                                    | 79.41 [62.10, 91.30]                               | 62.60 [53.72, 70.89]                             | 35.53 [29.40, 42.16]                                           | 92.13 [85.66, 95.83]                                           |
|                    | I-T | 0.838 [0.774, 0.889]                                        | 71.10 [63.73, 77.73]                                    | 89.74 [75.78, 97.13]                               | 65.67 [56.98, 73.65]                             | 43.21 [37.04, 49.60]                                           | 95.65 [89.61, 98.25]                                           |

Table 5: The subgroup performance in prediction of ALN status by DL-CNB+C model (N0 vs. N(+)).

| Characteristics   | Value             |         | AUC                                       | ACC (%)                                   | SENS (%)                                  | SPEC (%)                                  | PPV (%)                                   | NPV (%)                                   |      p |
|-------------------|-------------------|---------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|--------|
| Age ≤ 50          | Yes No            | I-T I-T | 0.918 [0.825, 0.971] 0.794 [0.720, 0.855] | 82.09 [70.80, 90.39] 66.89 [58.77, 74.32] | 93.33 [77.93, 99.18] 90.74 [79.70, 96.92] | 72.97 [55.88, 86.21] 53.61 [43.19, 63.8]  | 73.68 [62.05, 82.74] 52.13 [46.38, 57.82] | 93.10 [77.72, 98.12] 91.23 [81.56, 96.07] | 0.0151 |
| T stage           | T1 T2             | I-T I-T | 0.833 [0.754, 0.895] 0.814 [0.722, 0.886] | 71.90 [63.01, 79.69] 71.13 [61.05, 79.89] | 89.19 [74.58, 96.97] 93.62 [82.46, 98.66] | 64.29 [53.08, 74.45] 50.00 [35.53, 64.47] | 52.38 [44.70, 59.95] 63.77 [56.91, 70.11] | 93.10 [84.07, 97.19] 89.29 [72.93, 96.27] | 0.7426 |
| ER                | Positive Negative | I-T I-T | 0.853 [0.789, 0.903] 0.737 [0.596, 0.849] | 82.53 [75.88, 87.98] 67.31 [52.89, 79.67] | 89.23 [79.06, 95.56] 73.68 [48.80, 90.85] | 78.22 [68.90, 85.82] 63.64 [45.12, 79.60] | 72.50 [64.34, 79.39] 53.85 [40.83, 66.36] | 91.86 [84.76, 95.81] 80.77 [65.47, 90.30] | 0.1253 |
| PR                | Positive Negative | I-T I-T | 0.839 [0.772, 0.893] 0.811 [0.690, 0.900] | 72.61 [64.93, 79.42] 70.49 [57.43, 81.48] | 96.61 [88.29, 99.59] 80.00 [59.30, 93.17] | 58.16 [47.77, 68.05] 63.89 [46.22, 79.18] | 58.16 [52.28, 63.82] 60.61 [48.85, 71.25] | 96.61 [87.84, 99.12] 82.14 [66.92, 91.27] | 0.6591 |
| HER2              | Positive Negative | I-T I-T | 0.800 [0.677, 0.892] 0.842 [0.776, 0.895] | 66.67 [53.31, 78.31] 74.05 [66.49, 80.69] | 88.89 [65.29, 98.62] 92.42 [83.20, 97.49] | 57.14 [40.96, 72.28] 60.87 [50.14, 70.88] | 47.06 [37.68, 56.65] 62.89 [56.54, 68.81] | 92.31 [75.99, 97.85] 91.80 [82.60, 96.35] | 0.5238 |

Figure 1: Comparison of receiver operating characteristic (ROC) curves between different models for predicting disease-free axilla (N0) and low metastatic burden of axillary disease (N + (1-2)). Numbers in parentheses are areas under the receiver operating characteristic curve (AUCs).

<!-- image -->

Figure 2: Comparison of receiver operating characteristic (ROC) curves between different models for predicting disease-free axilla (N0) and low metastatic burden of axillary disease (N + ( ≥ 3)). Numbers in parentheses are areas under the receiver operating characteristic curve (AUCs).

<!-- image -->
