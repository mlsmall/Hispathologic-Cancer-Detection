# Hispathologic Cancer Detection
## Identifing metastatic tissue in histopathologic scans of lymph node sections

Lymph node metastases occur in most cancer types (e.g. breast, prostate, colon). Lymph nodes are small glands that filter lymph, the fluid that circulates through the lymphatic system. The lymph nodes in the underarms are the first place breast cancer is likely to spread. Metastatic involvement of lymph nodes is one of the most important prognostic variables in breast cancer. The diagnostic procedure for pathologists is, however, tedious and time-consuming and prone to misinterpretation. 

Automated detection of lymph node metastasis has a great potential to help the pathologist. Within the past few years, the field has been moving towards a fully automated analysis of whole-slide images to detect cancer, to predict prognosis or identify metastases. 

## Dataset

The [dataset](https://www.kaggle.com/c/histopathologic-cancer-detection/data) consists of 220,025 image patches of size 96x96 in the training set. 

The goal is to correctly predict the label (positive or negative) for the images in the test set. A positive label indicates that the center 32x32 region of a patch contains at least one pixel of tumor tissue. If there is a tumor on the tissue that is outside this 32x32 center region, then it is not considered for labeling purposes. 

## Visualizing the Data

Here's an example of a sample tissue with a 32x32 boudary box.

<img src="https://github.com/mlsmall/Hispathologic-Cancer-Detection/blob/master/sample%20tissue.png" width="200" />

And here's a picture of 64 different tissue samples.

<img src="https://github.com/mlsmall/Hispathologic-Cancer-Detection/blob/master/sample%20tissues.png" width='400' />
