# Hispathologic Cancer Detection
## Identifing metastatic tissue in histopathologic scans of lymph node sections

Lymph node metastases occur in most cancer types (e.g. breast, prostate, colon). Lymph nodes are small glands that filter lymph, the fluid that circulates through the lymphatic system. The lymph nodes in the underarms are the first place breast cancer is likely to spread. Metastatic involvement of lymph nodes is one of the most important prognostic variables in breast cancer. The diagnostic procedure for pathologists is, however, tedious and time-consuming and prone to misinterpretation. 

Automated detection of lymph node metastasis has a great potential to help the pathologist. Within the past few years, the field has been moving towards a fully automated analysis of whole-slide images to detect cancer, to predict prognosis or identify metastases. 

## Dataset

The [dataset](https://www.kaggle.com/c/histopathologic-cancer-detection/data) consists of 220,025 image patches of size 96x96 in the training set. 

The goal is to correctly predict the label (positive or negative) for the images in the test set. A positive label indicates that the center 32x32 region of a patch contains at least one pixel of tumor tissue. If there is a tumor on the tissue that is outside this 32x32 center region, then it is not considered for labeling purposes. 

## Visualizing the Data

Here's an example of a sample tissue with a 32x32 boudary box.

<img src="https://github.com/mlsmall/Hispathologic-Cancer-Detection/blob/master/sample%20tissue.png" width="180" />

Here is a variety of histopathologic scans with 64 different tissue samples.

<img src="https://github.com/mlsmall/Hispathologic-Cancer-Detection/blob/master/sample%20tissues.png" width='600' />

Next is a plot of four hyspatholigic samples of lymph nodes with their corresponding positive labels (containing a tumor) and four samples with their negative labels (not containing a tumor).

<img src="https://github.com/mlsmall/Hispathologic-Cancer-Detection/blob/master/positive%20and%20negative%20samples.png" width='1080' />

## Creating a Databunch

A databunch is used to prepare the data so it can be trained using the fastai [Learner class](https://docs.fast.ai/basic_train.html#Learner).

### Data Augmentation

To create the databunch, we first define the type of augmentation we would like to do on the images. Data augmentation is technique used to increase the amount of training data by using information only in our training dataset. It can be used as a way we to reduce overfitting on models. Instead of feeding the model with the same pictures every time, we do small random transformations that don't change what's inside the image (for the human eye) but change its pixel values. Models trained with data augmentation will then generalize better.

We will flip the images horizontally, vertically, add a little bit of brightness, zoom and lighting. For more information on doing data augmentation with the fastai library, refer to [this link](https://docs.fast.ai/vision.transform.html#Data-augmentation).

The code used is:
'''python
tfms=get_transforms(do_flip=True, flip_vert=True, max_rotate=45, max_zoom=0.15,
               max_lighting=0.5, max_warp=0., p_affine=0.5, p_lighting=0.50)'
'''
