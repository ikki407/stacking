#### Part of documentation regarding model ensemble for Santander of team "#1 Leustagos" - Ikki's solution

Ikki Tanaka
ikki0407@gmail.com

## 1. How to generate the predictions - Ikki's solution 

File list

Task --- File name

Feature engineering for XGBOOST --- ikki_feat_ver1.py, ikki_feat_ver2.py
Feature engineering for Neural nets --- ikki_feat_ver3.py
Train XGBOOST --- scripts/ikki_xgb_1.py, scripts/ikki_xgb_2.py
Train Neural nets --- scripts/ikki_NN_1.py
Combine prediction --- combine_pred.py
Model wrapper class -- base_fixed_fold.py



1. Original train and test data are in data/input

2. Open command line and change path to ikki

3. Run the script run_ikki.sh

4. The final output files will be saved in the folder data/output/train and
data/output/test. The created features will be saved in
ikki/data/output/features. The temporary files in of each models will be saved in ikki/data/output/temp.

It will generate files of indivisual models under /ikki/data/output/temp/ ( 2(train&test) * 20(fold sets) * 3(models) = 120 files), including 2 final files under /data/output/train/ & /data/output/test/ with predictions of 3 models - 2 xgboosts and 1 neural net model. 

Dependencies:
The model can be run on Linux Ubuntu 14.04 or Mac OS.

Python:
os, sys, pandas, numpy, xgboost, keras, sklearn


## 2. Preprocessing and models

## 2.1 Preprocessing and feature engineering

Following data cleaning and feature engineering tricks were used:
a) calculating zero-count per ID.
b) calculating count of integer variables.
c) creating dummy variables of var3 in the threshold (var3 >= 4 or 5).
d) creating doolean variable if var38 is peak value (117310.979016494).
e) calculating log of var38 and replace log(117310.979016494) with zero.
f) removing constant variables.
g) removing identical variables.
h) for neural nets, removing some one-hot-encoded categorical variables
i) for neural nets, limiting variables in test based on min/max of train.
j) for neural nets, applying log transformation to all variables
k) one-hot encoding of specific variables (some vars duplicated of the ones created in (c) are removed in model training).

This procedure can be found in ikki/ikki_feat_verXX.

## 2.2 Models

I used XGBOOST and neural networks. Each model was trained 20 times with 20 fold sets and averaged using rank transformation. I used slightly different parameters in each model, respectively. These parameters were chosen based on CV.
In neural nets, the number of layers were 3 or 4 and BatchNormalization was used.
LeakyReLU and Parameterized ReLU were used as activation functions. Optimizer of all models was Stochastic Gradient Decent(SGD). I also standarized the data by removing the mean value of each feature and dividing by their standard deviation. 

XGBOOST models are found in ikki/scripts/ikki_XGB_X.py and neural net model is found in ikki/scripts/ikki_NN_1.py.

And I created and used my class of models for stacking. This file is ikki/base_fixed_fold.py. This script helps to implement stacking easily as below:

1. specify the parameters and model to variable "m = ModelV1(~)"
2. then just run "m.run()"


## 3 Inportant remarks

Finaly I desided to use simple feature engineerings as shown above because I'd like to trust in local CV and public LB at almost the same weights, although plan of our team was to trust in CV.
But actually, the weights of CV are higher than the ones of public LB.


