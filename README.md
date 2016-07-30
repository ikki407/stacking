Stacking (stacked generalization)
====

[![PyPI version](https://badge.fury.io/py/stacking.svg)](https://badge.fury.io/py/stacking)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/ikki407/stacking/LICENSE)

## Overview

[ikki407/stacking](https://github.com/ikki407/stacking) - Simple and useful [stacking](https://en.wikipedia.org/wiki/Ensemble_learning#Stacking) library, written in Python.


## Description

Stacking (sometimes called stacked generalization) involves training a learning algorithm to combine the predictions of several other learning algorithms. The basic idea is to use a pool of base classifiers, then using another classifier to combine their predictions, with the aim of reducing the generalization error.


## Usage

**Working Example:**
 
 * [binary classification](https://github.com/ikki407/stacking/tree/master/examples/binary_class)
 * [multi-class classification](https://github.com/ikki407/stacking/tree/master/examples/multi_class)
 * [regression](https://github.com/ikki407/stacking/tree/master/examples/regression)

To run these examples, just run `sh run.sh`. Note that: 

1. Set train and test dataset under data/input

2. Created features from original dataset need to be under data/output/features

3. Models for stacking are defined in `scripts.py` under scripts folder

4. Need to define created features in that scripts

5. Just run `sh run.sh` (`python scripts/XXX.py`).


## Installation
To install stacking, `cd` to the stacking folder and run the install command**(up-to-date version, recommended)**:
```
sudo python setup.py install
```

You can also install stacking from PyPI:
```
pip install stacking
```


## Files

- [stacking/base.py](https://github.com/ikki407/stacking/blob/master/stacking/base.py) : stacking module
- examples/
 - [binary_class](https://github.com/ikki407/stacking/tree/master/examples/binary_class) : binary classification
 - [multi_class](https://github.com/ikki407/stacking/tree/master/examples/multi_class) : multi-class classification
 - [regression](https://github.com/ikki407/stacking/tree/master/examples/regression) : regression


## Details of scripts

- base.py: 
  - Base models for stacking are defined here (using sklearn.base.BaseEstimator).
  - Some models are defined here. e.g., XGBoost, Keras, Vowpal Wabbit.
  - These models are wrapped as scikit-learn like (using sklearn.base.ClassifierMixin, sklearn.base.RegressorMixin).
  - That is, model class has some methods, fit(), predict_proba(), and predict().

New user-defined models can be added here.

Scikit-learn models can be used.

Base model have some arguments.

- 's': Stacking. Saving oof(out-of-fold) prediction({model_name}_all_fold.csv) and average of test prediction based on train-fold models({model_name}_test.csv). These files will be used for next level stacking.

- 't': Training with all data and predict test({model_name}_TestInAllTrainingData.csv). In this training, no validation data are used.

- 'st': Stacking and then training with all data and predict test ('s' and 't').

- 'cv': Only cross validation without saving the prediction.


Define several models and its parameters used for stacking.
Define task details on the top of script.
Train and test feature set are defined here. 
Need to define CV-fold index.

Any level stacking can be defined.

## Reference

[1] [Wolpert, David H. Stacked generalization, Neural Networks, 5(2), 241-259](http://machine-learning.martinsewell.com/ensembles/stacking/Wolpert1992.pdf)

[2] [Ensemble learning(Stacking)](https://en.wikipedia.org/wiki/Ensemble_learning#Stacking)

[3] [KAGGLE ENSEMBLING GUIDE](http://mlwave.com/kaggle-ensembling-guide/)

