# Library for stacking

1. Set train and test dataset under data/input.

2. Created features from original dataset need to be under data/output/features.

3. Models for stacking are defined in scripts under scripts folder.

4. Need to define created features in that scripts.

5. Just run "python scripts/XXX.py"

## Tree of files

- base_fixed_fold.py (class of stacking)
- data/
  - input/
    - train.csv (train dataset)
    - test.csv (test dataset)
  - output/
    - features/
      - features.csv (features user created)
    - temp/
      - temp.csv (files saved in stacking)
- scripts/
  - script.csv (main script where concrete models defined)

## Installation
`python setup.py install`


## Details of scripts

* base.py: 
Base models for stacking are defined here (using sklearn.base.BaseEstimator).
Some models are defined here. e.g., XGBoost, Keras, Vowpal Wabbit.
These models are wrapped as scikit-learn like (using sklearn.base.ClassifierMixin, sklearn.base.RegressorMixin).
That is, model class has some methods, fit() and predict_proba().

New user-defined models can be added here.

Scikit-learn models can be used.

Base model have some arguments.
's': Stacking. Svaing a oof prediction({model_name}_all_fold.csv) and average of test prediction based on fold-train models({model_name}_test.csv). These files will be used for next level stacking.
't': Training with all data and predict test({model_name}_TestInAllTrainingData.csv). This is useful to get the single model performance.
'st': Stacking and then training with all data and predict test ('s' and 't').
'cv': Only cross validation without saving the prediction.

Define task details top of script.


* features.py:
Create features based on original dataset.

* scripts/XXX.py:
Define several models and its parameters used for stacking.
Train and test feature set are defined here.
Need to define CV-fold index.

Any level stacking can be defined.



## TODO LIST

Need to be more general library.

Please check isuues!!

