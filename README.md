# Library for stacking

1. Set train and test dataset under data/input.

2. Created features from original dataset need to be under data/output/features.

3. Models for stacking are defined in scripts under scripts folder.

4. Need to define created features in that scripts.

5. Just run "python scripts/XXX.py"

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

1. How to define target column in feature set. Target column need to be included in train dataset as column name 'target' now.

2. Multi-classification task is needed. Binary-classification and regression tasks now.

3. How to create CV-fold index.
* previous version for CV-fold file. Using index.
train: 
[2,3,4,5,6,7,8,9]
[0,1,4,5,6,7,8,9]
[0,1,2,3,6,7,8,9]
[0,1,2,3,4,5,8,9]
[0,1,2,3,4,5,6,7]
test:  
[0,1]
[2,3]
[4,5]
[6,7]
[8,9]
     
* current version for CV-fold file(better than previous one). Using fold ID.
[0,0,1,1,2,2,3,3,4,4]

But current BaseModel uses previous version architectures. 
If current version is used, it is changed to previous format.
So need to change that to using original format.
And need to change .ix to .iloc for stable behavior.

Need to change global CV-fold file name with new CV-fold file name, if new CV-fold be created.


4. How to design saving prediction as submittion format (save_pred_as_submit_format).

5. Change task-dependent functions to virtual functions? User need to define such functions themselves. (CV-fold index, save_pred_as_submit_format, )

6. In class BaseModel, scikit-learn models can be used, but should reconsider that approach.

7. In stacking, test data is not passed to models as validation data (XGBoost and NN). That is, validation scores are calculated after model training is done. It will be convenient to check the validation score every epoch. So need to pass fold-out data in model training as well.


