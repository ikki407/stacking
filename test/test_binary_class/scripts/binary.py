# -*- coding: utf-8 -*-

# ----- for creating dataset -----
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split

# ----- general import -----
import pandas as pd
import numpy as np

# ----- stacking library -----
from stacking.base import FOLDER_NAME, PATH, INPUT_PATH, TEMP_PATH,\
        FEATURES_PATH, OUTPUT_PATH, SUBMIT_FORMAT
# ----- utils -----
from stacking.base import load_data, save_pred_as_submit_format, create_cv_id, \
        eval_pred
# ----- classifiers -----
from stacking.base import BaseModel, XGBClassifier, KerasClassifier

# ----- keras -----
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2, l1l2, activity_l2

# ----- Set problem type!! -----
problem_type = 'classification'
classification_type = 'binary'
eval_type = 'auc'

BaseModel.set_prob_type(problem_type, classification_type, eval_type)



# ----- create dataset -----

# load data for binary
digits = load_digits(2)

# split data for train and test
data_train, data_test, label_train, label_test = train_test_split(digits.data, digits.target)

# concat data as pandas' dataframe format
data_train = pd.DataFrame(data_train)
label_train = pd.DataFrame(label_train, columns=['target'])
train = pd.concat([data_train, label_train], axis=1)

data_test = pd.DataFrame(data_test)
label_test = pd.DataFrame(label_test, columns=['target'])
test = data_test # not include target

# save data under /data/input.
train.to_csv(INPUT_PATH + 'train.csv', index=False)
test.to_csv(INPUT_PATH + 'test.csv', index=False)
label_test.to_csv(INPUT_PATH + 'label_test.csv', index=False)

# ----- END create dataset -----

# -----create features -----
train_log = train.iloc[:, :64].applymap(lambda x: np.log(x+1))
test_log = test.iloc[:, :64].applymap(lambda x: np.log(x+1))

train_log.columns = map(str, train_log.columns)
test_log.columns = map(str, test_log.columns)

train_log.columns += '_log'
test_log.columns += '_log'

# save data under /data/output/features/.
train_log.to_csv(FEATURES_PATH + 'train_log.csv', index=False)
test_log.to_csv(FEATURES_PATH + 'test_log.csv', index=False)

# ----- END create features -----



# ----- First stage stacking model-----

# FEATURE LISTS in Stage 1.
FEATURE_LIST_stage1 = {
                'train':(INPUT_PATH + 'train.csv',
                         FEATURES_PATH + 'train_log.csv',
                         
                        ),#target is in 'train'
                'test':(INPUT_PATH + 'test.csv',
                        FEATURES_PATH + 'test_log.csv',
                        ),
                }

# need to get input shape for NN now
X,y,test  = load_data(flist=FEATURE_LIST_stage1, drop_duplicates=True)
assert((False in X.columns == test.columns) == False)
nn_input_dim_NN = X.shape[1:]
del X, y, test



# Models in Stage 1
PARAMS_V1 = {
        'colsample_bytree':0.80,
        'learning_rate':0.1,"eval_metric":"auc",
        'max_depth':5, 'min_child_weight':1,
        'nthread':4,
        'objective':'binary:logistic','seed':407,
        'silent':1, 'subsample':0.60,
        }

class ModelV1(BaseModel):
        def build_model(self):
            return XGBClassifier(params=self.params, num_round=10)


PARAMS_V2 = {
            'batch_size':8,
            'nb_epoch':5,
            'verbose':1, 
            'callbacks':[],
            'validation_split':0.,
            'validation_data':None,
            'shuffle':True,
            #'show_accuracy':True,
            'class_weight':None,
            'sample_weight':None,
            'normalize':True,
            'categorize_y':True
            }

class ModelV2(BaseModel):
        def build_model(self):
            model = Sequential()
            model.add(Dense(64, input_shape=nn_input_dim_NN, init='he_normal'))
            model.add(LeakyReLU(alpha=.00001))
            model.add(Dropout(0.5))
                        
            model.add(Dense(2, init='he_normal'))
            model.add(Activation('softmax'))
            sgd = SGD(lr=0.1, decay=1e-5, momentum=0.9, nesterov=True)

            model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=["accuracy"])

            return KerasClassifier(nn=model,**self.params)

# ----- END first stage stacking model -----

# ----- Second stage stacking model -----

PARAMS_V1_stage2 = {
        'colsample_bytree':0.6,'colsample_bylevel':0.80,
        'learning_rate':0.05,"eval_metric":"auc",
        'max_depth':6, 'seed':1234,
        'nthread':8,'reg_lambda':3,'reg_alpha':0.01,
        'objective':'binary:logistic',
        'silent':1, 'subsample':0.60,
        }

class ModelV1_stage2(BaseModel):
        def build_model(self):
            return XGBClassifier(params=self.params, num_round=5)

# ----- END first stage stacking model -----

if __name__ == "__main__":
    
    # Create cv-fold index
    train = pd.read_csv(INPUT_PATH + 'train.csv')
    create_cv_id(train, n_folds_ = 5, cv_id_name='cv_id', seed=407)

    ######## stage1 Models #########
    print 'Start stage 1 training'

    m = ModelV1(name="v1_stage1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V1,
                kind = 'st'
                )
    m.run()


    m = ModelV2(name="v2_stage1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V2,
                kind = 'st'
                )
    m.run()

    print 'Done stage 1'
    print 
    ######## stage2 Models #########

    print 'Start stage 2 training'

    # FEATURE LISTS in Stage 2.
    # Need to define here because the outputs for NN dim. haven't been created yet.
    FEATURE_LIST_stage2 = {
                'train':(INPUT_PATH + 'train.csv',
                         FEATURES_PATH + 'train_log.csv',
                         
                         TEMP_PATH + 'v1_stage1_all_fold.csv',
                         TEMP_PATH + 'v2_stage1_all_fold.csv',


                        ),#targetはここに含まれる
                'test':(INPUT_PATH + 'test.csv',
                         FEATURES_PATH + 'test_log.csv',
                         
                         TEMP_PATH + 'v1_stage1_test.csv',
                         TEMP_PATH + 'v2_stage1_test.csv',
                       
                        ),
                }

    X,y,test  = load_data(flist=FEATURE_LIST_stage2, drop_duplicates=True)
    assert((False in X.columns == test.columns) == False)
    nn_input_dim_NN2 = X.shape[1]
    del X, y, test


    # Models
    m = ModelV1_stage2(name="v1_stage2",
                    flist=FEATURE_LIST_stage2,
                    params = PARAMS_V1_stage2,
                    kind = 'st',
                    )
    m.run()

    print 'Done stage 2'
    print 
    
    # averaging
    print 'Saving as submission format'
    #sample_sub = pd.read_csv('data/input/sample_submission.csv')
    label = pd.read_csv(INPUT_PATH + 'label_test.csv')
    testID = range(len(label))
    testID = pd.DataFrame(testID, columns=['ID'])
    pred = pd.read_csv(TEMP_PATH + 'v1_stage2_TestInAllTrainingData.csv')

    print 'Evaluation'
    auc = eval_pred(label.target, pred.iloc[:,0], eval_type=eval_type)
    pred = pd.concat([testID, pred], axis=1)
    pred.to_csv(TEMP_PATH + 'final_submission.csv', index=False)

    

