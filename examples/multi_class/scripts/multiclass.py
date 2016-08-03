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

# ----- scikit-learn -----
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

# ----- Set problem type!! -----
problem_type = 'classification'
classification_type = 'multi-class'
eval_type = 'logloss'

BaseModel.set_prob_type(problem_type, classification_type, eval_type)



# ----- create dataset -----

# load data for binary
digits = load_digits()

# split data for train and test
data_train, data_test, label_train, label_test = train_test_split(digits.data, digits.target)

# concat data as pandas' dataframe format
data_train = pd.DataFrame(data_train)
label_train = pd.DataFrame(label_train, columns=['target'])

data_test = pd.DataFrame(data_test)
label_test = pd.DataFrame(label_test, columns=['target'])

# save data under /data/input.
data_train.to_csv(INPUT_PATH + 'train.csv', index=False)
label_train.to_csv(INPUT_PATH + 'target.csv', index=False)
data_test.to_csv(INPUT_PATH + 'test.csv', index=False)
label_test.to_csv(INPUT_PATH + 'label_test.csv', index=False)

# ----- END create dataset -----

# -----create features -----
train_log = data_train.iloc[:, :64].applymap(lambda x: np.log(x+1))
test_log = data_test.iloc[:, :64].applymap(lambda x: np.log(x+1))

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
                'train':(
                         INPUT_PATH + 'train.csv',
                         FEATURES_PATH + 'train_log.csv',
                        ),

                'target':(
                         INPUT_PATH + 'target.csv',
                        ),

                'test':(
                         INPUT_PATH + 'test.csv',
                         FEATURES_PATH + 'test_log.csv',
                        ),
                }

# need to get input shape for NN now
X,y,test  = load_data(flist=FEATURE_LIST_stage1, drop_duplicates=False)
assert((False in X.columns == test.columns) == False)
nn_input_dim_NN = X.shape[1:]
output_dim = len(set(y))
del X, y, test



# Models in Stage 1
PARAMS_V1 = {
        'colsample_bytree':0.80,
        'learning_rate':0.1,
        "eval_metric":"mlogloss",
        'max_depth':5, 
        'min_child_weight':1,
        'nthread':4,
        'seed':407,
        'silent':1, 
        'subsample':0.60,
        'objective':'multi:softprob',
        'num_class':output_dim,
        }

class ModelV1(BaseModel):
        def build_model(self):
            return XGBClassifier(params=self.params, num_round=10)


PARAMS_V2 = {
            'batch_size':32,
            'nb_epoch':15,
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
                        
            model.add(Dense(output_dim, init='he_normal'))
            model.add(Activation('softmax'))
            sgd = SGD(lr=0.1, decay=1e-5, momentum=0.9, nesterov=True)

            model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])

            return KerasClassifier(nn=model,**self.params)

PARAMS_V3 = {
             'n_estimators':500, 'criterion':'gini', 'n_jobs':8, 'verbose':0,
             'random_state':407, 'oob_score':True,
             }

class ModelV3(BaseModel):
        def build_model(self):
            return RandomForestClassifier(**self.params)

PARAMS_V4 = {
             'n_estimators':550, 'criterion':'gini', 'n_jobs':8, 'verbose':0,
             'random_state':407,
             }

class ModelV4(BaseModel):
        def build_model(self):
            return ExtraTreesClassifier(**self.params)


PARAMS_V5 = {
             'n_estimators':300, 'learning_rate':0.05,'subsample':0.8,
             'max_depth':5, 'verbose':1, 'max_features':0.9,
             'random_state':407,
             }

class ModelV5(BaseModel):
        def build_model(self):
            return GradientBoostingClassifier(**self.params)

PARAMS_V6 = {
             'n_estimators':650, 'learning_rate':0.01,'subsample':0.8,
             'max_depth':5, 'verbose':1, 'max_features':0.82,
             'random_state':407,
             }

class ModelV6(BaseModel):
        def build_model(self):
            return GradientBoostingClassifier(**self.params)

# ----- END first stage stacking model -----

# ----- Second stage stacking model -----
'''
PARAMS_V1_stage2 = {
        'colsample_bytree':0.8,
        'learning_rate':0.05,
        "eval_metric":"mlogloss",
        'max_depth':4, 
        'seed':1234,
        'nthread':8,
        'reg_lambda':0.01,
        'reg_alpha':0.01,
        'silent':1, 
        'subsample':0.80,
        'objective':'multi:softprob',
        'num_class':output_dim,
        }

class ModelV1_stage2(BaseModel):
        def build_model(self):
            return XGBClassifier(params=self.params, num_round=40)
'''

PARAMS_V1_stage2 = {
                    'penalty':'l2',
                    'tol':0.0001, 
                    'C':1.0, 
                    'random_state':None, 
                    'verbose':0, 
                    'n_jobs':8
                    }

class ModelV1_stage2(BaseModel):
        def build_model(self):
            return LR(**self.params)


# ----- END first stage stacking model -----

if __name__ == "__main__":
    
    # Create cv-fold index
    target = pd.read_csv(INPUT_PATH + 'target.csv')
    create_cv_id(target, n_folds_ = 5, cv_id_name='cv_id', seed=407)

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

    m = ModelV3(name="v3_stage1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V3,
                kind = 'st'
                )
    m.run()

    m = ModelV4(name="v4_stage1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V4,
                kind = 'st'
                )
    m.run()

    m = ModelV5(name="v5_stage1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V5,
                kind = 'st'
                )
    m.run()

    m = ModelV6(name="v6_stage1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V6,
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
                         TEMP_PATH + 'v3_stage1_all_fold.csv',
                         TEMP_PATH + 'v4_stage1_all_fold.csv',
                         TEMP_PATH + 'v5_stage1_all_fold.csv',
                         TEMP_PATH + 'v6_stage1_all_fold.csv',
                        ),

                'target':(
                         INPUT_PATH + 'target.csv',
                        ),

                'test':(INPUT_PATH + 'test.csv',
                         FEATURES_PATH + 'test_log.csv',
                         
                         TEMP_PATH + 'v1_stage1_test.csv',
                         TEMP_PATH + 'v2_stage1_test.csv',
                         TEMP_PATH + 'v3_stage1_test.csv',
                         TEMP_PATH + 'v4_stage1_test.csv',
                         TEMP_PATH + 'v5_stage1_test.csv',
                         TEMP_PATH + 'v6_stage1_test.csv',                       
                        ),
                }

    X,y,test  = load_data(flist=FEATURE_LIST_stage2, drop_duplicates=False)
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

    print 'Test evaluation'
    mll = eval_pred(label.target, pred.values, eval_type=eval_type)

    print 'saving final results'
    pred.columns = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
    pred = pd.concat([testID, pred], axis=1)
    pred.to_csv(TEMP_PATH + 'final_submission.csv', index=False)

    

