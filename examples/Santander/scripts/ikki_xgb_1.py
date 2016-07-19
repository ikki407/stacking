#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Genaral packages
import os, sys
import pandas as pd
import numpy as np

sys.path.append(os.getcwd())

#各種PATH
from stacking.base import FOLDER_NAME, PATH, INPUT_PATH, OUTPUT_PATH, ORIGINAL_TRAIN_FORMAT, SUBMIT_FORMAT


np.random.seed(407)
#keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2, l1l2, activity_l2

#base_ver2 utils
from stacking.base import load_data, save_pred_as_submit_format, create_cv_id


#classifiers
from stacking.base import BaseModel, XGBClassifier, KerasClassifier


########### First stage ###########

# FEATURE LISTS in Stage 1.


FEATURE_LIST_stage1 = {
                'train':('data/output/features/ikki_features_train_ver1.csv',
                         'data/output/features/ikki_one_hot_encoder_train_ver1.csv',
                         
                        ),#target is in 'train'
                'test':('data/output/features/ikki_features_test_ver1.csv',
                        'data/output/features/ikki_one_hot_encoder_test_ver1.csv',
                        ),
                }

#X,y,test  = load_data(flist=FEATURE_LIST_stage1, drop_duplicates=True)
#assert((False in X.columns == test.columns) == False)
#nn_input_dim = X.shape[1]
#del X, y, test


# Models in Stage 1
PARAMS_V1 = {
        'colsample_bytree':0.83,'colsample_bylevel':0.9,
        'learning_rate':0.01,"eval_metric":"auc",
        'max_depth':5, 'min_child_weight':1,
        'nthread':8,'gamma':1.0,'reg_lambda':5.0,'reg_alpha':0.0001,
        'objective':'binary:logistic','seed':333111,
        'silent':1, 'subsample':0.60,'base_score':0.04
        }

class ModelV1(BaseModel):
        def build_model(self):
            return XGBClassifier(params=self.params, num_round=565)

PARAMS_V2 = {
        'colsample_bytree':0.74,
        'learning_rate':0.01,"eval_metric":"auc",
        'max_depth':6, 'min_child_weight':1,
        'nthread':8,'gamma':1.2,'reg_lambda':7.0,'reg_alpha':0.001,
        'objective':'binary:logistic','seed':123210,
        'silent':1, 'subsample':0.77,'base_score':0.04
        }

class ModelV2(BaseModel):
        def build_model(self):
            return XGBClassifier(params=self.params, num_round=595)




PARAMS_V3 = {
        'colsample_bytree':0.65,'colsample_bylevel':0.85,
        'learning_rate':0.02,"eval_metric":"auc",
        'max_depth':6, 'min_child_weight':3,
        'nthread':8,'gamma':2.5,'reg_lambda':6,'reg_alpha':0,
        'objective':'binary:logistic','seed':232323,
        'silent':1, 'subsample':0.86,'base_score':0.04
        }

class ModelV3(BaseModel):
        def build_model(self):
            return XGBClassifier(params=self.params, num_round=400)



PARAMS_V4 = {
        'colsample_bytree':0.8,'colsample_bylevel':0.7,
        'learning_rate':0.01,"eval_metric":"auc",
        'max_depth':6, 'min_child_weight':3,
        'nthread':8,'gamma':5,'reg_lambda':1,'reg_alpha':0,
        'objective':'binary:logistic','seed':407,
        'silent':1, 'subsample':0.7,'base_score':0.04
            }

class ModelV4(BaseModel):
        def build_model(self):
            return XGBClassifier(params=self.params, num_round=570)


PARAMS_V5 = {
        'colsample_bytree':0.62,'colsample_bylevel':0.825,
        'learning_rate':0.015,"eval_metric":"auc",
        'max_depth':6, 'min_child_weight':3,
        'nthread':8,'gamma':2,'reg_lambda':8,'reg_alpha':0.01,
        'objective':'binary:logistic','seed':404447,
        'silent':1, 'subsample':0.76,'base_score':0.04
        }

class ModelV5(BaseModel):
        def build_model(self):
            return XGBClassifier(params=self.params, num_round=580)

PARAMS_V6 = {
        'colsample_bytree':0.85,
        'learning_rate':0.01,"eval_metric":"auc",
        'max_depth':5, 'min_child_weight':1,
        'nthread':8,'gamma':0.0,'reg_lambda':1,'reg_alpha':0.0,
        'objective':'binary:logistic','seed':47707,
        'silent':1, 'subsample':0.736,'base_score':0.04
        }

class ModelV6(BaseModel):
        def build_model(self):
            return XGBClassifier(params=self.params, num_round=570)

PARAMS_V7 = {
        'colsample_bytree':0.62,'colsample_bylevel':0.8,
        'learning_rate':0.01,"eval_metric":"auc",
        'max_depth':5, 'min_child_weight':1,
        'nthread':8,'gamma':0.5,'reg_lambda':4.5,'reg_alpha':0.001,
        'objective':'binary:logistic','random_state':432434,
        'silent':1, 'subsample':0.80,'base_score':0.04
        }

class ModelV7(BaseModel):
        def build_model(self):
            return XGBClassifier(params=self.params, num_round=570)


PARAMS_V8 = {
        'colsample_bytree':0.7,'colsample_bylevel':0.8,
        'learning_rate':0.01,"eval_metric":"auc",
        'max_depth':6, 'min_child_weight':1,
        'nthread':8,'gamma':5,'reg_lambda':1,'reg_alpha':0,
        'objective':'binary:logistic','seed':403327,
        'silent':1, 'subsample':0.7,'base_score':0.04
            }

class ModelV8(BaseModel):
        def build_model(self):
            return XGBClassifier(params=self.params, num_round=570)


PARAMS_V9 = {
        'colsample_bytree':0.63,'colsample_bylevel':0.85,
        'learning_rate':0.015,"eval_metric":"auc",
        'max_depth':6, 'min_child_weight':3,
        'nthread':8,'gamma':2,'reg_lambda':8,'reg_alpha':0.01,
        'objective':'binary:logistic','seed':40723,
        'silent':1, 'subsample':0.76,'base_score':0.04
        }

class ModelV9(BaseModel):
        def build_model(self):
            return XGBClassifier(params=self.params, num_round=580)


PARAMS_V10 = {
        'colsample_bytree':0.62,'colsample_bylevel':0.9,
        'learning_rate':0.01,"eval_metric":"auc",
        'max_depth':5, 'min_child_weight':1,
        'nthread':8,'gamma':0.8,'reg_lambda':4.5,'reg_alpha':0.001,
        'objective':'binary:logistic','random_state':434,
        'silent':1, 'subsample':0.79,'base_score':0.04
        }

class ModelV10(BaseModel):
        def build_model(self):
            return XGBClassifier(params=self.params, num_round=570)

PARAMS_V11 = {
        'colsample_bytree':0.89,'colsample_bylevel':0.9,
        'learning_rate':0.01,"eval_metric":"auc",
        'max_depth':5, 'min_child_weight':1,
        'nthread':8,'gamma':1.0,'reg_lambda':5.0,'reg_alpha':0.0001,
        'objective':'binary:logistic','seed':333,
        'silent':1, 'subsample':0.60,'base_score':0.04
        }

class ModelV11(BaseModel):
        def build_model(self):
            return XGBClassifier(params=self.params, num_round=565)

PARAMS_V12 = {
        'colsample_bytree':0.74,
        'learning_rate':0.01,"eval_metric":"auc",
        'max_depth':6, 'min_child_weight':1,
        'nthread':8,'gamma':1.2,'reg_lambda':7.0,'reg_alpha':0.001,
        'objective':'binary:logistic','seed':10,
        'silent':1, 'subsample':0.75,'base_score':0.04
        }

class ModelV12(BaseModel):
        def build_model(self):
            return XGBClassifier(params=self.params, num_round=595)



PARAMS_V13 = {
        'colsample_bytree':0.55,'colsample_bylevel':0.85,
        'learning_rate':0.02,"eval_metric":"auc",
        'max_depth':6, 'min_child_weight':3,
        'nthread':8,'gamma':2.5,'reg_lambda':6,'reg_alpha':0,
        'objective':'binary:logistic','seed':23,
        'silent':1, 'subsample':0.86,'base_score':0.04
        }

class ModelV13(BaseModel):
        def build_model(self):
            return XGBClassifier(params=self.params, num_round=400)


PARAMS_V14 = {
        'colsample_bytree':0.8,'colsample_bylevel':0.8,
        'learning_rate':0.01,"eval_metric":"auc",
        'max_depth':6, 'min_child_weight':1,
        'nthread':8,'gamma':5,'reg_lambda':1,'reg_alpha':0,
        'objective':'binary:logistic','seed':407,
        'silent':1, 'subsample':0.7,'base_score':0.04
            }

class ModelV14(BaseModel):
        def build_model(self):
            return XGBClassifier(params=self.params, num_round=570)


PARAMS_V15 = {
        'colsample_bytree':0.6,'colsample_bylevel':0.85,
        'learning_rate':0.015,"eval_metric":"auc",
        'max_depth':6, 'min_child_weight':3,
        'nthread':8,'gamma':2,'reg_lambda':8,'reg_alpha':0.01,
        'objective':'binary:logistic','seed':407,
        'silent':1, 'subsample':0.76,'base_score':0.04
        }

class ModelV15(BaseModel):
        def build_model(self):
            return XGBClassifier(params=self.params, num_round=580)

PARAMS_V16 = {
        'colsample_bytree':0.85,
        'learning_rate':0.01,"eval_metric":"auc",
        'max_depth':5, 'min_child_weight':1,
        'nthread':8,'gamma':0.0,'reg_lambda':1,'reg_alpha':0.0,
        'objective':'binary:logistic','seed':407,
        'silent':1, 'subsample':0.736,'base_score':0.04
        }

class ModelV16(BaseModel):
        def build_model(self):
            return XGBClassifier(params=self.params, num_round=570)

PARAMS_V17 = {
        'colsample_bytree':0.62,'colsample_bylevel':0.9,
        'learning_rate':0.01,"eval_metric":"auc",
        'max_depth':5, 'min_child_weight':1,
        'nthread':8,'gamma':0.5,'reg_lambda':4.5,'reg_alpha':0.001,
        'objective':'binary:logistic','random_state':434,
        'silent':1, 'subsample':0.80,'base_score':0.04
        }

class ModelV17(BaseModel):
        def build_model(self):
            return XGBClassifier(params=self.params, num_round=570)

PARAMS_V18 = {
        'colsample_bytree':0.89,
        'learning_rate':0.01,"eval_metric":"auc",
        'max_depth':5, 'min_child_weight':1,
        'nthread':8,'gamma':1.0,'reg_lambda':5.0,'reg_alpha':0.0001,
        'objective':'binary:logistic','seed':333,
        'silent':1, 'subsample':0.60,'base_score':0.04
        }

class ModelV18(BaseModel):
        def build_model(self):
            return XGBClassifier(params=self.params, num_round=565)

PARAMS_V19 = {
        'colsample_bytree':0.75,
        'learning_rate':0.01,"eval_metric":"auc",
        'max_depth':6, 'min_child_weight':1,
        'nthread':8,'gamma':1.2,'reg_lambda':7.0,'reg_alpha':0.001,
        'objective':'binary:logistic','seed':0,
        'silent':1, 'subsample':0.75,'base_score':0.04
        }

class ModelV19(BaseModel):
        def build_model(self):
            return XGBClassifier(params=self.params, num_round=595)




PARAMS_V20 = {
        'colsample_bytree':0.45,'colsample_bylevel':0.85,
        'learning_rate':0.02,"eval_metric":"auc",
        'max_depth':6, 'min_child_weight':3,
        'nthread':8,'gamma':2.5,'reg_lambda':6,'reg_alpha':0,
        'objective':'binary:logistic','seed':3231,
        'silent':1, 'subsample':0.86,'base_score':0.04
        }

class ModelV20(BaseModel):
        def build_model(self):
            return XGBClassifier(params=self.params, num_round=400)




if __name__ == "__main__":
    
    m = ModelV1(name="v1_stage1_ver1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V1,
                kind = 's', fold_name='set1'
                )
    m.run()


    m = ModelV2(name="v2_stage1_ver1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V2,
                kind = 's', fold_name='set2'
                )
    m.run()


    m = ModelV3(name="v3_stage1_ver1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V3,
                kind = 's', fold_name='set3'
                )
    m.run()


    m = ModelV4(name="v4_stage1_ver1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V4,
                kind = 's', fold_name='set4'
                )
    m.run()


    m = ModelV5(name="v5_stage1_ver1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V5,
                kind = 's', fold_name='set5'
                )
    m.run()


    m = ModelV6(name="v6_stage1_ver1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V6,
                kind = 's', fold_name='set6'
                )
    m.run()


    m = ModelV7(name="v7_stage1_ver1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V7,
                kind = 's', fold_name='set7'
                )
    m.run()


    m = ModelV8(name="v8_stage1_ver1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V8,
                kind = 's', fold_name='set8'
                )
    m.run()


    m = ModelV9(name="v9_stage1_ver1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V9,
                kind = 's', fold_name='set9'
                )
    m.run()


    m = ModelV10(name="v10_stage1_ver1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V10,
                kind = 's', fold_name='set10'
                )
    m.run()


    m = ModelV11(name="v11_stage1_ver1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V11,
                kind = 's', fold_name='set11'
                )
    m.run()


    m = ModelV12(name="v12_stage1_ver1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V12,
                kind = 's', fold_name='set12'
                )
    m.run()


    m = ModelV13(name="v13_stage1_ver1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V13,
                kind = 's', fold_name='set13'
                )
    m.run()


    m = ModelV14(name="v14_stage1_ver1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V14,
                kind = 's', fold_name='set14'
                )
    m.run()


    m = ModelV15(name="v15_stage1_ver1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V15,
                kind = 's', fold_name='set15'
                )
    m.run()


    m = ModelV16(name="v16_stage1_ver1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V16,
                kind = 's', fold_name='set16'
                )
    m.run()


    m = ModelV17(name="v17_stage1_ver1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V17,
                kind = 's', fold_name='set17'
                )
    m.run()


    m = ModelV18(name="v18_stage1_ver1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V18,
                kind = 's', fold_name='set18'
                )
    m.run()


    m = ModelV19(name="v19_stage1_ver1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V19,
                kind = 's', fold_name='set19'
                )
    m.run()


    m = ModelV20(name="v20_stage1_ver1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V20,
                kind = 's', fold_name='set20'
                )
    m.run()


    
    print 'Done stage 1'

    print 'Start averaging'
    # averaging
    sample_sub = pd.read_csv('data/input/sample_submission.csv')
    testID = sample_sub[['ID']]
    a = pd.DataFrame()
    for i in ['data/output/temp/v1_stage1_ver1_test.csv',
              'data/output/temp/v2_stage1_ver1_test.csv',
              'data/output/temp/v3_stage1_ver1_test.csv',
              'data/output/temp/v4_stage1_ver1_test.csv',
              'data/output/temp/v5_stage1_ver1_test.csv',
              'data/output/temp/v6_stage1_ver1_test.csv',
              'data/output/temp/v7_stage1_ver1_test.csv',
              'data/output/temp/v8_stage1_ver1_test.csv',
              'data/output/temp/v9_stage1_ver1_test.csv',
              'data/output/temp/v10_stage1_ver1_test.csv',
              'data/output/temp/v11_stage1_ver1_test.csv',
              'data/output/temp/v12_stage1_ver1_test.csv',
              'data/output/temp/v13_stage1_ver1_test.csv',
              'data/output/temp/v14_stage1_ver1_test.csv',
              'data/output/temp/v15_stage1_ver1_test.csv',
              'data/output/temp/v16_stage1_ver1_test.csv',
              'data/output/temp/v17_stage1_ver1_test.csv',
              'data/output/temp/v18_stage1_ver1_test.csv',
              'data/output/temp/v19_stage1_ver1_test.csv',
              'data/output/temp/v20_stage1_ver1_test.csv',
              ]:
        x = pd.read_csv(i)
        a = pd.concat([a, x],axis=1)
    #x['TARGET'] = (a.rank().mean(1))/a.shape[0]
    # just averaging
    x['TARGET'] = a.mean(1)
    x = pd.concat([testID, x[['TARGET']]], axis=1)
    x.to_csv('data/output/temp/test_xgb_ikkiver1_variantA.csv', index=None)
    #pubLB: 

    # averaging
    a = pd.DataFrame()
    train = pd.read_csv('data/input/train.csv')
    targetID = train[['ID']]
    for i in ['data/output/temp/v1_stage1_ver1_all_fold.csv',
              'data/output/temp/v2_stage1_ver1_all_fold.csv',
              'data/output/temp/v3_stage1_ver1_all_fold.csv',
              'data/output/temp/v4_stage1_ver1_all_fold.csv',
              'data/output/temp/v5_stage1_ver1_all_fold.csv',
              'data/output/temp/v6_stage1_ver1_all_fold.csv',
              'data/output/temp/v7_stage1_ver1_all_fold.csv',
              'data/output/temp/v8_stage1_ver1_all_fold.csv',
              'data/output/temp/v9_stage1_ver1_all_fold.csv',
              'data/output/temp/v10_stage1_ver1_all_fold.csv',
              'data/output/temp/v11_stage1_ver1_all_fold.csv',
              'data/output/temp/v12_stage1_ver1_all_fold.csv',
              'data/output/temp/v13_stage1_ver1_all_fold.csv',
              'data/output/temp/v14_stage1_ver1_all_fold.csv',
              'data/output/temp/v15_stage1_ver1_all_fold.csv',
              'data/output/temp/v16_stage1_ver1_all_fold.csv',
              'data/output/temp/v17_stage1_ver1_all_fold.csv',
              'data/output/temp/v18_stage1_ver1_all_fold.csv',
              'data/output/temp/v19_stage1_ver1_all_fold.csv',
              'data/output/temp/v20_stage1_ver1_all_fold.csv',
              ]:
        x = pd.read_csv(i)
        a = pd.concat([a, x],axis=1)
    #x['TARGET'] = (a.rank().mean(1))/a.shape[0]
    # just averaging
    x['TARGET'] = a.mean(1)
    x = pd.concat([targetID, x[['TARGET']]], axis=1)
    x.to_csv('data/output/temp/train_xgb_ikkiver1_variantA.csv', index=None)
    #pubLB: 
    print 'Done averaging'

    print 'rank transformation with train and test'
    #rank trafo with train and test
    tr = pd.read_csv('data/output/temp/train_xgb_ikkiver1_variantA.csv')
    te = pd.read_csv('data/output/temp/test_xgb_ikkiver1_variantA.csv')
    tr_te = pd.concat([tr, te])
    tr_te['TARGET'] = tr_te['TARGET'].rank()
    # scale [0,1]
    tr_te['TARGET'] = (tr_te['TARGET'] - tr_te['TARGET'].min()) / (tr_te['TARGET'].max() - tr_te['TARGET'].min())
    tr = tr_te.iloc[:len(tr),:]
    te = tr_te.iloc[len(tr):,:]
    tr.to_csv('data/output/temp/train_xgb_ikki_ver1.csv', index=False)
    te.to_csv('data/output/temp/test_xgb_ikki_ver1.csv', index=False)
    print 'Done rank transformation'

    print 'CV of each model per fold and averaging'
    # CV of each model and averaging
    from sklearn.metrics import roc_auc_score as AUC
    a = pd.DataFrame()
    set_idnex = 1
    set_data = pd.read_csv('data/input/5fold_20times.csv')
    y = train.TARGET
    for i in ['data/output/temp/v1_stage1_ver1_all_fold.csv',
              'data/output/temp/v2_stage1_ver1_all_fold.csv',
              'data/output/temp/v3_stage1_ver1_all_fold.csv',
              'data/output/temp/v4_stage1_ver1_all_fold.csv',
              'data/output/temp/v5_stage1_ver1_all_fold.csv',
              'data/output/temp/v6_stage1_ver1_all_fold.csv',
              'data/output/temp/v7_stage1_ver1_all_fold.csv',
              'data/output/temp/v8_stage1_ver1_all_fold.csv',
              'data/output/temp/v9_stage1_ver1_all_fold.csv',
              'data/output/temp/v10_stage1_ver1_all_fold.csv',
              'data/output/temp/v11_stage1_ver1_all_fold.csv',
              'data/output/temp/v12_stage1_ver1_all_fold.csv',
              'data/output/temp/v13_stage1_ver1_all_fold.csv',
              'data/output/temp/v14_stage1_ver1_all_fold.csv',
              'data/output/temp/v15_stage1_ver1_all_fold.csv',
              'data/output/temp/v16_stage1_ver1_all_fold.csv',
              'data/output/temp/v17_stage1_ver1_all_fold.csv',
              'data/output/temp/v18_stage1_ver1_all_fold.csv',
              'data/output/temp/v19_stage1_ver1_all_fold.csv',
              'data/output/temp/v20_stage1_ver1_all_fold.csv',
              ]:
        x = pd.read_csv(i)
        a = pd.concat([a, x],axis=1)

        cv_index = {}
        set_name = 'set{}'.format(set_idnex)
        for i in xrange(5):
            train_cv = set_data.loc[(set_data[set_name]!=i).values, set_name].index
            test_cv = set_data.loc[(set_data[set_name]==i).values, set_name].index
            cv_index[i] = {}
            cv_index[i]['train'] = train_cv.values
            cv_index[i]['test'] = test_cv.values

        skf = pd.DataFrame(cv_index).stack().T
        auc = []
        for i in xrange(5):
            #print AUC(y.ix[skf['test'][i]].values, x.ix[skf['test'][i]].values) 
            auc.append(AUC(y.ix[skf['test'][i]].values, x.ix[skf['test'][i]].values))

        set_idnex += 1
    print 'Per model, mean: {} std: {}'.format(np.mean(auc), np.std(auc))
    print 'Averaging AUC:{}'.format(AUC(y.values,a.mean(1).values))
    #AUC:0.8426893 

