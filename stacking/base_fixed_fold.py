#!/usr/bin/env python
# -*- coding: utf-8 -*-

######### General #########
import numpy as np
import pandas as pd
import os, sys, re
import tables
import itertools
import logging

######### Problem Type #########
######### Change!!!!!! #########
eval_type_list = ('logloss', 'auc', 'rmse')

problem_type_list = ('classification','regression')

classification_type_list = ('binary', 'multi-class')



######### PATH #########
######### Change main folder name #########
FOLDER_NAME = ''
PATH = ''
INPUT_PATH = 'data/input/' #path of original data and fold_index
OUTPUT_PATH = 'data/output/'
TEMP_PATH = 'data/output/temp/' #path of saving each stacking prediction
FEATURES_PATH = 'data/output/features/' #path of dataset created in feat_verXX.py


# for saving the submitted format file(save_pred_as_submit_format())
SUBMIT_FORMAT = 'sample_submission.csv'




######### BaseEstimator ##########
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.base import TransformerMixin

######### Keras #########
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import Callback

######### XGBoost #########
import xgboost as xgb

######### Evaluation ##########
from sklearn.metrics import log_loss as ll
from sklearn.metrics import roc_auc_score as AUC
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error

######### Vowpal Wabbit ##########
#import wabbit_wappa as ww
import os
from time import asctime, time
import subprocess
import csv













######### CV index #########
cv_id_name='cv_id' #change if using fixed cv_index file
n_folds = 5

def create_cv_id(train, n_folds_ = 5, cv_id_name=cv_id_name, seed=407):
    try:
        a = StratifiedKFold(train['target'],n_folds=n_folds_, shuffle=True, random_state=seed)
        print 'Done StratifiedKFold'
    except:
        a = KFold(len(train),n_folds=n_folds_, shuffle=True, random_state=seed)
        print 'Done Kfold'
    cv_index = a.test_folds
    np.save(INPUT_PATH + cv_id_name, cv_index)
    return 

######### Utils #########

#feature listを渡してデータを作成するutil関数
def load_data(flist, drop_duplicates=False):
    '''
    flistにシリアライゼーションを渡すことでより効率的に
    data構造をここで考慮
    '''
    flist_len = len(flist['train'])
    X_train = pd.DataFrame()
    test = pd.DataFrame()
    for i in xrange(flist_len):
        X_train = pd.concat([X_train,pd.read_csv(PATH+flist['train'][i])],axis=1)
        test = pd.concat([test,pd.read_csv(PATH+flist['test'][i])],axis=1)

    y_train = X_train['target']
    del X_train['target']
    #del test['t_id']
    #print X_train.columns
    #print test.columns
    assert( (False in X_train.columns == test.columns) == False)
    print 'train shape :{}'.format(X_train.shape)
    if drop_duplicates == True:
        #add for ver.12. Use later version than ver.12.
        #delete identical columns
        unique_col = X_train.T.drop_duplicates().T.columns
        X_train = X_train[unique_col]
        test = test[unique_col]
        assert( all(X_train.columns == test.columns))
        print 'train shape after concat and drop_duplicates :{}'.format(X_train.shape)

    # drop constant features
    #X_train = X_train.loc[:, (X_train != X_train.ix[0]).any()] 
    #test = test.loc[:, (test != test.ix[0]).any()] 

    #common_col = list(set(X_train.columns.tolist()) and set(test.columns.tolist()))
    #X_train = X_train[common_col]
    #test = test[common_col]
    #print 'shape after dropping constant features: {}'.format(X_train.shape)

    return X_train, y_train, test 

#最終予測結果を提出フォーマットで保存する
# ID is different by problem. So this function is disabled.
def save_pred_as_submit_format(pred_path, output_file, col_name=('ID', "TARGET")):
    print 'writing prediction as submission format'
    print 'read prediction <{}>'.format(pred_path)
    pred = pd.read_csv(pred_path).values
    #(((test.mean(1) - test.mean(1).mean())/test.mean(1).std()/100. + 0.5).values + pred)/2.0
    submission = pd.read_csv(INPUT_PATH+SUBMIT_FORMAT)
    submission[col_name[1]] = pred
    submission.to_csv( output_file, columns = col_name, index = None )
    print 'done writing'
    return

#evalation function
def eval_pred( y_true, y_pred, eval_type):
    if eval_type == 'logloss':#eval_typeはここに追加
        print "logloss: ", ll( y_true, y_pred )
        return ll( y_true, y_pred )             
    
    elif eval_type == 'auc':
        print "AUC: ", AUC( y_true, y_pred )
        return AUC( y_true, y_pred )             
    
    elif eval_type == 'rmse':
        print "rmse: ", np.sqrt(mean_squared_error(y_true, y_pred))
        return np.sqrt(mean_squared_error(y_true, y_pred))




######### BaseModel Class #########

class BaseModel(BaseEstimator):
    """
    Parameters of fit
    ----------
    FEATURE_LIST = {
                    'train':('flist_train.csv'),#targetはここに含まれる
                    'test':('flist_test.csv'),
                    }

    Note
    ----
    init: compiled model
    

    
    (Example)
    from base import BaseModel, XGBClassifier
    FEATURE_LIST = ["feat.group1.blp"]
    PARAMS = {
            'n_estimator':700,
            'sub_sample': 0.8,
            'seed': 71
        }
    class ModelV1(BaseModel):
         def build_model(self):
         return XGBClassifier(**self.params)


    if __name__ == "__main__":
        m = ModelV1(name="v1",
                    flist=FEATURE_LIST,
                    params=PARAMS,
                    kind='s')
        m.run()
   
    """
    
    # Problem type(class variables)
    # Need to be set by BaseModel.set_prob_type()
    problem_type = ''
    classification_type = ''
    eval_type = ''


    def __init__(self, name="", flist={}, params={}, kind='s', fold_name=cv_id_name):
        '''
        name: Model name
        flist: Feature list
        params: Parameters
        kind: Kind of run() 
        {'s': Stacking only. Svaing a oof prediction({}_all_fold.csv)
              and average of test prediction based on fold-train models({}_test.csv).
         't': Training all data and predict test({}_TestInAllTrainingData.csv).
         'st': Stacking and then training all data and predict test
               Using save final model with cross-validation
         'cv': Only cross validation without saving the prediction

        '''
        if BaseModel.problem_type == 'classification':
            if not ((BaseModel.classification_type in classification_type_list)
                     and (BaseModel.eval_type in eval_type_list)):
                raise ValueError('Problem Type, Classification Type, and Evaluation Type\
                        should be set before model defined')

        elif BaseModel.problem_type == 'regression':
            if not BaseModel.eval_type in eval_type_list:
                raise ValueError('Problem Type, and Evaluation Type\
                        should be set before model defined')

        else:
            raise ValueError('Problem Type, Classification Type, and Evaluation Type\
                        should be set before model defined')

        self.name = name
        self.flist = flist
        self.params = params
        self.kind = kind
        self.fold_name = fold_name
        assert(self.kind in ['s', 't', 'st', 'cv'])
        
    @classmethod
    def set_prob_type(cls, problem_type, classification_type, eval_type):
        """ Set problem type """
        assert problem_type in problem_type_list, 'Need to set Problem Type'
        if problem_type == 'classification':
            assert classification_type in classification_type_list,\
                                            'Need to set Classification Type'
        assert eval_type in eval_type_list, 'Need to set Evaluation Type'
        
        cls.problem_type = problem_type
        cls.classification_type = classification_type
        cls.eval_type = eval_type
        
        if cls.problem_type == 'classification':
            print 'Setting Problem:{}, Type:{}, Eval:{}'.format(cls.problem_type,
                                                                cls.classification_type,
                                                                cls.eval_type)

        elif cls.problem_type == 'regression':
            print 'Setting Problem:{}, Eval:{}'.format(cls.problem_type,
                                                        cls.eval_type)

        return



    def build_model(self):
        return None

    def make_multi_cols(self, num_class, name):
        '''make cols for multi-class predictions'''
        cols = ['c' + str(i) + '_' for i in xrange(num_class)]
        cols = map(lambda x: x + name, cols)
        return cols


    def run(self):
        print 'running model: {}'.format(self.name)
        X, y, test = self.load_data()
        num_class = len(set(y)) # only for multi-class classification
        #print X.shape, test.shape
        #print X.dropna().shape, test.dropna().shape

        if self.kind == 't':
            clf = self.build_model()
            clf.fit(X, y)
            
            if BaseModel.problem_type == 'classification':
                y_submission = clf.predict_proba(test)#[:,1]#multi-class => 消す #コード変更
                
                if BaseModel.classification_type == 'binary':
                    y_submission = pd.DataFrame(y_submission,columns=['{}_pred'.format(self.name)])
                    y_submission.to_csv(TEMP_PATH+'{}_TestInAllTrainingData.csv'.format(self.name),index=False)

                elif BaseModel.classification_type == 'multi-class':
                    saving_cols = self.make_multi_cols(num_class, '{}_pred'.format(self.name))
                    y_submission = pd.DataFrame(y_submission,columns=saving_cols)
                    y_submission.to_csv(TEMP_PATH+'{}_TestInAllTrainingData.csv'.format(self.name),index=False)


            elif BaseModel.problem_type == 'regression':
                y_submission = clf.predict(test)#[:,1]#multi-class => 消す #コード変更
                y_submission = pd.DataFrame(y_submission,columns=['{}_pred'.format(self.name)])
                y_submission.to_csv(TEMP_PATH+'{}_TestInAllTrainingData.csv'.format(self.name),index=False)
            #save_pred_as_submit_format(TEMP_PATH+'{}_TestInAllTrainingData.csv'.format(self.name), TEMP_PATH+'OK_{}_TestInAllTrainingData.csv'.format(self.name))
            return 0 #保存して終了
        
        #print 'using fold: {}'.format(self.fold_name)
        #skf = pd.read_pickle(INPUT_PATH+cv_id_filename)
        print 'loading cv_fold file'
        a = np.load(INPUT_PATH + self.fold_name + '.npy')
        #cv_index = {}
        #set_name = self.fold_name
        #creating cv_index format
        #for i in xrange(5):
        #    train_cv = a.loc[(a[set_name]!=i).values, set_name].index
        #    test_cv = a.loc[(a[set_name]==i).values, set_name].index
        #    cv_index[i] = {}
        #    cv_index[i]['train'] = train_cv.values
        #    cv_index[i]['test'] = test_cv.values

        #skf = pd.DataFrame(cv_index).stack().T
        clf = self.build_model()
        print "Creating train and test sets for stacking."
        #print "\nLevel 0"

        ############# for binary #############
        if BaseModel.problem_type == 'regression' or BaseModel.classification_type == 'binary':
            dataset_blend_train = np.zeros(X.shape[0]) #trainの予測結果の保存
            dataset_blend_test = np.zeros(test.shape[0]) #testの予測結果の保存
    
            #stacked_data_columns = X.columns.tolist()
            dataset_blend_test_j = np.zeros((test.shape[0], n_folds))
        
        ############# for multi-class #############
        elif BaseModel.classification_type == 'multi-class':
            #TODO
            #trainの予測結果の保存
            dataset_blend_train = np.zeros(X.shape[0]*num_class).reshape((X.shape[0],num_class))
            #testの予測結果の保存
            dataset_blend_test = np.zeros(test.shape[0]*num_class).reshape((test.shape[0],num_class))
            #stacked_data_columns = X.columns.tolist()
            #dataset_blend_test_j = np.zeros((test.shape[0], n_folds))



        evals = []
        for i in xrange(n_folds):# of n_folds
            train_fold = (a!=i)
            test_fold = (a==i)
            print "Fold", i
            #print X
            #print train_fold
            X_train = X[train_fold].dropna(how='all')
            y_train = y[train_fold].dropna(how='all')
            X_test = X[test_fold].dropna(how='all')
            y_test = y[test_fold].dropna(how='all')
            
            #print X_train,y_train,X_test,y_test
            clf.fit(X_train, y_train)

            if BaseModel.problem_type == 'classification' and BaseModel.classification_type == 'binary':            
                #if using the mean of the prediction of each n_fold
                #print str(type(clf))
                if 'sklearn' in str(type(clf)):
                    y_submission = clf.predict_proba(X_test)[:,1]
                else:
                    y_submission = clf.predict_proba(X_test)

            elif BaseModel.problem_type == 'classification' and BaseModel.classification_type == 'multi-class':
                if 'sklearn' in str(type(clf)):
                    y_submission = clf.predict_proba(X_test) #Check!!
                else:
                    y_submission = clf.predict_proba(X_test)

            elif BaseModel.problem_type == 'regression':      
                y_submission = clf.predict(X_test)

            #add .values for numpy.
            #print test_fold
            #print y_submission
            #print dataset_blend_train
            #dataset_blend_train[test_fold.values] = y_submission
            #depend version of numpy
            try:
                dataset_blend_train[test_fold] = y_submission
            except:
                dataset_blend_train[test_fold.values] = y_submission
            
            #外に持ってく
            evals.append(eval_pred(y_test, y_submission, BaseModel.eval_type))

            ############ binary classification ############
            if BaseModel.problem_type == 'classification' and BaseModel.classification_type == 'binary':            
                #if using the mean of the prediction of each n_fold
                if 'sklearn' in str(type(clf)):
                    dataset_blend_test += clf.predict_proba(test)[:,1]
                else:
                    dataset_blend_test += clf.predict_proba(test)

            ############ multi-class classification ############
            elif BaseModel.problem_type == 'classification' and BaseModel.classification_type == 'multi-class':            
                #if using the mean of the prediction of each n_fold
                dataset_blend_test += clf.predict_proba(test)
                pass

            ############ regression ############
            elif BaseModel.problem_type == 'regression':      
                #if using the mean of the prediction of each n_fold
                dataset_blend_test += clf.predict(test)


        dataset_blend_test /= n_folds
        
        for i in xrange(n_folds):
            print 'Fold{}: {}'.format(i+1, evals[i])
        print '{} Mean: '.format(BaseModel.eval_type), np.mean(evals), ' Std: ', np.std(evals)

        #Saving 上でモデルの保存も追加できる
        if self.kind != 'cv':
            print 'Saving results'

            if (BaseModel.problem_type == 'classification' and BaseModel.classification_type == 'binary') or (BaseModel.problem_type == 'regression'):
                dataset_blend_train = pd.DataFrame(dataset_blend_train,columns=['{}_stack'.format(self.name)])
                dataset_blend_train.to_csv(TEMP_PATH+'{}_all_fold.csv'.format(self.name),index=False)
                dataset_blend_test = pd.DataFrame(dataset_blend_test,columns=['{}_stack'.format(self.name)])
                dataset_blend_test.to_csv(TEMP_PATH+'{}_test.csv'.format(self.name),index=False)
                

            elif BaseModel.problem_type == 'classification' and BaseModel.classification_type == 'multi-class':
                saving_cols = self.make_multi_cols(num_class, '{}_stack'.format(self.name))
                dataset_blend_train = pd.DataFrame(dataset_blend_train,columns=saving_cols)
                dataset_blend_train.to_csv(TEMP_PATH+'{}_all_fold.csv'.format(self.name),index=False)

                dataset_blend_test = pd.DataFrame(dataset_blend_test,columns=saving_cols)
                dataset_blend_test.to_csv(TEMP_PATH+'{}_test.csv'.format(self.name),index=False)
                


        
        if self.kind == 'st':
            #Stacking(cross-validation)後に全データで学習
            clf = self.build_model()
            clf.fit(X, y)
            if BaseModel.problem_type == 'classification':

                if BaseModel.classification_type == 'binary':
                    if 'sklearn' in str(type(clf)):
                        y_submission = clf.predict_proba(test)[:,1]#multi-class => 消す #コード変更
                    else:
                        y_submission = clf.predict_proba(test)

                    y_submission = pd.DataFrame(y_submission,columns=['{}_pred'.format(self.name)])
                    y_submission.to_csv(TEMP_PATH+'{}_TestInAllTrainingData.csv'.format(self.name),index=False)

                elif BaseModel.classification_type == 'multi-class':
                    y_submission = clf.predict_proba(test)
                    saving_cols = self.make_multi_cols(num_class, '{}_pred'.format(self.name))
                    y_submission = pd.DataFrame(y_submission,columns=saving_cols)
                    y_submission.to_csv(TEMP_PATH+'{}_TestInAllTrainingData.csv'.format(self.name),index=False)

            elif BaseModel.problem_type == 'regression':
                y_submission = clf.predict(test)#[:,1]#multi-class => 消す #コード変更
            
                y_submission = pd.DataFrame(y_submission,columns=['{}_pred'.format(self.name)])
                y_submission.to_csv(TEMP_PATH+'{}_TestInAllTrainingData.csv'.format(self.name),index=False)

        return



    def load_data(self):
        '''
        flistにシリアライゼーションを渡すことでより効率的に
        data構造をここで考慮
        '''
        return load_data(self.flist, drop_duplicates=True )
        




################################################
######### Wrapper Class of Classifiers #########
################################################



class IntervalEvaluation(Callback):
    def __init__(self, validation_data=(), interval=10):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict_proba(self.X_val, verbose=0)
            score = AUC(self.y_val, y_pred)
            #logging.info("interval evaluation - epoch: {:d} - score: {:.6f}".format(epoch, score))
            print "interval evaluation - epoch: {:d} - score: {:.6f}".format(epoch, score)


class KerasClassifier(BaseEstimator, ClassifierMixin):
    """
    (Example)
    from base import KerasClassifier
    class KerasModelV1(KerasClassifier):
        ###
        #Parameters for lerning
        #    batch_size=128,
        #    nb_epoch=100,
        #    verbose=1, 
        #    callbacks=[],
        #    validation_split=0.,
        #    validation_data=None,
        #    shuffle=True,
        #    class_weight=None,
        #    sample_weight=None,
        #    normalize=True,
        #    categorize_y=False
        ###
        
        def __init__(self,**params):
            model = Sequential()
            model.add(Dense(input_dim=X.shape[1], output_dim=100, init='uniform', activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(input_dim=50,output_dim=2, init='uniform'))
            model.add(Activation('softmax'))
            model.compile(optimizer='adam', loss='binary_crossentropy',class_mode='binary')

            super(KerasModelV1, self).__init__(model,**params)
    
    KerasModelV1(batch_size=8, nb_epoch=10, verbose=1, callbacks=[], validation_split=0., validation_data=None, shuffle=True, class_weight=None, sample_weight=None, normalize=True, categorize_y=True)
    KerasModelV1.fit(X_train, y_train,validation_data=[X_test,y_test])
    KerasModelV1.predict_proba(X_test)[:,1]
    """

    def __init__(self,nn,batch_size=128, nb_epoch=100, verbose=1, callbacks=[],
            validation_split=0., validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None, normalize=True, categorize_y=False):
        self.nn = nn
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.verbose = verbose
        self.callbacks = callbacks
        self.validation_split = validation_split
        self.validation_data = validation_data
        self.shuffle = shuffle
        self.class_weight = class_weight
        self.sample_weight = sample_weight
        self.normalize = normalize
        self.categorize_y = categorize_y
        #set initial weights
        self.init_weight = self.nn.get_weights()

    def fit(self, X, y, validation_data=None):
        X = X.values#Need for Keras
        y = y.values#Need for Keras
        if validation_data != None:
            self.validation_data = validation_data
            if self.normalize:
                self.validation_data[0] = (validation_data[0] - np.mean(validation_data[0],axis=0))/np.std(validation_data[0],axis=0)
            if self.categorize_y:
                self.validation_data[1] = np_utils.to_categorical(validation_data[1])

        if self.normalize:
            self.mean = np.mean(X,axis=0)
            self.std = np.std(X,axis=0) + 1 #CAUSION!!!
            X = (X - self.mean)/self.std
        if self.categorize_y:
            y = np_utils.to_categorical(y)
        
        #set initial weights
        self.nn.set_weights(self.init_weight)

        #set callbacks
        #self.callbacks = [IntervalEvaluation(validation_data=(X, y), interval=5)]
        #print self.callbacks

        #print all(pd.DataFrame(np.isfinite(X)))
        #print X.shape
        return self.nn.fit(X, y, batch_size=self.batch_size, nb_epoch=self.nb_epoch, verbose=self.verbose, callbacks=self.callbacks, validation_split=self.validation_split, validation_data=self.validation_data, shuffle=self.shuffle, class_weight=self.class_weight, sample_weight=self.sample_weight)

    def predict_proba(self, X, batch_size=128, verbose=1):
        X = X.values#Need for Keras
        if self.normalize:
            X = (X - self.mean)/self.std
        
        if BaseModel.classification_type == 'binary':
            return self.nn.predict_proba(X, batch_size=batch_size, verbose=verbose)[:,1]#multi-class => 消す #コード変更
        elif BaseModel.classification_type == 'multi-class':
            return self.nn.predict_proba(X, batch_size=batch_size, verbose=verbose)


class XGBClassifier(BaseEstimator, ClassifierMixin):
    '''
    (Example)
    from base import XGBClassifier
    class XGBModelV1(XGBClassifier):
        def __init__(self,**params):
            super(XGBModelV1, self).__init__(**params)

    a = XGBModelV1(colsample_bytree=0.9, learning_rate=0.01,max_depth=5, min_child_weight=1,n_estimators=300, nthread=-1, objective='binary:logistic', seed=0,silent=True, subsample=0.8)
    a.fit(X_train, y_train, eval_metric='logloss',eval_set=[(X_train, y_train),(X_test, y_test)])
    
    '''
    def __init__(self, params={}, num_round=50 ):
        self.params = params
        self.num_round = num_round

        self.clf = xgb
        
    def fit(self, X, y=[], sample_weight=None, eval_set=None, eval_metric=None,
            early_stopping_rounds=None, verbose=True):
        
        dtrain = xgb.DMatrix(X, label=y,missing=-999)
        
        watchlist  = [(dtrain,'train')]
        
        self.clf = xgb.train(self.params, dtrain, self.num_round, watchlist)
        return self.clf

    def predict_proba(self, X, output_margin=False, ntree_limit=0):
        dtest = xgb.DMatrix(X,missing=-999)
        #return self.clf.predict(X, output_margin=output_margin, ntree_limit=ntree_limit)
        
        return self.clf.predict(dtest)


class VWClassifier(BaseEstimator, ClassifierMixin):
    """
    PARAMS = {
    'trainCommand' : ("vw --loss_function logistic --l2 0.001 --learning_rate 0.015 --link=logistic --passes 20 --decay_learning_rate 0.97 --power_t 0 -d {}train_vw.data --cache_file vw.cache -f {}vw.model -b 20".format(TEMP_PATH, TEMP_PATH)).split(' '), \

    'predictCommand': ("vw -t -d {}test_vw.data -i {}vw.model -p {}vw.predict".format(TEMP_PATH,TEMP_PATH,TEMP_PATH)).split(' ')
    
    }
    """
    
    
    def __init__(self, trainCommand="", predictCommand="", train_vw_data="train_vw.data"):
        self.trainCommand = trainCommand
        self.predictCommand = predictCommand
        self.environmentDict = dict(os.environ, LD_LIBRARY_PATH='/usr/local/lib') 
        self.train_vw_data = train_vw_data

    def genTrainInstances(self, aRow):  
        #index = str(aRow['index'])
        #urlid = str(aRow['urlid'])
        y_row = str(int(float(aRow['target']))  )
        #rowtag = userid
        #rowText = (y_row + " 1.0  " + index)
        rowText = y_row
        col_names = aRow.index
        for i in col_names:
            if i in ['index','target']:
                continue
            rowText += " |{} {}:".format(i,i) + str(aRow[i])
        return  rowText

    def genTestInstances(self, aRow):  
        y_row = str(1)
        #index = str(aRow['index'])
        #urlid = str(aRow['urlid'])
        #rowtag = userid
        #rowText = (y_row + " 1.0  " + index)
        rowText = y_row
        col_names = aRow.index
        for i in col_names:
            if i in ['index','target']:
                continue
            rowText += " |{} {}:".format(i,i) + str(aRow[i])
        return  rowText

    def fit(self, X, y=[]):
        #delete vw.cache 
        subprocess.call(['rm','-f','{}vw.cache'.format(PATH)])
        #global df_train, trainCommand, environmentDict
        y = y.apply(lambda x: -1 if x < 1 else 1)
        X = pd.concat([X, y], axis=1)
        X = X.reset_index()
        #if os.path.isfile(TEMP_PATH+self.train_vw_data) == False:
        print "Generating VW Training Instances: ", asctime()
        X['TrainInstances'] = X.apply(self.genTrainInstances, axis=1)
        print "Finished Generating Train Instances: ", asctime()

        #if os.path.isfile(TEMP_PATH+self.train_vw_data) == False:
        print "Writing Train Instances To File: ", asctime()
        trainInstances = list(X['TrainInstances'].values)
        f = open(TEMP_PATH+'train_vw.data','w')
        f.writelines(["%s\n" % row  for row in trainInstances])
        f.close()
        print "Finished Writing Train Instances: ", asctime()
        #else:
        #    print 'already generated {}'.format(train_vw_data)

        subprocess.call(self.trainCommand)
        print "Finished Training: ", asctime()      
        return

    def readPredictFile(self):
        parseStr = lambda x: float(x) if '.' in x else int(x)
        y_pred = []
        with open(TEMP_PATH+'vw.predict', 'rb') as csvfile:
            predictions = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in predictions:
                pred = parseStr(row[0])
                y_pred.append(pred)
        return np.asarray(y_pred)  


    def predict_model(self,test):
        #global environmentDict, predictCommand, df_test
        test = test.reset_index()
        print "Building Test Instances: ", asctime()
        test['TestInstances'] = test.apply(self.genTestInstances, axis=1)
        print "Finished Generating Test Instances: ", asctime()

        print "Writing Test Instances: ", asctime()
        testInstances = list(test['TestInstances'].values)
        f = open(TEMP_PATH+'test_vw.data','w')
        f.writelines(["%s\n" % row  for row in testInstances])
        f.close()
        print "Finished Writing Test Instances: ", asctime()

        subprocess.call(self.predictCommand, env=self.environmentDict)

        #df_test['y_pred'] = readPredictFile()
        return self.readPredictFile()

    def predict_proba(self, X):
        if BaseModel.classification_type == 'binary':
            return self.predict_model(X)
        elif BaseModel.classification_type == 'multi-class':
            return self.predict_model(X) #Check!



###########################################
######### Regressor Wrapper Class #########
###########################################

class KerasRegressor(BaseEstimator, RegressorMixin):
    """
    (Example)
    from base import KerasClassifier
    class KerasModelV1(KerasClassifier):
        ###
        #Parameters for lerning
        #    batch_size=128,
        #    nb_epoch=100,
        #    verbose=1, 
        #    callbacks=[],
        #    validation_split=0.,
        #    validation_data=None,
        #    shuffle=True,
        #    class_weight=None,
        #    sample_weight=None,
        #    normalize=True,
        #    categorize_y=False
        ###
        
        def __init__(self,**params):
            model = Sequential()
            model.add(Dense(input_dim=X.shape[1], output_dim=100, init='uniform', activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(input_dim=50,output_dim=1, init='uniform'))
            model.add(Activation('linear')
            #################### CAUSION ####################
            # Change the output of last layer to 1          #
            # Change the loss to mse or mae                 #
            # Using mse loss results in faster convergence  #
            #################################################
            model.compile(optimizer='rmsprop', loss='mean_absolute_error')#'mean_squared_error'

            super(KerasModelV1, self).__init__(model,**params)
    
    KerasModelV1(batch_size=8, nb_epoch=10, verbose=1, callbacks=[], validation_split=0., validation_data=None, shuffle=True, class_weight=None, sample_weight=None, normalize=True, categorize_y=True)
    KerasModelV1.fit(X_train, y_train,validation_data=[X_test,y_test])
    KerasModelV1.predict_proba(X_test)[:,1]
    """

    def __init__(self,nn,batch_size=128, nb_epoch=100, verbose=1, callbacks=[],
            validation_split=0., validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None, 
            normalize=True, categorize_y=False, random_sampling=None):
        self.nn = nn
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.verbose = verbose
        self.callbacks = callbacks
        self.validation_split = validation_split
        self.validation_data = validation_data
        self.shuffle = shuffle
        self.class_weight = class_weight
        self.sample_weight = sample_weight
        self.normalize = normalize
        self.categorize_y = categorize_y
        self.random_sampling = random_sampling
        #set initial weights
        self.init_weight = self.nn.get_weights()

    def fit(self, X, y, validation_data=None):
        
        if self.random_sampling != None:
            self.sampling_col = np.random.choice(range(X.shape[1]),self.random_sampling,replace=False)
            X = X.iloc[:,self.sampling_col].values#Need for Keras
        else:
            X = X.values#Need for Keras

        y = y.values#Need for Keras
        if validation_data != None:
            self.validation_data = validation_data
            if self.normalize:
                self.validation_data[0] = (validation_data[0] - np.mean(validation_data[0],axis=0))/np.std(validation_data[0],axis=0)
            #if self.categorize_y:
            #    self.validation_data[1] = np_utils.to_categorical(validation_data[1])

        if self.normalize:
            self.mean = np.mean(X,axis=0)
            self.std = np.std(X,axis=0) + 1 #CAUSION!!!
            X = (X - self.mean)/self.std

        #if self.categorize_y:
        #    y = np_utils.to_categorical(y)
            
        #set initial weights
        self.nn.set_weights(self.init_weight)
        print X.shape
        return self.nn.fit(X, y, batch_size=self.batch_size, nb_epoch=self.nb_epoch, verbose=self.verbose, callbacks=self.callbacks, validation_split=self.validation_split, validation_data=self.validation_data, shuffle=self.shuffle, class_weight=self.class_weight, sample_weight=self.sample_weight)

    def predict(self, X, batch_size=128, verbose=1):
        if self.random_sampling != None:
            X = X.iloc[:,self.sampling_col].values
        else:
            X = X.values#Need for Keras
        if self.normalize:
            X = (X - self.mean)/self.std
        
        return [ pred_[0] for pred_ in self.nn.predict(X, batch_size=batch_size, verbose=verbose)]
    

class XGBRegressor(BaseEstimator, RegressorMixin):
    """
    (Example)
    from base import XGBClassifier
    class XGBModelV1(XGBClassifier):
        def __init__(self,**params):
            super(XGBModelV1, self).__init__(**params)

    a = XGBModelV1(colsample_bytree=0.9, learning_rate=0.01,max_depth=5, min_child_weight=1,n_estimators=300, nthread=-1, objective='reg:linear', seed=0,silent=True, subsample=0.8)
    a.fit(X_train, y_train, eval_metric='logloss',eval_set=[(X_train, y_train),(X_test, y_test)])
    
    #boosterを指定したいのでRegressorだけ先にxgb.trainを使用する
    """
    def __init__(self, params={}, num_round=50 ):
        self.params = params
        self.num_round = num_round

        self.clf = xgb
        
    def fit(self, X, y=[], sample_weight=None, eval_set=None, eval_metric=None,
            early_stopping_rounds=None, verbose=True):
        #return self.clf.fit(X,y, sample_weight=sample_weight, eval_set=eval_set, eval_metric=eval_metric,early_stopping_rounds=early_stopping_rounds, verbose=verbose)
        
        #weights = y.apply(lambda x: 0.5 - (y.max() - x)/float(y.max() - y.min())*0.5 + 0.5)
        #dtrain = xgb.DMatrix(X, label=y,missing=-999, weight=weights)
        
        dtrain = xgb.DMatrix(X, label=y,missing=-999)
        
        watchlist  = [(dtrain,'train')]
        
        #num_round = self.num_round
        #print self.clf
        self.clf = xgb.train(self.params, dtrain, self.num_round, watchlist)
        return self.clf

    def predict(self, X, output_margin=False, ntree_limit=0):
        dtest = xgb.DMatrix(X,missing=-999)
        #return self.clf.predict(X, output_margin=output_margin, ntree_limit=ntree_limit)
        
        return self.clf.predict(dtest)


class VWRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

