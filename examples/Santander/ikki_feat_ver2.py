#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
#os.chdir('/Users/IkkiTanaka/Documents/kaggle/Santander/')

#各種PATH
from stacking.base import FOLDER_NAME, PATH, INPUT_PATH, OUTPUT_PATH, ORIGINAL_TRAIN_FORMAT, SUBMIT_FORMAT


#import bloscpack
from datetime import date

from sklearn.manifold import TSNE

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, PolynomialFeatures, MinMaxScaler



########### Feature Engineering ############

######### Reading data ###########
ori_train = pd.read_csv('data/input/train.csv')
ori_test = pd.read_csv('data/input/test.csv')
sample_submit = pd.read_csv('data/input/sample_submission.csv')

ori_train['target'] = ori_train['TARGET']
ori_train['t_id'] = ori_train["ID"]
ori_test['t_id'] = ori_test["ID"]

del ori_train['TARGET'], ori_train["ID"], ori_test["ID"]



########## Cleaning data based on reverse feature engineering ###########

def cleaning_rfe(ori_train, ori_test):
    print 'cleaning...'
    # bad features for delete such as A = {1,2,3} * B, 
    bad = ['num_var6_0', 'num_var6', 'num_var8', 'num_var13_medio_0', 'num_var13_medio', 'num_var18_0', 'num_var18', 'num_var20_0', 'num_var20', 'num_var29_0', 'num_var29', 'num_var34_0', 'num_var34', 'num_var44', 'delta_imp_amort_var18_1y3', 'delta_imp_amort_var34_1y3', 'num_var7_emit_ult1', 'num_meses_var13_medio_ult3']

    for i in bad:
        del ori_train[i], ori_test[i]
    assert( all(ori_train.columns == ori_test.columns))

    
    # Dataframe for saving
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()


    # temporal information?
    # (ori_train[['num_var45_hace3','num_var45_hace2','num_var45_ult1']] + 1).T.pct_change().T
    tinfo = [
            ['num_op_var40_hace3','num_op_var40_hace2','num_op_var40_ult1','num_op_var40_ult3'],
            ['num_op_var41_hace3','num_op_var41_hace2','num_op_var41_ult1','num_op_var41_ult3'],
            ['num_var22_hace3', 'num_var22_hace2', 'num_var22_ult1', 'num_var22_ult3'],
            ['num_var45_hace3', 'num_var45_hace2', 'num_var45_ult1', 'num_var45_ult3'],
            ]

    #pct_change and sub
    for i in tinfo:
        a_tr = (ori_train[i] + 0.01).T.pct_change().T.iloc[:,1:]
        a_tr.columns += '_pct_change'
        a_te = (ori_test[i] + 0.01).T.pct_change().T.iloc[:,1:]
        a_te.columns += '_pct_change'
        train_df = pd.concat([train_df, a_tr], axis=1)
        test_df = pd.concat([test_df, a_te], axis=1)
        #sub
        train_df['{}-{}'.format(i[3],i[2])] = ori_train[i[3]] - ori_train[i[2]]
        train_df['{}-{}'.format(i[3],i[1])] = ori_train[i[3]] - ori_train[i[1]]
        train_df['{}-{}'.format(i[3],i[0])] = ori_train[i[3]] - ori_train[i[0]]
        test_df['{}-{}'.format(i[3],i[2])] = ori_test[i[3]] - ori_test[i[2]]
        test_df['{}-{}'.format(i[3],i[1])] = ori_test[i[3]] - ori_test[i[1]]
        test_df['{}-{}'.format(i[3],i[0])] = ori_test[i[3]] - ori_test[i[0]]
    assert( all(train_df.columns == test_df.columns))



    # comer
    comer = [
            ['imp_op_var39_comer_ult1','imp_op_var40_comer_ult1','imp_op_var41_comer_ult1'],
            ['imp_op_var39_comer_ult3','imp_op_var40_comer_ult3','imp_op_var41_comer_ult3'],
            ['num_op_var39_comer_ult1','num_op_var40_comer_ult1','num_op_var41_comer_ult1'],
            ['num_op_var39_comer_ult3','num_op_var40_comer_ult3','num_op_var41_comer_ult3'],
            ]

    def percent_row(data):
        #print data
        lhs = data.index[0]
        sum_ = data[1:].sum()
        data.index += '_pct_imp_op_' + lhs
        return pd.Series(data[1:])/(sum_+0.1)

    for i in xrange(len(comer)):
        train_df = pd.concat([train_df, ori_train[comer[i]].apply(percent_row, axis=1)],axis=1)
        test_df = pd.concat([test_df, ori_test[comer[i]].apply(percent_row, axis=1)],axis=1)
    assert( all(train_df.columns == test_df.columns))



    # efect
    efect = [
            ['imp_op_var39_efect_ult1','imp_op_var40_efect_ult1','imp_op_var41_efect_ult1'],
            ['imp_op_var39_efect_ult3','imp_op_var40_efect_ult3','imp_op_var41_efect_ult3'],
            ['num_op_var39_efect_ult1','num_op_var40_efect_ult1','num_op_var41_efect_ult1'],
            ['num_op_var39_efect_ult3','num_op_var40_efect_ult3','num_op_var41_efect_ult3'],
            ]

    def percent_row(data):
        #print data
        lhs = data.index[0]
        sum_ = data[1:].sum()
        data.index += '_pct_imp_op_' + lhs
        return pd.Series(data[1:])/(sum_+0.1)

    for i in xrange(len(efect)):
        train_df = pd.concat([train_df, ori_train[efect[i]].apply(percent_row, axis=1)],axis=1)
        test_df = pd.concat([test_df, ori_test[efect[i]].apply(percent_row, axis=1)],axis=1)
    assert( all(train_df.columns == test_df.columns))



    # imp_op
    imp_op = [['imp_op_var39_ult1','imp_op_var40_ult1','imp_op_var41_ult1']]

    def percent_row(data):
        #print data
        lhs = data.index[0]
        sum_ = data[1:].sum()
        data.index += '_pct_imp_op_' + lhs
        return pd.Series(data[1:])/(sum_+0.1)

    for i in xrange(len(imp_op)):
        train_df = pd.concat([train_df, ori_train[imp_op[i]].apply(percent_row, axis=1)],axis=1)
        test_df = pd.concat([test_df, ori_test[imp_op[i]].apply(percent_row, axis=1)],axis=1)
    assert( all(train_df.columns == test_df.columns))



    # num_op
    num_op = [
            ['num_op_var39_hace2','num_op_var40_hace2','num_op_var41_hace2'],
            ['num_op_var39_hace3','num_op_var40_hace3','num_op_var41_hace3'],
            ['num_op_var39_ult1','num_op_var40_ult1','num_op_var41_ult1'],
            ['num_op_var39_ult3','num_op_var40_ult1','num_op_var40_hace2','num_op_var40_hace3','num_op_var41_ult1','num_op_var41_hace2','num_op_var41_hace3'],
            ]

    def percent_row(data):
        #print data
        lhs = data.index[0]
        sum_ = data[1:].sum()
        data.index += '_pct_num_op_' + lhs
        return pd.Series(data[1:])/(sum_+0.1)

    for i in xrange(len(num_op)):
        train_df = pd.concat([train_df, ori_train[num_op[i]].apply(percent_row, axis=1)], axis=1)
        test_df = pd.concat([test_df, ori_test[num_op[i]].apply(percent_row, axis=1)], axis=1)
    assert( all(train_df.columns == test_df.columns))


    #make features of var39, var40, and var41
    varl = ['var39', 'var40', 'var41']
    for i in varl:
        #train
        #imp_op_var39_ult1
        train_df['imp_op_{}_efect_ult1/imp_op_{}_ult1'.format(i,i)] = ori_train['imp_op_{}_efect_ult1'.format(i)] / (ori_train['imp_op_{}_ult1'.format(i)] + 0.01)
        train_df['imp_op_{}_comer_ult1/imp_op_{}_ult1'.format(i,i)] = ori_train['imp_op_{}_comer_ult1'.format(i)] / (ori_train['imp_op_{}_ult1'.format(i)] + 0.01)
        train_df['imp_op_{}_efect_ult1/imp_op_{}_comer_ult1'.format(i,i)] = ori_train['imp_op_{}_efect_ult1'.format(i)] / (ori_train['imp_op_{}_comer_ult1'.format(i)] + 0.01)

        #train_df['imp_op_{}_efect_ult3/imp_op_{}_ult3'.format(i,i)] = ori_train['imp_op_{}_efect_ult3'.format(i)] / (ori_train['imp_op_{}_ult3'.format(i)] + 0.01)
        #train_df['imp_op_{}_comer_ult3/imp_op_{}_ult3'.format(i,i)] = ori_train['imp_op_{}_comer_ult3'.format(i)] / (ori_train['imp_op_{}_ult3'.format(i)] + 0.01)
        train_df['imp_op_{}_efect_ult3/imp_op_{}_comer_ult3'.format(i,i)] = ori_train['imp_op_{}_efect_ult3'.format(i)] / (ori_train['imp_op_{}_comer_ult3'.format(i)] + 0.01)

        #num_op_var39_ult1
        train_df['num_op_{}_efect_ult1/num_op_{}_ult1'.format(i,i)] = ori_train['num_op_{}_efect_ult1'.format(i)] / (ori_train['num_op_{}_ult1'.format(i)] + 0.01)
        train_df['num_op_{}_comer_ult1/num_op_{}_ult1'.format(i,i)] = ori_train['num_op_{}_comer_ult1'.format(i)] / (ori_train['num_op_{}_ult1'.format(i)] + 0.01)
        train_df['num_op_{}_efect_ult1/num_op_{}_comer_ult1'.format(i,i)] = ori_train['num_op_{}_efect_ult1'.format(i)] / (ori_train['num_op_{}_comer_ult1'.format(i)] + 0.01)

        train_df['num_op_{}_efect_ult3/num_op_{}_ult3'.format(i,i)] = ori_train['num_op_{}_efect_ult3'.format(i)] / (ori_train['num_op_{}_ult3'.format(i)] + 0.01)
        train_df['num_op_{}_comer_ult3/num_op_{}_ult3'.format(i,i)] = ori_train['num_op_{}_comer_ult3'.format(i)] / (ori_train['num_op_{}_ult3'.format(i)] + 0.01)
        train_df['num_op_{}_efect_ult3/num_op_{}_comer_ult3'.format(i,i)] = ori_train['num_op_{}_efect_ult3'.format(i)] / (ori_train['num_op_{}_comer_ult3'.format(i)] + 0.01)

        #num/imp_op_var39_ult1
        train_df['imp_op_{}_efect_ult1/num_op_{}_ult1'.format(i,i)] = ori_train['imp_op_{}_efect_ult1'.format(i)] / (ori_train['num_op_{}_ult1'.format(i)] + 0.01)
        train_df['imp_op_{}_comer_ult1/num_op_{}_ult1'.format(i,i)] = ori_train['imp_op_{}_comer_ult1'.format(i)] / (ori_train['num_op_{}_ult1'.format(i)] + 0.01)
        train_df['imp_op_{}_efect_ult1/num_op_{}_comer_ult1'.format(i,i)] = ori_train['imp_op_{}_efect_ult1'.format(i)] / (ori_train['num_op_{}_comer_ult1'.format(i)] + 0.01)

        train_df['imp_op_{}_efect_ult3/num_op_{}_ult3'.format(i,i)] = ori_train['imp_op_{}_efect_ult3'.format(i)] / (ori_train['num_op_{}_ult3'.format(i)] + 0.01)
        train_df['imp_op_{}_comer_ult3/num_op_{}_ult3'.format(i,i)] = ori_train['imp_op_{}_comer_ult3'.format(i)] / (ori_train['num_op_{}_ult3'.format(i)] + 0.01)
        train_df['imp_op_{}_efect_ult3/num_op_{}_comer_ult3'.format(i,i)] = ori_train['imp_op_{}_efect_ult3'.format(i)] / (ori_train['num_op_{}_comer_ult3'.format(i)] + 0.01)

        #test
        #imp_op_var39_ult1
        test_df['imp_op_{}_efect_ult1/imp_op_{}_ult1'.format(i,i)] = ori_test['imp_op_{}_efect_ult1'.format(i)] / (ori_test['imp_op_{}_ult1'.format(i)] + 0.01)
        test_df['imp_op_{}_comer_ult1/imp_op_{}_ult1'.format(i,i)] = ori_test['imp_op_{}_comer_ult1'.format(i)] / (ori_test['imp_op_{}_ult1'.format(i)] + 0.01)
        test_df['imp_op_{}_efect_ult1/imp_op_{}_comer_ult1'.format(i,i)] = ori_test['imp_op_{}_efect_ult1'.format(i)] / (ori_test['imp_op_{}_comer_ult1'.format(i)] + 0.01)

        #test_df['imp_op_{}_efect_ult3/imp_op_{}_ult3'.format(i,i)] = ori_test['imp_op_{}_efect_ult3'.format(i)] / (ori_test['imp_op_{}_ult3'.format(i)] + 0.01)
        #test_df['imp_op_{}_comer_ult3/imp_op_{}_ult3'.format(i,i)] = ori_test['imp_op_{}_comer_ult3'.format(i)] / (ori_test['imp_op_{}_ult3'.format(i)] + 0.01)
        test_df['imp_op_{}_efect_ult3/imp_op_{}_comer_ult3'.format(i,i)] = ori_test['imp_op_{}_efect_ult3'.format(i)] / (ori_test['imp_op_{}_comer_ult3'.format(i)] + 0.01)

        #num_op_var39_ult1
        test_df['num_op_{}_efect_ult1/num_op_{}_ult1'.format(i,i)] = ori_test['num_op_{}_efect_ult1'.format(i)] / (ori_test['num_op_{}_ult1'.format(i)] + 0.01)
        test_df['num_op_{}_comer_ult1/num_op_{}_ult1'.format(i,i)] = ori_test['num_op_{}_comer_ult1'.format(i)] / (ori_test['num_op_{}_ult1'.format(i)] + 0.01)
        test_df['num_op_{}_efect_ult1/num_op_{}_comer_ult1'.format(i,i)] = ori_test['num_op_{}_efect_ult1'.format(i)] / (ori_test['num_op_{}_comer_ult1'.format(i)] + 0.01)

        test_df['num_op_{}_efect_ult3/num_op_{}_ult3'.format(i,i)] = ori_test['num_op_{}_efect_ult3'.format(i)] / (ori_test['num_op_{}_ult3'.format(i)] + 0.01)
        test_df['num_op_{}_comer_ult3/num_op_{}_ult3'.format(i,i)] = ori_test['num_op_{}_comer_ult3'.format(i)] / (ori_test['num_op_{}_ult3'.format(i)] + 0.01)
        test_df['num_op_{}_efect_ult3/num_op_{}_comer_ult3'.format(i,i)] = ori_test['num_op_{}_efect_ult3'.format(i)] / (ori_test['num_op_{}_comer_ult3'.format(i)] + 0.01)

        #num/imp_op_var39_ult1
        test_df['imp_op_{}_efect_ult1/num_op_{}_ult1'.format(i,i)] = ori_train['imp_op_{}_efect_ult1'.format(i)] / (ori_train['num_op_{}_ult1'.format(i)] + 0.01)
        test_df['imp_op_{}_comer_ult1/num_op_{}_ult1'.format(i,i)] = ori_train['imp_op_{}_comer_ult1'.format(i)] / (ori_train['num_op_{}_ult1'.format(i)] + 0.01)
        test_df['imp_op_{}_efect_ult1/num_op_{}_comer_ult1'.format(i,i)] = ori_train['imp_op_{}_efect_ult1'.format(i)] / (ori_train['num_op_{}_comer_ult1'.format(i)] + 0.01)

        test_df['imp_op_{}_efect_ult3/num_op_{}_ult3'.format(i,i)] = ori_train['imp_op_{}_efect_ult3'.format(i)] / (ori_train['num_op_{}_ult3'.format(i)] + 0.01)
        test_df['imp_op_{}_comer_ult3/num_op_{}_ult3'.format(i,i)] = ori_train['imp_op_{}_comer_ult3'.format(i)] / (ori_train['num_op_{}_ult3'.format(i)] + 0.01)
        test_df['imp_op_{}_efect_ult3/num_op_{}_comer_ult3'.format(i,i)] = ori_train['imp_op_{}_efect_ult3'.format(i)] / (ori_train['num_op_{}_comer_ult3'.format(i)] + 0.01)


    assert( all(train_df.columns == test_df.columns))



    # saldo_var13 = 1 * saldo_var13_corto + 1 * saldo_var13_largo + 1 * saldo_var13_medio #
    saldo = [
            ['saldo_var13','saldo_var13_corto','saldo_var13_medio','saldo_var13_largo'],
            ]

    def percent_row(data):
        #print data
        lhs = data.index[0]
        sum_ = data.sum()
        data.index += '_pct_saldo_' + lhs 
        return pd.Series(data)/(sum_+0.1)

    train_df = pd.concat([train_df, ori_train[saldo[0]].apply(percent_row, axis=1)],axis=1)
    test_df = pd.concat([test_df, ori_test[saldo[0]].apply(percent_row, axis=1)],axis=1)
    assert( all(train_df.columns == test_df.columns))



    #num_var
    num_var = [
            ['num_var13_0','num_var13_corto_0','ind_var13_medio_0','num_var13_largo_0'],
            ['num_var13','num_var13_corto','ind_var13_medio','num_var13_largo'],
            ]

    def percent_row(data):
        #print data
        lhs = data.index[0]
        sum_ = data.sum()
        data.index += '_pct_num_var' + lhs
        return pd.Series(data)/(sum_+0.1)

    for i in xrange(len(num_var)):
        train_df = pd.concat([train_df, ori_train[num_var[i]].apply(percent_row, axis=1)],axis=1)
        test_df = pd.concat([test_df, ori_test[num_var[i]].apply(percent_row, axis=1)],axis=1)

    assert( all(train_df.columns == test_df.columns))

    ori_train = pd.concat([ori_train, train_df], axis=1)
    ori_test = pd.concat([ori_test, test_df], axis=1)
    

    return ori_train, ori_test





def main_feat(train, test, sample_submit=None):
    
    train_target = train['target']
    del train['target']

    #delete id
    del train['t_id'], test['t_id']

    # 0 count per ID
    def countZero(data):
        return np.sum(data == 0)

    train['count0'] = train.apply(countZero, axis=1)
    test['count0'] = test.apply(countZero, axis=1)

    # add count features of integer columns
    int_col = (train.dtypes == int)[(train.dtypes == int).values].index
    train_test = pd.concat([train,test])
    for i in int_col:
        tmp_cnt = train_test[i].value_counts()
        tmp_cnt = tmp_cnt.to_frame(name=i+'_cnt')
        tmp_cnt[i] = tmp_cnt.index
        tmp_cnt.reset_index(drop=True, inplace=True)
        train = train.reset_index().merge(tmp_cnt, how='left', on=i).sort('index').drop('index', axis=1)
        test = test.reset_index().merge(tmp_cnt, how='left', on=i).sort('index').drop('index', axis=1)
        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)
    del train_test

    ###### cleaing data using reverse feature engineering ######
    # To enable cleaning with reverse feature engineering,     #
    # comment out the next lines                               #
    ############################################################

    #print 'starting cleaning_rfe...'
    #train, test = cleaning_rfe(ori_train=train.copy(), ori_test=test.copy())
    #print 'done cleaning_rfe'

    # make dummy variables of var3 in the threshold(>=5)
    var3_cnt = train.var3.value_counts()
    #threshold is different from feat_ver407(var3_cnt>=5)
    index_var3_th = var3_cnt[(var3_cnt>=4).values].index
    train['var3_tmp'] = train.var3.apply(lambda x: x if x in index_var3_th else np.nan)
    test['var3_tmp'] = test.var3.apply(lambda x: x if x in index_var3_th else np.nan)
    
    train_test = pd.concat([train,test])
    #train_test.reset_index(drop=True, inplace=True)
    tmp = pd.get_dummies(train_test['var3_tmp'], prefix='ohe_var3', prefix_sep='_')

    train = pd.concat([train, tmp.iloc[:len(train),:]], axis=1)
    test = pd.concat([test, tmp.iloc[len(train):,:]], axis=1)
    del train['var3_tmp'], test['var3_tmp']

    # add feature of var38
    train['var38mc'] = np.isclose(train.var38, 117310.979016)
    train['logvar38'] = train.loc[~train['var38mc'], 'var38'].map(np.log)
    train.loc[train['var38mc'], 'logvar38'] = 0

    test['var38mc'] = np.isclose(test.var38, 117310.979016)
    test['logvar38'] = test.loc[~test['var38mc'], 'var38'].map(np.log)
    test.loc[test['var38mc'], 'logvar38'] = 0

    train['var38mc'] = train['var38mc'].astype(int)

    test['var38mc'] = test['var38mc'].astype(int)

    #delete constant features
    for i in train.columns:
        if len(set(train[i].values)) == 1:
            del train[i], test[i]
    assert( all(train.columns == test.columns))

    #delete identical columns
    unique_col = train.T.drop_duplicates().T.columns
    train = train[unique_col]
    test = test[unique_col]
    assert( all(train.columns == test.columns))


    train['target'] = train_target

    train.to_csv('data/output/features/ikki_features_train_ver2.csv',index=None)
    test.to_csv('data/output/features/ikki_features_test_ver2.csv',index=None)

def main_feat_NN():
    train = pd.read_csv('data/output/features/ikki_features_train_ver2.csv')
    test = pd.read_csv('data/output/features/ikki_features_test_ver2.csv')

    train_target = train['target']
    del train['target']

    #delete id
    #del train['t_id'], test['t_id']
    ohe_col = ['num_var13_corto','num_var13_corto_0','num_meses_var12_ult3','num_meses_var13_corto_ult3','num_meses_var39_vig_ult3','num_meses_var5_ult3','num_var24_0','num_var12','var36','num_var5','num_var5_0','num_var12_0','num_var13','num_var13_0','num_var42','num_var4','num_var42_0','num_var30','num_var39_0','num_var41_0']

    #delete categorical columns
    #because OHEncoder is ismplemented in another func.
    for i in ohe_col:
        del train[i], test[i]

    #delete var3 because var3 is OHEncoded in main_feature() 
    del train['var3']
    del test['var3']

    #replace min/max in test with min/max in train
    for i in train.columns:
        min_val = train[i].min()
        test.loc[(test[i] < min_val).values,i] = min_val

        max_val = train[i].max()
        test.loc[(test[i] > max_val).values,i] = max_val


    
    #log transformation
    train_test = pd.concat([train, test])
    train_test_min = train_test.min()
    train_test = train_test - train_test_min
    train = train_test.iloc[:len(train),:]
    test = train_test.iloc[len(train):,:]

    train = train.applymap(lambda x: np.log(x + 1))
    test = test.applymap(lambda x: np.log(x + 1))
    assert( all(train.columns == test.columns))
    
    train['target'] = train_target

    train.to_csv('data/output/features/ikki_features_train_NN_ver2.csv',index=None)
    test.to_csv('data/output/features/ikki_features_test_NN_ver2.csv',index=None)


#one hot encoder
def one_hot_encoder(train, test):

    ohe_col = ['num_var13_corto','num_var13_corto_0','num_meses_var12_ult3','num_meses_var13_corto_ult3','num_meses_var39_vig_ult3','num_meses_var5_ult3','num_var24_0','num_var12','var36','num_var5','num_var5_0','num_var12_0','num_var13','num_var13_0','num_var42','num_var4','num_var42_0','num_var30','num_var39_0','num_var41_0']
    
    train_test = pd.concat([train,test])
    train_test.reset_index(drop=True, inplace=True)
    ohe_data = pd.DataFrame()
    for i in train_test.columns:
        if i in ohe_col:
            tmp = pd.get_dummies(train_test[i], prefix='ohe_'+i, prefix_sep='_')
            ohe_data = pd.concat([ohe_data, tmp], axis=1)
    
    train = ohe_data.iloc[:len(train),:]
    test = ohe_data.iloc[len(train):,:]

    train.to_csv('data/output/features/ikki_one_hot_encoder_train_ver2.csv',index=None)
    test.to_csv('data/output/features/ikki_one_hot_encoder_test_ver2.csv',index=None)


if __name__ == '__main__':
    print 'Creating dataset (Feature engineering)'
    main_feat(train=ori_train.copy(), test=ori_test.copy())
    #main_feat_NN()
    one_hot_encoder(train=ori_train.copy(), test=ori_test.copy())
    print 'Done dataset creation'




