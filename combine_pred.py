import pandas as pd
import os, sys
import subprocess

# Change the current directory
#os.chdir(os.getcwd())

# train
xgb1_train = pd.read_csv('data/output/temp/train_xgb_ikki_ver1.csv')
xgb2_train = pd.read_csv('data/output/temp/train_xgb_ikki_ver2.csv')
nn1_train = pd.read_csv('data/output/temp/train_NN_ikki_ver3.csv')

final_train = xgb1_train.copy()
final_train = final_train.merge(xgb2_train, how='left', on='ID')
final_train = final_train.merge(nn1_train, how='left', on='ID')
final_train.columns = ('ID', 'ikki_xgb1', 'ikki_xgb2', 'ikki_NN1')
final_train.to_csv('data/output/train/ikki_models_train.csv', index=False)

# test
xgb1_test = pd.read_csv('data/output/temp/test_xgb_ikki_ver1.csv')
xgb2_test = pd.read_csv('data/output/temp/test_xgb_ikki_ver2.csv')
nn1_test = pd.read_csv('data/output/temp/test_NN_ikki_ver3.csv')

final_test = xgb1_test.copy()
final_test = final_test.merge(xgb2_test, how='left', on='ID')
final_test = final_test.merge(nn1_test, how='left', on='ID')
final_test.columns = ('ID', 'ikki_xgb1', 'ikki_xgb2', 'ikki_NN1')
final_test.to_csv('data/output/test/ikki_models_test.csv', index=False)

