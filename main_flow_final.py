import os, sys
import subprocess


# Change the current directory
os.chdir(os.getcwd())

# flag if run all model training or just generate my final prediction
run_flag = 1 #run all model training

if run_flag == 1:
    print 'Start model training and prediction'
    print 
    print 'Starting ver1 (XGB)'
    # Version 1 prediction
    # Create train(test)_xgb_ikki_ver1.csv
    subprocess.call("python ikki_feat_ver1.py".split(' '))
    subprocess.call("python scripts/ikki_xgb_1.py".split(' '))

    print 
    print 'Starting ver2 (XGB)'
    # Version 2 prediction
    # Create train(test)_xgb_ikki_ver2.csv
    subprocess.call("python ikki_feat_ver2.py".split(' '))
    subprocess.call("python scripts/ikki_xgb_2.py".split(' '))

    print 
    print 'Starting ver3 (NN)'
    # Version 3 prediction
    # Create train(test)_NN_ikki_ver3.csv
    subprocess.call("python ikki_feat_ver3.py".split(' '))
    subprocess.call("python scripts/ikki_NN_1.py".split(' '))


print 'Combining the data for final ensembling'
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

