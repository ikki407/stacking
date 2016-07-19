#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Genaral packages
import os, sys
import pandas as pd
import numpy as np

sys.path.append(os.getcwd())
#os.chdir('/Users/IkkiTanaka/Documents/kaggle/Santander/')

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
                'train':('data/output/features/ikki_features_train_NN_ver3.csv',
                         'data/output/features/ikki_one_hot_encoder_train_ver3.csv',
                        ),#target is in 'train'
                'test':('data/output/features/ikki_features_test_NN_ver3.csv',
                        'data/output/features/ikki_one_hot_encoder_test_ver3.csv',
                        ),
                }
X,y,test  = load_data(flist=FEATURE_LIST_stage1, drop_duplicates=True)
assert((False in X.columns == test.columns) == False)
nn_input_dim_NN = X.shape[1]
del X, y, test



# Models in Stage 1
PARAMS_V1 = {
            'batch_size':256,
            'nb_epoch':35,
            'verbose':1, 
            'callbacks':[],
            'validation_split':0.,
            'validation_data':None,
            'shuffle':True,
            'show_accuracy':True,
            'class_weight':None,#{0:0.0396, 1:0.9604},
            'sample_weight':None,
            'normalize':True,
            'categorize_y':True
            }

class ModelV1(BaseModel):
        def build_model(self):
            model = Sequential()
            model.add(Dropout(0.2, input_shape=(nn_input_dim_NN,)))
            model.add(Dense(input_dim=nn_input_dim_NN, output_dim=120, init='uniform'))
            model.add(LeakyReLU(alpha=.00001))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(input_dim=120,output_dim=280, init='uniform'))
            model.add(LeakyReLU(alpha=.00001))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(input_dim=280,output_dim=100, init='uniform', activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.4))
            model.add(Dense(input_dim=100,output_dim=2, init='uniform', activation='softmax'))    
            #model.add(Activation('softmax'))
            sgd = SGD(lr=0.015, decay=1e-6, momentum=0.9, nesterov=True)

            model.compile(optimizer=sgd, loss='binary_crossentropy',class_mode='binary')

            return KerasClassifier(nn=model,**self.params)


PARAMS_V2 = {
            'batch_size':512,
            'nb_epoch':70,
            'verbose':1, 
            'callbacks':[],
            'validation_split':0.,
            'validation_data':None,
            'shuffle':True,
            'show_accuracy':True,
            'class_weight':None,#{0:0.0396, 1:0.9604},
            'sample_weight':None,
            'normalize':True,
            'categorize_y':True
            }

class ModelV2(BaseModel):
        def build_model(self):
            model = Sequential()
            model.add(Dropout(0.1, input_shape=(nn_input_dim_NN,)))
            model.add(Dense(input_dim=nn_input_dim_NN, output_dim=112, init='he_normal'))
            model.add(LeakyReLU(alpha=.00001))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(input_dim=112,output_dim=128, init='he_normal'))
            model.add(LeakyReLU(alpha=.00001))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(input_dim=128,output_dim=68, init='he_normal'))
            model.add(LeakyReLU(alpha=.00003))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            model.add(Dense(input_dim=68,output_dim=2, init='he_normal'))
            model.add(Activation('softmax'))
            sgd = SGD(lr=0.01, decay=1e-10, momentum=0.99, nesterov=True)

            model.compile(optimizer=sgd, loss='binary_crossentropy',class_mode='binary')

            return KerasClassifier(nn=model,**self.params)

PARAMS_V3 = {
            'batch_size':128,
            'nb_epoch':72,
            'verbose':1, 
            'callbacks':[],
            'validation_split':0.,
            'validation_data':None,
            'shuffle':True,
            'show_accuracy':True,
            'class_weight':None,#{0:0.0396, 1:0.9604},
            'sample_weight':None,
            'normalize':True,
            'categorize_y':True
            }

class ModelV3(BaseModel):
        def build_model(self):
            model = Sequential()
            model.add(Dropout(0.1, input_shape=(nn_input_dim_NN,)))
            model.add(Dense(input_dim=nn_input_dim_NN, output_dim=310, init='he_normal'))
            model.add(LeakyReLU(alpha=.001))
            model.add(BatchNormalization())
            model.add(Dropout(0.6))
            model.add(Dense(input_dim=310,output_dim=252, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(input_dim=252,output_dim=128, init='he_normal'))
            model.add(LeakyReLU(alpha=.001))
            model.add(BatchNormalization())
            model.add(Dropout(0.4))
            model.add(Dense(input_dim=128,output_dim=2, init='he_normal', activation='softmax'))
            #model.add(Activation('softmax'))
            sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)

            model.compile(optimizer=sgd, loss='binary_crossentropy',class_mode='binary')

            return KerasClassifier(nn=model,**self.params)

PARAMS_V4 = {
            'batch_size':128,
            'nb_epoch':56,
            'verbose':1, 
            'callbacks':[],
            'validation_split':0.,
            'validation_data':None,
            'shuffle':True,
            'show_accuracy':True,
            'class_weight':None,#{0:0.0396, 1:0.9604},
            'sample_weight':None,
            'normalize':True,
            'categorize_y':True
            }

class ModelV4(BaseModel):
        def build_model(self):
            model = Sequential()
            model.add(Dropout(0.1, input_shape=(nn_input_dim_NN,)))
            model.add(Dense(input_dim=nn_input_dim_NN, output_dim=62, init='he_normal'))
            model.add(LeakyReLU(alpha=.001))
            model.add(Dropout(0.3))
            model.add(Dense(input_dim=62,output_dim=158, init='he_normal'))
            model.add(LeakyReLU(alpha=.001))
            model.add(Dropout(0.25))
            model.add(Dense(input_dim=158,output_dim=20, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(Dropout(0.2))
            model.add(Dense(input_dim=20,output_dim=2, init='he_normal', activation='softmax'))
            #model.add(Activation('softmax'))
            sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)

            model.compile(optimizer=sgd, loss='binary_crossentropy',class_mode='binary')

            return KerasClassifier(nn=model,**self.params)


PARAMS_V5 = {
            'batch_size':216,
            'nb_epoch':90,
            'verbose':1, 
            'callbacks':[],
            'validation_split':0.,
            'validation_data':None,
            'shuffle':True,
            'show_accuracy':True,
            'class_weight':None,#{0:0.0396, 1:0.9604},
            'sample_weight':None,
            'normalize':True,
            'categorize_y':True
            }

class ModelV5(BaseModel):
        def build_model(self):
            model = Sequential()
            model.add(Dropout(0.1, input_shape=(nn_input_dim_NN,)))
            model.add(Dense(input_dim=nn_input_dim_NN, output_dim=100, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(Dropout(0.2))
            model.add(Dense(input_dim=100,output_dim=380, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(Dropout(0.6))
            model.add(Dense(input_dim=380,output_dim=50, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(Dropout(0.6))
            model.add(Dense(input_dim=50,output_dim=20, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(Dropout(0.2))
            model.add(Dense(input_dim=20,output_dim=2, init='he_normal', activation='softmax'))    
            sgd = SGD(lr=0.01, decay=1e-10, momentum=0.9, nesterov=True)

            model.compile(optimizer=sgd, loss='binary_crossentropy',class_mode='binary')

            return KerasClassifier(nn=model,**self.params)

PARAMS_V6 = {
            'batch_size':216,
            'nb_epoch':72,
            'verbose':1, 
            'callbacks':[],
            'validation_split':0.,
            'validation_data':None,
            'shuffle':True,
            'show_accuracy':True,
            'class_weight':None,#{0:0.0396, 1:0.9604},
            'sample_weight':None,
            'normalize':True,
            'categorize_y':True
            }

class ModelV6(BaseModel):
        def build_model(self):
            model = Sequential()
            model.add(Dropout(0.1, input_shape=(nn_input_dim_NN,)))
            model.add(Dense(input_dim=nn_input_dim_NN, output_dim=105, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(input_dim=105,output_dim=280, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.6))
            model.add(Dense(input_dim=280,output_dim=60, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.6))
            model.add(Dense(input_dim=60,output_dim=20, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(input_dim=20,output_dim=2, init='he_normal', activation='softmax'))    
            sgd = SGD(lr=0.01, decay=1e-10, momentum=0.99, nesterov=True)

            model.compile(optimizer=sgd, loss='binary_crossentropy',class_mode='binary')

            return KerasClassifier(nn=model,**self.params)

PARAMS_V7 = {
            'batch_size':128,
            'nb_epoch':65,
            'verbose':1, 
            'callbacks':[],
            'validation_split':0.,
            'validation_data':None,
            'shuffle':True,
            'show_accuracy':True,
            'class_weight':None,#{0:0.0396, 1:0.9604},
            'sample_weight':None,
            'normalize':True,
            'categorize_y':True
            }

class ModelV7(BaseModel):
        def build_model(self):
            model = Sequential()
            model.add(Dropout(0.2, input_shape=(nn_input_dim_NN,)))
            model.add(Dense(input_dim=nn_input_dim_NN, output_dim=100, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(input_dim=100,output_dim=180, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.6))
            model.add(Dense(input_dim=180,output_dim=50, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(input_dim=50,output_dim=30, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(input_dim=30,output_dim=2, init='he_normal', activation='softmax'))    
            sgd = SGD(lr=0.01, decay=1e-10, momentum=0.9, nesterov=True)

            model.compile(optimizer=sgd, loss='binary_crossentropy',class_mode='binary')

            return KerasClassifier(nn=model,**self.params)


PARAMS_V8 = {
            'batch_size':216,
            'nb_epoch':89,
            'verbose':1, 
            'callbacks':[],
            'validation_split':0.,
            'validation_data':None,
            'shuffle':True,
            'show_accuracy':True,
            'class_weight':None,#{0:0.0396, 1:0.9604},
            'sample_weight':None,
            'normalize':True,
            'categorize_y':True
            }

class ModelV8(BaseModel):
        def build_model(self):
            model = Sequential()
            model.add(Dropout(0.2, input_shape=(nn_input_dim_NN,)))
            model.add(Dense(input_dim=nn_input_dim_NN, output_dim=140, init='uniform'))
            model.add(LeakyReLU(alpha=.00001))
            model.add(BatchNormalization())
            model.add(Dropout(0.6))
            model.add(Dense(input_dim=140,output_dim=250, init='uniform'))
            model.add(LeakyReLU(alpha=.00001))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(input_dim=250,output_dim=90, init='uniform', activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.4))
            model.add(Dense(input_dim=90,output_dim=2, init='uniform', activation='softmax'))    
            #model.add(Activation('softmax'))
            sgd = SGD(lr=0.013, decay=1e-6, momentum=0.9, nesterov=True)

            model.compile(optimizer=sgd, loss='binary_crossentropy',class_mode='binary')

            return KerasClassifier(nn=model,**self.params)


PARAMS_V9 = {
            'batch_size':512,
            'nb_epoch':90,
            'verbose':1, 
            'callbacks':[],
            'validation_split':0.,
            'validation_data':None,
            'shuffle':True,
            'show_accuracy':True,
            'class_weight':None,#{0:0.0396, 1:0.9604},
            'sample_weight':None,
            'normalize':True,
            'categorize_y':True
            }

class ModelV9(BaseModel):
        def build_model(self):
            model = Sequential()
            model.add(Dropout(0.1, input_shape=(nn_input_dim_NN,)))
            model.add(Dense(input_dim=nn_input_dim_NN, output_dim=100, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(input_dim=100,output_dim=380, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.6))
            model.add(Dense(input_dim=380,output_dim=50, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.6))
            model.add(Dense(input_dim=50,output_dim=20, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(input_dim=20,output_dim=2, init='he_normal', activation='softmax'))    
            sgd = SGD(lr=0.01, decay=1e-10, momentum=0.9, nesterov=True)

            model.compile(optimizer=sgd, loss='binary_crossentropy',class_mode='binary')

            return KerasClassifier(nn=model,**self.params)


PARAMS_V10 = {
            'batch_size':216,
            'nb_epoch':80,
            'verbose':1, 
            'callbacks':[],
            'validation_split':0.,
            'validation_data':None,
            'shuffle':True,
            'show_accuracy':True,
            'class_weight':None,#{0:0.0396, 1:0.9604},
            'sample_weight':None,
            'normalize':True,
            'categorize_y':True
            }

class ModelV10(BaseModel):
        def build_model(self):
            model = Sequential()
            model.add(Dropout(0.1, input_shape=(nn_input_dim_NN,)))
            model.add(Dense(input_dim=nn_input_dim_NN, output_dim=100, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(input_dim=100,output_dim=360, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.6))
            model.add(Dense(input_dim=360,output_dim=50, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.6))
            model.add(Dense(input_dim=50,output_dim=20, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(input_dim=20,output_dim=2, init='he_normal', activation='softmax'))    
            sgd = SGD(lr=0.01, decay=1e-10, momentum=0.9, nesterov=True)

            model.compile(optimizer=sgd, loss='binary_crossentropy',class_mode='binary')

            return KerasClassifier(nn=model,**self.params)


PARAMS_V11 = {
            'batch_size':384,
            'nb_epoch':80,
            'verbose':1, 
            'callbacks':[],
            'validation_split':0.,
            'validation_data':None,
            'shuffle':True,
            'show_accuracy':True,
            'class_weight':None,#{0:0.0396, 1:0.9604},
            'sample_weight':None,
            'normalize':True,
            'categorize_y':True
            }

class ModelV11(BaseModel):
        def build_model(self):
            model = Sequential()
            model.add(Dropout(0.1, input_shape=(nn_input_dim_NN,)))
            model.add(Dense(input_dim=nn_input_dim_NN, output_dim=110, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(input_dim=110,output_dim=350, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.6))
            model.add(Dense(input_dim=350,output_dim=50, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.6))
            model.add(Dense(input_dim=50,output_dim=20, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(input_dim=20,output_dim=2, init='he_normal', activation='softmax'))    
            sgd = SGD(lr=0.01, decay=1e-10, momentum=0.9, nesterov=True)

            model.compile(optimizer=sgd, loss='binary_crossentropy',class_mode='binary')

            return KerasClassifier(nn=model,**self.params)

PARAMS_V12 = {
            'batch_size':216,
            'nb_epoch':82,
            'verbose':1, 
            'callbacks':[],
            'validation_split':0.,
            'validation_data':None,
            'shuffle':True,
            'show_accuracy':True,
            'class_weight':None,#{0:0.0396, 1:0.9604},
            'sample_weight':None,
            'normalize':True,
            'categorize_y':True
            }

class ModelV12(BaseModel):
        def build_model(self):
            model = Sequential()
            model.add(Dropout(0.1, input_shape=(nn_input_dim_NN,)))
            model.add(Dense(input_dim=nn_input_dim_NN, output_dim=110, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            model.add(Dense(input_dim=110,output_dim=300, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(input_dim=300,output_dim=60, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.6))
            model.add(Dense(input_dim=60,output_dim=20, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(input_dim=20,output_dim=2, init='he_normal', activation='softmax'))    
            sgd = SGD(lr=0.01, decay=1e-10, momentum=0.9, nesterov=True)

            model.compile(optimizer=sgd, loss='binary_crossentropy',class_mode='binary')

            return KerasClassifier(nn=model,**self.params)

PARAMS_V13 = {
            'batch_size':512,
            'nb_epoch':90,
            'verbose':1, 
            'callbacks':[],
            'validation_split':0.,
            'validation_data':None,
            'shuffle':True,
            'show_accuracy':True,
            'class_weight':None,#{0:0.0396, 1:0.9604},
            'sample_weight':None,
            'normalize':True,
            'categorize_y':True
            }

class ModelV13(BaseModel):
        def build_model(self):
            model = Sequential()
            model.add(Dropout(0.1, input_shape=(nn_input_dim_NN,)))
            model.add(Dense(input_dim=nn_input_dim_NN, output_dim=100, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(Dropout(0.1))
            model.add(Dense(input_dim=100,output_dim=300, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(Dropout(0.6))
            model.add(Dense(input_dim=300,output_dim=50, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(Dropout(0.6))
            model.add(Dense(input_dim=50,output_dim=20, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(Dropout(0.2))
            model.add(Dense(input_dim=20,output_dim=2, init='he_normal', activation='softmax'))    
            sgd = SGD(lr=0.01, decay=1e-10, momentum=0.9, nesterov=True)

            model.compile(optimizer=sgd, loss='binary_crossentropy',class_mode='binary')

            return KerasClassifier(nn=model,**self.params)

PARAMS_V14 = {
            'batch_size':216,
            'nb_epoch':72,
            'verbose':1, 
            'callbacks':[],
            'validation_split':0.,
            'validation_data':None,
            'shuffle':True,
            'show_accuracy':True,
            'class_weight':None,#{0:0.0396, 1:0.9604},
            'sample_weight':None,
            'normalize':True,
            'categorize_y':True
            }

class ModelV14(BaseModel):
        def build_model(self):
            model = Sequential()
            model.add(Dropout(0.1, input_shape=(nn_input_dim_NN,)))
            model.add(Dense(input_dim=nn_input_dim_NN, output_dim=105, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(input_dim=105,output_dim=200, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.6))
            model.add(Dense(input_dim=200,output_dim=60, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(input_dim=60,output_dim=20, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.1))
            model.add(Dense(input_dim=20,output_dim=2, init='he_normal', activation='softmax'))    
            sgd = SGD(lr=0.01, decay=1e-10, momentum=0.99, nesterov=True)

            model.compile(optimizer=sgd, loss='binary_crossentropy',class_mode='binary')

            return KerasClassifier(nn=model,**self.params)

PARAMS_V15 = {
            'batch_size':128,
            'nb_epoch':65,
            'verbose':1, 
            'callbacks':[],
            'validation_split':0.,
            'validation_data':None,
            'shuffle':True,
            'show_accuracy':True,
            'class_weight':None,#{0:0.0396, 1:0.9604},
            'sample_weight':None,
            'normalize':True,
            'categorize_y':True
            }

class ModelV15(BaseModel):
        def build_model(self):
            model = Sequential()
            model.add(Dropout(0.2, input_shape=(nn_input_dim_NN,)))
            model.add(Dense(input_dim=nn_input_dim_NN, output_dim=100, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(input_dim=100,output_dim=180, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(input_dim=180,output_dim=50, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(input_dim=50,output_dim=40, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(input_dim=40,output_dim=2, init='he_normal', activation='softmax'))    
            sgd = SGD(lr=0.01, decay=1e-10, momentum=0.9, nesterov=True)

            model.compile(optimizer=sgd, loss='binary_crossentropy',class_mode='binary')

            return KerasClassifier(nn=model,**self.params)


PARAMS_V16 = {
            'batch_size':216,
            'nb_epoch':89,
            'verbose':1, 
            'callbacks':[],
            'validation_split':0.,
            'validation_data':None,
            'shuffle':True,
            'show_accuracy':True,
            'class_weight':None,#{0:0.0396, 1:0.9604},
            'sample_weight':None,
            'normalize':True,
            'categorize_y':True
            }

class ModelV16(BaseModel):
        def build_model(self):
            model = Sequential()
            model.add(Dropout(0.2, input_shape=(nn_input_dim_NN,)))
            model.add(Dense(input_dim=nn_input_dim_NN, output_dim=140, init='uniform'))
            model.add(LeakyReLU(alpha=.00001))
            model.add(BatchNormalization())
            model.add(Dropout(0.6))
            model.add(Dense(input_dim=140,output_dim=250, init='uniform'))
            model.add(LeakyReLU(alpha=.00001))
            model.add(BatchNormalization())
            model.add(Dropout(0.6))
            model.add(Dense(input_dim=250,output_dim=90, init='uniform', activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.4))
            model.add(Dense(input_dim=90,output_dim=2, init='uniform', activation='softmax'))    
            #model.add(Activation('softmax'))
            sgd = SGD(lr=0.013, decay=1e-6, momentum=0.9, nesterov=True)

            model.compile(optimizer=sgd, loss='binary_crossentropy',class_mode='binary')

            return KerasClassifier(nn=model,**self.params)


PARAMS_V17 = {
            'batch_size':512,
            'nb_epoch':90,
            'verbose':1, 
            'callbacks':[],
            'validation_split':0.,
            'validation_data':None,
            'shuffle':True,
            'show_accuracy':True,
            'class_weight':None,#{0:0.0396, 1:0.9604},
            'sample_weight':None,
            'normalize':True,
            'categorize_y':True
            }

class ModelV17(BaseModel):
        def build_model(self):
            model = Sequential()
            model.add(Dropout(0.1, input_shape=(nn_input_dim_NN,)))
            model.add(Dense(input_dim=nn_input_dim_NN, output_dim=140, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(input_dim=140,output_dim=380, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.6))
            model.add(Dense(input_dim=380,output_dim=50, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.6))
            model.add(Dense(input_dim=50,output_dim=20, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(input_dim=20,output_dim=2, init='he_normal', activation='softmax'))    
            sgd = SGD(lr=0.01, decay=1e-10, momentum=0.9, nesterov=True)

            model.compile(optimizer=sgd, loss='binary_crossentropy',class_mode='binary')

            return KerasClassifier(nn=model,**self.params)


PARAMS_V18 = {
            'batch_size':216,
            'nb_epoch':80,
            'verbose':1, 
            'callbacks':[],
            'validation_split':0.,
            'validation_data':None,
            'shuffle':True,
            'show_accuracy':True,
            'class_weight':None,#{0:0.0396, 1:0.9604},
            'sample_weight':None,
            'normalize':True,
            'categorize_y':True
            }

class ModelV18(BaseModel):
        def build_model(self):
            model = Sequential()
            model.add(Dropout(0.1, input_shape=(nn_input_dim_NN,)))
            model.add(Dense(input_dim=nn_input_dim_NN, output_dim=100, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(input_dim=100,output_dim=360, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.6))
            model.add(Dense(input_dim=360,output_dim=50, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.6))
            model.add(Dense(input_dim=50,output_dim=20, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.1))
            model.add(Dense(input_dim=20,output_dim=2, init='he_normal', activation='softmax'))    
            sgd = SGD(lr=0.007, decay=1e-10, momentum=0.9, nesterov=True)

            model.compile(optimizer=sgd, loss='binary_crossentropy',class_mode='binary')

            return KerasClassifier(nn=model,**self.params)


PARAMS_V19 = {
            'batch_size':384,
            'nb_epoch':80,
            'verbose':1, 
            'callbacks':[],
            'validation_split':0.,
            'validation_data':None,
            'shuffle':True,
            'show_accuracy':True,
            'class_weight':None,#{0:0.0396, 1:0.9604},
            'sample_weight':None,
            'normalize':True,
            'categorize_y':True
            }

class ModelV19(BaseModel):
        def build_model(self):
            model = Sequential()
            model.add(Dropout(0.1, input_shape=(nn_input_dim_NN,)))
            model.add(Dense(input_dim=nn_input_dim_NN, output_dim=110, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(input_dim=110,output_dim=350, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.6))
            model.add(Dense(input_dim=350,output_dim=150, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.6))
            model.add(Dense(input_dim=150,output_dim=20, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(input_dim=20,output_dim=2, init='he_normal', activation='softmax'))    
            sgd = SGD(lr=0.02, decay=1e-10, momentum=0.9, nesterov=True)

            model.compile(optimizer=sgd, loss='binary_crossentropy',class_mode='binary')

            return KerasClassifier(nn=model,**self.params)

PARAMS_V20 = {
            'batch_size':216,
            'nb_epoch':82,
            'verbose':1, 
            'callbacks':[],
            'validation_split':0.,
            'validation_data':None,
            'shuffle':True,
            'show_accuracy':True,
            'class_weight':None,#{0:0.0396, 1:0.9604},
            'sample_weight':None,
            'normalize':True,
            'categorize_y':True
            }

class ModelV20(BaseModel):
        def build_model(self):
            model = Sequential()
            model.add(Dropout(0.1, input_shape=(nn_input_dim_NN,)))
            model.add(Dense(input_dim=nn_input_dim_NN, output_dim=110, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            model.add(Dense(input_dim=110,output_dim=200, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(input_dim=200,output_dim=60, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.6))
            model.add(Dense(input_dim=60,output_dim=80, init='he_normal'))
            model.add(PReLU(init='zero'))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            model.add(Dense(input_dim=80,output_dim=2, init='he_normal', activation='softmax'))    
            sgd = SGD(lr=0.01, decay=1e-10, momentum=0.9, nesterov=True)

            model.compile(optimizer=sgd, loss='binary_crossentropy',class_mode='binary')

            return KerasClassifier(nn=model,**self.params)



if __name__ == "__main__":
    
    m = ModelV1(name="v1_stage1_ver3",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V1,
                kind = 's', fold_name='set1'
                )
    m.run()


    m = ModelV2(name="v2_stage1_ver3",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V2,
                kind = 's', fold_name='set2'
                )
    m.run()


    m = ModelV3(name="v3_stage1_ver3",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V3,
                kind = 's', fold_name='set3'
                )
    m.run()


    m = ModelV4(name="v4_stage1_ver3",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V4,
                kind = 's', fold_name='set4'
                )
    m.run()


    m = ModelV5(name="v5_stage1_ver3",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V5,
                kind = 's', fold_name='set5'
                )
    m.run()


    m = ModelV6(name="v6_stage1_ver3",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V6,
                kind = 's', fold_name='set6'
                )
    m.run()


    m = ModelV7(name="v7_stage1_ver3",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V7,
                kind = 's', fold_name='set7'
                )
    m.run()


    m = ModelV8(name="v8_stage1_ver3",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V8,
                kind = 's', fold_name='set8'
                )
    m.run()


    m = ModelV9(name="v9_stage1_ver3",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V9,
                kind = 's', fold_name='set9'
                )
    m.run()
    

    m = ModelV10(name="v10_stage1_ver3",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V10,
                kind = 's', fold_name='set10'
                )
    m.run()


    m = ModelV11(name="v11_stage1_ver3",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V11,
                kind = 's', fold_name='set11'
                )
    m.run()


    m = ModelV12(name="v12_stage1_ver3",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V12,
                kind = 's', fold_name='set12'
                )
    m.run()


    m = ModelV13(name="v13_stage1_ver3",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V13,
                kind = 's', fold_name='set13'
                )
    m.run()


    m = ModelV14(name="v14_stage1_ver3",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V14,
                kind = 's', fold_name='set14'
                )
    m.run()


    m = ModelV15(name="v15_stage1_ver3",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V15,
                kind = 's', fold_name='set15'
                )
    m.run()


    m = ModelV16(name="v16_stage1_ver3",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V16,
                kind = 's', fold_name='set16'
                )
    m.run()


    m = ModelV17(name="v17_stage1_ver3",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V17,
                kind = 's', fold_name='set17'
                )
    m.run()


    m = ModelV18(name="v18_stage1_ver3",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V18,
                kind = 's', fold_name='set18'
                )
    m.run()


    m = ModelV19(name="v19_stage1_ver3",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V19,
                kind = 's', fold_name='set19'
                )
    m.run()


    m = ModelV20(name="v20_stage1_ver3",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V20,
                kind = 's', fold_name='set20'
                )
    m.run()


    
    print 'Done stage 1'

    print 'Averaging'
    # averaging
    sample_sub = pd.read_csv('data/input/sample_submission.csv')
    testID = sample_sub[['ID']]
    a = pd.DataFrame()
    for i in ['data/output/temp/v1_stage1_ver3_test.csv',
              'data/output/temp/v2_stage1_ver3_test.csv',
              'data/output/temp/v3_stage1_ver3_test.csv',
              'data/output/temp/v4_stage1_ver3_test.csv',
              'data/output/temp/v5_stage1_ver3_test.csv',
              'data/output/temp/v6_stage1_ver3_test.csv',
              'data/output/temp/v7_stage1_ver3_test.csv',
              'data/output/temp/v8_stage1_ver3_test.csv',
              'data/output/temp/v9_stage1_ver3_test.csv',
              'data/output/temp/v10_stage1_ver3_test.csv',
              'data/output/temp/v11_stage1_ver3_test.csv',
              'data/output/temp/v12_stage1_ver3_test.csv',
              'data/output/temp/v13_stage1_ver3_test.csv',
              'data/output/temp/v14_stage1_ver3_test.csv',
              'data/output/temp/v15_stage1_ver3_test.csv',
              'data/output/temp/v16_stage1_ver3_test.csv',
              'data/output/temp/v17_stage1_ver3_test.csv',
              'data/output/temp/v18_stage1_ver3_test.csv',
              'data/output/temp/v19_stage1_ver3_test.csv',
              'data/output/temp/v20_stage1_ver3_test.csv',
              ]:
        x = pd.read_csv(i)
        a = pd.concat([a, x],axis=1)
    #x['TARGET'] = (a.rank().mean(1))/a.shape[0]
    # just averaging
    x['TARGET'] = a.mean(1)
    x = pd.concat([testID, x[['TARGET']]], axis=1)
    x.to_csv('data/output/temp/test_NN_ikkiver3_variantA.csv', index=None)
    #pubLB: 


    # averaging
    a = pd.DataFrame()
    train = pd.read_csv('data/input/train.csv')
    targetID = train[['ID']]
    for i in ['data/output/temp/v1_stage1_ver3_all_fold.csv',
              'data/output/temp/v2_stage1_ver3_all_fold.csv',
              'data/output/temp/v3_stage1_ver3_all_fold.csv',
              'data/output/temp/v4_stage1_ver3_all_fold.csv',
              'data/output/temp/v5_stage1_ver3_all_fold.csv',
              'data/output/temp/v6_stage1_ver3_all_fold.csv',
              'data/output/temp/v7_stage1_ver3_all_fold.csv',
              'data/output/temp/v8_stage1_ver3_all_fold.csv',
              'data/output/temp/v9_stage1_ver3_all_fold.csv',
              'data/output/temp/v10_stage1_ver3_all_fold.csv',
              'data/output/temp/v11_stage1_ver3_all_fold.csv',
              'data/output/temp/v12_stage1_ver3_all_fold.csv',
              'data/output/temp/v13_stage1_ver3_all_fold.csv',
              'data/output/temp/v14_stage1_ver3_all_fold.csv',
              'data/output/temp/v15_stage1_ver3_all_fold.csv',
              'data/output/temp/v16_stage1_ver3_all_fold.csv',
              'data/output/temp/v17_stage1_ver3_all_fold.csv',
              'data/output/temp/v18_stage1_ver3_all_fold.csv',
              'data/output/temp/v19_stage1_ver3_all_fold.csv',
              'data/output/temp/v20_stage1_ver3_all_fold.csv',
              ]:
        x = pd.read_csv(i)
        a = pd.concat([a, x],axis=1)
    #x['TARGET'] = (a.rank().mean(1))/a.shape[0]
    # just averaging
    x['TARGET'] = a.mean(1)
    x = pd.concat([targetID, x[['TARGET']]], axis=1)
    x.to_csv('data/output/temp/train_NN_ikkiver3_variantA.csv', index=None)
    #pubLB: 
    print 'Done averaging'

    print 'rank transformation with train and test'
    #rank trafo with train and test
    tr = pd.read_csv('data/output/temp/train_NN_ikkiver3_variantA.csv')
    te = pd.read_csv('data/output/temp/test_NN_ikkiver3_variantA.csv')
    tr_te = pd.concat([tr, te])
    tr_te['TARGET'] = tr_te['TARGET'].rank()
    # scale [0,1]
    tr_te['TARGET'] = (tr_te['TARGET'] - tr_te['TARGET'].min()) / (tr_te['TARGET'].max() - tr_te['TARGET'].min())
    tr = tr_te.iloc[:len(tr),:]
    te = tr_te.iloc[len(tr):,:]
    tr.to_csv('data/output/temp/train_NN_ikki_ver3.csv', index=False)
    te.to_csv('data/output/temp/test_NN_ikki_ver3.csv', index=False)
    print 'Done rank transformation'

    print 'CV of each model per fold and averaging'
    # CV of each model and averaging
    from sklearn.metrics import roc_auc_score as AUC
    a = pd.DataFrame()
    set_idnex = 1
    set_data = pd.read_csv('data/input/5fold_20times.csv')
    y = train.TARGET
    for i in ['data/output/temp/v1_stage1_ver3_all_fold.csv',
              'data/output/temp/v2_stage1_ver3_all_fold.csv',
              'data/output/temp/v3_stage1_ver3_all_fold.csv',
              'data/output/temp/v4_stage1_ver3_all_fold.csv',
              'data/output/temp/v5_stage1_ver3_all_fold.csv',
              'data/output/temp/v6_stage1_ver3_all_fold.csv',
              'data/output/temp/v7_stage1_ver3_all_fold.csv',
              'data/output/temp/v8_stage1_ver3_all_fold.csv',
              'data/output/temp/v9_stage1_ver3_all_fold.csv',
              'data/output/temp/v10_stage1_ver3_all_fold.csv',
              'data/output/temp/v11_stage1_ver3_all_fold.csv',
              'data/output/temp/v12_stage1_ver3_all_fold.csv',
              'data/output/temp/v13_stage1_ver3_all_fold.csv',
              'data/output/temp/v14_stage1_ver3_all_fold.csv',
              'data/output/temp/v15_stage1_ver3_all_fold.csv',
              'data/output/temp/v16_stage1_ver3_all_fold.csv',
              'data/output/temp/v17_stage1_ver3_all_fold.csv',
              'data/output/temp/v18_stage1_ver3_all_fold.csv',
              'data/output/temp/v19_stage1_ver3_all_fold.csv',
              'data/output/temp/v20_stage1_ver3_all_fold.csv',
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
    #AUC:
