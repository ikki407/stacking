#!/bin/bash

echo "Start model training and prediction"

#version1(XGB)
echo "Starting ver1(XGB)"
python ikki_feat_ver1.py
python scripts/ikki_xgb_1.py
echo "Done ver1"

#version2(XGB)
echo "Starting ver2(XGB)"
python ikki_feat_ver2.py
python scripts/ikki_xgb_2.py
echo "Done ver2"

#version3(NN)
echo "Starting ver2(NN)"
python ikki_feat_ver3.py
python scripts/ikki_NN_1.py
echo "Done ver3"

#Combining the data for final ensembling
echo "Combining the data for final ensembling"
python combine_pred.py
echo "Done combining"
