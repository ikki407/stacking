#!/bin/bash

echo "Start model training and prediction..."
python test_binary_class/scripts/binary.py

echo "Done...\n"

echo "Start model training and prediction..."
python test_multi_class/scripts/multiclass.py

echo "Done...\n"

echo "Start model training and prediction..."
python test_regression/scripts/regression.py

echo "Done...\n"

