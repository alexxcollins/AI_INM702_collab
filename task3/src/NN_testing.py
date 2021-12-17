#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 11:07:20 2021

@author: suenchihang
"""

from NN import NN
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


path = '.'

train_data = pd.read_csv(path + '/fashion-mnist_train.csv')
test_data = pd.read_csv(path + '/fashion-mnist_test.csv')

y_train = train_data['label']/10
X_train = train_data.drop(['label'], axis = 1)/255
y_test = test_data['label']/10
X_test = test_data.drop(['label'], axis = 1)/255

#y = pd.get_dummies(target['label'], prefix='label')
#print(y.head())


test_model = NN(784)
test_model.add(784, 'sigmoid')
test_model.add(784, 'sigmoid')
test_model.add(784, 'sigmoid')
test_model.add(1, 'softmax')

print(test_model.layer)
test_model.hyper(0.001, epochs=2)
test_model.fit(X_train.to_numpy(), y_train.to_numpy(), visible=True)
test_model.predict(X_test.to_numpy())

print('testing done')