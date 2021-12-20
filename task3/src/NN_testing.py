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
from timeit import timeit
from tensorflow.keras import datasets

"""
Labels

Each training and test example is assigned to one of the following labels:

0 T-shirt/top
1 Trouser
2 Pullover
3 Dress
4 Coat
5 Sandal
6 Shirt
7 Sneaker
8 Bag
9 Ankle boot
"""


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

path = '.'

train_data = pd.read_csv(path + '/fashion-mnist_train.csv')
test_data = pd.read_csv(path + '/fashion-mnist_test.csv')

y_label = train_data['label'].to_numpy()
X_train = train_data.drop(['label'], axis = 1).to_numpy()/255
y_label_test = test_data['label'].to_numpy()
X_test = test_data.drop(['label'], axis = 1).to_numpy()/255

y_train = np.zeros((np.shape(y_label)[0], 10))
y_train[np.arange(np.shape(y_label)[0]), y_label] = 1
y_test = np.zeros((np.shape(y_label_test)[0], 10))
y_test[np.arange(np.shape(y_label_test)[0]), y_label_test] = 1
#y = pd.get_dummies(target['label'], prefix='label')
#print(y.head())
print(y_test)


test_model = NN(784)
test_model.add(784, 'relu')
test_model.add(10, 'softmax')

print(test_model.layer)
print(y_test)
test_model.hyper(0.1, epochs=100, weight_scale='Xavier')

test_model.fit(X_train, y_train, visible=True)
print('predict: '+str(test_model.predict(X_test)))
test_model.evaluate(X_test, y_test)

print('testing done')