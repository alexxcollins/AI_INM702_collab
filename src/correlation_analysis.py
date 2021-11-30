#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 20:03:06 2021

@author: suenchihang
"""

from regression import GenerateData, UniformX
from colinearity import ColinearX
from simulation import print_coef, simulation, change_factor
import numpy as np
from numpy.random import default_rng

import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


rng = np.random.default_rng()

#initialize for object of GenerateData sub class ColinearX
test_colinear=ColinearX(N=1000, random_seed=None)

#set list of changing correlation
correlation_list = [i/10 for i in range(-9,10)]

#store result of simulation over the list of changing correlation
result = change_factor(test_colinear, 1000,factor={"cov":correlation_list} )

#plot the result
print('testing impact of correlation between X1, X2')
for estimate_key in ["b0_mean", "b1_mean", "b2_mean", "b0_variance", "b1_variance", "b2_variance","score_mean", "score_variance", "ssr_mean", "ssr_variance"]:
    plt.plot(correlation_list, result[estimate_key], label = "colinear")   
    plt.title(estimate_key)
    plt.xlabel('Correlation')
    plt.legend()
    plt.show()

