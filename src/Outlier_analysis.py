#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 14:50:01 2021

@author: suenchihang
"""

from regression import GenerateData, UniformX
from Outlier import Outlier, Outlier_position
from simulation import print_coef, simulation, change_factor
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng()

        
#testing impact

#test1 - test impact of Num of Outliers, assuming evenly spread
set1=UniformX(N=1000)
set1.generate_dataset()
X=set1.X
y=set1.y
test1=Outlier(N=1000)
positions=[]

Number_of_outliers = np.arange(0,10)
positions.append(Outlier_position([10,10],[-10,-10],N=i)) for i in Number_of_outliers
    
        
d=change_factor(test1, 1000,factor={"positions":positions},original_X=X, original_y=y, original_beta=set1.beta, magnitude=500 )
print("score_mean: "+str(d["score_mean"]))
print(d["b0_mean"])
print(d["b1_mean"])
        




