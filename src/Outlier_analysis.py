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
from sklearn.model_selection import train_test_split
from yellowbrick.regressor import ResidualsPlot

rng = np.random.default_rng()

        
#testing impact

#test1 - test impact of Num of Outliers, 
#assuming i)evenly spread, ii)high end value of input X, iii)low end, iv)centre value of input X

#initialize for object of GenerateData sub class Outlier
set1=UniformX(N=1000)
set1.generate_dataset()
X=set1.X
y=set1.y
test1=Outlier(N=1000)
outlier_magnitude = 500

#initialize 4 different scenarios of outlier positions, and set varying number of outliers for each scenario
Number_of_outliers = np.arange(0,11)
p = ['even', 'high','low','centre']
pos = {k:None for k in p}
num = {k:None for k in p}
pos['even']=[Outlier_position([10,10],[-10,-10],N=i) for i in Number_of_outliers]
pos['high']=[Outlier_position([10,10],N=i) for i in Number_of_outliers]
pos['low']=[Outlier_position([-10,-10],N=i) for i in Number_of_outliers]
pos['centre']=[Outlier_position([0,0],N=i) for i in Number_of_outliers]

#create dictionary of dictionary storing results based on varying number of outliers for 4 position scenarios
for position in p:
    num[position] = change_factor(test1, 1000,factor={"positions":pos[position]}, magnitude=outlier_magnitude, original_X=X, original_y=y, original_beta=set1.beta,)

#create chart to show impact on coefficients against number of outliers
print('testing impact of number of outliers')
for estimate_key in ["b0_mean", "b1_mean", "b2_mean", "b0_variance", "b1_variance", "b2_variance","score_mean", "score_variance", "ssr_mean", "ssr_variance"]:
    for position in p:
        plt.plot(Number_of_outliers, num[position][estimate_key], label = position)   
    plt.title(estimate_key)
    plt.xlabel('Number of Outliers')
    plt.legend()
    plt.show()
        
    
    
    
    

#test2 - test impact of magnitude of Outliers, 
#assuming i)evenly spread, ii)high end value of input X, iii)low end, iv)centre value of input X

#initialize for object of GenerateData sub class Outlier
test2=Outlier(N=1000)
outlier_number = 10

#initialize 4 different scenarios of outlier positions, and set varying number of outliers for each scenario
Magnitude_of_outliers = np.arange(-1000,1000, step=100)
p = ['even', 'high','low','centre']
pos = {k:None for k in p}
mag = {k:None for k in p}
pos['even']=Outlier_position([10,10],[-10,-10], N=outlier_number)
pos['high']=Outlier_position([10,10], N=outlier_number)
pos['low']=Outlier_position([-10,-10], N=outlier_number)
pos['centre']=Outlier_position([0,0], N=outlier_number) 

#create dictionary of dictionary storing results based on varying magnitude of outliers for 4 position scenarios
for position in p:
    mag[position] = change_factor(test2, 1000,factor={"magnitude":Magnitude_of_outliers},positions=pos[position], original_X=X, original_y=y, original_beta=set1.beta,)

#create chart to show impact on coefficients against magnitude of outliers
print('testing impact of magnitude of outliers')
for estimate_key in ["b0_mean", "b1_mean", "b2_mean", "b0_variance", "b1_variance", "b2_variance","score_mean", "score_variance", "ssr_mean", "ssr_variance"]:
    for position in p:
        plt.plot(Magnitude_of_outliers, mag[position][estimate_key], label = position)   
    plt.title(estimate_key)
    plt.xlabel('Magnitude of Outliers')
    plt.legend()
    plt.show()
    
    

#how to detect outliers
#initialize for object of GenerateData sub class Outlier
test3=Outlier(N=1000)
outlier_number = 10
outlier_magnitude = 200
positions=Outlier_position([-10,-10], N=outlier_number)
test3.generate_dataset(magnitude=outlier_magnitude, original_X=X, original_y=y, original_beta=set1.beta, positions=positions)
test3.fit()
test3.plot2D()

#residual plot - library reference: https://www.scikit-yb.org/en/latest/api/regressor/residuals.html
#initialize for object of GenerateData sub class Outlier
test4=Outlier(N=1000)
test4.generate_dataset(magnitude=outlier_magnitude, original_X=X, original_y=y, original_beta=set1.beta, positions=positions)

X=test4.X
y=test4.y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # split the train and test data
model = LinearRegression() # Instantiate the linear model and visualizer
visualizer = ResidualsPlot(model)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show()                 # Finalize and render the figure

