#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 15:00:30 2021

@author: alexxcollins
"""

from regression import UniformX
from colinearity import ColinearX
from matplotlib import pyplot as plt
import numpy as np


#%% visualisation tests
def d2test():
    test = UniformX(beta = (2, 0.7))
    test.generate_dataset()
    test.fit()
    test.plot2D()
    return test

def d3test():
    test = UniformX()
    test.generate_dataset()
    test.fit()
    test.plot3D()
    return test

def subsetTest():
    t1 = UniformX(beta = (2, 0.7))
    t1.generate_dataset()
    t1.fit()
    t1.plot2D()
    
    t2 = UniformX(beta = (2, 0.7, 2))
    t2.generate_dataset()
    t2.fit()
    t2.plot2D(i=1)
    t2.plot2D(i=2)
    t2.plot3D()
    
    t3 = UniformX(beta = (2, 2, 0.7))
    t3.generate_dataset()
    t3.fit()
    t3.plot2D(i=1)
    t3.plot2D(i=2)
    t3.plot3D()
    
    return t1, t2, t3

#%% colinear data tests
def pairwise_test(mean=None, cov=0.3):
    test = ColinearX(beta=(0, 3, 1))
    test.generate_dataset(cov=cov)
    test.fit()
    test.plot2D()
    test.plot2D(i=2)
    test.plot3D()
    return test

def pairwise_test2(mean=None, cov=0.3):
    test = ColinearX(beta=(0, 3, 1, 2, 2))
    test.generate_dataset(cov=cov)
    test.fit()
    test.plot2D()
    test.plot2D(i=2)
    test.plot2D(i=3)
    test.plot2D(i=4)
    print('test score is {}'.format(test.score))
    print('predicted b is {}'.format(test.b_pred))
    print('beta of underlying distribution is {}'.format(test.beta))
    return test

def co_error(mean=None, cov=0.3, plot_dim=0):
    test = ColinearX(beta=(1,2,3,4,2))
    test.generate_dataset(mean=mean, cov=cov)
    test.fit()
    test.plot2D()
    test.plot2D(i=2)
    test.plot2D(i=3)
    test.plot2D(i=4)
    
    fig, ax = plt.subplots()
    ax.scatter(test.y_pred.reshape(test.y_pred.size), test.X_test[:,plot_dim])
    ax.scatter(test.y_test.reshape(test.y_pred.size), test.X_test[:,plot_dim])
    
    return test

def plot(regr=[]):
    for r in regr:
        fig, axs = plt.subplots(2,2)
        for i, ax in enumerate(axs.flatten()):
            ax.hist(r.X[:,i], bins=20)
        plt.show()
        for i in range(4):
            r.plot2D(i+1)


#%% test fit lines when there is a really massive outlier
def outrageous_outlier():
    test = UniformX(N=100, beta=(0,1))
    test.generate_dataset()
    # now make a massive outlier in the training dataset
    test.X_train[0] = -250
    test.y_train[0] = 250
    test.fit()
    test.plot2D()
    print('test score is {}'.format(test.score))
    print('predicted b is {}'.format(test.b_pred))
    print('beta of underlying distribution is {}'.format(test.beta))
    return test

