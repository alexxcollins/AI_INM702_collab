#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 12:01:05 2021

@author: alexxcollins
"""

#%% to test if I can create ABC which defintes how to generate y with x
from numpy.random import default_rng
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GenerateData(ABC):
    '''
    The base class will generate random Y given parameters for sample size
    beta (coefficients for intercept plus independent variables), variance of
    epsilon and optional X. 
    The X generated by this base class will match dimension of beta and will
    be generated from uniform distribution along each dimension. Subclasses can
    be used to generate or supply difference Xs, with outliers, co-depence etc.
    '''

    def __init__(self, N=1000, beta=(1,2,3), noise_var=1):
        # not sure where to set random seed. Should we do this in Jupyter notebook?
        # and pass into the class?
        self.rng = default_rng(42)
        self.N = N
        self._p = len(beta)-1
        self._p1 = len(beta)
        self.beta = np.array(beta).reshape((self._p1,1))
        # we often use p and p' in regression formulas, so I'm defining both
        # at the moment discourage changing with underscore rather than setters
        self.e_var = 1
        
        
    def _generate_epsilon(self):
        return self.rng.normal(0, self.e_var**(1/2), 
                               size=(self.N,1))
    
    @abstractmethod
    def generate_X(self, low=None, high=None):
        pass
    
    def generate_y(self):
        self.e = self._generate_epsilon()
        self.y = np.matmul(self.X1, self.beta) + self.e  
        
    def generate_dataset(self, **kwargs):
        self.generate_X(**kwargs)
        self.generate_y()
             
    def line2D(self):
        '''
        this is designed to work for a one factor regression
        
        takes the beta (intercept + x coefficient) of the original data to 
        create a visualisation of the line showing the relationship between
        x and y, as well as a scatter plot of the data.
        
        ###### ToDo
        put line of best fit from regression results
        ######
        '''
        if self._p != 1:
            raise Exception('dimension of X must be 1. X had dimension {}'
                            .format(self._p))
        
        x = np.linspace(self.X.min(), self.X.max(), 100)
        y = self.beta[0] + self.beta[1] * x
        
        fig, ax = plt.subplots()
        ax.plot(x, y, color='r')
        ax.scatter(self.X, self.y, alpha=0.2)

            
class UniformX(GenerateData):
    
    def generate_X(self, low=-10, high=10):
        self.X = self.rng.uniform(low=low, high=high, size=(self.N,self._p))
        self.X0 = np.ones((self.N,1))
        self.X1 =  np.concatenate([self.X0, self.X], axis=1)

