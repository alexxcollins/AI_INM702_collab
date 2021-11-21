#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 12:01:05 2021

@author: alexxcollins
"""

#%% import packages
from numpy.random import default_rng
from abc import ABC, abstractmethod
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#%% GnerateData
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
        
    def train_test_split(self, test_size=0.25, shuffle=True):
        '''
        splits data into test and train datasets.
        
        will shuffle dataset if shuffle=True. Test size should be between
        0 and 1.0 and is the proportion of the sample held back for testing.

        '''
        if shuffle:
            index = np.arange(self.N)
            self.rng.shuffle(index)
            self.X = self.X[index]
            self.y = self.y[index]
        
        test_N = np.round(self.N*test_size).astype(int)
        self.X_test = self.X[:test_N]
        self.y_test = self.y[:test_N]
        self.X_train = self.X[test_N:]
        self.y_train = self.y[test_N:]
        
    def generate_dataset(self, test_size=0.25, shuffle = True, **kwargs):
        self.generate_X(**kwargs)
        self.generate_y()
        self.train_test_split(test_size=test_size, shuffle=True)
        
    def fit(self, fit_intercept=True):
        '''
        Use sklearn to fit the linear regression and "save" results as object
        attributes. 
        This function has to be run after .train_test_split method.
        It calculates:
            predicted y, for the X_test data
            score - which is R^2 as in sklearn LinearRegression
            intercept, coef from sklearn LinearRegression
            b_pred which is concatenation of intercept and coe and same shape
                as self.beta

        '''
        reg = LinearRegression(fit_intercept=fit_intercept)
        reg = reg.fit(self.X_train, self.y_train)
        self.y_pred = reg.predict(self.X_test)
        self.score = reg.score(self.X_test, self.y_test)
        # intercept and coef attributes are equivalent to scikit learn 
        # attributes
        self.intercept = reg.intercept_
        self.coef = reg.coef_
        # we also set b_pred attribute. This is equivalent to self.beta: it
        # includes intercept and other coeffs and is same shape as self.beta
        b = np.concatenate((self.intercept, 
                            np.resize(self.coef, (self.coef.size))
                            ))
        self.b_pred = b[:, np.newaxis]
             
    def plot2D(self, i=1, fitted_line=True, true_beta_line=True):
        '''
        plot the scatter of y and X_i. The user can select the column of X
        to plot by setting i.
        
        takes the beta (intercept + x_i coefficient) of the original data to 
        create a visualisation of the line showing the relationship between
        X and y, as well as a scatter plot of the data.
        
        ###### ToDo
        options to print title for chart and equation for line of best fit
        ######
        
        Parameters
        ----------
        i : int
            optional parameter to select on column from X matrix. i indexed at
            1, so choose 1 for X_1 and p for final column of X
            
        '''
        # variable below is the selected column of X data
        X_i = self.X[:,i-1,np.newaxis]
        
        X = np.linspace(X_i.min(), X_i.max(), 100)
        y_beta = self.beta[0] + self.beta[i] * X
        y_fitted = self.b_pred[0] + self.b_pred[i] * X
        
        fig, ax = plt.subplots()
        if true_beta_line:
            ax.plot(X, y_beta, color='r')
        if fitted_line:
            ax.plot(X, y_fitted, color='g')
        ax.scatter(X_i, self.y, color='b', alpha=0.2)
        
    # next two functions used to generate integer range around (a, b)
    # use round_down(a) - returns integer below a, and works if a is +ve or -ve
    # similarly for round_up(b)
    # rounds number "up". 9.8 -> 10; -9.8 -> -9
    def round_up(self, x): return int(x) + (x % 1 > 0)*(x>0)

    # rounds number "down". 9.8 -> 9; -9.8 -> -10
    def round_down(self, x): return int(x) - (x % 1 > 0)*(x<0)
    
    def plot3D(self):
        '''
        this is designed to work for a regression with two input features
        
        takes the beta (intercept + x coefficients) of the original data to 
        create a visualisation of the line showing the relationship between
        x and y, as well as a scatter plot of the data.
        
        ###### ToDo
        put line of best fit from regression results
        
        options to print title for chart and equation for line of best fit
        ######
        '''
        if self._p != 2:
            raise Exception('dimension of X must be 2. X had dimension {}'
                            .format(self._p))
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        X1 = np.linspace(self.round_down(self.X[:,0].min()),
                         self.round_up(self.X[:,0].max()), 2)
        X2 = np.linspace(self.round_down(self.X[:,1].min()),
                         self.round_up(self.X[:,1].max()), 2)
        X1, X2 = np.meshgrid(X1, X2)
        y = self.beta[0] + self.beta[1] * X1 + self.beta[2] * X2

        ax.scatter(self.X[:,0], self.X[:,1], self.y, alpha=0.2)
        ax.plot_surface(X1, X2, y, alpha=0.2, color='r')

#%% UniformX            
class UniformX(GenerateData):
    
    def generate_X(self, low=-10, high=10):
        self.X = self.rng.uniform(low=low, high=high, size=(self.N,self._p))
        self.X0 = np.ones((self.N,1))
        self.X1 =  np.concatenate([self.X0, self.X], axis=1)

