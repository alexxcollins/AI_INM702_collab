#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 23:26:41 2021

@author: suenchihang
"""

from regression import GenerateData, UniformX
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng()

#generate data with Outliers given orginal X and y
class Outlier(GenerateData): 
    
    def generate_X(self, original_X, original_y, original_beta, positions, magnitude):
        if positions==[]:
            self.X = original_X
        else:
            self.X = np.vstack((original_X,np.array((positions))))
        self.N = len(original_X)+len(positions)
        self.X0 = np.ones((self.N,1))
        self.X1 =  np.concatenate([self.X0, self.X], axis=1)
        self.original_X=original_X
        self.original_y=original_y
        self.original_beta=original_beta
        self.positions = positions
        self.magnitude = magnitude
    def generate_y(self):
        if self.positions==[]:
            self.y = self.original_y
        else:
            self.e_add = self.rng.normal(0, self.e_var**(1/2), size=(len(self.positions),1))
            self.y_add = np.matmul(self.positions, self.original_beta[1:])+ self.e_add + (self.magnitude + self.original_beta[0])
            self.y = np.vstack((self.original_y, self.y_add))
 
#spread positions given start and end points, 
#and handle trivial cases when N=0, N=1 or when end-point not given
def Outlier_position(start_position, end_position=[[float("inf"),float("inf")]], N=1):
    if N==0:
        positions = []
    else:
        if end_position==[[float("inf"),float("inf")]]:
            positions=[start_position for n in range(N)]
        else:
            if N==1:
                positions=[((np.array(start_position) + np.array(end_position))/2).tolist()]
            else:
                increment=(np.array(end_position) - np.array(start_position))/(N-1)
                positions=[]
                for n in range(N):
                    new = np.add(np.array(start_position) , np.array(n * increment)).tolist()
                    positions.append(new)
            
    return positions
    

def print_coef(X, y, to_print=True):
    reg = LinearRegression().fit(X, y)
    score=reg.score(X, y)
    coef=reg.coef_
    intercept=reg.intercept_
    ss_residual = np.linalg.norm(y - reg.predict(X)) ** 2
    if to_print:
        print("intercept: "+str(intercept)+"; coef: "+str(coef))
    return     intercept, coef, score, ss_residual
        

          
        
    
    
    




