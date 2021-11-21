#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 12:43:24 2021

@author: alexxcollins

different types of colinearity to investigate:
    (1) pairwise colinearity between X1 and X2
    (2) where p' > 2, "colinearity" between intercept and X1
    (3) Xi which is linear combination of other X
"""

from regression import GenerateData, UniformX
from numpy.random import default_rng
from abc import ABC, abstractmethod
import numpy as np
        
class ColinearX(GenerateData):
    
    def generate_X(self, co_type='pairwise correlation', **kwargs):
        
        if co_type == 'pairwise correlation':
            self.pairwiseX(**kwargs)
            
    def pairwiseX(self, mean = None, cov=0.3):
        
        if mean == None:
            mean = np.zeros(self._p)
        else:
            mean = np.array(mean)
        
        self.X1X2_cov = cov
        cov = np.diag(np.ones(self._p))
        cov[0,1], cov[1,0] = self.X1X2_cov, self.X1X2_cov
        
        self.X = self.rng.multivariate_normal(mean=mean, cov=cov, size=self.N)
        self.X0 = np.ones((self.N, 1))
        self.X1 = np.concatenate([self.X0, self.X], axis=1)
        