#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 12:43:24 2021

@author: alexxcollins
"""

from regression import GenerateData, UniformX

class UniformX(GenerateData):
    
    def generate_X(self, low=-10, high=10):
        X = self.rng.uniform(low=low, high=high, size=(self.N,self._p))
        X0 = np.ones((self.N,1))
        self.X =  np.concatenate([X0, X], axis=1)

    def generate_y(self):
        return np.matmul(self.X, self.beta) + self.e
    
    def generate_dataset(self):
        self.y = self.generate_y()
        
        