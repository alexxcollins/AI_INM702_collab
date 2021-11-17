#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 18:57:02 2021

@author: alexxcollins
"""
import numpy as np

from regression import GenerateData, UniformX

class Outlier(UniformX):
    
    def generate_X(self, original_X, original_y, original_beta, positions, magnitude):
        self.X = np.vstack((original_X,np.array((positions))))

