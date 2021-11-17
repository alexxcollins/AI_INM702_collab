#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 18:57:02 2021

@author: alexxcollins
"""
import numpy as np

from regression import GenerateData, UniformX

class Outlier(UniformX):
    
    # Alex: I changed the class Outlier inherits from to UniformX, so that
    # you can generate X data from uniform distribution
    # the problem with that is that "generate_X" below shares the same name,
    # so the uniform X data can't be generated. I think if name of method 
    # below is changed it should all work
    
    def generate_X(self, original_X, original_y, original_beta, positions, magnitude):
        self.X = np.vstack((original_X,np.array((positions))))

