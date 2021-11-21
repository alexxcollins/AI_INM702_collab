#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 15:00:30 2021

@author: alexxcollins
"""

from regression import UniformX
from colinearity import ColinearX

#%% visualisation tests
def d2test():
    test = UniformX(beta = (2, 0.7))
    test.generate_dataset()
    test.plot2D()
    return test

def d3test():
    test = UniformX()
    test.generate_dataset()
    test.plot3D()
    return test

def subsetTest():
    t1 = UniformX(beta = (2, 0.7))
    t1.generate_dataset()
    t1.plot2D()
    
    t2 = UniformX(beta = (2, 0.7, 2))
    t2.generate_dataset()
    t2.plot2D(i=1)
    t2.plot2D(i=2)
    t2.plot3D()
    
    t3 = UniformX(beta = (2, 2, 0.7))
    t3.generate_dataset()
    t3.plot2D(i=1)
    t3.plot2D(i=2)
    t3.plot3D()
    
    return t1, t2, t3

#%% colinear data tests
def pairwise_test(mean=None, cov=0.3):
    test = ColinearX(beta=(0, 1, 1))
    test.generate_dataset(cov=cov)
    test.plot2D()
    test.plot2D(i=2)
    test.plot3D()
    return test

