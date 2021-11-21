#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 15:00:30 2021

@author: alexxcollins
"""

from regression import UniformX
from colinearity import ColinearX

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

def pairwise_test():
    test = ColinearX(N=5)
    test.generate_dataset()
    test.plot3D()
    return test

