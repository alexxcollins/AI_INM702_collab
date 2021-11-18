#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 12:42:14 2021

@author: alexxcollins


"""

from regression import GenerateData, UniformX
from numpy.random import default_rng
from abc import ABC, abstractmethod
import numpy as np

class Colinear(GenerateData):
    
    def generate_X(self, strin = 'hello'):
        self.X = 1
        print(strin)