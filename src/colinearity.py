#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 12:43:24 2021

@author: alexxcollins
"""

from regression import GenerateData, UniformX
from numpy.random import default_rng
from abc import ABC, abstractmethod
import numpy as np
        
class ColinearX(GenerateData):
    
    def generate_X(self):
        