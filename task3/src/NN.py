#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 16:14:54 2021

@author: suenchihang
"""

import numpy as np

"""
As R2 regularization and dropout treat weight and bias separately, we will use "separate"-style notations:
    Z = WX + b
    X: input or values activated from previous layer
    W: the weight matrix excl bias
    b: bias vector
    Z: linear combination value just before activation

"""

def sigmoid(Z):
    
    return A
    
    
def relu():
    pass

def forward_linear():
    
    
def forward_activate():
    
    
    
def backprop():


def loss_function():
    

def mask(shape, dropout):

    

class NN():
    def __init__(self, input_dim, dropout=0):
        """
        input_dim: dimension of input for the neural network
        dropout: dropout ratio of the input layer, from 0 to 1
        """
        self.input_dim = input_dim
        self.layer=0 #current number of layers
        self.N = [input_dim] #list of no of nodes for each layer
        self.activation=['na']  #list of activation type for each layer
        self.dropout=[dropout]  #list of dropout ratio for each layer


        

    def add(self, N, activation='sigmoid', dropout=0):
        """
        adding one layer to neural network
        input arguments:
        N: no of nodes of the layer
        activation: type of activation of the layer
        dropout: dropout ratio of the layer, from 0 to 1
    
        """
        self.layer += 1
        self.N.append(N)
        self.activation.append(activation)
        self.dropout.append(dropout)
        
        
    def hyper(self, learning_rate=0.01, optimizer='SGD', epochs=100, batch_size=10, weight_scale=0.01, seed=1):
        

    
    def fit(self, X_train, y_train):

        
        
    def predict(self, X_enquiry):
        
        
        
    def evaluate(self, X_test, y_test):
        pass
    
        
        
        
        
                
    