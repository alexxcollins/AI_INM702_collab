#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 16:14:54 2021

@author: suenchihang
"""

"""

"""

import numpy as np

"""
As L2 regularization treats weight and bias separately, we will use "separate"-style notations:
    Z = WX + b
    X: input or values activated from previous layer
    W: the weight matrix excl bias
    b: bias vector
    Z: linear combination value just before activation
    
    A = g(Z), where A is the activation value and g() is the activation function
    
    n: no of nodes of a particular layer
    m: no of samples
    l: layer number

"""

def sigmoid(Z):
    """
    Parameters
    ----------
    Z: scalr or array-like
    
    Returns
    -------
    A: sigmoid value to each element of Z, same shape as Z
    """
    A = 1 / (1 + np.exp(Z))
    return A
    
def softmax(Z):
    """

    Parameters
    ----------
    Z : array-like of one dimension

    Returns
    -------
    A: softmax values corresponding to each element of Z

    """
    A = np.exp(Z)/np.sum(np.exp(Z))
    return A
    
def relu():
    pass


def init_parameter(n_current, n_prev, scale):
    """
    To initialize weight and bias parameter

    Parameters
    ----------
    n_current : no of nodes of current layer
    n_prev : no of nodeds of previous layer
    scale: scale to multiply to uniform random variable, or method of initialization

    Returns
    -------
    W: weight of node connection, shape(size of current layer, size of previous layer)
    b: bias for current layer

    """
    if scale =='Xavier': #Xavier initilization
        scale = (1/n_prev)**0.5
    else if scale =="He": #He initilization
        scale = (2/n_prev)**0.5
        
    W = np.random.normal(size = (n_current, n_prev))*scale
    b = np.zeros((n_current, 1))
    
    return W, b

def forward_linear(W, X, b):
    """
    

    Parameters
    ----------
    W : 2-D array of shape (size of current layer, size of previous layer), Weight of nodes
    X : 2-D array, input data or activations from previous layer of shape (size of previous layer, no of samples)
    b : bias, of array shape (size of current layer, 1)

    Returns
    -------
    Z: linear combination of weight and previous activated output plus bias, shape (size of current layer, no of samples)

    """
    Z = np.matmul(W, X) + b
    
    
    return Z
    
    
def forward_activate(Z, activation):
    """

    Parameters
    ----------
    Z : array-like of one dimension
    activation: type of activation

    Returns
    -------
    A: activation values corresponding to each element of Z

    """
    if activation == 'sigmoid':
        A = sigmoid(Z)
    else if activation == 'relu':
        A = relu(Z)
    else if activation == 'softmax':
        A = softmax(Z)
    
    return A
    
    
def backprop_activate(dA, activation):
    if activation == "sigmoid":
        
    
    
    return dZ
    
def backprop_linear():
    
    
    return dA_prev, dW, db



  


def cost_function(prob_predict, y_true, m, regularization_lambda):
    """
    

    Parameters
    ----------
    prob_predict : predicted probability
    y_true : target value for training
    m : no of samples
    regularization_lambda : lambda of L2 regularization method

    Returns
    -------
    J : TYPE
        DESCRIPTION.

    """

    r = 0.5 * regularization_lambda * np.sum(W**2)/m

    J = - np.sum(np.multiply(y_true, log(prob_predict)))/m + r
    
    return J
    
    
    
    

def mask(shape, dropout):

    

class NN():
    def __init__(self, input_dim, dropout=0):
        """
        initialize neural network
        
        Parameters
        ----------
        input_dim: dimension of input for the neural network
        dropout: dropout ratio of the input layer, from 0 to 1
        """
        self.input_dim = input_dim
        self.layer=0 #current number of layers
        self.N = [input_dim] #list of no of nodes for each layer, with self.N[0] set as input_dim
        self.activation=['na']  #list of activation type for each layer
        self.dropout=[dropout]  #list of dropout ratio for each layer
        
        #create empty dictionary for W, b, Z, A
        self.W={} #weight of nodes
        self.b={} #bias
        self.Z={} #pre-activation values, linear combination Z = WX + b
        self.A={} #activation values


        

    def add(self, N, activation='sigmoid', dropout=0):
        """
        adding one layer to neural network
        
        Parameters
        ----------
        N: no of nodes of the layer
        activation: type of activation of the layer
        dropout: dropout ratio of the layer, from 0 to 1
    
        """
        self.layer += 1
        self.N.append(N)
        self.activation.append(activation)
        self.dropout.append(dropout)
        
        
    def hyper(self, learning_rate=0.01, optimizer='none', regularization_lambda=0, epochs=100, batch_size=10, weight_scale=0.01, seed=1):
        """
        Setting hyper-parameters of neural network
        
        Parameters
        ----------
        learning_rate: learning rate for updating weight by gradient
        optimizer: type of optimization method
        regularization: boolean type, whether or not to use L2 regularization 
        epochs: max number of epochs for training
        batch_size: batch size for training
        weight_scale: the scale used for initializing weight, multiplied to standard normal random variable
        seed: random seed for training
        
        """
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.regularization_lambda = regularization_lambda
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_scale = weight_scale
        self.seed = seed

    
    def fit(self, X_train, y_train):
        """
        Training the neural network given the traiing data

        Parameters
        ----------
        X_train : array of shape (no of samples, no of features), data for training
        y_train : array of shape (no of samples), target value

        Returns
        -------
        None.

        """
        self.X_train = X_train
        self.y_train = y_train
        
        #initialize weight and bias
        np.random.seed(self.seed)
        for l in range(1, self.layer):
            self.W[l], self.b[l] = init_parameter(n_current=self.N[l], n_prev=self.N[l-1], scale=self.weight_scale)
        
        #initialize "activation" of layer 0
        self.A[0]=self.X_train
        # forward propagation
        for l in range(1, self.layer):
            self.Z[l] = forward_linear(self.W[l], self.A[l-1], self.b[l])
            self.A[l] = forward_activate(self.Z[l], self.activation[l])
        
        #compute the cost
        self.m = len(self.y_train)
        J = cost_function(self.A[self.layer], self.y_train, self.m, self.regularization_lambda)
        
        #backpropagation
        

        
        
    def predict(self, X_enquiry):
        #initialize "activation" of layer 0
        self.A[0]=X_enquiry
        # forward propagation
        for l in range(1, self.layer):
            self.Z[l] = forward_linear(self.W[l], self.A[l-1], self.b[l])
            self.A[l] = forward_activate(self.Z[l], self.activation[l])
        
        y_predict = self.A[self.layer]
        
        return y_predict
        
    def evaluate(self, X_test, y_test):
        pass
    
        
        
        
        
                
    