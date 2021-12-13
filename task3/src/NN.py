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
    X: input or values activated from previous layer, shape(no of features, no of samples)
    W: the weight matrix excl bias, shape(size of current layer, size of previous layer)
    b: bias vector, shape(size of current layer, 1)
    Z: linear combination value just before activation, shape(size of current layer, no of samples)
    
    A = g(Z), where A is the activation value and g() is the activation function, same shape as Z
    
    n: no of nodes of a particular layer
    m: no of samples
    l: layer number

"""

def sigmoid(Z):
    """
    Parameters
    ----------
    Z: scalar or array-like
    
    Returns
    -------
    A: sigmoid value to each element of Z, same shape as Z
    """
    A = 1 / (1 + np.exp(-Z))
    return A

def sigmoid_derivative(Z, A='none'):
    """
    Parameters
    ----------
    Z: scalar or array-like
    A: activation on Z, same shape as Z
    
    Returns
    -------
    derivative: compute derivative element-vise to Z
    """
    if A='none':
        derivative = sigmoid(Z) * (1 - sigmoid(Z))
    else:
        derivative = A * (1 - A)
    return derivative

def softmax(Z):
    """

    Parameters
    ----------
    Z : array of shape (no of nodes or classes, no of samples)

    Returns
    -------
    A: softmax values of shape (no of nodes or classes, no of samples)

    """
    eZ = np.exp(Z)
    sum_eZ = np.sum(eZ, axis=0)  # this gives shape of (1, m)
    A = np.divide(eZ, sum_eZ)    # matrix broadcasting
    return A
    

def softmax_derivative(Z, A='none'):
    """
    Parameters
    ----------
    Z : array of shape (no of nodes or classes, no of samples)
    A: activation on Z, same shape as Z
     
    Returns
    -------
    derivative: compute derivative element-vise to Z
    """
    #maths behind softmax derivative: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    if A='none':
        derivative = softmax(Z) * (1 - softmax(Z))
    else:
        derivative = A * (1 - A)
    return derivative


def relu():
    pass

def relu_derivative():
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
    if scale =='Xavier': #Xavier initialization
        scale = (1/n_prev)**0.5
    else if scale =="He": #He initialization
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
    Z : array-like shape of (no of nodes, no of samples), pre-activation
    activation: type of activation

    Returns
    -------
    A: activation values corresponding to each element of Z

    """
    if activation == 'sigmoid':
        A = sigmoid(Z)
    if activation == 'relu':
        A = relu(Z)
    if activation == 'softmax':
        A = softmax(Z)
    
    return A
    
    
def backprop_activate(dA, Z, activation, A):
    """

    Parameters
    ----------
    dA: gradient of A, same shape as A
    Z : array-like shape of (no of nodes, no of samples), pre-activation
    activation: type of activation
    A: activation values corresponding to each element of Z, same shape as Z

    Returns
    -------
    dZ: gradient of Z, same shape as Z

    """
    if activation == "sigmoid":
        derivative = sigmoid_derivative(Z, A)
    if activation == "softmax":
        derivative = softmax_derivative(Z, A)
    if activation == "relu":
        derivative = relu_derivative(Z, A) 
    dZ = np.multiply(dA, derivative)
    
    return dZ
    

def backprop_linear(dZ, W, A_prev, regularization_lambda):
    """

    Parameters
    ----------
    dZ : gradient of Z, shape same as Z
    W : weight of current layer, shape (size of current layer, size of previous layer)
    A_prev : gradient of A of previous layer
    regularization_lambda : lamda of L2 regularization

    Returns
    -------
    dA_prev : gradient of A of previous layer
    dW : gradient of W
    db : gradient of b

    """
    #formula reference/cross check:  Summary of gradient descent, Andrew Ng
    dA_prev = np.matmul(np.transpose(W), dZ)
    dW = np.matmul(dZ, np.transpose(A_prev))/m + regularization_lambda * W/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    return dA_prev, dW, db


def update_parameter(W, b, dW, db, learning_rate):
    """
    

    Parameters
    ----------
    W : Weight, shape of (size of current layer, size of previous layer)
    b : bias, shape of (size of current layer, 1)
    dW : gradient of W, same shape as W
    db : gradient of b, same shape of b
    learning_rate : learning rate applied to gradient for updating parameter

    Returns
    -------
    W : Weight, shape of (size of current layer, size of previous layer)
    b : bias, shape of (size of current layer, 1)

    """
    W = W - dW * learning_rate
    b = b - dB * learning_rate
    return W, b
  


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
    J : cost reflecting the prediction error

    """

    r = 0.5 * regularization_lambda * np.sum(W**2)/m
    J = - np.sum(np.multiply(y_true, log(prob_predict)))/m + r
    
    return J
    
    
    
    

def mask(shape, dropout):
    pass

    

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
        
        #create empty dictionary for corresponding gradient
        self.dW={}
        self.db={}
        self.dZ={}
        self.dA={}
        

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
        X_train : data input of shape (no of samples, no of features)
            Note: transpose to shape (no of features, no of samples) as self.X_train for internal training
        y_train : target value of shape (no of samples, no of classes)
            Note: transpose to shape (no of classes, no of samples) as self.y_train for internal training

        Returns
        -------
        None.

        """
        self.X_train = np.transpose(X_train)
        self.y_train = np.transpose(y_train)
        self.J =[]
        self.m = len(self.y_train)  #get no of samples
        self.m_mini = batch_size    #Alex to work on mini batch or other optimizers
        
        #initialize weight and bias
        np.random.seed(self.seed)
        for l in range(1, self.layer):
            self.W[l], self.b[l] = init_parameter(n_current=self.N[l], n_prev=self.N[l-1], scale=self.weight_scale)
        
        #initialize A[0]
        self.A[0]=self.X_train

        #training begins
        
        for e in range(self.epochs):

            # forward propagation
            for l in range(1, self.layer):
                self.Z[l] = forward_linear(self.W[l], self.A[l-1], self.b[l])
                self.A[l] = forward_activate(self.Z[l], self.activation[l])
        
            #compute the cost and store in self.J
            self.J.append(cost_function(self.A[self.layer], self.y_train, self.m, self.regularization_lambda))
        
            #backpropagation
            self.dA[self.layer] = -np.divide(self.y_train, self.A[self.layer])
            for l in reversed(range(1, self.layer+1)):
                self.dZ[l] = backprop_activate(self.dA[l], self.Z[l], self.activation[l], self.A[l])
                self.dA[l-1], self.dW[l], self.db[l] = backprop_linear(self.dZ[l], self.W[l], self.A[l-1], self.regularization_lambda)
                self.W[l], self.b[l] = update_parameter(self.W[l], self.b[l], self.dW[l], self.db[l], self.learning_rate)

        
        
    def predict(self, X_enquiry):
        """
        Predict the target values

        Parameters
        ----------
        X_enquiry : data input of shape (no of samples, no of features)
            Note: transpose to shape (no of features, no of samples) for internal calculation
       
        Returns
        -------
        y_predict: array of shape (no of samples, no of classes)

        """
        #initialize "activation" of layer 0
        self.A[0]=np.transpose(X_enquiry)
        # forward propagation
        for l in range(1, self.layer):
            self.Z[l] = forward_linear(self.W[l], self.A[l-1], self.b[l])
            self.A[l] = forward_activate(self.Z[l], self.activation[l])
        
        a = np.transpose(self.A[self.layer]) #transpose predicted probability
        y_predict = np.zeros_like(a)
        y_predict[np.arange(len(a)), a.argmax(1)] = 1
        
        return y_predict
        
    
    def evaluate(self, X_test, y_test):
        pass
    
        
        
        
        
                
    