#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 11:40:43 2021

@author: alexxcollins

I can't work out what's going wrong with our NN.py. This is an attempt to
create a really simple NN from scratch to see if it's working
"""
import numpy as np


class NN:
    def __init__(self, learning_rate=0.001, random_seed=42):
        self.lr = learning_rate
        self.layers = 0
        self.activations = []
        self.nodes = [None]  # List of node sizes. First entry will hold number
        # of features when fit called
        self.rng = np.random.default_rng(random_seed)
        self.L = []  # empty list to store loss function output for epochs

        # create empty dictionary for W, b, Z, A
        self.W = {}  # weight of nodes
        self.b = {}  # bias
        self.Z = {}  # pre-activation values, linear combination Z = WX + b
        self.A = {}  # activation values

        # create empty dictionary for corresponding gradient
        self.dW = {}
        self.db = {}
        self.dZ = {}
        self.dA = {}

    def add(self, nodes, activation):
        """
        Add dense layer with specified number of nodes

        Parameters
        ----------
        nodes : integer
            Number of nodes
        activation: function
            default value is self.ReLu

        Returns
        -------
        None. Updates NN attributes

        """
        self.layers += 1
        self.nodes.append(nodes)
        self.activations.append(activation)

    def model(self, X_train, y_train, X_test=None, y_test=None, verbose=True):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.no_samples = y_train.shape[0]
        self.nodes[0] = X_train.shape[1]
        self.batch_size = self.no_samples  # not actively setting this yet - maybe
        # best to do somewhere else
        self.init_params()
        if verbose:
            print("number of samples = {}".format(self.no_samples))
            print("training data has {} features".format(self.nodes[0]))
            print("learning rate = {}".format(self.lr))
            for i in range(self.layers):
                print("\nhidden layer {}:".format(i))
                print("nodes in previous layer: {}".format(self.nodes[i]))
                print("nodes in this layer: {}".format(self.nodes[i + 1]))
                print("weight shape: {}".format(self.W[i].shape))
                print("bias shape: {}".format(self.b[i].shape))
                print("activation function is {}".format(self.activations[i]))

    def fit(self, epochs=10, verbose=True):
        self.epochs = epochs
        # set self.y for softmax activation function to take as input
        self.y = self.y_train
        for e in range(epochs):
            # forward propagation
            self.fwd_pass(X=self.X_train)

            # store loss function output
            self.L.append(self.xe_loss(X=self.A[self.layers], y=self.y))

            # back propogation. Calculate last layer first and then iterate
            l = self.layers - 1
            self.dZ[l] = self.softmax(X=self.A[self.layers], gradient=True)
            self.dW[l] = self.A[l].T @ self.dZ[l] / self.batch_size
            self.db[l] = self.dZ[l].sum(axis=0, keepdims=True) / self.batch_size
            self.dA[l] = self.dZ[l] @ self.W[l].T
            for l in reversed(range(self.layers - 1)):
                self.dZ[l] = (
                    self.activate(self.activations[l], X=self.Z[l], gradient=True)
                    * self.dA[l + 1]
                )
                self.dW[l] = (self.A[l].T @ self.dZ[l]) / self.batch_size
                self.db[l] = self.dZ[l].sum(axis=0) / self.batch_size
                self.dA[l] = self.dZ[l] @ self.W[l].T

            # update weights and biases
            self.update_params()

            # print training information
            train_pred = self.predict(self.X_train)
            train_accuracy = self.accuracy(train_pred, self.y_train)

            if self.X_test is not None:
                test_pred = self.predict(self.X_test)
                test_accuracy = self.accuracy(test_pred, self.y_test)

            if verbose:
                print(
                    "epoch {} loss: {:.3f}; train accy: {:.3f}; test accy: {:.3f}".format(
                        e, self.L[e], train_accuracy, test_accuracy
                    )
                )

    def fwd_pass(self, X, train=True):

        if train:
            self.A[0] = X
            for l in range(self.layers):
                self.Z[l] = self.A[l] @ self.W[l] + self.b[l]
                self.A[l + 1] = self.activate(
                    self.activations[l], X=self.Z[l], gradient=False
                )
        else:
            Z = {}
            A = {}
            A[0] = X
            for l in range(self.layers):
                Z[l] = A[l] @ self.W[l] + self.b[l]
                A[l + 1] = self.activate(self.activations[l], X=Z[l], gradient=False)
            return A, Z

    def predict(self, X):
        """calculates predicted y from X"""
        A, _ = self.fwd_pass(X, train=False)
        return np.argmax(A[self.layers], axis=1)

    def accuracy(self, y_pred, y_true, one_hot=True):
        if one_hot:
            y_true = np.argmax(y_true, axis=1)
        return np.where(y_true == y_pred, True, False).sum() / y_true.size

    def init_params(self, scale="He"):
        """
        Take first layer as example. Input is n samples with p features.
        X array is shape (n, p)
        Create weight matrix of shape (p, n_1) where n_1 are the nodes in the
        first hidden layer.
        We want to ensure that at each stage, the array X@W + b is shape
        (no of samples, no of nodes)

        Very naive setting of weight magnitudes to start with

        Returns
        -------
        Weight and bias matrices

        """
        for i in range(self.layers):
            if scale == "Xavier":  # Xavier initialization used for Sigmoid?
                scale = (1 / self.nodes[i]) ** 0.5
            if scale == "He":  # He initialization used for ReLu?
                scale = (2 / self.nodes[i]) ** 0.5
            self.W[i] = self.rng.normal(size=(self.nodes[i], self.nodes[i + 1])) * scale
            # for ReLu need to initiate with small positive bias
            self.b[i] = np.ones(shape=(1, self.nodes[i + 1])) * scale

    def update_params(self):
        # for NN work with n layers, there are n weights indexed from 0 to n-1
        for i in range(self.layers):
            temp = self.W[i][0, 0]
            self.W[i] -= self.dW[i] * self.lr
            self.b[i] -= self.db[i] * self.lr

    def activate(self, activation, **kwargs):
        return activation(self, **kwargs)

    def ReLu(self, X, gradient=False):
        """
        Relu activation
        Returns either forward pass value or gradient

        Parameters
        ----------
        X : numpy array of shape (nodes in current layer, 1)

        Returns
        -------
        Transformed array, A, of same shape as Z

        """
        if gradient:
            return np.where(X > 0, 1, 0)
        else:
            return np.where(X > 0, X, 0)

    def sigmoid(self, X, gradient=False):
        """
        Sigmoid activation
        Returns either forward pass value or gradient

        Parameters
        ----------
        X : numpy array of shape (nodes in current layer, 1)

        Returns
        -------
        Transformed array, A, of same shape as Z

        """
        if gradient:
            return 1 / (1 + np.exp(-X))
        else:
            return X * (1 - X)

    def softmax(self, X, gradient=False):
        """
        softmax activation

        Parameters
        ----------
        X : numpy array of shape (nodes in current layer, 1)

        Returns
        -------
        Transformed array, A, of same shape as X

        """
        if gradient:
            return X - self.y
        else:
            shiftX = X - X.max(axis=1, keepdims=True)
            exps = np.exp(shiftX)
            return exps / np.sum(exps, axis=1, keepdims=True)

    def xe_loss(self, X, y):
        """
        Calculate cross entropy loss.
        Normalises by batch size.
        """
        epsilon = 0.0000000001
        return -(y * np.log(X + epsilon)).sum() / self.batch_size
