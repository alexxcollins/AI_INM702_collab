#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 11:40:43 2021
@author: alexxcollins
I can't work out what's going wrong with our NN.py. This is an attempt to
create a really simple NN from scratch to see if it's working
"""
import numpy as np
import matplotlib.pyplot as plt


class NN:
    def __init__(
        self, learning_rate=0.01, regularization=None, reg_alpha=None, random_seed=42
    ):
        """Intialiase NN class. Set learning rate.
        regularization is None or string. Currenlty "L1" and "L2" allowed
        reg_alpha is float in (0,1)
        random_seed sets random seed for object methods.
        Example of setting up a neural network:
        To run a neural net first intialise
               model = NN(learning_rate=0.2, random_seed=42)
        Then add layers to the model. For categorisatoin, last layer will need
        nodes = number of classes, and activation=NN.softmax.
        See .add() method for more details.
               model.add(nodes = 512, activation=NN.ReLu)
               model.add(nodes = 10, activation=NN.softmax)
        Then initialise the model with training and optional test data:
               model.model(X_train, y_train, X_test, y_test, minibatch_size=32)
        If verbose=True (the default value) then .model() will print a summary
        of the model with output for each layer of nodes, size of weights and
        biases parameters etc.
            number of samples = 60000
            training data has 784 features
            learning rate = 0.25
            regularization is None with parameter None
            minibatch size is 32
            hidden layer 0:
            nodes in previous layer: 784
            nodes in this layer: 1024
            weight shape: (784, 1024)
            bias shape: (1, 1024)
            activation function is <function NN.ReLu at 0x12ff843a0>
            hidden layer 1:
            nodes in previous layer: 1024
            nodes in this layer: 10
            weight shape: (1024, 10)
            bias shape: (1, 10)
            activation function is <function NN.softmax at 0x12ff844c0>
        Then fit the model. Set the maximum number of epochs, stopping criteria
        and whether you want verbose output or not.
              model.fit(epochs=15, min_epochs=2, patience=2,
                        stopping_metric="valid", verbose=True)
        To plot history of loss per minibatch, run model.plot_error()
        """
        self.lr = learning_rate
        self.regularization = regularization
        self.reg_alpha = reg_alpha
        self.layers = 0
        self.activations = []
        self.nodes = [None]  # List of node sizes. First entry will hold number
        # of features when fit called
        self.rng = np.random.default_rng(random_seed)

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
        Add dense layer with specified number of nodes.
        If adding the final layer, need to use activation = NN.softmax and
        nodes equal to number of classes to classify.
        Parameters
        ----------
        nodes : integer
            Number of nodes
        activation: function, user options - NN.ReLu, NN.sigmoid, NN.softmax
            default value is self.ReLu
        Returns
        -------
        None. Updates NN attributes
        """
        self.layers += 1
        self.nodes.append(nodes)
        self.activations.append(activation)

    def model(
        self,
        X_train,
        y_train,
        X_test=None,
        y_test=None,
        minibatch_size=None,
        scale="He",
        verbose=True,
    ):
        """
        After the neural net has been initialised and layers added, model()
        initialises the model parameters, sets meta data for training and
        optionally prints out model architecture.
        Parameters
        ----------
        X_train : np array, shape (number samples, number features)
            training data
        y_train : np array, shape (number of samples, number of classes)
            one-hot encoded labels for training data
        X_test : np array, shape (number samples, number features)
            test data
        y_test : np array, shape (number of samples, number of classes)
            one-hot encoded labels for test data
        minibatch_size : integer or None, optional
            If None then batch-SGD is done (i.e. all samples used for each
            epoch). If set to one then SGD used with weights reset after cost
            and gradients calculated for each sample.)
            Minibatches are randomly sampled at each epoch.
        scale : string. Either 'He' or 'Xavier'.
            Method to initialize weights. He is better for ReLu and 'Xavier'
            for Sigmoid.
        verbose : Bool, optional
            If True then prints out model arhitecture. The default is True.
        Returns
        -------
        None. Prints out model architecture if verbose=True
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.no_samples = y_train.shape[0]
        self.nodes[0] = X_train.shape[1]
        if minibatch_size is None:
            self.minibatch_size = self.no_samples
            self.num_batches = 1
        else:
            self.minibatch_size = minibatch_size
            # if minibatch size doesn't divide sample size perfectly then
            self.num_batches = np.ceil(self.no_samples / self.minibatch_size).astype(
                int
            )

        self.init_params(scale=scale)
        if verbose:
            print("number of samples = {}".format(self.no_samples))
            print("training data has {} features".format(self.nodes[0]))
            print("learning rate = {}".format(self.lr))
            print(
                "regularization is {} with parameter {}".format(
                    self.regularization, self.reg_alpha
                )
            )
            if minibatch_size is not None:
                print("minibatch size is {}".format(self.minibatch_size))
            for i in range(self.layers):
                print("\nhidden layer {}:".format(i))
                print("nodes in previous layer: {}".format(self.nodes[i]))
                print("nodes in this layer: {}".format(self.nodes[i + 1]))
                print("weight shape: {}".format(self.W[i].shape))
                print("bias shape: {}".format(self.b[i].shape))
                print("activation function is {}".format(self.activations[i]))

    def fit(
        self, epochs=10, min_epochs=5, patience=1, stopping_metric="valid", verbose=True
    ):
        """
        Fit the model.
        Parameters
        ----------
        epochs : int, optional
            Max number of epochs to run. If early stopping criteria is used
            method may not reach max epochs. The default is 10.
        min_epochs : integer, optional
            Minimum number of epochs to run. Will not stop early before this.
            The default is 5.
        patience : integer, optional
            Number of epochs to wait before seeing an improvement. Early sopping
            will compare loss at current epoch with loss at patience epochs before.
            The default is 1.
        stopping_metric : string, either "valid" or "train", optional
            whether to use training or test loss as stopping criteria.
            The default is "valid".
        verbose : boolean, optional
            If True then print out metrics for each epoch. The default is True.
        Returns
        -------
        None.
        """
        self.epochs = epochs
        self.min_epochs = min_epochs
        self.L = []  # empty list to store loss function output for within epoch
        self.Lreg = (
            []
        )  # empty list to store loss function with regularization term within epoch

        # create an empty array to store losses. Each row will hold losses for one
        # epoch. Initialise "empty" array of appropriate size and append rows for each epoch
        self.L_ar = np.array([]).reshape(0, self.num_batches)
        self.Lreg_ar = np.array([]).reshape(0, self.num_batches)
        # create empty array for keeping track of training loss after each epoch
        # self.L_train = np.array([]).reshape(0, 1)
        # create empty array for keeping track of validation loss after each epoch
        self.L_valid = np.array([]).reshape(0, 1)
        # set self.y for softmax activation function to take as input
        self.y = self.y_train

        # loop through all epochs (unless stopping criteria are met)
        for e in range(epochs):
            # shuffle the data to ensure different minibatches are returned
            # for each epoch.
            if self.num_batches > 1:
                mask = self.rng.permutation(self.no_samples)
                X_epoch = self.X_train[mask]
                y_epoch = self.y_train[mask]
            else:
                X_epoch = self.X_train
                y_epoch = self.y_train

            # loop over batches:
            for i in range(self.num_batches):
                X = X_epoch[i * self.minibatch_size : (i + 1) * self.minibatch_size]
                y = y_epoch[i * self.minibatch_size : (i + 1) * self.minibatch_size]
                m = y.shape[0]  # this could be less than
                # self.minibatch_size for last iteration

                # forward propagation
                self.fwd_pass(X=X)

                # store loss function output
                xe_loss, xe_loss_reg = self.xe_loss(X=self.A[self.layers], y=y, m=m)
                # Loss without regularisation term - so models with different regularization
                # params can be compared
                self.L.append(xe_loss)
                # Loss with regularization term
                self.Lreg.append(xe_loss_reg)

                # back propogation.
                for l in reversed(range(self.layers)):
                    self.dZ[l] = self.de_Z(l, y)
                    self.dW[l] = self.de_W(l, m)
                    self.db[l] = self.de_b(l, m)
                    self.dA[l] = self.de_A(l)

                # update weights and biases
                self.update_params()

            # append losses to Loss array. Array has one row per epoch, and row
            # contains losses for all minibatches
            self.L_ar = np.concatenate((self.L_ar, np.array(self.L).reshape(1, -1)))
            self.Lreg_ar = np.concatenate(
                (self.Lreg_ar, np.array(self.Lreg).reshape(1, -1))
            )
            self.L = []
            self.Lreg = []

            # prepare to print training information
            train_pred = self.predict(self.X_train)
            train_accuracy = self.accuracy(train_pred, self.y_train)

            if self.X_test is not None:
                test_pred = self.predict(self.X_test)
                test_accuracy = self.accuracy(test_pred, self.y_test)
                # update validation loss for epoch
                A, _ = self.fwd_pass(self.X_test, train=False)
                test_loss, _ = self.xe_loss(
                    X=A[self.layers], y=self.y_test, m=self.y_test.shape[0]
                )
                self.L_valid = np.concatenate(
                    (self.L_valid, np.array(test_loss).reshape(1, 1))
                )
            else:
                test_accuracy = 0.0
                test_loss = 0.0

            if verbose:
                # calculate mean loss across all minibatches in epoch.
                avg_loss = self.L_ar[-1, :].mean()
                print(
                    "epoch {} average loss: {:.3f}; train accy: {:.3f}; test accy: {:.3f}; test loss: {:.3f}".format(
                        e, avg_loss, train_accuracy, test_accuracy, self.L_valid[e, 0]
                    )
                )

            # early stopping test
            if self.stop(
                e, self.L_ar, self.min_epochs, patience, metric=stopping_metric
            ):
                print("average loss improvement is small: early stop")
                break

    def fwd_pass(self, X, train=True):
        """
        Run forward pass. Either changes attributes for all layers: Z and A
        or if running on test data it just returns the final Z and A layers.
        For final layers, A is an array of size (batch samples, numnber of
        classes) with probabilities summing to 1 across eac row.
        Parameters
        ----------
        X : np array of shape (batch samples, number of features)
            Input samples
        train : Boolean, optional
            Function runds differently depending on whether it is called in
            training or testing. If training, then all Z and A matirces in
            network are updated. The default is True.
        Returns
        -------
        If train is False:
            A : np array. Shape (number of samples in batch, 10)
            Z : np array. Shape (number of samples in batch, 1)
        Otherwise returns nothing.
        """
        # if fwd_pass done in training mode, update attributes corresponding
        # to the Zs and As in layers.
        # note that Z and A arrays are only stored for each epoch - so the same
        # arrays will get over-written by each mini-batch loop. This would be
        # somthing to fix in a future iteration of the code, but it works.
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
        """
        Calculate accuracy metric. [num correct predictions] / [sample size]
        Parameters
        ----------
        y_pred : array. Size (number of samples, 1) or
                        (number of samples, number of categories)
            array of predictions. Either one-hot encoded, or an array of
            intergers corresponding to categories
        y_true : array. Same size as y_pred.
            array of actual labels. Either one-hot encoded or array of
            integers corresponding to categories
        one_hot : Boolean optional
            if y_pred and y_true are one-hot encoded then this should be set
            to True; otherwise False. The default is True.
        Returns
        -------
        float - [num correct predictions] / [sample size]
        """
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
        Very naive setting of weight magnitudes to start with.
        Parameters
        ----------
        scale : string. Either 'He' or 'Xavier'. He is better for ReLu and
                'Xavier' for Sigmoid.
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
        """
        Update parameters after gradients have been calculted in back propogation
        stage.
        Returns
        -------
        None. Updates attributes containing weights. Uses gradients calculated
        by back propogation.
        """
        # for NN work with n layers, there are n weights indexed from 0 to n-1
        for i in range(self.layers):
            self.W[i] -= self.dW[i] * self.lr
            self.b[i] -= self.db[i] * self.lr

    def activate(self, activation, **kwargs):
        # this function isn't really needed. Was created initially to wrap more
        # complex functionality. In future iteration we could get rid of this.
        return activation(self, **kwargs)

    def ReLu(self, X, y=None, gradient=False):
        """
        Relu activation
        Returns either forward pass value or gradient
        Parameters
        ----------
        X : numpy array of shape (no of samples in batch, nodes in current layer)
        y : None
            Not great design. Needs a named variable y so that activation
            functions can be called in general way from loop. It works though.
            Could be fixed in future design
        gradient : Boolean
            Set to True if want deritive returned, False if want forward pass
            value returned
        Returns
        -------
        Transformed array, A, of same shape as X
        """
        if gradient:
            return np.where(X > 0, 1, 0)
        else:
            return np.where(X > 0, X, 0)

    def sigmoid(self, X, y=None, gradient=False):
        """
        Sigmoid activation
        Returns either forward pass value or gradient
        Parameters
        ----------
        X : numpy array of shape (no of samples in batch, nodes in current layer)
        y : None
            Not great design. Needs a named variable y so that activation
            functions can be called in general way from loop. It works though.
            Could be fixed in future design
        gradient : Boolean
            Set to True if want deritive returned, False if want forward pass
            value returned
        Returns
        -------
        Transformed array, A, of same shape as X
        """
        if gradient:
            return X * (1 - X)
        else:
            return 1 / (1 + np.exp(-X))
            
        

    def softmax(self, X, y=None, gradient=False):
        """
        softmax activation
        Parameters
        ----------
        X : numpy array of shape (no of samples in batch, nodes in current layer)
        y : numpy array of shape (no of samples in batch, no of categories)
        gradient : Boolean
            Set to True if want deritive returned, False if want forward pass
            value returned. The derivative in this case is the derivative of
            the softmax and cross-entropy loss chained together.
        Returns
        -------
        Transformed array, A, of same shape as X
        """
        if gradient:
            return X - y
        else:
            shiftX = X - X.max(axis=1, keepdims=True)
            exps = np.exp(shiftX)
            return exps / np.sum(exps, axis=1, keepdims=True)

    def xe_loss(self, X, y, m):
        """
        Calculate cross entropy loss.
        Parameters
        ----------
        X : array, features
        y : array, one-hot encoded labels
        m : int
            batch size
        If L1 or L2 regularization is used, we return two values: it is helpful
        to keep track of loss without regularization to compare betweed models
        with different regularization or different parameters.
        Returns
        -------
        tuple. (xe-loss without regularization loss, xe-loss with regularization loss)
        The two values will be the same if no regularization is used.
        """
        epsilon = 0.0000000001
        if self.regularization == "L1":
            # add terms for weights in each layer
            loss_reg = 0
            for i in range(self.layers):
                loss_reg += (
                    self.reg_alpha
                    * np.where(self.W[i] > 0, self.W[i], -self.W[i]).sum()
                )
        elif self.regularization == "L2":
            # add terms for weights in each layer
            loss_reg = 0
            for i in range(self.layers):
                loss_reg += self.reg_alpha * 0.5 * (self.W[i] ** 2).sum() / m
        else:
            loss_reg = 0
        xel = -(y * np.log(X + epsilon)).sum() / m
        return xel, xel + loss_reg

    def de_Z(self, l, y):
        """
        This returns dZ/dA in the general case, but for the final layer returns
        dL/dA where L is the loss function
        l is an integer representing layer
        y is one-hot encoded numpy array of size (number of samples, number of categories)
        """
        if l == self.layers - 1:
            return self.softmax(X=self.A[self.layers], y=y, gradient=True)
        else:
            dZ = (
                self.activate(self.activations[l], X=self.Z[l], gradient=True)
                * self.dA[l + 1]
            )
            return dZ

    def de_W(self, l, m):
        """Returns dW_l/dZ_l"""
        if self.regularization == "L1":
            reg = self.reg_alpha * np.where(self.W[l] > 0, 1, -1)
        elif self.regularization == "L2":
            reg = self.reg_alpha * self.W[l]
        else:
            reg = 0
        return (self.A[l].T @ self.dZ[l]) / m + reg

    def de_b(self, l, m):
        """Returns db_l/dZ_l"""

        return self.dZ[l].sum(axis=0, keepdims=True) / m

    def de_A(self, l):
        """Returns dA_l/dZ_l"""
        return self.dZ[l] @ self.W[l].T

    def stop(self, epoch, L_ar, n=5, patience=1, metric="valid"):
        """
        Decide whether to stop. Will stop if at least n epochs have been run
        epoch : int
            the number epoch we have just finished (0-indexed)
        L_ar : np.array
            the self.L_ar array: array of losses for each minibatch and epoch
        n : int
            minimum number of epochs to run for
        patience : int
            number of epochs to wait before seeing an improvement
        metric: str
            whether to stop based on test or train loss
        """
        n = max(n, 1 + patience)
        if epoch >= (n - 1):
            if metric == "train":
                loss = self.L_ar
            elif metric == "valid":
                loss = self.L_valid

            if loss[epoch].mean() > loss[epoch - patience].mean() * 0.975:
                return True
            else:
                return False
        else:
            return False

    def plot_error(self, miss_first=0):
        """
        Generate plot of errors.
        miss_first : int, optional
            The number of minibatches (or epochs if batch size = sample size)
            to avoid in the chart. Sometimes the first few runs produce large
            errors and we would rather not plot them. The default is 0.
        """
        # matplotlib gives warning when setting axis formatting. Repress this warning.
        import warnings

        warnings.filterwarnings("ignore")

        ax = plt.axes()
        ax.plot(self.L_ar.flatten()[miss_first:], "b.", alpha=0.1)
        ax.grid()
        ax.set_xticklabels(ax.get_xticks(), rotation=45)
        ax.xaxis.set_major_locator(plt.MultipleLocator(self.num_batches))
        ax.xaxis.set_major_formatter("epoch {pos}")

