# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 12:43:24 2021

@author: alexxcollins

different types of colinearity to investigate:
    (1) pairwise colinearity between X1 and X2
    (2) where p' > 2, "colinearity" between intercept and X1
    (3) Xi which is linear combination of other X
"""

from regression import GenerateData
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression


class ColinearX(GenerateData):
    """docstring.

    some more docstring.
    """

    def generate_X(self, co_type="pairwise correlation", **kwargs):
        """
        Generate X.

        Different behaviour depending on which 'co_type' -
        correlation type is chosen.

        Parameters
        ----------
        co_type : string, optional
            'pairwise correlation' creates a multivariate distribution with
            X_1 and X_2 correlated with covariance cov. Use can set the means
            of the X's and also the covariance of X_1, X_2. For example:
                generate_X(co_type='pairwise correlation', mean=(1,1,1),
                           cov = 0.5)

        Function sets self.X, self.X0, self.X1, self.mean, self.X1X2_cov
        """
        if co_type == "pairwise correlation":
            self.pairwiseX(**kwargs)

    def pairwiseX(self, mean=None, cov=0.3):
        """
        Create a multivariate distribution with X_1 and X_2 correlated.

        Parameters
        ----------
        mean : array-like, must have size equal to self._p - dimensions of X
        cov : float, optional
            The covariance of X_1, X_2, The default is 0.3.

        Function sets self.X, self.X0, self.X1, self.mean, self.X1X2_cov
        """
        if mean is None:
            mean = np.zeros(self._p)
        else:
            mean = np.array(mean)

        self.mean = mean
        self.X1X2_cov = cov
        cov = np.diag(np.ones(self._p))
        cov[0, 1], cov[1, 0] = self.X1X2_cov, self.X1X2_cov

        self.X = self.rng.multivariate_normal(mean=mean, cov=cov, size=self.N)
        self.X0 = np.ones((self.N, 1))
        self.X1 = np.concatenate([self.X0, self.X], axis=1)

    def to_uniform(self, i_list=[1]):
        """
        Transform members of X from normal to uniform.

        Parameters
        ----------
        i : list or list-like of integers, optional
            The X variables to tranform from normal to uniform.
        """
        for i in i_list:
            self.X1[:, i] = norm.cdf(self.X1[:, i])
            self.X[:, i - 1] = self.X1[:, i]

    def remove_Xi(self, i=1):
        """
        Remove a column from X.

        Usecase is to remove potentially redundantX_i from the model - e.g.
        if there is a colinearity.

        Adjust all relevant attributes of GenerateData object.

        Parameters
        ----------
        i : integer, optional
            The 1-indexed dimension of X to remove. i can be 1, ... p.
            The default is 1.
        """
        self._p -= 1
        self._p1 -= 1
        # create mask to remove desired row/column from beta, X etc
        mask = np.ones(self.beta.size, bool)
        mask[i] = False
        self.beta = self.beta[mask]
        self.X1 = self.X1[:, mask]
        # .X is just .X1 without the first column of ones
        self.X = self.X1[:, 1:]
        # if .train_test_split() has been run then adjust X_train and X_test
        try:
            # self.X is of size one less than self.X1 so needs a smaller mask
            mask = np.ones(self.X_test.shape[1], bool)
            mask[i - 1] = False
            self.X_test = self.X_test[:, mask]
            self.X_train = self.X_train[:, mask]
        except AttributeError:
            pass

            # if model has been fit, wipe the results
            self.y_pred = None
            self.score = None
            self.intercept = None
            self.coef = None
            self.b_pred = None

    def add_linear_combination(self, i_list=[], beta=(), y_beta=1, noise_var=0.5):
        """
        Create a new column of X as a linear combination of existing cols.

        This method should be run after X has been created and before
        .generate_y() method has been run.

        Adjust all relevant attributes of GenerateData object, and set
        parameters for the creation of the new column of X.

        Parameters
        ----------
        i : list of integers
            The 1-indexed dimension of X_i's to combine. i can be 1, ... p.
        beta : array_like of floats
               The beta array (intercept plus coefficients) used to create
               the new feature.
        y_beta : float
                 We need to add a beta coefficient for the new variable.
        noise_var : float
                    The variance of the N(0, var) noise term in the model.
        """
        self._p += 1
        self._p1 += 1
        i_list = [0] + i_list
        beta = np.array(beta)[:, np.newaxis]
        epsilon = self._generate_epsilon(noise_var=noise_var)
        new = np.matmul(self.X1[:, i_list], beta) + epsilon
        self.X = np.concatenate([self.X, new], axis=1)
        self.X1 = np.concatenate([self.X1, new], axis=1)
        # change the beta used to create y - different form beta used to
        # create the new X_i
        self.beta = np.concatenate([self.beta, np.array(y_beta).reshape(1, 1)])

    def variance_inflation_factor(self, X):
        """Calculate variance inflation factors for X."""

        # create attibute to hold inflation factors
        self.var_inf_factor = np.zeros(shape=X.shape[1])
        X = self._normalise(X)
        # loop over rows of X
        for i in range(X.shape[1]):
            Xi = X[:, i]
            x_rest = np.delete(X, i, axis=1)
            reg = LinearRegression()
            reg = reg.fit(x_rest, Xi)
            score = reg.score(x_rest, Xi)
            self.var_inf_factor[i] = 1 / (1 - score)

        return self.var_inf_factor

    def convert_feature_to_residuals(self, X, i):
        """Convert a feature from original values to the residuals of linear
        regression of the other features.

        X: array of features of the model
        i: 0-indexed feature to convert to residuals.
        """

        # select the feature to change
        Xi = X[:, i]
        x_rest = np.delete(X, i, axis=1)

        reg = LinearRegression()
        reg = reg.fit(x_rest, Xi)
        Xi_pred = reg.predict(x_rest)
        residuals = Xi - Xi_pred
        self.X[:, i] = residuals
        self.X1[:, i + 1] = residuals

    def _normalise(self, X):
        """normalises vector array X so that all features (columns) have
        unit length and zero mean.
        """
        X = X - X.mean(axis=0)
        norm = np.linalg.norm(X, axis=0)
        eps = 0.0001
        # if norm < eps:
        #     return X / eps
        return X / norm
