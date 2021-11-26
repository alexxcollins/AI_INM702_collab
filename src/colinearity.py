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

    def variance_inflation_factor():
        """Calculate variance inflation factors for X."""
