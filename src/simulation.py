#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 12:51:11 2021

@author: suenchihang
"""
from regression import GenerateData, UniformX
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng()


def print_coef(X, y, to_print=True):
    reg = LinearRegression().fit(X, y)
    score=reg.score(X, y)
    coef=reg.coef_
    intercept=reg.intercept_
    ss_residual = np.linalg.norm(y - reg.predict(X)) ** 2
    if to_print:
        print("intercept: "+str(intercept)+"; coef: "+str(coef))
    return     intercept, coef, score, ss_residual
        

def simulation(object_GenerateData, frequency, *arg, **kwarg):
    data = object_GenerateData
    beta=[]
    scores=[]
    ssr=[] #sum of sqaures of residuals
    for i in range (frequency):
        data.generate_dataset(*arg, **kwarg)
        intercept, coef, score, ss_residual = print_coef(data.X, data.y, to_print=False)
        beta.append([intercept, *coef])
        scores.append(score)
        ssr.append(ss_residual)
        
    #adjust for sample variance by frequency/(frequency-1)
    beta_mean = np.average(beta, axis=0) 
    beta_variance = np.var(beta, axis=0)*frequency/(frequency-1)   
    score_mean = np.average(scores)
    score_variance = np.var(scores)*frequency/(frequency-1)
    ssr_mean = np.average(ssr)
    ssr_variance = np.var(ssr)*frequency/(frequency-1)


    sim_result = {
        "beta_mean":beta_mean,
        "beta_variance":beta_variance,
        "score_mean":score_mean,
        "score_variance":score_variance,
        "ssr_mean":ssr_mean,
        "ssr_variance":ssr_variance,
        "beta":beta,
        "score":scores,
        "ssr":ssr,
        }
    return sim_result
        