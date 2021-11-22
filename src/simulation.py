#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 12:51:11 2021

@author: suenchihang
"""
from regression import GenerateData, UniformX
import numpy as np
from numpy.random import default_rng
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
        
#given object of sub-class of GenerateData and frequency of simulation, output mean and variance of coefficients and estimates
#*arg, **kwarg - same input as those in the corresponding sub-class of GenerateData
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

#allow input with one change factor to vary, given a list of changes expressed as dictionary, e.g. factor={"positions":positions}
#*arg, **kwarg - same input as those in the corresponding sub-class of GenerateData, except the factor to be varied
def change_factor(object_GenerateData, frequency, factor, *arg, **kwarg):
    test = object_GenerateData
    d={}
    for key in ("score_mean", "score_variance", "ssr_mean", "ssr_variance"):
        d[key]=[]
    for l in range(len(test.beta)):
        d["b"+str(l)+"_mean"]=[]
        d["b"+str(l)+"_variance"]=[]
    for key_factor, change_value in factor.items():
        for change in change_value:
            kw = {key_factor: change}
            result=simulation(test,frequency, *arg, **kw, **kwarg)
            for key in ("score_mean", "score_variance", "ssr_mean", "ssr_variance"):
                d[key].append(result[key])
            d["b0_mean"].append(*result["beta_mean"][0])
            d["b0_variance"].append(*result["beta_variance"][0])
            for l in range(len(test.beta)-1):
                d["b"+str(l+1)+"_mean"].append(result["beta_mean"][1][l])
                d["b"+str(l+1)+"_variance"].append(result["beta_variance"][1][l])
    return d
