# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 20:01:48 2021

@author: visam
"""
#https://en.wikipedia.org/wiki/Bhattacharyya_distance
import numpy as np
import pandas as pd

def probabilities_for_bin(x_input, nro_bins = 50, bins = ""):
    n = x_input.shape[0]
    x = np.array(x_input).reshape(n,)

    if bins == "":
        left_edge = np.min(x)
        right_edge = np.max(x)
        bins = np.linspace(left_edge, right_edge, nro_bins).reshape(nro_bins,)
    d = pd.cut(x, bins, include_lowest=True).describe() 
    
    d = d.reset_index()
    d = d.rename(columns={"levels": "interval", "counts": "numbers", "counts": "numbers"})
    return(d, bins)

#Continuous variable
def Bhattacharyya_distance(bins_probs_P,bins_probs_Q):
    #merge
    bins_probs = bins_probs_P.merge(bins_probs_Q, left_on='categories', right_on='categories')

    #Form new variable
    bins_probs = bins_probs.assign(sqrt_P_times_Q=np.sqrt(bins_probs['freqs_x']*bins_probs['freqs_y']))
    
    #arvot välillä 0-inf. inf jos ei ollenkaan overlappia
    res = -1*np.log(float(bins_probs[['sqrt_P_times_Q']].sum()))
    return(res)

#Categorical variable
def Bhattacharyya_distance_cat(x_dict, model, col):
    probs_P = x_dict[col]
    n = probs_P.shape[0]
    probs_Q = model.synthetic_data(n)[col]
    
    #
    res = -1*np.log(np.sum(np.sqrt(np.mean(probs_P, axis = 0)*np.mean(probs_Q, axis = 0))))

    return(res)

#B-distance for each variable of the data set
def Bhattacharyya_distance_total(X, X_dict, model, variable_types):
    n = X.shape[0]
    res = []
    for i, col in enumerate(model.columns):
        if variable_types[i] != 'cat':
            bins_probs_P, bins = probabilities_for_bin(X[[col]])
            bins_probs_Q, bins = probabilities_for_bin(model.synthetic_data(n)[col],bins=bins)
            B_dist = Bhattacharyya_distance(bins_probs_P,bins_probs_Q)
        else:
            B_dist = Bhattacharyya_distance_cat(X_dict, model, col)
        res.append(B_dist)
    return(np.sum(res))
