# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 15:35:11 2021

@author: visam
"""
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

#Includes functions for :
#1) Creating a dictionary with one hot encoder for each categorical variable of the data set
#2) Converting data to one hot form and back to original form
#3) Converting data to dictionary and back to original form

#Input: Input data in original form
#Output: dictionary of OneHotEncoders for each categorical variable
def data_dictionary(data):
    encoders = {}
    for column in data.columns:
        lb = OneHotEncoder(handle_unknown='ignore')
        lb.fit(data[[column]])
        encoders[column] = lb
    return(encoders)


#Inputs: original data, list of OneHotEncoders and probability distributions for each variable
#Outputs:
# X_input : Data in onehot form as a matrix
# X_dict : Dat in onehot form as a dictinary, where keys are variable names
#Realisation_counts : A list containing realisation counts for each categorical variable. For other variable types 1
def get_inputs_outputs(X, data_dict, variables_types):
    X_input = []
    X_dict = {}
    realisation_counts = []

    n = X.shape[0]
    for i, col in enumerate(X.columns):
        if variables_types[i] == 'cat':
            X_input.append(data_dict[col].transform(X[[col]]).toarray())
            X_dict[col] = data_dict[col].transform(X[[col]]).toarray()
            realisation_counts.append(data_dict[col].transform(X[[col]]).toarray().shape[1])
        else:
            X_input.append(np.array(X[col]).reshape(n,1))
            X_dict[col] = np.array(X[col]).reshape(n,1)
            realisation_counts.append(1)
    
    X_input = np.concatenate((X_input), axis = 1)
    
    return(X_input, X_dict, realisation_counts)

#Maps dictionary of one hot data to matrix of one hot data
def dict_to_array(dictionary):
    res_array = []
    for i, col in enumerate(dictionary.keys()):
        res_array.append(dictionary[col])
        
    res_array = np.concatenate((res_array), axis = 1)
    return(res_array)

#Shuffles dictionary of one hot data. Used in a training loop after each epoch
def shuffle_dictionary(dictionary):
    n = list(dictionary.values())[1].shape[0]
    foo = range(n)
    rand_indx = random.sample(foo, n)
    
    res_dict = {}
    for i, col in enumerate(dictionary.keys()):
        res_dict[col] = dictionary[col][rand_indx]
        
    return(res_dict)

#Calculates decoder output size for integer variables
def decoder_int_output_layer_size(variable_types):
    res = []
    for variable_type in variable_types:
        if variable_type == 'int_negBin':
            res.append(2)
        if variable_type == 'int_Poisson':
            res.append(1)
        if variable_type == 'real_Normal':
            res.append(2)
    return(sum(res))

#Converts numeric synthetic data to original form
def synthetic_Data_to_original(Xs, columns, data_dict, variable_types):
    res = []
    for i, col in enumerate(columns):
        if variable_types[i] == "cat":
            res.append(data_dict[col].inverse_transform(Xs[col]))
        else:
            res.append(Xs[col])
    res_array = np.concatenate(res, axis = 1)
    res_pd = pd.DataFrame(res_array)
    res_pd.columns = columns
    return(res_pd)

