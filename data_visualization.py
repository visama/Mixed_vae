# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 23:15:13 2021

@author: visam
"""
import seaborn as sns
import numpy as np
import pandas as pd

#Feed data as dictionaries, variables of interest name and data_dictionary  
def plot_marginal_marginal_distr_for_realValued_variables(X, col, model):
    var_synth =  model.synthetic_data(X[col].shape[0])[col]
    var_real = X[col]

    array_plot_s = np.concatenate((var_synth, np.array(['synth']*len(var_synth)).reshape(len(var_synth),1)), axis = 1)
    array_plot_r = np.concatenate((var_real, np.array(['real']*len(var_real)).reshape(len(var_real),1)), axis = 1)

    array_plot = np.concatenate((array_plot_r, array_plot_s), axis = 0)
    x_plot = pd.DataFrame(array_plot, columns = ['value', 'label'])
    
    x_plot['value'] = pd.to_numeric(x_plot['value'],errors='coerce')
    #ax=
    return(x_plot)
    

#input on nyt int ja output float. Tämä toimii siitä huolimatta, mutta olisi muokattava kuntoon
def plot_histogram(X, model, col, data_dict, numeric = False):
    X_synth = model.synthetic_data(X[col].shape[0])
    if numeric == True:
        var_real = X[col].astype(int)
        var_synth = X_synth[col].astype(int)
    else:
        var_synth = data_dict[col].inverse_transform(X_synth[col])
        var_real = data_dict[col].inverse_transform(X[col])

    array_plot_s = np.concatenate((var_synth, np.array(['synth']*len(var_synth)).reshape(len(var_synth),1)), axis = 1)
    array_plot_r = np.concatenate((var_real, np.array(['real']*len(var_real)).reshape(len(var_real),1)), axis = 1)

    array_plot = np.concatenate((array_plot_r, array_plot_s), axis = 0)
    x_plot = pd.DataFrame(array_plot, columns = ['value', 'label'])
    
    if numeric == True:
        x_plot['value'] = x_plot['value'].astype(int)
        x_plot = x_plot.sort_values('value', axis=0, ascending=True)
    
    return(x_plot)
    
    
#plot_histogram(X_dict, model, 'readmitted', data_dict, numeric = False)

#Nimeä uudestaan. Histogrammi int-muuttujille. Kun int muuttuja mallinnetaan cat-jakaumalla käsiteltävä erikseen.TEE!!!!!!!!!!!!!!!!!!
def plot_histogram_Int(X, model, col, data_dict, numeric = False):
    X_synth = model.synthetic_data(X[col].shape[0])
    if numeric == True:
        var_real = X[col].astype(int)
        var_synth = X_synth[col].astype(int)
    else:
        var_synth = data_dict[col].inverse_transform(X_synth[col]).astype(int)
        var_real = X[col].astype(int)

    array_plot_s = np.concatenate((var_synth, np.array(['synth']*len(var_synth)).reshape(len(var_synth),1)), axis = 1)
    array_plot_r = np.concatenate((np.array(var_real).reshape(len(var_real),1), np.array(['real']*len(var_real)).reshape(len(var_real),1)), axis = 1)

    array_plot = np.concatenate((array_plot_r, array_plot_s), axis = 0)
    x_plot = pd.DataFrame(array_plot, columns = ['value', 'label'])
    
    if numeric == True:
        x_plot['value'] = x_plot['value'].astype(int)
        x_plot = x_plot.sort_values('value', axis=0, ascending=True)
    
    ax=sns.histplot(data=x_plot,x="value",hue="label",palette="deep", stat = 'probability', common_norm=False)
    
    
#plot_histogram_Int(X_dict, model, 'num_procedures', data_dict, numeric = False)

#Data one hot muodossa. Tarvitaan muita keinoja, jos mukana muuttujia, joilla paljon realisaatioita
#Voisi vain suodattaa arvot väliltä (h,h), -1<h<1 pois
def plot_correlation_matrices(X, model):
    x_1hot = X
    xs_1hot = dict_to_array(model.synthetic_data(X.shape[0]))

    corr_matrix = pd.DataFrame(x_1hot).corr()
    corr_matrix_s = pd.DataFrame(xs_1hot).corr()

    fig, axs = plt.subplots(ncols=2)
    sns.heatmap(corr_matrix, ax=axs[0])
    sns.heatmap(corr_matrix_s, ax=axs[1])

#plot_correlation_matrices(X_input, model)



