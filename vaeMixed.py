# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 15:33:59 2021

@author: visam
"""

import scipy
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from tensorflow.keras import Input, layers, Model
import seaborn as sns
import tensorflow_probability as tfp
tfd = tfp.distributions

#Probability distributions available for data types
#Integer variables: Poisson, Negative binomial or Categorical distributions
#Categorical variables: Categorical distribution
#Real valued: Normal distribution

def decoder_outputs(columns, realisation_counts, variables_types, decoder_output_cat, decoder_output_int, decoder_intermediate_layer):
    list_outputs = {}
    j = 0
    k = 0
    for i, col in enumerate(columns):
        d = realisation_counts[i]
        if variables_types[i] == 'cat':
            list_outputs[col] = decoder_output_cat(decoder_intermediate_layer)[:,j:(j+d)]
            j = j + d                
        elif variables_types[i] == 'int_negBin':
            list_outputs[col] = decoder_output_int(decoder_intermediate_layer)[:,k:(k+2)] 
            k = k + 2
        elif variables_types[i] == 'int_Poisson':
            list_outputs[col] = decoder_output_int(decoder_intermediate_layer)[:,k] 
            k = k + 1
        elif variables_types[i] == 'real_Normal':
            list_outputs[col] = decoder_output_int(decoder_intermediate_layer)[:,k:(k+2)] 
            k = k + 2
    return(list_outputs)
                
                
def prob_distribution_integer(distr_name, dec_logits):
    if distr_name == 'int_negBin':
        decoder_distr = tfd.NegativeBinomial(total_count=tf.nn.relu(dec_logits[:,0]) + tf.constant(0.00001), logits = dec_logits[:,1])
    elif distr_name == 'int_Poisson':
        decoder_distr = tfd.Poisson(log_rate=dec_logits)
    elif distr_name == 'real_Normal':
        decoder_distr = tfd.Normal(loc=dec_logits[:,0], scale = tf.nn.relu(dec_logits[:,1]) + tf.constant(0.00001))
        
    return(decoder_distr)

class VAE_mixed(Model):
    
    def __init__(self, X, X_dict, z_dim, enc_size, dec_size, realisation_counts, variables_types, decoder_int_size, beta = 1):
        super(VAE_mixed, self).__init__()
        
        #Model hyperparameters etc.
        self.columns = X.columns
        self.beta = beta
        self.z_dim = z_dim
        self.realisation_counts = realisation_counts
        self.variables_types = variables_types
        
        #Layers encoder
        self.encoder_intermediate = layers.Dense(enc_size, activation = "tanh")
        self.encoder_mean = layers.Dense(z_dim)
        self.encoder_std = layers.Dense(z_dim, activation = 'relu')
        
        #Decoder
        self.decoder_hidden_layer = layers.Dense(dec_size, activation = "tanh")
        self.decoder_output_cat = layers.Dense(sum(np.array(realisation_counts)[np.array(realisation_counts) != 1]))
        self.decoder_output_int = layers.Dense(decoder_int_size)

        
    def call(self, inputs):
            
        #Encoder
        x_rep = self.encoder_intermediate(inputs)
        z_mean = self.encoder_mean(x_rep)
        z_log_var = self.encoder_std(x_rep)
        
        #Reparametrisation
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        z_sample = z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
        #Decoder  
        decoder_intermediate_layer = self.decoder_hidden_layer(z_sample)
        
        #Decoder output         
        list_outputs = decoder_outputs(self.columns, self.realisation_counts, self.variables_types, self.decoder_output_cat, self.decoder_output_int, decoder_intermediate_layer)
           
        #KL-loss
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        
        #Add KL-loss to model loss
        self.add_loss(self.beta*kl_loss)
        
        return(list_outputs)
    
    def custom_loss(self, logits, outputs):
        
        for i, col in enumerate(self.columns):
            y_pred = logits[col]
            y_true = outputs[col]
            
            if self.variables_types[i] == 'cat':
                if i == 0:
                    decoder_distr = tfd.OneHotCategorical(logits = y_pred)
                    likelihood_loss = - tf.reduce_mean(decoder_distr.log_prob(y_true))
                else:
                    decoder_distr = tfd.OneHotCategorical(logits = y_pred)
                    likelihood_loss = likelihood_loss - tf.reduce_mean(decoder_distr.log_prob(y_true))
            else:
                #Reshape
                y_true = tf.reshape(y_true, shape = (-1,))
                
                if i == 0:
                    distr_name = self.variables_types[i]
                    decoder_distr = prob_distribution_integer(distr_name, y_pred)
                    likelihood_loss = - tf.reduce_mean(decoder_distr.log_prob(y_true))
                else:
                    distr_name = self.variables_types[i]
                    decoder_distr = prob_distribution_integer(distr_name, y_pred)
                    likelihood_loss = likelihood_loss - tf.reduce_mean(decoder_distr.log_prob(y_true))
                    
        return likelihood_loss
         
    
    def synthetic_data(self, n):
        
        #Sample from N(0,I)
        z_sample = tf.keras.backend.random_normal(shape=(n, self.z_dim))
        
        #Decoder  
        decoder_intermediate_layer = self.decoder_hidden_layer(z_sample)
        
        #Decoder output         
        list_output_params = decoder_outputs(self.columns, self.realisation_counts, self.variables_types, self.decoder_output_cat, self.decoder_output_int, decoder_intermediate_layer)
       
        #Decoder outputs parametrize distributions
        list_outputs = {}
        for i, col in enumerate(self.columns):
            d = self.realisation_counts[i]
            y_pred = list_output_params[col]
            if self.variables_types[i] == 'cat':
                decoder_distr = tfd.OneHotCategorical(logits = y_pred)
                list_outputs[col] = decoder_distr.sample(1).numpy()[0]
            else:
                distr_name = self.variables_types[i]
                decoder_distr = prob_distribution_integer(distr_name, y_pred)
                list_outputs[col] = decoder_distr.sample(1).numpy()[0].reshape(n,1)
            
        return(list_outputs)
    
    def likelihood(self, X_dict):
        #Size of input
        n = list(X_dict.values())[1].shape[0]
        
        #Sample from N(0,I)
        z_sample = tf.keras.backend.random_normal(shape=(n, self.z_dim))
        
        #Decoder  
        decoder_intermediate_layer = self.decoder_hidden_layer(z_sample)
        
        #Decoder output         
        list_output_params = decoder_outputs(self.columns, self.realisation_counts, self.variables_types, self.decoder_output_cat, self.decoder_output_int, decoder_intermediate_layer)
       
        
        for i, col in enumerate(self.columns):
            d = self.realisation_counts[i]
            y_true = X_dict[col]
            y_true = tf.convert_to_tensor(y_true, dtype = 'float32')
            y_pred = list_output_params[col]
            if i == 0:
                if self.variables_types[i] == 'cat':
                    decoder_distr = tfd.OneHotCategorical(logits = y_pred)
                    sum_of_log_probabilities = decoder_distr.log_prob(y_true)              
                else:
                    y_true = tf.reshape(y_true, shape = (-1,))
                    distr_name = self.variables_types[i]
                    decoder_distr = prob_distribution_integer(distr_name, y_pred)
                    sum_of_log_probabilities = decoder_distr.log_prob(y_true)
            else:
                if self.variables_types[i] == 'cat':
                    decoder_distr = tfd.OneHotCategorical(logits = y_pred)
                    sum_of_log_probabilities = sum_of_log_probabilities + decoder_distr.log_prob(y_true)
                else:
                    y_true = tf.reshape(y_true, shape = (-1,))
                    distr_name = self.variables_types[i]
                    decoder_distr = prob_distribution_integer(distr_name, y_pred)
                    sum_of_log_probabilities = sum_of_log_probabilities + decoder_distr.log_prob(y_true)
            
        return(sum_of_log_probabilities.numpy())
    
    def encode(self, inputs):
        #Encoder
        x_rep = self.encoder_intermediate(inputs)
        z_mean = self.encoder_mean(x_rep)
        z_log_var = self.encoder_std(x_rep)
        
        #Reparametrisation
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        z_sample = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        return(z_sample)
    
    def reconstruct_logProbs(self, inputs, inputs_dict):
       
         #Encoder
        x_rep = self.encoder_intermediate(inputs)
        z_mean = self.encoder_mean(x_rep)
        z_log_var = self.encoder_std(x_rep)
        
        #Reparametrisation
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        z_sample = z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
        #Decoder  
        decoder_intermediate_layer = self.decoder_hidden_layer(z_sample)
        
        #Decoder output         
        list_output_params = decoder_outputs(self.columns, self.realisation_counts, self.variables_types, self.decoder_output_cat, self.decoder_output_int, decoder_intermediate_layer)
       
        #
        list_outputs = {}
        for i, col in enumerate(self.columns):
            d = self.realisation_counts[i]
            y_true = tf.convert_to_tensor(inputs_dict[col], dtype = 'float32')
            y_pred = list_output_params[col] 
            if self.variables_types[i] == 'cat':
                decoder_distr = tfd.OneHotCategorical(logits = y_pred)
                list_outputs[col] = decoder_distr.log_prob(y_true).numpy()
            else:
                y_true = tf.reshape(y_true, shape = (-1,))
                distr_name = self.variables_types[i]
                decoder_distr = prob_distribution_integer(distr_name, y_pred)
                list_outputs[col] = decoder_distr.log_prob(y_true).numpy()
            
        return(list_outputs)