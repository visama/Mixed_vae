# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 15:41:30 2021

@author: visam
"""
import tensorflow as tf
from data_processing import shuffle_dictionary, dict_to_array
from Bhattacharyya_distance import Bhattacharyya_distance_total
import numpy as np

#Function for training loop
def training_loop(model, X, epochs, batch_size, optimizer_learning_rate, x_input, x_output): #Lisäparametrejä epsilon, delta, optimizer
    X_input = x_input
    X_dict = x_output
    n = X_input.shape[0]
    optimizer = tf.keras.optimizers.Adam(optimizer_learning_rate)
    
    losses = []
    for epoch in range(epochs):
        print(epoch)
        X_dict_tr = shuffle_dictionary(X_dict)
        X_input_tr = dict_to_array(X_dict_tr)
         
        for iteration in range(0, int(n/batch_size)):
            x_input = X_input_tr[iteration*batch_size:(iteration*batch_size + batch_size)]
            x_input = tf.convert_to_tensor(x_input, dtype = "float32")
            x_dict = {}
            for i, col in enumerate(X.columns):
                x_dict[col] = tf.convert_to_tensor(X_dict_tr[col][iteration*batch_size:(iteration*batch_size + batch_size)], dtype = "float32")
        
            with tf.GradientTape() as tape:

                logits = model(x_input, training=True)  

                # Compute the loss value for this minibatch.
                loss = model.custom_loss(logits, x_dict)
                loss += sum(model.losses)
            
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            losses.append(loss.numpy())
            
    return(model, losses)

#data quality loss in loop
def training_loop_B_loss(model, X, epochs, batch_size, optimizer_learning_rate, x_input, x_output): #Lisäparametrejä epsilon, delta, optimizer
    X_input = x_input
    X_dict = x_output
    n = X_input.shape[0]
    optimizer = tf.keras.optimizers.Adam(optimizer_learning_rate)
    
    losses = []
    B_loss = []
    for epoch in range(epochs):
        print(epoch)
        X_dict_tr = shuffle_dictionary(X_dict)
        X_input_tr = dict_to_array(X_dict_tr)
         
        for iteration in range(0, int(n/batch_size)):
            x_input = X_input_tr[iteration*batch_size:(iteration*batch_size + batch_size)]
            x_input = tf.convert_to_tensor(x_input, dtype = "float32")
            x_dict = {}
            for i, col in enumerate(X.columns):
                x_dict[col] = tf.convert_to_tensor(X_dict_tr[col][iteration*batch_size:(iteration*batch_size + batch_size)], dtype = "float32")
        
            with tf.GradientTape() as tape:

                logits = model(x_input, training=True)  

                # Compute the loss value for this minibatch.
                loss = model.custom_loss(logits, x_dict)
                loss += sum(model.losses)
            
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            losses.append(loss.numpy())
        B_loss.append(Bhattacharyya_distance_total(X, x_output, model, variable_types))
            
    return(model, losses, B_loss)

#Joka sadannen batchin kohdalla lasketaan loss
def training_loop_B_loss(model, X, epochs, batch_size, optimizer_learning_rate, x_input, x_output): #Lisäparametrejä epsilon, delta, optimizer
    X_input = x_input
    X_dict = x_output
    n = X_input.shape[0]
    optimizer = tf.keras.optimizers.Adam(optimizer_learning_rate)
    
    losses = []
    B_loss = []
    for epoch in range(epochs):
        print(epoch)
        X_dict_tr = shuffle_dictionary(X_dict)
        X_input_tr = dict_to_array(X_dict_tr)
         
        for iteration in range(0, int(n/batch_size)):
            x_input = X_input_tr[iteration*batch_size:(iteration*batch_size + batch_size)]
            x_input = tf.convert_to_tensor(x_input, dtype = "float32")
            x_dict = {}
            for i, col in enumerate(X.columns):
                x_dict[col] = tf.convert_to_tensor(X_dict_tr[col][iteration*batch_size:(iteration*batch_size + batch_size)], dtype = "float32")
        
            with tf.GradientTape() as tape:

                logits = model(x_input, training=True)  

                # Compute the loss value for this minibatch.
                loss = model.custom_loss(logits, x_dict)
                loss += sum(model.losses)
            
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            losses.append(loss.numpy())
            if iteration % 100 == 0:
                B_loss.append(Bhattacharyya_distance_total(X, x_output, model, variable_types))
        
            
    return(model, losses, B_loss)


#data quality loss in loop
def training_loop_no_epochs_B_loss(model, X, nro_batches, batch_size, optimizer_learning_rate, x_input, x_output, variable_types, M_samples = 1): #Lisäparametrejä epsilon, delta, optimizer
    X_input = x_input
    X_dict = x_output
    n = X_input.shape[0]
    optimizer = tf.keras.optimizers.Adam(optimizer_learning_rate)
    
    losses = []
    B_loss = []
    batch = 0
    while batch < nro_batches:
        
        X_dict_tr = shuffle_dictionary(X_dict)
        X_input_tr = dict_to_array(X_dict_tr)
         
        ###One epoch
        for iteration in range(0, int(n/batch_size)):
            x_input = X_input_tr[iteration*batch_size:(iteration*batch_size + batch_size)]
            x_input = tf.convert_to_tensor(x_input, dtype = "float32")
            x_dict = {}
            for i, col in enumerate(X.columns):
                x_dict[col] = tf.convert_to_tensor(X_dict_tr[col][iteration*batch_size:(iteration*batch_size + batch_size)], dtype = "float32")
        
            with tf.GradientTape() as tape:

                logits = model(x_input, training=True)  

                # Compute the loss value for this minibatch.
                loss = model.custom_loss(logits, x_dict)
                loss += sum(model.losses)
            
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            losses.append(loss.numpy())
            
            ith = 10
            if batch % ith == 0 and batch > 0:
                b_loss = 0
                for m_sample in range(M_samples):
                    b_loss = b_loss + Bhattacharyya_distance_total(X, x_output, model, variable_types) / M_samples
                B_loss.append(b_loss)
                print(b_loss)
            batch += 1
        ###One epoch
    return(model, losses, B_loss)

def training_loop_no_epochs_B_loss_hparams(model, X, nro_batches, batch_size, optimizer_learning_rate, x_input, x_output, M_samples = 10): #Lisäparametrejä epsilon, delta, optimizer
    X_input = x_input
    X_dict = x_output
    n = X_input.shape[0]
    optimizer = tf.keras.optimizers.Adam(optimizer_learning_rate)
    
    batch = 0
    while batch < nro_batches:
        
        X_dict_tr = shuffle_dictionary(X_dict)
        X_input_tr = dict_to_array(X_dict_tr)
         
        ###One epoch
        for iteration in range(0, int(n/batch_size)):
            x_input = X_input_tr[iteration*batch_size:(iteration*batch_size + batch_size)]
            x_input = tf.convert_to_tensor(x_input, dtype = "float32")
            x_dict = {}
            for i, col in enumerate(X.columns):
                x_dict[col] = tf.convert_to_tensor(X_dict_tr[col][iteration*batch_size:(iteration*batch_size + batch_size)], dtype = "float32")
        
            with tf.GradientTape() as tape:

                logits = model(x_input, training=True)  

                # Compute the loss value for this minibatch.
                loss = model.custom_loss(logits, x_dict)
                loss += sum(model.losses)
            
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            
            ith = 10
            if nro_batches - batch == 1:
                b_loss = 0
                for m_sample in range(M_samples):
                    b_loss = b_loss + Bhattacharyya_distance_total(X, x_output, model, variable_types) / M_samples
                print(b_loss)
            batch += 1
        ###One epoch
    return(b_loss)