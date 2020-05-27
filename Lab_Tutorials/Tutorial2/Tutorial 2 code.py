# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 10:34:48 2017

@author: Aujasvi
"""

from numpy import *
import matplotlib.pyplot as plt
#Load the data set
data = loadtxt('nonlinear_classification.data')
X = data [:, :2]
T = data [:,2]
N, d = X.shape

#Parameters
eta = .3 #Learning rate
K = 15 #Number of hidden neurons

#splitting the data into training and testing set
X_train_data = X[:int((len(X))*.70)]
X_test_data = X[int(len(X)*.70):]
T_train_data = T[:int(len((T))*.70)]
T_test_data = T[int(len((T))*.70):]
#weights and biases
max_val = .1
W_hid = random.uniform(-max_val, max_val,(d,K)) #all are small function
b_hid = random.uniform(-max_val, max_val, K)
W_out = random.uniform(-max_val, max_val, K)
b_out = random.uniform(-max_val,max_val, 1) 

#Logistic transfer function for the hidden neurons
def logistic(X_train_data):
    return 1.0/(1.0 + exp(-X_train_data))
#Threshold transfer function for the output neuron
def threshold (X_train_data):
    P = X_train_data.copy
    P[P > 0.] = 1.
    P[P < 0.] = -1.
    return P
def feedforward (X_train_data, W_hid, b_hid, W_out, b_out):
    #Hidden layer
    Y = logistic (dot(X_train_data, W_hid) + b_hid)
   #Output layer
    O = threshold (dot(Y, W_out) + b_out) 
    return Y, O

#Backproppgation Algo
errors = []
for epoch in range (100):
    nb_errors = 0
    for i in range (N):
        x = X_train_data[i,:]
        t = T[i]
        Y, O = feedforward (X_train_data,W_hid, b_hid, W_out, b_out)
        if t != O:
            nb_errors +=1
        delta_out = (t-O)
        delta_hidden = Y*(1-Y)*delta_out*W_out
        W_out += eta*Y*delta_out
        b_out += eta*delta_out
        for k in range (K):
            W_hid[ :, k] +=  eta*x*delta_hidden[k]
        b_hid += eta*delta_hidden
    errors.append(nb_errors/float(N))
plt.plot(errors)
         
M = mean(epoch)
print(M)

V = var(epoch)
print(V)
            
#Question 2 : Convergence speed as no of hidden neurons are increasing
#Question 3:          
#Question 4: When weights are initialised between at max_val = 0 then error is 
#incresing.   
#Question 5: Mean of the number of the epochs needed is 9 and variance is 0.   


