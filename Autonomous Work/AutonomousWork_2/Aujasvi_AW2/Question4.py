# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 13:55:10 2017

@author: Aujasvi
"""

from numpy import*
import matplotlib.pyplot as plt
#Load the dataset
data = loadtxt('polynome.data')
#Seprate the input from the output
X = data[:, 0]
Y = data[:, 1]

#Spliting the data into training & test data
X_train_data = X[:int((len(X))*.70)]
X_test_data = X[int(len(X)*.70):]
Y_train_data = Y[:int((len(Y))*.70)]
Y_test_data = Y[int(len(Y)*.70):]

print('X_train_data :', X_train_data)
print('X_test_data :', X_test_data)
print('Y_train_data :', Y_train_data)
print('Y_test_data :',Y_test_data)

for order in range(1,21):
    w = polyfit(X_train_data,Y_train_data,order)
    y = polyval (w,X_test_data)
    #Genearlization Error on Test set
    mse = ((Y_test_data-y)**2).mean(axis=None) 
    print('order='+str(order)+', mse='+str(mse))
    
#In case of overfitting train error is low, test error is high.
#In case of order 4 the test error is minimum & it is generalizing well but when we
#increasing the degree of polynome then the curve tends to overfit.
