# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 21:32:25 2017

@author: Aujasvi
"""
import numpy as np
#K-Fold validation
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import KFold
#Load the dataset
data = loadtxt('polynome.data')
#Seprate the input from the output
X = data[:, 0]
Y = data[:, 1]

X_split_data = hsplit(X,8)
Y_split_data = hsplit(Y,8)

for order in range(1,21):
    sum = 0
    for omit in range(0,8):
        X_train = hstack((X_split_data[i] for i in range(8) if not i==omit))
        Y_train = hstack((Y_split_data[i] for i in range(8) if not i==omit))
        X_test = X_split_data[omit]
        Y_test = Y_split_data[omit]
        w = polyfit(X_train,Y_train,order)
        y = polyval (w,X_test)
        #Genearlization Error on Test set
        mse = ((Y_test-y)**2).mean(axis=None) 
        sum = sum + mse
        
    average_of_mse = sum/8
    print('order='+str(order)+', average_of_mse='+str(average_of_mse))

