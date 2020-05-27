# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 12:15:57 2017

@author: Aujasvi
"""

from numpy import*
import matplotlib.pyplot as plt
#Load the dataset
data = loadtxt('polynome.data')
#Seprate the input from the output
X = data[:, 0]
Y = data[:, 1]

 #Apply the polynomial regression of order i on data
for order in range(1,21):
    w = polyfit(X,Y,order)
    y = polyval (w,X)
#Quadratic_error (mse) on the training set 
    mse = ((Y-y)**2).mean(axis=None)
    print('order='+str(order)+', mse='+str(mse))
#As the order of the polynome is increasing the quadratic error is decreasing 
#(Overfitting)