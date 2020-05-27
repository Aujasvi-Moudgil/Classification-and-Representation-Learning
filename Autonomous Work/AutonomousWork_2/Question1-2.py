# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 22:20:24 2017

@author: Aujasvi
"""

from numpy import*
import matplotlib.pyplot as plt
#Load the dataset
data = loadtxt('polynome.data')
#Seprate the input from the output
X = data[:, 0]
Y = data[:, 1]
N = len (X)

def visualize (w): #Plot the data
    plt.plot(X,Y,'r.')
#Plot the fitted curve
    x = linspace (0.,1., 100)
    y = polyval (w,x)
    plt.plot(x,y,'g-')
    plt.title('Polynomial regression with order ' + str(len(w)-1))
    plt.show()
#Apply the polynomial regression of order i on data from 1 to 20
for order in range(1,21):
    w = polyfit(X,Y,order)
    #Visualise the fit
    visualize(w) 


