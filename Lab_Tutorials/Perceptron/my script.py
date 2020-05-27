from numpy import *
import matplotlib.pyplot as plt
#Load the data set
data = loadtxt('./Data/linear.data')
#Seprate the input from the output
X = data [:, 0:-1]
Y = data[:, -1]
N, d=X.shape
#Seprate the positive from the negative class
positive_class = X[Y==1., :]
negative_class = X[Y==-1., :]
#print(positive_class)
def visualize(w,b):
    plt.plot(positive_class[:,0], positive_class[:,1],'.r')
    plt.plot(negative_class[:,0],negative_class[:,1],'.b')
    X = linspace(0.,1.,100)
    Y = -(w[0]*X+b)/w[1]
    plt.plot(X,Y,'g')
    plt.show()

w = zeros(d)  
b = 0.   
eta = 0.001

def perceptron (w,b,eta):
    epoch = 1
    while True:
        for i in range(N):
            f_i = sign(dot(X[i,:],w)+b)
            w += eta*(Y[i]-f_i)*X[i,:]
            b += eta*(Y[i]-f_i)
        error,gamma = compute_error(w,b)
        print('Epoch:', epoch)
        print('w:', w)
        print('b', b)
        print('Error:', error)
        print('Functional margin:', gamma)
        if gamma >=0:
            print('Positive functional margin reached')
            break
        epoch +=1
      
    return w,b
   
        
    
def compute_error(w,b):
    nb_error = 0
    gamma = 1000.
    for i in range (N):
        scalar_product = dot(X[i,:],w)+b
        gamma = min(gamma, Y[i]*scalar_product)
        if sign(scalar_product) != Y[i]:
            nb_error +=1

    return nb_error/N, gamma
        
    
weight,bias = perceptron(w,b,eta)
print('weight:',weight)
print('bias:',bias)
visualize(weight,bias)


import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))






