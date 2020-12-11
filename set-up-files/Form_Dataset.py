# All imports
from __future__ import print_function
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, SGDRegressor, ElasticNet, LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import math
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_predict, cross_validate
from sklearn.kernel_ridge import KernelRidge
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from sklearn.ensemble import AdaBoostRegressor
from numpy import genfromtxt

# Form dataset being able to adjust number of masses/overlap integrals and size of dataset

n_iterations = 1000 #Rows of Matrix
Ntau = 10
n_masses = 1  #Number of masses and overlap integrals
Design = []
Target =[]
for i in range(n_iterations):                                                       
    masses = []                                                                     #Empty list for masses
    amplitudes = []   #Empty List for amplitudes
    mass = []
    amplitude = []
    masses.append(np.random.uniform(low=0, high=5, size=n_masses))                         #Add random values to masses
    amplitudes.append(np.random.uniform(low=0, high=1, size=n_masses))                     #Add random values to amplitudes/overlap integral
    mass.append(-np.sort(-masses[0]))
    amplitude.append(-np.sort(-amplitudes[0]))
    for j in range(Ntau):                                
        G = []
        for k in range(len(mass)):
            G.append(amplitude[k]*np.exp(-mass[k]*(j+1)))                            #Correlation Function
        Design.append((np.log(sum(G[0])))+np.random.uniform(low=-10e-3, high=10e-3, size=1))       #Add artificial noise                                                    #Sum all values upto n_masses

    Target.append(mass)
    Target.append(amplitude)  
X1 = np.array(Design)
X1.shape = (n_iterations, Ntau)
y = np.array(Target)
y.shape = (n_iterations, 2*n_masses)
X = np.array(X1)
x = np.concatenate((X,y),axis=1)
np.savetxt("Dataset.csv", x, delimiter=",")                 #Save as full dataset

#Split dataset into features and target as well as training and test. train_test_split would also work as well as this method.
#Scale the data using StandardScaler()

scaler = StandardScaler()
n_samples_train = int(0.9*n_iterations )
X_train, X_test = X[:n_samples_train], X[n_samples_train:]   
y_train, y_test = y[:n_samples_train], y[n_samples_train:]

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print(y_train)

np.savetxt("DatasetX_train.csv", X_train, delimiter=",")
np.savetxt("DatasetX_test.csv", X_test, delimiter=",")
np.savetxt("Datasety_train.csv", y_train, delimiter=",")
np.savetxt("Datasety_test.csv", y_test, delimiter=",")

# Code to retrieve datasets for regression models

X_train = genfromtxt('DatasetX_train.csv', delimiter=',')
X_test = genfromtxt('DatasetX_test.csv', delimiter=',')
y_train = genfromtxt('Datasety_train.csv', delimiter=',')
y_test = genfromtxt('Datasety_test.csv', delimiter=',')
