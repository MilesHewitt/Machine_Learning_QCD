# Imports
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

########################################################################################################################################
# Applying randomised search and cross validating to find best model
size_dist = 500
n_iter_search = 20
param_dist1 = {'alpha':np.random.uniform(low=0, high=1e-10, size=size_dist),'gamma': np.random.uniform(low=0, high=1, size=size_dist)}
            
# create and fit a ridge regression model, testing each alpha
model_random = KernelRidge(kernel = 'rbf')
grid_random = RandomizedSearchCV(estimator=model_random, param_distributions = param_dist1 , cv=5, scoring = 'neg_mean_squared_error', n_iter = n_iter_search, verbose=3)
grid_random.fit(X_train, y_train)
print('The Random Grid Search is:',grid_random)
MSE_Random = grid_random.best_score_
# summarize the results of the grid search
print('The best RMSE is:',np.sqrt(np.absolute(MSE_Random)))
print('The best Alpha is:',grid_random.best_estimator_.alpha)
print('The best Gamma is:',grid_random.best_estimator_.gamma)
print('The best Kernel is:',grid_random.best_estimator_.kernel)

##################################################################################################################################################
# Best randomised search model
Model_Rand = KernelRidge(kernel=grid_random.best_estimator_.kernel,alpha=grid_random.best_estimator_.alpha,gamma = grid_random.best_estimator_.gamma )
Model_Rand.fit(X_train,y_train)
RS_Prediction = Model_Rand.predict(X_test)
