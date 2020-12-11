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

####################################################################################################################################
# Grid Search a model of your choice (kernel ridge in the case below) then k-fold cross validation
parameter_grid = {'alpha':[1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-11,1e-12], 'gamma':[1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7], 'kernel':['rbf', 'linear']}
        
# create and fit a ridge regression model, testing each alpha
model_Grid = KernelRidge()
grid = GridSearchCV(estimator=model_Grid, param_grid = parameter_grid , cv=5, scoring = 'neg_mean_squared_error', verbose=2, n_jobs=-2)
grid.fit(X_train, y_train)
print('The Grid Search is:',grid)
# summarize the results of the grid search
MSE_Grid = grid.best_score_
print('The best RMSE is:',np.sqrt(np.absolute(MSE_Grid)))
print('The best Alpha is:',grid.best_estimator_.alpha)
print('The best Gamma is:',grid.best_estimator_.gamma)
print('The best Kernel:',grid.best_estimator_.kernel)
#################################################################################################################################################################
#Best grid search model to use
Model_Grid = KernelRidge(kernel = grid.best_estimator_.kernel,alpha = grid.best_estimator_.alpha,gamma = grid.best_estimator_.gamma )
Model_Grid.fit(X_train,y_train)
GS_Prediction = Model_Grid.predict(X_test)
