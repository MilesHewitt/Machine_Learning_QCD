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
# Cross validate surface plot model
scores_OptiKR = cross_validate(Optimum_KR, X_train, y_train, cv=5, scoring = ('r2', 'neg_mean_squared_error'))
print('MSE is:', np.absolute(np.mean(scores_OptiKR['test_neg_mean_squared_error'])))
print('R2 score is:', np.mean(scores_OptiKR['test_r2']))

# Cross validate grid search model
scores_Grid = cross_validate(Model_Grid, X_train, y_train, cv=5, scoring = ('r2', 'neg_mean_squared_error'))
print('MSE is:', np.absolute(np.mean(scores_Grid['test_neg_mean_squared_error'])))
print('R2 score is:', np.mean(scores_Grid['test_r2']))

# Cross validate randomised search model
scores_Rand = cross_validate(Model_Rand, X_train, y_train, cv=5, scoring = ('r2', 'neg_mean_squared_error'))
print('MSE is:', np.absolute(np.mean(scores_Rand['test_neg_mean_squared_error'])))
print('R2 score is:', np.mean(scores_Rand['test_r2']))
