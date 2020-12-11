#Imports
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

#####################################################################################################################
#Code to find optimal value of alpha (model hyperparameter) - Same code adapted for LASSO and elastic net

print(__doc__)
alphas = np.logspace(-20, 15, 10) 

%matplotlib inline

trainER = []
testER = []
train_errors1 = []
test_errors1 = []

for a in alphas:
    Ridge_Reg = MultiOutputRegressor(Ridge(alpha=a, fit_intercept=True))
    Ridge_Reg.fit(X_train,y_train)
    trainER.append(Ridge_Reg.score(X_train, y_train, sample_weight=None))
    testER.append(Ridge_Reg.score(X_test, y_test, sample_weight=None))
    train_MSE = Ridge_Reg.predict(X_train)
    test_MSE = Ridge_Reg.predict(X_test)
    train_errors1.append(mean_squared_error(y_train, train_MSE))
    test_errors1.append(mean_squared_error(y_test, test_MSE))



#plt.semilogx(alphas, trainER, 'b',label='Train R2 (Ridge)')
#plt.semilogx(alphas, testER, '--b',label='Test R2 (Ridge)')
plt.semilogx(alphas, train_errors1, 'g',label='Train MSE (Ridge)')
plt.semilogx(alphas, test_errors1, '--g',label='Test MSE (Ridge)')


plt.legend(loc='upper right')
plt.xlabel(r'$\alpha$',fontsize=18)
plt.ylabel('MSE')
plt.grid()
#plt.savefig("Hyperparam_Ridge_1000.png", dpi=1000)
plt.show()

################################################################################################################
# Plot a learning curve to compare how the algorithm learns over training set size, both training and testing
def plot_learning_curves(model, X, y):
    X_trainz, X_val, y_trainz, y_val = train_test_split(X, y, test_size=0.1)
    train_errors, val_errors = [], []
    
    for m in range(1, len(X_train)):
        model.fit(X_trainz[:m], y_trainz[:m])
        y_train_predict = model.predict(X_trainz[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_trainz[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)   
    plt.xlabel("Training set size", fontsize=14) 
    plt.ylabel("RMSE", fontsize=14)              
    
    
plot_learning_curves(Optimum_Ridge, X, y)
plt.axis([0,200,0,2])
#plt.figure(figsize=(10,10))
#plt.savefig('Learning_Curves_Ridge.png', dpi=1000)
plt.show()
