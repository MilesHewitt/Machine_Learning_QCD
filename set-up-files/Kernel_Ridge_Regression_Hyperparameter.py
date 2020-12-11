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

##################################################################################################################################
# Surface Plot to obtain best hyperparameters
%matplotlib notebook
gammas= np.logspace(-4, 0, 5) 
alphas = np.logspace(-12, -9, 5) 
alf=[]
gam=[]
trainKR = []
testKR = []
train_MSE_KR = []
test_MSE_KR = []
for c in gammas:
    for a in alphas:
        KR_Reg = KernelRidge(alpha=a, kernel='rbf', gamma=c)
        KR_Reg.fit(X_train,y_train)
        #trainKR.append(KR_Reg.score(X_train, y_train, sample_weight=None))
        #testKR.append(KR_Reg.score(X_test, y_test, sample_weight=None))
        training_MSE = KR_Reg.predict(X_train)
        testing_MSE = KR_Reg.predict(X_test)
        train_MSE_KR.append(np.absolute(mean_squared_error(y_train, training_MSE)))
        test_MSE_KR.append(np.absolute(mean_squared_error(y_test, testing_MSE)))
        alf.append(a)
        gam.append(c)

alpha = np.reshape(np.array(alf), (len(gammas), len(alphas)))
gamma = np.reshape(np.array(gam), (len(gammas), len(alphas)))
minis = np.reshape(np.array(test_MSE_KR), (len(gammas), len(alphas)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot a basic surface.
ax.plot_surface(np.log(alpha), np.log(gamma), np.log(minis), cmap=cm.coolwarm)

ax.set_xlabel('log alpha')
ax.set_ylabel('log gamma')
ax.set_zlabel('log MSE')
ax.view_init(elev=38, azim=37)
plt.show()
#plt.savefig('3D_surface_plot.png', dpi=1000)

# Obtain best values from surface plot
result = np.where(np.array(test_MSE_KR) == np.amin(np.array(test_MSE_KR)))
print('Best RMSE is:',np.sqrt(np.array(test_MSE_KR).min()))
print('Best model is placed:',result[0])
print('Best RMSE is:',np.sqrt(np.array(test_MSE_KR)[result[0]]),'Best Gamma is:',np.array(gam)[result[0]],'Best Alpha is:',np.array(alf)[result[0]])
Best_KR_Gamma = np.array(gam)[result[0]]
Best_KR_Alpha = np.array(alf)[result[0]]
############################################################################################################
