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
from matplotlib import collections as matcoll

##########################################################################################################################################
# Last set of masses and overlap integrals that'll be plotted on delta function
last_number = len(y_test)-1
#######################################################################################################################################
# Find x and y errors for each regression model (in this case randomised search KRR predictions)
masses_pred = []
amplitudes_pred = []
masses_Res = []
amplitudes_Res = []

for i in range(len(RS_Prediction)):
    masses_pred.append(RS_Prediction[i][:n_masses])
    amplitudes_pred.append(RS_Prediction[i][n_masses:])
    masses_Res.append(y_test[i][:n_masses])
    amplitudes_Res.append(y_test[i][n_masses:])
    
np.concatenate(masses_pred, axis=0 )
np.concatenate(amplitudes_pred, axis=0 )
np.concatenate(masses_Res, axis=0 )
np.concatenate(amplitudes_Res, axis=0 )

x_error = np.sqrt(mean_squared_error(masses_pred, masses_Res))
y_error = np.sqrt(mean_squared_error(amplitudes_pred, amplitudes_Res))
################################################################################################################################################
# Set out the test example that is plotted in the delta function
full_pred = RS_Prediction[last_number]
full_result = y_test[last_number]
mass_pred = full_pred[:n_masses]
amp_pred = full_pred[n_masses:]
mass_res = full_result[:n_masses]
amp_res = full_result[n_masses:]
##################################################################################################################################################
# Plot delta function using previously calculated
lines_pred = []
for i in range(len(mass_pred)):
    pair=[(mass_pred[i],0), (mass_pred[i], amp_pred[i])]
    lines_pred.append(pair)

linecoll_pred = matcoll.LineCollection(lines_pred)
fig, ax = plt.subplots()
ax.add_collection(linecoll_pred)

lines_res = []
for i in range(len(mass_res)):
    pair=[(mass_res[i],0), (mass_res[i], amp_res[i])]
    lines_res.append(pair)

linecoll_res = matcoll.LineCollection(lines_res, colors='r')
ax.add_collection(linecoll_res)

plt.errorbar(mass_pred,amp_pred,xerr=x_error,yerr=y_error, fmt='o', ecolor='black',
            capsize=4, barsabove=True, markersize=0.1)
plt.scatter(mass_res,amp_res, s=0.1)

#plt.xticks(mass_pred)
plt.ylim(0,1)
plt.xlabel("Mass")
plt.ylabel("Overlap integral")
plt.legend(loc='upper right')
#plt.savefig("Delta_function_10000.png", dpi=1000)
plt.show()
