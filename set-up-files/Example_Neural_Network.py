# Imports
%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_predict, cross_validate
from sklearn.kernel_ridge import KernelRidge
import os
import numpy as np
from sklearn import metrics
from scipy.stats import zscore
import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.client import device_lib
########################################################################################################################################
# Example neural network used, multiple others were used and run on my own GPU and a supercomputer, with tensorboard callback
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras import regularizers
from keras import optimizers
NAME = "Model-4_L-1000_N-0.0001_lr-0.001_l2"
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

adam = optimizers.Adam(lr=0.0001)
###

inputs = Input(shape=(100,))

hidden1 = Dense(1000, activation='relu', 
                kernel_regularizer=regularizers.l2(0.001))(inputs)
hidden2 = Dense(1000, activation='relu',
                kernel_regularizer=regularizers.l2(0.001))(hidden1)
hidden3 = Dense(500, activation='relu',
               kernel_regularizer=regularizers.l2(0.001))(hidden2)
hidden4 = Dense(200, activation='relu',
               kernel_regularizer=regularizers.l2(0.001))(hidden3)


output = Dense(6)(hidden4)
###
Dnn = Model(inputs, output)

Dnn.compile(optimizer=adam, loss='mse', metrics=['accuracy'])

Dnn.fit(X_train, y_train, epochs=500, verbose=1, callbacks=[tensorboard], validation_split = 0.2)

Predictions = Dnn.predict(X_test)

# Can now be used on delta function python code
