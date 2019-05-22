import numpy as np
np.random.seed(43)
import pandas as pd
import os
from scipy import stats
from time import time
from keras.layers import Input, Dense,Dropout
from keras.models import Model
from keras.callbacks import TensorBoard
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras import optimizers, regularizers, backend as K
import seaborn as sn
import keras
from matplotlib import pyplot as plt
%matplotlib inline
import pickle

import seaborn as sn

import warnings
warnings.filterwarnings('ignore')

from preprocessing import get_data
import Utils

import warnings
warnings.filterwarnings('ignore')

train ,test = get_data()

train_normal = (train[train["label"]==1]).drop(["label","weight"],axis=1)

train_label = train.label
train = train.drop(["label","weight"],axis=1)
Scaler1 = StandardScaler()
train = Scaler1.fit_transform(train.values)[np.where(train_label == 1)]

#tune activation function and optimizer
def fit_model(params,X):
    input_dim = train_normal.shape[1]
    latent_space_size = 15
    K.clear_session()
    input_ = Input(shape = (input_dim, ))

    layer_1 = Dense(115, activation=params[0])(input_)
    layer_2 = Dense(100, activation=params[0])(layer_1)
    layer_3 = Dense(75, activation=params[0])(layer_2)
    layer_4 = Dense(50, activation='relu')(layer_3)
    layer_5 = Dense(25, activation=params[0])(layer_4)

    encoding = Dense(latent_space_size,activation=None)(layer_5)

    layer_6 = Dense(25, activation=params[0])(encoding)
    layer_7 = Dense(50, activation='relu')(layer_6)
    layer_8 = Dense(75, activation=params[0])(layer_7)
    layer_9 = Dense(100, activation=params[0])(layer_8)
    layer_10 = Dense(115, activation=params[0])(layer_9)

    decoded = Dense(input_dim,activation=None)(layer_10)

    autoencoder = Model(inputs=input_ , outputs=decoded)

    autoencoder.compile(metrics=['accuracy'],loss='mean_squared_error',optimizer=params[1])
    #create TensorBoard
    tb = TensorBoard(log_dir=f'./Logs/logs30/{params[0]}_{params[1]}',histogram_freq=0,write_graph=False,write_images=False)

    autoencoder.fit(X, X,epochs=50,validation_split=0.2,batch_size=100,shuffle=True,verbose=0,callbacks=[tb])

    return autoencoder

# for param in comb:
#     print(param)
#     fit_model(param,train_normal)
# model = fit_model(["tanh","Adam"],train_normal)
model = fit_model(["tanh","Adam"],train)

losses = Utils.get_losses(model, train)
thresholds = Utils.confidence_intervals(losses,0.95)

#choose the upper interval as threshold
threshold = thresholds[1]

xtest , ytest = Scaler1.transform(test.drop(["label","weight"],axis=1)), test.label.values

pred = Utils.predictAnomaly(model,xtest,threshold)

Utils.performance(ytest,pred)
