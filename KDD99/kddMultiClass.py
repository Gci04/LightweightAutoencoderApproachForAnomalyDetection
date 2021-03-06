import numpy as np
import pandas as pd
np.random.seed(43)

import os, sys, keras, pickle, warnings
from scipy import stats
from time import time

import tensorflow as tf
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.callbacks import TensorBoard
from keras import optimizers, regularizers, backend as K

warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import seaborn as sn
from matplotlib import pyplot as plt

from preprocessing import get_kdd_data
import Utils

train ,test ,indx = get_kdd_data("multiclass")
train_label = train.label
train = train.drop(["label"],axis=1)

Scaler = StandardScaler()
train = Scaler.fit_transform(train.values)[np.where(train_label == 1)]

xtest , ytest = Scaler.transform(test.drop(["label"],axis=1)), test.label.values

def fit_kdd_AE(X):
    input_dim = X.shape[1]
    latent_space_size = 12
    K.clear_session()
    input_ = Input(shape = (input_dim, ))

    layer_1 = Dense(100, activation="tanh")(input_)
    layer_2 = Dense(50, activation="tanh")(layer_1)
    layer_3 = Dense(25, activation="tanh")(layer_2)

    encoding = Dense(latent_space_size,activation=None)(layer_3)

    layer_5 = Dense(25, activation="tanh")(encoding)
    layer_6 = Dense(50, activation="tanh")(layer_5)
    layer_7 = Dense(100, activation='tanh')(layer_6)

    decoded = Dense(input_dim,activation=None)(layer_7)

    autoencoder = Model(inputs=input_ , outputs=decoded)
    # opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    autoencoder.compile(metrics=['accuracy'],loss='mean_squared_error',optimizer="adam")
    # autoencoder.summary()

    #create TensorBoard
    tb = TensorBoard(log_dir="kdd99logs/{}".format(time()),histogram_freq=0,write_graph=True,write_images=False)

    # Fit autoencoder
    autoencoder.fit(X, X,epochs=100,validation_split=0.1 ,batch_size=100,shuffle=False,verbose=0,callbacks=[tb])

    return autoencoder

model = fit_kdd_AE(train)

losses = Utils.get_losses(model, train)
thresholds = Utils.confidence_intervals(losses,0.95)
threshold = thresholds[1]
pred = Utils.predictAnomaly(model,xtest,threshold)
true = np.where(ytest == "normal", 1,0)
Utils.performance(pred,true)
#1 : normal , 0 : Anomal
for key in indx.keys():
    if(key != "normal"):
        print('-'*35)
        print(' '*18 + key)
        print('-'*35)
        temp = np.ones(len(pred))
        mask = indx[key]
        np.put(temp,mask,0)
        temp_pred = np.ones(len(pred))
        np.put(temp_pred,mask,pred[mask])
        res = classification_report(temp,temp_pred,output_dict=True)["0.0"]
        print("{:<12s}{:<12s}{:<12s}".format("precision", "recall" ,"f1-score"))
        print("{:<12.2f} {:<12.2f} {:<12.2f}".format(res["precision"],res["recall"],res["f1-score"]))
        print()
