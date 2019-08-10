import sys
sys.path.insert(0, './KDD99/')
import numpy as np
# np.random.seed(43)
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

from preprocessing import get_kdd_data
import Utils

# train ,test ,indx = get_kdd_data("multiclass")
train ,test = get_kdd_data("Binary")

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
    # opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    autoencoder.compile(metrics=['accuracy'],loss='mean_squared_error',optimizer="Adam")
    # autoencoder.summary()

    #create TensorBoard
    tb = TensorBoard(log_dir="./kdd99logs/{}".format(time()),histogram_freq=0,write_graph=True,write_images=False)

    # Fit autoencoder
    start= time()
    autoencoder.fit(X, X,epochs=20,validation_split=0.2 ,batch_size=100,shuffle=True,verbose=0,callbacks=[tb])
    print(time() - start)
    return autoencoder

model = fit_kdd_AE(train)
# with open('KDD99/kdd99_ep20_bs100_l12_samp50.pickle', 'wb') as f:
            # pickle.dump(model, f)
# with open('KDD99/kdd99_ep100_bs100_l12_samp50_final.pickle', 'rb') as fid:
#     model = pickle.load(fid)
losses = Utils.get_losses(model, train)
thresholds = Utils.confidence_intervals(losses,0.95)
#choose the upper interval as threshold
threshold = thresholds[1]
pred = Utils.predictAnomaly(model,xtest,threshold)
Utils.performance(ytest,pred)
