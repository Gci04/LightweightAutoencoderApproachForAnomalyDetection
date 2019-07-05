# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, './UNSW-NB15/')
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard
import tensorflow as tf
from keras import regularizers, backend as K
import Utils
from preprocessing import get_data
from matplotlib import pyplot as plt

train, test = get_data('Binary')

train_label = train.label

train.drop(["label"],axis=1,inplace=True)
#0 : normal, 1 : anomal
Scaler = StandardScaler()
train = Scaler.fit_transform(train.values)[np.where(train_label == 0)]

test, ytest = Scaler.transform(test.drop(["label"],axis=1)) , test.label.values


#AUTOENCODER
def fit_model(X):
    input_dim = X.shape[1]
    latent_space_size = 12
    K.clear_session()
    input_ = Input(shape = (input_dim, ))

    layer_1 = Dense(100, activation='tanh')(input_)
    layer_2 = Dense(50, activation='tanh')(layer_1)
    layer_3 = Dense(25, activation='tanh')(layer_2)

    encoding = Dense(latent_space_size,activation=None)(layer_3)

    layer_5 = Dense(25, activation='tanh')(encoding)
    layer_6 = Dense(50, activation='tanh')(layer_5)
    layer_7 = Dense(100, activation='tanh')(layer_6)

    decoded = Dense(input_dim,activation=None)(layer_7)

    autoencoder = Model(inputs=input_ , outputs=decoded)
    dim_reducer = Model(inputs = input_, outputs = encoding)

    autoencoder.compile(metrics=['accuracy'],loss='mean_squared_error',optimizer='adam')
    #create TensorBoard
    tb = TensorBoard(log_dir=f'./logs5',histogram_freq=0,write_graph=False,write_images=False)

    hist = autoencoder.fit(X, X,epochs=100,validation_split=0.2,batch_size=100,shuffle=True,verbose=1,callbacks=[tb])

    return autoencoder, dim_reducer, hist

autoencoder, dim_reducer, hist = fit_model(train)

"""
autoencoder, dim_reducer, hist = fit_model(train_normal)
with open('autoenc100.pickle', 'wb') as f:
            pickle.dump(autoencoder, f)
with open('dimred100.pickle', 'wb') as f:
            pickle.dump(dim_reducer, f)


with open('autoenc100.pickle', 'rb') as fid:
    autoencoder = pickle.load(fid)

with open('dimred100.pickle', 'rb') as fid:
    dim_reducer = pickle.load(fid)
"""

#TEST SET PERFORMANCE
losses = Utils.get_losses(autoencoder, train)
thresholds = Utils.confidence_intervals(losses,0.95)
threshold = thresholds[1]
test_pred = Utils.predictAnomaly(autoencoder,test,threshold)
Utils.performance(ytest, test_pred)

#tSNE
# Utils.get_TSNE()
