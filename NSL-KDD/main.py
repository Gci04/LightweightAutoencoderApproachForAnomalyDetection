import numpy as np
import pandas as pd
import sys, os, pickle, keras, warnings

np.random.seed(43)
warnings.filterwarnings('ignore')

from scipy import stats
from time import time

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense,Dropout
from keras import optimizers, regularizers, backend as K

from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import seaborn as sn
# %matplotlib inline

from preprocessing import get_data
import Utils

# train ,test ,indx = get_data("multiclass")
train ,test = get_data("Binary")

train_label = train.label
train = train.drop(["label"],axis=1)

Scaler = StandardScaler()
train = Scaler.fit_transform(train.values)[np.where(train_label == 1)]

xtest , ytest = Scaler.transform(test.drop(["label"],axis=1)), test.label.values

def fit_model(params,X,latent=10,BS=250,ep = 100):
    input_dim = train.shape[1]
    latent_space_size = latent
    K.clear_session()
    input_ = Input(shape = (input_dim, ))

    layer_1 = Dense(100, activation=params[0])(input_)
    layer_2 = Dense(50, activation=params[0],kernel_regularizer=regularizers.l2(0.01))(layer_1)
    layer_3 = Dense(25, activation=params[0])(layer_2)

    encoding = Dense(latent_space_size,activation=None)(layer_3)

    layer_6 = Dense(25, activation=params[0])(encoding)
    layer_7 = Dense(50, activation=params[0],kernel_regularizer=regularizers.l2(0.01))(layer_6)
    layer_8 = Dense(100, activation=params[0])(layer_7)

    decoded = Dense(input_dim,activation=None)(layer_8)

    autoencoder = Model(inputs=input_ , outputs=decoded)
    opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    autoencoder.compile(metrics=['accuracy'],loss='mean_squared_error',optimizer=opt)

    print("Training Started ....")
    autoencoder.fit(X, X,epochs=ep,validation_split=0.2,batch_size=BS,shuffle=True,verbose=0)
    print("Training Finished ....")

    return autoencoder

if os.path.exists('models/nslkdd_reg_ep100_bs250_final.pickle'):
    with open('models/model_tanh_Adam_reg_ep95_bs250.pickle', 'rb') as fid:
        model = pickle.load(fid)

else:
    model = fit_model(["tanh","Adam"],X=train,latent=12,BS=250,ep=95)
    with open('models/nslkdd_reg_ep100_bs250_final.pickle', 'wb') as f:
                pickle.dump(model, f)

losses = Utils.get_losses(model, train)

thresholds = Utils.confidence_intervals(losses,0.95)
#choose the upper interval as threshold
threshold = thresholds[1]
pred = Utils.predictAnomaly(model,xtest,threshold)
Utils.performance(ytest,pred)
