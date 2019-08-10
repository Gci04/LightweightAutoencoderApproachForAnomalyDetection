# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, './UNSW-NB15/')

import numpy as np
np.random.seed(43)
import pandas as pd

import tensorflow as tf
tf.set_random_seed(7)
import pandas as pd

import pickle
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard
import tensorflow as tf
from keras import regularizers,optimizers, backend as K
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from preprocessing import get_data
import Utils

train, test = get_data(encoding="Hash_encoder")

train_label = train.label.values
train.drop(["label"],axis=1,inplace=True)

#0 : normal, 1 : anomal
Scaler = StandardScaler()
train = Scaler.fit_transform(train.values)[np.where(train_label == 0)]

test, ytest = Scaler.transform(test.drop(["label"],axis=1)) , test.label.values

#AUTOENCODER
def fit_model(X,lr=0.001,l2=0.001,ep=100, bs=50):
    input_dim = X.shape[1]
    latent_space_size = 15
    K.clear_session()
    input_ = Input(shape = (input_dim, ))

    layer_1 = Dense(100, activation='tanh')(input_)
    layer_2 = Dense(50, activation='tanh',kernel_regularizer=regularizers.l2(l2))(layer_1)
    layer_3 = Dense(25, activation='tanh',kernel_regularizer=regularizers.l2(l2))(layer_2)

    encoding = Dense(latent_space_size,activation=None,kernel_regularizer=regularizers.l2(0.01))(layer_3)

    layer_5 = Dense(25, activation='tanh',kernel_regularizer=regularizers.l2(l2))(encoding)
    layer_6 = Dense(50, activation='tanh',kernel_regularizer=regularizers.l2(l2))(layer_5)
    layer_7 = Dense(100, activation='tanh')(layer_6)

    decoded = Dense(input_dim,activation=None)(layer_7)

    autoencoder = Model(inputs=input_ , outputs=decoded)

    #dim_reducer = Model(inputs = input_, outputs = encoding)

    #opt = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    autoencoder.compile(metrics=['accuracy'],loss='mean_squared_error',optimizer="adam")
    #create TensorBoard
    #tb = TensorBoard(log_dir=f'./logs5',histogram_freq=0,write_graph=False,write_images=False)

    autoencoder.fit(X, X,epochs=ep,validation_split=0.2,batch_size=bs,shuffle=True,verbose=1)
    #hist = autoencoder.fit(X, X,epochs=50,validation_split=0.2,batch_size=100,shuffle=True,verbose=1,callbacks=[tb])

    #return autoencoder, dim_reducer, hist
    return autoencoder

# autoencoder = fit_model(train,lr=0.001,l2=0.001,ep=100, bs=64)
autoencoder = fit_model(train)

#TEST SET PERFORMANCE
losses = Utils.get_losses(autoencoder, train)
thresholds = Utils.confidence_intervals(losses,0.95)
threshold = thresholds[1]
test_pred = Utils.predictAnomaly(autoencoder,test,threshold)
Utils.performance(ytest, test_pred)
from collections import defaultdict
df = pd.read_csv("UNSW-NB15/dataset/part_training_testing_set/UNSW_NB15_testing-set.csv")
attack_cat = df.attack_cat
indices = defaultdict(dict)
for label in np.unique(attack_cat) :
    indices[label] = np.where(attack_cat == label)[0]
from sklearn.metrics import classification_report
for key in indices.keys():
    if(key != "Normal"):
        print('-'*35)
        print(' '*18 + key)
        print('-'*35)
        temp = np.ones(len(test_pred))
        mask = indices[key]
        np.put(temp,mask,0)
        temp_pred = np.ones(len(test_pred))
        np.put(temp_pred,mask,test_pred[mask])
        res = classification_report(temp,temp_pred,output_dict=True)["0.0"]
        print("{:<12s}{:<12s}{:<12s}".format("precision", "recall" ,"f1-score"))
        print("{:<12.2f} {:<12.2f} {:<12.2f}".format(res["precision"],res["recall"],res["f1-score"]))
        print()
        
