import pandas as pd
import numpy as np
np.random.seed(43)
import sys, os, pickle, keras, warnings

warnings.filterwarnings('ignore')

from tensorflow import set_random_seed
set_random_seed(1)

from scipy import stats
from time import time

from keras.layers import Input, Dense
from keras.models import Model
import tensorflow as tf
from keras import optimizers, regularizers, backend as K

from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import seaborn as sn
# %matplotlib inline

from preprocessing import get_data
import Utils


#for NLS-KDD
train ,test ,indexes = get_data("multiclass")

train_label = train.label
train = train.drop(["label"],axis=1)

Scaler = StandardScaler()
train = Scaler.fit_transform(train.values)[np.where(train_label == 1)]

xtest , ytest = Scaler.transform(test.drop(["label"],axis=1)), test.label.values

def fit_model(params,X,latent=10,BS=250,ep = 95):

  input_dim = X.shape[1]
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

  autoencoder.fit(X, X,epochs=ep,validation_split=0.2,batch_size=BS,shuffle=True,verbose=0)

  return autoencoder

# model = fit_model(["tanh","Adam"],X=train,latent=10,BS=250,ep=95)
if os.path.exists('models/model_tanh_Adam_reg_ep95_bs250.pickle'):
    with open('models/model_tanh_Adam_reg_ep95_bs250.pickle', 'rb') as fid:
        model = pickle.load(fid)

else:
    model = fit_model(["tanh","Adam"],X=train,latent=12,BS=250,ep=95)
    with open('models/model_tanh_Adam_reg_ep95_bs250.pickle', 'wb') as f:
                pickle.dump(model, f)

losses = Utils.get_losses(model, train)
thresholds = Utils.confidence_intervals(losses,0.95)
threshold = thresholds[1]
pred = Utils.predictAnomaly(model,xtest,threshold)
true = np.where(ytest == "normal", 1,0)
Utils.performance(pred,true)
#1 : normal , 0 : Anomal
for key in indexes.keys():
    if(key != "normal"):
        print('-'*35)
        print(' '*18 + key)
        print('-'*35)
        temp_true = np.ones(len(pred))
        mask = indexes[key]
        np.put(temp_true,mask,0)
        temp_pred = np.ones(len(pred))
        np.put(temp_pred,mask,pred[mask])
        res = classification_report(temp_true,temp_pred,output_dict=True)["0.0"]
        print("{:<12s}{:<12s}{:<12s}".format("precision", "recall" ,"f1-score"))
        print("{:<12.2f} {:<12.2f} {:<12.2f}".format(res["precision"],res["recall"],res["f1-score"]))
        print()
