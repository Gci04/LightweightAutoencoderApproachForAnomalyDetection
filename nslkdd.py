import numpy as np
np.random.seed(1233)
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
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
import seaborn as sn

import warnings
warnings.filterwarnings('ignore')

from preprocessing import get_data
import Utils

train ,test = get_data()

train_normal = (train[train["label"]==1]).drop(["label","weight"],axis=1)

Scaler = StandardScaler()

train_normal = Scaler.fit_transform(train_normal)

#for Hyperparameters turning
# optimizrs = ['RMSprop', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
# activation = ['softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
# # comb = [[x,y] for x in optimizrs for y in activation]
#
# comb = [['tanh','Adam'],['tanh','Adamax'],['tanh','Adadelta'],['tanh','Nadam'],
# ['softsign','Adamax'],['relu','RMSprop'],['relu','Adam'],['relu','Nadam'],['softplus','Nadam'],
# ['softplus','RMSprop'],['softplus','Adam'],['softsign','Nadam'],['softsign','Adam'],['softsign','RMSprop']]

#tune activation function and optimizer
def fit_model(params,X):
    input_dim = train_normal.shape[1]
    latent_space_size = 15
    K.clear_session()
    input_ = Input(shape = (input_dim, ))

    layer_1 = Dense(100, activation=params[0])(input_)
    layer_2 = Dense(50, activation=params[0])(layer_1)
    layer_3 = Dense(25, activation=params[0])(layer_2)

    encoding = Dense(latent_space_size,activation=None)(layer_3)

    layer_5 = Dense(25, activation=params[0])(encoding)
    layer_6 = Dense(50, activation=params[0])(layer_5)
    layer_7 = Dense(100, activation=params[0])(layer_6)

    decoded = Dense(input_dim,activation=None)(layer_7)

    autoencoder = Model(inputs=input_ , outputs=decoded)

    autoencoder.compile(metrics=['accuracy'],loss='mean_squared_error',optimizer=params[1])
    #create TensorBoard
    tb = TensorBoard(log_dir=f'./logs10/{params[0]}_{params[1]}',histogram_freq=0,write_graph=False,write_images=False)

    autoencoder.fit(X, X,epochs=20,validation_split=0.2,batch_size=100,shuffle=True,verbose=0,callbacks=[tb])

    return autoencoder

# for param in comb:
#     print(param)
#     fit_model(param,train_normal)

# model = fit_model(["tanh","Adam"],train_normal)

# with open('model_tanh_Adam.pickle', 'wb') as f:
#             pickle.dump(model, f)
with open('model_tanh_Adam.pickle', 'rb') as fid:
    model = pickle.load(fid)
losses = Utils.get_losses(model, train_normal)
loss_df = pd.DataFrame(losses,columns=["loss"])

thresholds = Utils.confidence_intervals(losses,0.95)
# loss_df.describe()

#choose the upper interval as threshold
threshold = thresholds[1]
xtest , ytest = test.drop(["label","weight"],axis=1), test.label.values
xtest = StandardScaler().fit_transform(xtest.values)
pred = Utils.predictAnomaly(model,xtest,threshold)

def performance(true,pred,title="confusion matrix"):
    acc = accuracy_score(pred,true)
    print("Accurary : ",acc)
    f1 = f1_score(true,pred)
    print("F1 Score : ",f1)

    print("Classification report")
    print(classification_report(pred,true))
    df_cm = pd.DataFrame(confusion_matrix(true,pred), index = ["Anomal","normal"],
                      columns = ["Anomal","Normal"])
    plt.figure(figsize = (10,7))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(title,fontsize=20)
    fig = sn.heatmap(df_cm,fmt='g',annot=True,annot_kws={"size": 20})
np.where(ytest==0)[0].shape
performance(ytest,pred)



# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
#
# bestfeatures = SelectKBest(score_func=chi2, k=10)
# fit = bestfeatures.fit(train.drop(["label","weight"],axis=1).values,train.label.values)
# dfscores = pd.DataFrame(fit.scores_)
# dfscores
# dfcolumns = pd.DataFrame(train.drop(["label","weight"],axis=1).columns)
# featureScores = pd.concat([dfcolumns,dfscores],axis=1)
# featureScores.columns = ['Specs','Score']
# print(featureScores.nlargest(10,'Score'))
# featureScores.Specs.values[np.argsort(featureScores.Score.values)]
# print(featureScores.nlargest(20,'Score'))
