#!curl -o data.gz -L "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz"
import os
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

def get_data():
    features = None
    with open('features.txt', 'r') as f:
      features = f.read().split('\n')
    features.append("weight")
    attacks = None
    with open('attacks.txt', 'r') as f:
      attacks = f.read().split('\n')
    # attacks
    train = pd.read_csv("NSL-KDD/KDDTrain+.txt",header=None)
    test = pd.read_csv("NSL-KDD/KDDTest+.txt",header=None)

    test.columns = features
    train.columns= features

    test.label = np.where(test.label == "normal",1,0)
    train.label = np.where(train.label == "normal",1,0)

    nTrain = train.shape[0]
    nTest = test.shape[0]

    combined = pd.get_dummies(pd.concat((train,test),axis=0), prefix=["protocol_type","service","flag"])

    train = combined.iloc[:nTrain]
    test = combined.iloc[nTest:]

    return train,test
