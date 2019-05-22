import os
import pandas as pd
import numpy as np
from collections import defaultdict

import warnings
warnings.filterwarnings('ignore')

def get_data(classification = 'Binary'):
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

    if(classification == 'multiclass'):

        nTrain = train.shape[0]
        nTest = test.shape[0]

        combined = pd.concat((train,test),axis=0)
        labels_ind = defaultdict(dict)
        for attack in np.unique(combined.label.values):
            labels_ind[attack] = np.where(combined.label == attack)[0]

        combined.label = np.where(combined.label == "normal",1,0)

        combined = pd.get_dummies(combined, prefix=["protocol_type","service","flag"])

        # combined = pd.get_dummies(pd.concat((train,test),axis=0), prefix=["protocol_type","service","flag"])

        train = combined.iloc[:nTrain]
        test = combined.iloc[nTrain:]

        return train,test,labels_ind
    else:
        test.label = np.where(test.label == "normal",1,0)
        train.label = np.where(train.label == "normal",1,0)

        nTrain = train.shape[0]
        nTest = test.shape[0]

        combined = pd.get_dummies(pd.concat((train,test),axis=0), prefix=["protocol_type","service","flag"])

        train = combined.iloc[:nTrain]
        test = combined.iloc[nTrain:]

        return train,test
