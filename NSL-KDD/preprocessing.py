import sys
sys.path.insert(0, './NSL-KDD/')
import os
import pandas as pd
import numpy as np
from collections import defaultdict

import warnings
warnings.filterwarnings('ignore')

def get_data(classification = 'Binary'):
    features = None
    with open('NSL-KDD/features.txt', 'r') as f:
      features = f.read().split('\n')
    features.append("weight")

    train = pd.read_csv("NSL-KDD/NSL-KDD_Data/KDDTrain+.txt",names=features)
    # train.drop_duplicates(inplace=True)
    test = pd.read_csv("NSL-KDD/NSL-KDD_Data/KDDTest+.txt",names=features)

    if(classification == 'multiclass'):

        nTrain = train.shape[0]

        combined = pd.concat((train,test),axis=0)
        combined = pd.get_dummies(combined, columns=["protocol_type","service","flag"])

        train = combined.iloc[:nTrain]
        train.label = np.where(train.label == "normal",1,0)

        test = combined.iloc[nTrain:]
        test.reset_index(inplace = True,drop=True)

        train_att = pd.read_csv('NSL-KDD/attacks_types.txt',sep=" ",header=None,index_col=0)
        attack_map = train_att.to_dict('dict')[1]

        test["label"] = test["label"].apply(lambda x: attack_map.get(x,"Unknown"))

        indexes = defaultdict(dict)
        for label in np.unique(test.label):
            indexes[label] = np.where(test.label.values == label)[0]

        return train,test,indexes
    else:
        test.label = np.where(test.label == "normal",1,0)
        train.label = np.where(train.label == "normal",1,0)

        nTrain = train.shape[0]

        combined = pd.get_dummies(pd.concat((train,test),axis=0), prefix=["protocol_type","service","flag"])

        train = combined.iloc[:nTrain]
        test = combined.iloc[nTrain:]

        return train,test
