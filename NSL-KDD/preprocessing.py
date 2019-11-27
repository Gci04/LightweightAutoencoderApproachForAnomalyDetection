import sys
sys.path.insert(0, './NSL-KDD/')
import os
import pandas as pd
import numpy as np
from collections import defaultdict

import warnings
warnings.filterwarnings('ignore')

def get_data(classification = 'Binary',data_folder="Data"):

    train = pd.read_csv(data_folder+"/KDDTrain.csv")
    test = pd.read_csv(data_folder+"/KDDTest.csv")

    if(classification == 'multiclass'):

        nTrain = train.shape[0]

        combined = pd.concat((train,test),axis=0)
        combined = pd.get_dummies(combined, columns=["protocol_type","service","flag"])

        train = combined.iloc[:nTrain]
        train.label = np.where(train.label == "normal",1,0)

        test = combined.iloc[nTrain:]
        test.reset_index(inplace = True,drop=True)

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
