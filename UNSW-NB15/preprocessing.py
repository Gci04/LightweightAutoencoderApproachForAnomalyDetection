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
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from collections import defaultdict

import category_encoders as ce

#label 0 - normal, 1 - anomal
def get_data(classification="Binary",encoding="OneHot"):

    test = pd.read_csv("UNSW-NB15/dataset/part_training_testing_set/UNSW_NB15_testing-set.csv")
    train = pd.read_csv("UNSW-NB15/dataset/part_training_testing_set/UNSW_NB15_testing-set.csv")

    #drop id .. service has 53% nan values!!!
    nTrain = train.shape[0]

    attack_cat = test.attack_cat

    if encoding == "OneHot":
        combined = pd.concat((train,test),axis=0)
        combined = pd.get_dummies(combined.drop(["attack_cat"],axis=1), columns=["proto","service","state"])

        train = combined.iloc[:nTrain]
        train.reset_index(inplace = True,drop=True)

        test = combined.iloc[nTrain:]
        test.reset_index(inplace = True,drop=True)
    else:
        hash_encoder = ce.HashingEncoder(cols= ["proto","service","state"],n_components = 13)
        train = hash_encoder.fit_transform(train.drop(["attack_cat"],axis=1),train.label)
        test = hash_encoder.transform(test.drop(["attack_cat"],axis=1))

    if(classification != "Binary"):
        indices = defaultdict(dict)
        for label in np.unique(attack_cat) :
            indices[label] = np.where(attack_cat == label)[0]

        return train, test, indices

    return train, test
