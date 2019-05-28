import sys
sys.path.insert(0, './KDD99/')
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import ks_2samp,chi2,chisquare

import warnings
warnings.filterwarnings('ignore')

def get_kdd_data(classification = 'Binary'):
    features = None
    with open('KDD99/features.txt', 'r') as f:
      features = f.read().split('\n')

    train = pd.read_csv('KDD99/KDD99_Data/kdd_train.gz', compression='gzip',names=features ,sep=',')
    train.drop_duplicates(inplace=True)


    test = pd.read_csv('KDD99/KDD99_Data/kdd_test.gz', compression='gzip',names=features ,sep=',')
    test.drop_duplicates(inplace=True)

    train = get_sample(train)
    train.reset_index(inplace = True,drop=True)

    if(classification == 'Binary'):
        test.label = np.where(test.label == "normal.",1,0)
        train.label = np.where(train.label == "normal.",1,0)

        nTrain = train.shape[0]

        combined = pd.get_dummies(pd.concat((train,test),axis=0), prefix=["protocol_type","service","flag"])

        train = combined.iloc[:nTrain]
        test = combined.iloc[nTrain:]

        return train,test

    else:
        nTrain = train.shape[0]

        combined = pd.concat((train,test),axis=0)
        combined = pd.get_dummies(combined, columns=["protocol_type","service","flag"])

        train = combined.iloc[:nTrain]
        train.label = np.where(train.label == "normal.",1,0)

        test = combined.iloc[nTrain:]
        test.reset_index(inplace = True,drop=True)

        attack_types = pd.read_csv('KDD99/attacks_types.txt',sep=" ",header=None,index_col=0)
        attack_map = attack_types.to_dict('dict')[1]

        test["label"] = test["label"].apply(lambda x: attack_map.get(x.replace(".", ""),"Unknown"))

        indexes = defaultdict(dict)
        for label in np.unique(test.label):
            indexes[label] = np.where(test.label.values == label)[0]

        return train,test,indexes
def get_sample(X,fraction = 0.5):
    overall = defaultdict(dict)
    for c in X.columns:
        if not(X[c].dtype == np.float64 or X[c].dtype == np.int64):
            val , counts = np.unique(X[c],return_counts=True)
            for k in np.asarray((val, counts)).T:
                overall[c][k[0]] = k[1]
    i = 0
    while(i < 100):
        sample = X.sample(frac=fraction,replace=False)
        passed = np.zeros(sample.shape[1],dtype='bool')
        j = 0
        for c in X.columns:
            if c not in ["protocol_type","service","flag","label"]:
                st,p = ks_2samp(X[c].values, sample[c].values)
                if p > 0.03 :
                    passed[j] = True
                    j += 1
                else:
                    passed[j] = False
                    j += 1
            else:
                temp = np.fromiter(overall[c].values(), dtype=np.int64)
                expected = np.fromiter(overall[c].values(), dtype=np.int64)
                val , counts = np.unique(sample[c],return_counts=True)
                val1 = list(overall[c].keys())
                diff = np.setdiff1d(val1,val)
                observed_val = val.tolist()
                observed_count = counts.tolist()
                if len(diff) > 0 :
                    val = np.array(observed_val + diff.tolist())
                    counts = np.array(observed_count + ([0] * len(diff)))
                #argsort
                obs_sort_ind = np.argsort(val)
                exp_sort_ind = np.argsort(val1)

                observed = counts[obs_sort_ind]
                expected = expected[exp_sort_ind]

                st,p = chisquare(f_obs= observed,f_exp= expected)
                if p < 0.03:
                    passed[j] = True
                    j += 1
                else:
                    passed[j] = False
                    j += 1
        i += 1
        if all(passed):
            return sample
