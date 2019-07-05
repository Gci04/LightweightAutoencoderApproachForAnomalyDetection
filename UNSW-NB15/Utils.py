import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn
from sklearn.manifold import TSNE
# %matplotlib inline

def mse(pred,true):
    result = []
    for sample1,sample2 in zip(pred,true):
        error = sum((sample1.astype("float") - sample2.astype("float")) ** 2)
        error /= float(len(sample2))
        result.append(error)
    return np.array(result)

def get_losses(model, x):
    reconstruct_err = []
    pred = model.predict(x)
    err = mse(x, pred)
    return err
def confidence_intervals(data, confidence=0.97):
    n = len(data)
    # mean & standard deviation
    mean, std_dev = np.mean(data), data.std()
    z_critical = stats.norm.ppf(q = confidence)
    margin_of_error = z_critical * (std_dev/np.sqrt(n))
    return [mean-margin_of_error, mean+margin_of_error]
def predictAnomaly(model,x,threshold):
    pred = model.predict(x)
    MSE = mse(pred,x)
    res = np.where(MSE < threshold,0,1) #anomaly : 1, normal : 0
    return res
def performance(true,pred,title="confusion matrix"):
    acc = accuracy_score(pred,true)
    print("Accurary : ",acc)
    f1 = f1_score(true,pred)
    print("F1 Score : ",f1)

    print("Classification report")
    print(classification_report(pred,true))
    df_cm = pd.DataFrame(confusion_matrix(true,pred), index = ["Anomal","Normal"],
                      columns = ["Anomal","Normal"])
    plt.figure(figsize = (10,7))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(title,fontsize=20)
    fig = sn.heatmap(df_cm,fmt='g',annot=True,annot_kws={"size": 20})
    plt.show()
def get_TSNE(dim_reducer,x,labels):

    test_reduced = dim_reducer.predict(x)
    test_embedded = TSNE(n_components=2).fit_transform(test_reduced)

    # with open('test_tsne.pickle', 'wb') as f:
    #             pickle.dump(test_embedded, f)

    plt.scatter(test_embedded[:,0], test_embedded[:,1], c = labels)
    plt.show()
