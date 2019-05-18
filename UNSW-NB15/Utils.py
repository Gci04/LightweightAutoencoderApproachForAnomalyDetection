import numpy as np
from scipy import stats
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
    res = np.where(MSE < threshold,1,0) #anomaly : 0, normal : 1
    return res
