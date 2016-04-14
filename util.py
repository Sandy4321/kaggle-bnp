import numpy as np 
from sklearn.metrics import mutual_info_score

def logloss(preds, targets):
    res = 0
    for pred, target in zip(preds, targets):
        pred = np.min(pred, 1 - 10**(-3))
        pred = np.max(pred, 10**(-3))
        if pred == 1.0:
            pred = 0.999
        if pred == 0.0:
            pred = 0.001
        res += target * np.log(pred) + (1 - target) * np.log(1 - pred)
    res /= len(targets)
    res = -res

    return res


def get_params():
    params = {}
    params["objective"] = "binary:logistic"
    params["eval_metric"] = "logloss"
    params["eta"] = 0.1
    params["min_child_weight"] = 1
    params["subsample"] = 1
    params["colsample_bytree"] = 0.3
    params["silent"] = 1
    params["max_depth"] = 7
    return params

def log(f, line):
   f.write(line + '\n') 


# calculate mutual information
def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi
