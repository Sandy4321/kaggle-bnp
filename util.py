import numpy as np 

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
