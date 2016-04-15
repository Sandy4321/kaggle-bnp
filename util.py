import numpy as np 
from sklearn.metrics import mutual_info_score
import sys

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

def discret(val, bins):
    for idx in xrange(len(bins)):
        try:
            if val >= bins[idx] and val <= bins[idx+1] :
                return idx
        except:
            print bins
            print val
            sys.exit(-1)

def calc_MI_feat_target(column, target, num_bins):

    p_neg = 0.238801
    p_pos = 0.761199
    column = column.round(5)
    min_val = np.min(column)
    max_val = np.max(column)
    column.fillna(-999, inplace=True)
    values = column.values
    try:
        bins = [-999] + np.arange(min_val, max_val, (max_val-min_val)/float(num_bins)).tolist()
    except:
        print min_val
        print max_val
        print (max_val-min_val)/num_bins
        sys.exit(-1)
    bins[-1] = max_val
    densitys, bin_edges = np.histogram(values, bins=bins, density=True)
    dist_vals = []
    for val in column.values:
        dist_val = discret(val, bins)
        dist_vals.append(dist_val)
        
    final_mi = 0
    dist_vals = np.array(dist_vals)
    for level in xrange(len(bins)):
        p_cate_pos = np.sum((dist_vals == level) & (target == 1)) / float(column.shape[0])
        p_cate_neg = np.sum((dist_vals == level) & (target == 0)) / float(column.shape[0])
        p_cate = np.sum((dist_vals == level)) / float(column.shape[0])
        if p_cate_pos == 0 or p_cate_neg == 0:
            continue
        final_mi += p_cate_pos * np.log2(p_cate_pos / (p_cate * p_pos))
        final_mi += p_cate_neg * np.log2(p_cate_neg / (p_cate * p_neg))

    return final_mi
