import pandas as pd
import numpy as np
import xgboost as xgb
import cPickle as pk

def get_params():
    params = {}
    params["objective"] = "binary:logistic"
    params["eta"] = 0.1
    params["min_child_weight"] = 10 
    params["subsample"] = 1.0 
    params["colsample_bytree"] = 0.2
    params["silent"] = 1
    params["max_depth"] = 6 
    plst = list(params.items())
    return plst

            
xgb_num_rounds = 1000000
#xgb_num_rounds = 100
num_classes = 2

with open('data/train_target_0.005') as f:
    train_target = pk.load(f)
with open('data/train_rf_feat_0.005') as f:
    train_feat = pk.load(f)




xgtrain = xgb.DMatrix(train_feat, train_target)

# get the parameters for xgboost
plst = get_params()
print(plst)

# train model
#model = xgb.train(plst, xgtrain, xgb_num_rounds)
cv_results = xgb.cv(plst, xgtrain, num_boost_round=xgb_num_rounds,
		nfold=5, metrics='logloss', show_progress=True)

