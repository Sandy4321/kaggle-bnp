import pandas as pd
import numpy as np
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss, make_scorer
from sklearn.grid_search import GridSearchCV
import random; random.seed(2016)
import xgboost as xgb

def get_params_list():
    params = {}
    params["objective"] = "binary:logistic"
    params["eval_metric"] = "logloss"
    params["eta"] = 0.05
    params["min_child_weight"] = 1
    params["subsample"] = 0.6
    params["colsample_bytree"] = 0.6
    params["silent"] = 1
    params["max_depth"] = 2 
    plst = list(params.items())

    return plst

xgb_num_rounds = 1000000


train_file = 'stack_data/train_ensemble_1.csv'
test_file = 'stack_data/test_ensemble_1.csv'
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

y_train = train.target
X_train = train.drop(['target'], axis=1)

xgtrain = xgb.DMatrix(X_train.values, y_train.values)


plst = get_params_list()
results = xgb.cv(plst, xgtrain, num_boost_round=xgb_num_rounds,
        nfold=5, metrics='logloss', verbose_eval=True, early_stopping_rounds=10)

