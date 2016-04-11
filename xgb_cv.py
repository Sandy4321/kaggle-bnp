import pandas as pd
import numpy as np
import xgboost as xgb
import cPickle as pk
from preprocess import find_delimiter
from util import get_params
log_file = open(__file__ + '_log', 'w')



def get_params_list():
    params = {}
    params["objective"] = "binary:logistic"
    params["eval_metric"] = "logloss"
    params["eta"] = 0.01
    params["min_child_weight"] = 1 
    params["subsample"] = 0.9 
    params["colsample_bytree"] = 0.9
    params["silent"] = 1
    params["max_depth"] = 10 
    plst = list(params.items())
    return plst

def factorize_category(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = pd.factorize(df[column])[0]
            

high_corr_columns = ['v8', 'v23', 'v25', 'v36', 'v37', 'v46', 'v51', 'v53', 'v54', 'v63', 'v73', 'v81', 'v82', 'v89', 'v92', 'v95', 'v105', 'v107', 'v108', 'v109', 'v116', 'v117', 'v118', 'v119', 'v123', 'v124', 'v128']
train_columns_to_drop = ['ID', 'target'] + high_corr_columns
print train_columns_to_drop  
test_columns_to_drop = ['ID'] + high_corr_columns

#train_columns_to_drop = ['ID', 'target', 'v107', 'v110']
#test_columns_to_drop = ['ID', 'v107', 'v110']
xgb_num_rounds = 1000000
#xgb_num_rounds = 100
num_classes = 2
print 'load data'
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')


train_feat = train.drop(train_columns_to_drop, axis=1) 
test_feat = test.drop(test_columns_to_drop, axis=1) 
factorize_category(train_feat)
factorize_category(test_feat)
train_feat.fillna(-1,inplace=True)
test_feat.fillna(-1,inplace=True)


train_feat_final = train_feat


xgtrain = xgb.DMatrix(train_feat_final, train['target'].values)


# grid search
params = get_params()
params["eta"] = 0.05

min_child_weight_list = [1, 5]
subsample_list = [0.6, 0.8, 1]
colsample_bytree_list = [0.6, 0.8, 1]
max_depth_list = [6, 8, 10]
params_list = []
for min_child_weight in min_child_weight_list:
    for subsample in subsample_list:
        for colsample_bytree in colsample_bytree_list:
            for max_depth in max_depth_list:

                params['min_child_weight'] = min_child_weight
                params['subsample'] = subsample
                params['colsample_bytree'] = colsample_bytree
                params['max_depth'] = max_depth

                plst = list(params.items())
                params_list.append(plst)

for plst in params_list:

    cv_results = xgb.cv(plst, xgtrain, num_boost_round=xgb_num_rounds,
	nfold=5, metrics='logloss', verbose_eval=True, early_stopping_rounds=10)


'''
valid_preds = model.predict(xgvalid, ntree_limit=model.best_iteration)
valid_target = train[ind==1]['target'].values
res = 0
for preds, target in zip(valid_preds, valid_target):
    res += target * np.log(preds) + (1 - target) * np.log(1 - preds) 
res /= len(valid_target)
res = -res
print res
'''
