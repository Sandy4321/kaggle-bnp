import pandas as pd
import numpy as np
import xgboost as xgb
import cPickle as pk
from preprocess import find_delimiter, compute_nan_feat, add_na_bin_pca, add_cate_comb, add_num_comb
from util import get_params, log

log_file = open(__file__ + '_log', 'w')
#log_file = open('nohup.out', 'w')


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

train = compute_nan_feat(train)
test = compute_nan_feat(test)


################################################
# add combination feat
train_test = pd.concat([train, test])
train_test = add_cate_comb(train_test)
train_test = add_num_comb(train_test)
train = train_test[train_test.target.isnull() == False]
test = train_test[train_test.target.isnull() == True]
test.drop(['target'], axis=1)


'''
################################################
# inverse feature scaling
#train_id = train['ID'].values
#train_target = train['target'].values
num_vars = ['v1', 'v2', 'v4', 'v5', 'v6', 'v7', 'v9', 'v10', 'v11',
            'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20',
            'v21', 'v26', 'v27', 'v28', 'v29', 'v32', 'v33', 'v34', 'v35', 'v38',
            'v39', 'v40', 'v41', 'v42', 'v43', 'v44', 'v45', 'v48', 'v49', 'v50',
            'v55', 'v57', 'v58', 'v59', 'v60', 'v61', 'v62', 'v64', 'v65', 'v67',
            'v68', 'v69', 'v70', 'v72', 'v76', 'v77', 'v78', 'v80', 'v83', 'v84', 
            'v85', 'v86', 'v87', 'v88', 'v90', 'v93', 'v94', 'v96', 'v97', 'v98', 
            'v99', 'v100', 'v101', 'v102', 'v103', 'v104', 'v106', 'v111', 'v114',
            'v115', 'v120', 'v121', 'v122', 'v126', 'v127', 'v129', 'v130', 'v131']

vs = pd.concat([train, test])
for c in num_vars:
    if c not in train.columns:
        continue
    
    train.loc[train[c].round(5) == 0, c] = 0
    test.loc[test[c].round(5) == 0, c] = 0

    delimiter = find_delimiter(vs, c)
    train[c] *= 1/delimiter
    test[c] *= 1/delimiter
################################################
'''

train_feat = train.drop(train_columns_to_drop, axis=1) 
test_feat = test.drop(test_columns_to_drop, axis=1) 
factorize_category(train_feat)
train_feat.fillna(-1,inplace=True)
test_feat.fillna(-1,inplace=True)


train_feat_final = train_feat


xgtrain = xgb.DMatrix(train_feat_final, train['target'].values)


# grid search
params = get_params()
params["eta"] = 0.05

min_child_weight_list = [1]
subsample_list = [0.8, 1]
colsample_bytree_list = [0.4, 0.6, 0.8, 1]
max_depth_list = [6, 8, 10, 12]

#min_child_weight_list = [1, 5, 10]
#subsample_list = [0.6, 0.8, 1]
#colsample_bytree_list = [0.6, 0.8, 1]
#max_depth_list = [8, 10, 12]
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
    log(log_file, str(plst))
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
