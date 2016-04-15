from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import xgboost as xgb
import cPickle as pk
import gc
import logging
logging.basicConfig(filename='xgb_cv_onehot.log',level=logging.INFO)
logger = logging.getLogger('xgb_cv_onehot')
import time

nan = 100000

def get_params_list():
    params = {}
    params["objective"] = "binary:logistic"
    params["eval_metric"] = "logloss"
    params["eta"] = 0.1
    params["min_child_weight"] = 1 
    params["subsample"] = 1 
    params["colsample_bytree"] = 0.3 
    params["silent"] = 1
    params["max_depth"] = 7
    plst = list(params.items())
    return plst

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

def factorize_category(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = pd.factorize(df[column])[0]
            
def factorize_category_onehot(df):
    str_columns = []
    for column in df.columns:
        if cmp(column, 'v22') == 0:
     	    continue
        if df[column].dtype == 'object':
	    str_columns.append(column)
    
    df_onehot = pd.DataFrame(columns=str_columns)
    for column in str_columns:
        df_onehot[column] = pd.factorize(df[column], na_sentinel=nan)[0]


    enc = OneHotEncoder()
    onehot_feat = enc.fit_transform(df_onehot)
    # print onehot_feat.shape
    # print onehot_feat.toarray().shape
    del df_onehot
    gc.collect()

    df = df.drop(str_columns, axis=1)

    # return onehot_feat.toarray()
    onehot_feat = onehot_feat.toarray()
    #df_res['onehot'] = onehot_feat
    return onehot_feat, df

high_corr_columns = ['v8', 'v23', 'v25', 'v36', 'v37', 'v46', 'v51', 'v53', 'v54', 'v63', 'v73', 'v81', 'v82', 'v89', 'v92', 'v95', 'v105', 'v107', 'v108', 'v109', 'v116', 'v117', 'v118', 'v119', 'v123', 'v124', 'v128']

train_columns_to_drop = ['ID', 'target'] + high_corr_columns
print train_columns_to_drop  
test_columns_to_drop = ['ID'] + high_corr_columns

print 'load data'
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

train_drop = train.drop(train_columns_to_drop, axis=1) 
test_drop = test.drop(test_columns_to_drop, axis=1) 

onehot_feat, train_drop  = factorize_category_onehot(train_drop)
train_drop['v22'] = pd.factorize(train_drop.v22, na_sentinel=nan)[0]

'''
# bayes encoding v22
with open('data/v22_bayes.pkl', 'r') as f:
    train_drop['v22'] = pk.load(f)
'''

train_drop.fillna(nan,inplace=True)

#train_columns_to_drop = ['ID', 'target', 'v107', 'v110']
#test_columns_to_drop = ['ID', 'v107', 'v110']
xgb_num_rounds = 1000000
#xgb_num_rounds = 100
num_classes = 2

#train_id = train['ID'].values
#train_target = train['target'].values



train_feat_final = np.hstack([train_drop.values, onehot_feat])

#valid_feat_final = model.transform(valid_feat_final)

xgtrain = xgb.DMatrix(train_feat_final, train['target'].values)

# params to be searched 
params = get_params()
min_child_weight_list = [1] 
subsample_list = [1]
colsample_bytree_list = [0.6]
max_depth_list = [10]
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

# train model
#model = xgb.train(plst, xgtrain, xgb_num_rounds)
#cv_results = xgb.cv(plst, xgtrain, num_boost_round=xgb_num_rounds,
#	nfold=5, metrics='logloss', verbose_eval=True)
for plst in params_list:
    logger.info(str(plst) + time.strftime("%Y-%m-%d %H:%M:%S"))
    cv_results = xgb.cv(plst, xgtrain, num_boost_round=xgb_num_rounds,
	    nfold=5, metrics='logloss', verbose_eval=True, early_stopping_rounds=10)
