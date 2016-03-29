import pandas as pd
import numpy as np
import xgboost as xgb
from preprocess import factorize_category 

def get_params():
    params = {}
    params["objective"] = "binary:logistic"
    params["eta"] = 0.1
    params["min_child_weight"] = 20 
    params["subsample"] = 1
    params["colsample_bytree"] = 0.2
    params["silent"] = 0
    params["max_depth"] = 8 
    plst = list(params.items())
    return plst

'''
def factorize_category(train, test):
    for column in train.columns:
        if train[column].dtype == 'object':
            train[column], tmp_indexer = pd.factorize(train[column])
	    test[column] = tmp_indexer.get_indexer(test[column])
'''
        
columns_to_drop = ['ID', 'target']
xgb_num_rounds = 10 
num_classes = 2
print 'load data'
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
submission = pd.read_csv('./data/sample_submission.csv')

train_id = train['ID'].values
train_target = train['target'].values

train_feat = train.drop(columns_to_drop, axis=1)
test_feat = test.drop('ID', axis=1)

factorize_category(train_feat, test_feat)
#factorize_category(test_feat)
train_feat.fillna(-1,inplace=True)
test_feat.fillna(-1,inplace=True)
#all_data = train.append(test)
#all_data.fillna(-1, inplace=True)
#train = all_data[all_data['Response']>0].copy()
#test = all_data[all_data['Response']<1].copy()
xgtrain = xgb.DMatrix(train_feat, train['target'].values)
xgtest = xgb.DMatrix(test_feat)

# get the parameters for xgboost
plst = get_params()
print(plst)

# train model
model = xgb.train(plst, xgtrain, xgb_num_rounds)
test_preds = model.predict(xgtest, ntree_limit=model.best_iteration)

preds_out = pd.DataFrame({"ID": test['ID'].values, "PredictedProb": test_preds})
preds_out = preds_out.set_index('ID')
preds_out.to_csv('xgb_fix_factor.csv')
print 'finish'
