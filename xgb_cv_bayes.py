import pandas as pd
import numpy as np
import xgboost as xgb
from preprocess import factorize_category_both, category_to_ratio_bayes 
from preprocess import bayes_encoding, nan
from util import logloss

def get_params():
    params = {}
    params["objective"] = "binary:logistic"
    params["eta"] = 0.1
    params["min_child_weight"] = 1 
    params["subsample"] = 0.9
    params["colsample_bytree"] = 0.9
    params["silent"] = 0
    params["max_depth"] = 10 
    plst = list(params.items())
    return plst

'''
def factorize_category(train, test):
    for column in train.columns:
        if train[column].dtype == 'object':
            train[column], tmp_indexer = pd.factorize(train[column])
	    test[column] = tmp_indexer.get_indexer(test[column])
'''
high_corr_columns = ['v8', 'v23', 'v25', 'v36', 'v37', 'v46', 'v51', 'v53', 'v54', 'v63', 'v73', 'v81', 'v82', 'v89', 'v92', 'v95', 'v105', 'v107', 'v108', 'v109', 'v116', 'v117', 'v118', 'v119', 'v123', 'v124', 'v128']
train_columns_to_drop = ['ID', 'target'] + high_corr_columns
print train_columns_to_drop
test_columns_to_drop = ['ID'] + high_corr_columns

xgb_num_rounds = 5 
num_classes = 2
print 'load data'
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
submission = pd.read_csv('./data/sample_submission.csv')

train_id = train['ID'].values
train_target = train['target'].values

train_feat = train.drop(train_columns_to_drop, axis=1)
test_feat = test.drop(test_columns_to_drop, axis=1)

valid_prob = 0.2
ind = [np.random.binomial(1,valid_prob) for i in xrange(len(train_feat))]
ind = np.array(ind)
train_feat_final = train_feat[ind == 0]
valid_feat_final = train_feat[ind == 1]

# change v22 to bayes encoding
res, v22_count_dict, v22_nan_ratio = bayes_encoding(train_feat_final.v22, train[ind==0].target)
#with open('data/v22_bayes_res.pkl', 'r') as f:
#   v22_count_dict, v22_nan_ratio = pk.load(f)
train_feat_final.v22 = category_to_ratio_bayes(train_feat_final.v22, v22_count_dict, v22_nan_ratio)
valid_feat_final.v22 = category_to_ratio_bayes(valid_feat_final.v22, v22_count_dict, v22_nan_ratio)
factorize_category_both(train_feat_final, valid_feat_final)
train_feat_final.fillna(nan,inplace=True)
valid_feat_final.fillna(nan,inplace=True)

xgtrain = xgb.DMatrix(train_feat_final, train[ind==0]['target'].values)
xgvalid = xgb.DMatrix(valid_feat_final, train[ind==1]['target'].values)

# get the parameters for xgboost
plst = get_params()
print(plst)

# train model
model = xgb.train(plst, xgtrain, xgb_num_rounds)
valid_preds = model.predict(xgvalid, ntree_limit=model.best_iteration)
print valid_preds


print logloss(valid_preds, train[ind==1]['target'].values)



'''
test_preds = model.predict(xgtest, ntree_limit=model.best_iteration)

preds_out = pd.DataFrame({"ID": test['ID'].values, "PredictedProb": test_preds})
preds_out = preds_out.set_index('ID')
preds_out.to_csv('xgb_drop_feat.csv')
print 'finish'
'''
