import pandas as pd
import numpy as np
import xgboost as xgb
import cPickle as pk

def get_params():
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

#train_id = train['ID'].values
#train_target = train['target'].values

train_feat = train.drop(train_columns_to_drop, axis=1) 
test_feat = test.drop(test_columns_to_drop, axis=1) 
factorize_category(train_feat)
factorize_category(test_feat)
train_feat.fillna(-1,inplace=True)
test_feat.fillna(-1,inplace=True)
#all_data = train.append(test)
#all_data.fillna(-1, inplace=True)
#train = all_data[all_data['Response']>0].copy()
#test = all_data[all_data['Response']<1].copy()


train_feat_final = train_feat

#valid_feat_final = model.transform(valid_feat_final)

xgtrain = xgb.DMatrix(train_feat_final, train['target'].values)

# get the parameters for xgboost
plst = get_params()
print(plst)

# train model
#model = xgb.train(plst, xgtrain, xgb_num_rounds)
cv_results = xgb.cv(plst, xgtrain, num_boost_round=xgb_num_rounds,
		nfold=5, metrics='logloss', show_progress=True)


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
