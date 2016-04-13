import pandas as pd
import numpy as np
import xgboost as xgb
import cPickle as pk
def get_params():
    params = {}
    params["objective"] = "binary:logistic"
    params["eta"] = 0.01
    params["min_child_weight"] = 10 
    params["subsample"] = 1.0 
    params["colsample_bytree"] = 0.2
    params["silent"] = 1
    params["max_depth"] = 6 
    plst = list(params.items())
    return plst

def factorize_category(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = pd.factorize(df[column])[0]
            
        
columns_to_drop = ['ID', 'target']
xgb_num_rounds = 1000000
#xgb_num_rounds = 100
num_classes = 2
print 'load data'
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
submission = pd.read_csv('./data/sample_submission.csv')

#train_id = train['ID'].values
#train_target = train['target'].values

train_feat = train.drop(columns_to_drop, axis=1)
test_feat = test.drop('ID', axis=1)

factorize_category(train_feat)
factorize_category(test_feat)
train_feat.fillna(-1,inplace=True)
test_feat.fillna(-1,inplace=True)
#all_data = train.append(test)
#all_data.fillna(-1, inplace=True)
#train = all_data[all_data['Response']>0].copy()
#test = all_data[all_data['Response']<1].copy()


train_feat_final = train_feat
print 'loading feature selection model'
f = open('model/rf_feat_select.pickle')
model = pk.load(f)

print 'transforming data'
train_feat_final = model.transform(train_feat_final)
#valid_feat_final = model.transform(valid_feat_final)
print 'shape of feature after feature selection'
print train_feat_final.shape

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
