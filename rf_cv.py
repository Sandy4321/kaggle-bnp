import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

def get_params():
    params = {}
    params["objective"] = "binary:logistic"
    params["eta"] = 0.05
    params["min_child_weight"] = 20 
    params["subsample"] = 1.0 
    params["colsample_bytree"] = 0.2
    params["silent"] = 1
    params["max_depth"] = 8 
    plst = list(params.items())
    return plst

def factorize_category(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = pd.factorize(df[column])[0]
            
        
columns_to_drop = ['ID', 'target']
xgb_num_rounds = 1000000
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
train_feat.fillna(-1,inplace=True)
#all_data = train.append(test)
#all_data.fillna(-1, inplace=True)
#train = all_data[all_data['Response']>0].copy()
#test = all_data[all_data['Response']<1].copy()
valid_prob = 0.2
ind = [np.random.binomial(1,valid_prob) for i in xrange(len(train_feat))]
ind = np.array(ind)
train_feat_final = train_feat[ind == 0]
valid_feat_final = train_feat[ind == 1]

xgtrain = xgb.DMatrix(train_feat_final, train[ind==0]['target'].values)
xgvalid = xgb.DMatrix(valid_feat_final, train[ind==1]['target'].values)
xgtest = xgb.DMatrix(test_feat)

# get the parameters for xgboost
plst = get_params()
print(plst)

# train model
#model = xgb.train(plst, xgtrain, xgb_num_rounds)
clf = RandomForestClassifier(n_estimators=100)
clf = clf.fit(train_feat_final, train[ind==0]['target'].values)
valid_target = train[ind==1]['target'].values
valid_preds = clf.predict_proba(valid_feat_final)
print valid_preds[:,1]
valid_preds = valid_preds[:,1]

res = 0
for preds, target in zip(valid_preds, valid_target):
    preds = np.min(preds, 1 - 10**(-3))
    preds = np.max(preds, 10**(-3))
    if preds == 1.0:
        preds = 0.99
    if preds == 0.0:
        preds = 0.01
    #print preds
    #print np.log(preds)
    #print np.log(1-preds)
    res += target * np.log(preds) + (1 - target) * np.log(1 - preds)

res /= len(valid_target)
res = -res
print res

