import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from preprocess import factorize_category_both
import cPickle as pk

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

factorize_category_both(train_feat, test_feat)
train_feat.fillna(-1,inplace=True)
test_feat.fillna(-1,inplace=True)
#all_data = train.append(test)
#all_data.fillna(-1, inplace=True)
#train = all_data[all_data['Response']>0].copy()
#test = all_data[all_data['Response']<1].copy()


# get the parameters for xgboost
plst = get_params()
print(plst)

# train model
#model = xgb.train(plst, xgtrain, xgb_num_rounds)
clf = RandomForestClassifier(n_estimators=200)
clf = clf.fit(train_feat, train['target'].values)
feat_imp = clf.feature_importances_
feat_imp = np.array(feat_imp)
np.savetxt('feat_imp.txt',feat_imp)
#model = SelectFromModel(clf, prefit=True)
#model_file = open('model/rf_feat_select.pickle', 'w')

'''
model_file = open('model/rf_200.pickle', 'w')
pk.dump(clf, model_file)
'''

