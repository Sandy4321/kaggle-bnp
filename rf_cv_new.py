# Based on : https://www.kaggle.com/chabir/bnp-paribas-cardif-claims-management/extratreesclassifier-score-0-45-v5/code
import pandas as pd

import numpy as np
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss, make_scorer
from sklearn.grid_search import GridSearchCV
import random; random.seed(2016)


def flog_loss(ground_truth, predictions):
    flog_loss_ = -log_loss(ground_truth, predictions) #, eps=1e-15, normalize=True, sample_weight=None)
    return flog_loss_
logloss_metric  = make_scorer(flog_loss, greater_is_better=False)




def find_delimiter(df, col):
    """
    Function that trying to find an approximate delimiter used for scaling.
    So we can undo the feature scaling.
    """
    vals = df[col].dropna().sort_values().round(8)
    vals = pd.rolling_apply(vals, 2, lambda x: x[1] - x[0])
    vals = vals[vals > 0.000001]
    return vals.value_counts().idxmax() 

print('Load data...')
train = pd.read_csv("./data/train.csv")
target = train['target'].values
train = train.drop(['ID','target','v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)
test = pd.read_csv("./data/test.csv")
id_test = test['ID'].values
test = test.drop(['ID','v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)

print('Clearing...')
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

for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems()):
    if train_series.dtype == 'O':
        #for objects: factorize
        train[train_name], tmp_indexer = pd.factorize(train[train_name])
        test[test_name] = tmp_indexer.get_indexer(test[test_name])
        #but now we have -1 values (NaN)
    else:
        #for int or float: fill NaN
        tmp_len = len(train[train_series.isnull()])
        if tmp_len>0:
            #print "mean", train_series.mean()
            train.loc[train_series.isnull(), train_name] = -999 
        #and Test
        tmp_len = len(test[test_series.isnull()])
        if tmp_len>0:
            test.loc[test_series.isnull(), test_name] = -999

#X_train, X_valid, y_train, y_valid = train_test_split(
#    train, target, test_size=0.2, random_state=42)
X_train, y_train = (train, target)

print('Training...')
rfc = RandomForestClassifier(n_estimators=1000,max_features= 50,criterion= 'gini',min_samples_split= 4,
                            max_depth= 35, min_samples_leaf= 2, n_jobs = -1)      

clf = {'rfc':rfc}
rfc_param_grid = [
  {'n_estimators': [1, 10], 'max_features': [25], 'criterion': ['gini'], 'max_depth': [25], 'min_samples_leaf': [2]},
 ]
rfc_param_grid = [
  {'n_estimators': [500, 1000, 2000, 2500], 'max_features': [25, 50], 'criterion': ['gini', 'entropy'], 'max_depth': [25, 35, 50], 'min_samples_leaf': [2, 4]},
 ]
for c in clf:
    model = GridSearchCV(estimator=clf[c], param_grid=rfc_param_grid, n_jobs =-1, cv=2, verbose=1, scoring=logloss_metric)
    model.fit(X_train,y_train) 
    print("Best parameters set found on development set:")
    print()
    print(model.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in model.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))



'''
print('Predict...')
valid_pred = rfc.predict_proba(X_valid)

print log_loss(y_valid, valid_pred)
'''
