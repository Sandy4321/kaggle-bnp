import pandas as pd
import numpy as np
import xgboost as xgb
from preprocess import factorize_category_both
from preprocess import find_delimiter, compute_nan_feat, add_na_bin_pca, add_cate_comb, add_cate_comb_reindex 
import sys


def get_params():
    params = {}
    params["objective"] = "binary:logistic"
    params["eta"] = 0.01
    params["min_child_weight"] = 1 
    params["subsample"] = 1   
    params["colsample_bytree"] = 0.6
    params["silent"] = 1
    params["max_depth"] = 10 
    plst = list(params.items())
    return plst

# index_file - index file for new sequence of train cate comb features
def run(train_file, test_file, predict_res_file, index_file=''):


    high_corr_columns = ['v8', 'v23', 'v25', 'v36', 'v37', 'v46', 'v51', 'v53', 'v54', 'v63', 'v73', 'v81', 'v82', 'v89', 'v92', 'v95', 'v105', 'v107', 'v108', 'v109', 'v116', 'v117', 'v118', 'v119', 'v123', 'v124', 'v128']
    train_columns_to_drop = ['ID', 'target'] + high_corr_columns
    print train_columns_to_drop
    test_columns_to_drop = ['ID'] + high_corr_columns


        
    xgb_num_rounds = 1450 
    #xgb_num_rounds = 2 
    num_classes = 2
    print 'load data'
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    # if using stacking
    if 'train' in test_file:
        test = test.drop(['target'], axis=1)

    train = compute_nan_feat(train)
    test = compute_nan_feat(test)

    ################################################
    # add category combination feat
    train_test = pd.concat([train, test])
    # if using stacking
    if 'train' in test_file:
        train_test = add_cate_comb_reindex(train_test, index_file)
    else:
        train_test = add_cate_comb(train_test)
    train = train_test[train_test.target.isnull() == False]
    # so test must drop target!!!!!!!
    test = train_test[train_test.target.isnull() == True]
    test = test.drop(['target'], axis=1)


    train_id = train['ID'].values
    train_target = train['target'].values

    train_feat = train.drop(train_columns_to_drop, axis=1)
    test_feat = test.drop(test_columns_to_drop, axis=1)

    factorize_category_both(train_feat, test_feat)
    train_feat.fillna(-1,inplace=True)
    test_feat.fillna(-1,inplace=True)
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
    preds_out.to_csv(predict_res_file)
    print 'finish'

if __name__ == '__main__':
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    predict_res_file = sys.argv[3]
    if len(sys.argv) == 5:
        index_file = sys.argv[4]
        run(train_file, test_file, predict_res_file, index_file)
    else:
        run(train_file, test_file, predict_res_file)

