import pandas as pd
import numpy as np

nan = 100000

def factorize_category_both(train, test):
    for column in train.columns:
        if train[column].dtype == 'object':
            train[column], tmp_indexer = pd.factorize(train[column], na_sentinel=nan)
            test[column] = tmp_indexer.get_indexer(test[column])

def factorize_category(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = pd.factorize(df[column], na_sentinel=nan)[0]

def category_to_ratio_bayes(column, count_dict, nan_ratio):
    ratio_list = []
    for val in column:
        if val is np.nan or not count_dict.has_key(val):
            ratio_list.append(nan_ratio)
        else:
            ratio_list.append(count_dict[val])

    return np.array(ratio_list)

# input - Series
def bayes_encoding(column, target):
    cate_list = column.unique()
    count_dict = {}
    nan_ratio = 0
    for cate in cate_list:
	#if cmp(cate,'NAN') == 0:
	# if cate == 'NAN' or cate == 'nan':
        if cate is np.nan:
            nan_idx = np.array([val is np.nan for val in column])
            cate_target = target[nan_idx]
            nan_ratio = np.sum(cate_target == 1)/float(cate_target.shape[0]) 
            continue

        cate_target = target[column == cate]
        ratio = np.sum(cate_target == 1)/float(cate_target.shape[0]) 
        #print '%s:%d:%f' % (cate, cate_target.shape[0], ratio)
        # print cate_target 
        count_dict[cate] = ratio
    
    ratio_arr = category_to_ratio_bayes(column, count_dict, nan_ratio)
    
    return ratio_arr, count_dict, nan_ratio


    
