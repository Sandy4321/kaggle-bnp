import pandas as pd
import numpy as np
import cPickle as pk
from sklearn.preprocessing import OneHotEncoder

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

def find_delimiter(df, col):
    """
    Function that trying to find an approximate delimiter used for scaling.
    So we can undo the feature scaling.
    """
    vals = df[col].dropna().sort_values().round(8)
    #print vals
    vals = pd.rolling_apply(vals, 2, lambda x: x[1] - x[0])
    vals = vals[vals > 0.000001]
    #print vals.values
    #print vals.value_counts()
    return vals.value_counts().idxmax() 

def compute_nan_feat(df):
    sum_list = []
    var_list = []
    for dummy_i in xrange(df.shape[0]):
        if dummy_i % 100 == 0:
            print 'row = %d' % dummy_i
        row = df.iloc[dummy_i, :]
        isnull = row.isnull()
        s = np.sum(isnull)
        v = np.var(isnull)
        sum_list.append(s)
        var_list.append(v)

    df['sum_nan'] = pd.Series(sum_list, index=df.index)
    # df['var_nan'] = pd.Series(var_list, index=df.index)

    return df



def add_na_bin_pca(train_test):
    with open('data/na_bin_tsvd_2.pkl', 'r') as f:
    	na_bin_pca = pk.load(f)
    	na_df = pd.DataFrame(na_bin_pca)

    for c in na_df.columns:
    	train_test[c] = na_df[c]

    return train_test

 
def add_na_bin(train_test):
    with open('data/na_bin.pkl', 'r') as f:
    	na_bin = pk.load(f)
    	na_df = pd.DataFrame(na_bin)

    na_df.columns = range(len(na_df.columns))
    for c in na_df.columns:
    	train_test[c] = na_df[c]

    return train_test

def add_cate_comb(train_test):
    #with open('data/comb_cate.pkl', 'r') as f:
    with open('data/comb_cate_v22_v56.pkl', 'r') as f:
    	df = pk.load(f)

    for c in df.columns:
    	train_test[c] = df[c]

    return train_test

def add_num_comb(train_test):
    #with open('data/comb_cate.pkl', 'r') as f:
    with open('data/comb_num_1.pkl', 'r') as f:
    	df = pk.load(f)

    for c in df.columns:
    	train_test[c] = df[c]

    return train_test

def add_v22_onehot(train_test):
    #with open('data/comb_cate.pkl', 'r') as f:
    with open('data/v22_onehot0.00005.pkl', 'r') as f:
    	df = pk.load(f)

    for c in df.columns:
    	train_test[c] = df[c]

    return train_test

def compute_onehot_feat(column):
    df_onehot = pd.DataFrame(columns=[column.name])
    df_onehot[column.name] = pd.factorize(column, na_sentinel=nan)[0]

    enc = OneHotEncoder()
    onehot_feat = enc.fit_transform(df_onehot)

    return onehot_feat.toarray() 
