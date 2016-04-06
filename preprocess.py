import pandas as pd
import numpy as np

def factorize_category(train, test):
    for column in train.columns:
        if train[column].dtype == 'object':
            train[column], tmp_indexer = pd.factorize(train[column])
            test[column] = tmp_indexer.get_indexer(test[column])

# input - Series
def bayes_encoding(column, target):
    cate_list = column.unique()
    count_dict = {}
    for cate in cate_list:
	#if cmp(cate,'NAN') == 0:
	# if cate == 'NAN' or cate == 'nan':
        if cate == np.nan:
            continue
        print cate
        cate_target = target[column == cate]
        ratio = (cate_target == 1).shape[0]/float(cate_target.shape[0]) 
        print '%s:%d:%f' % (cate, cate_target.shape[0], ratio)
        # print cate_target 
    
