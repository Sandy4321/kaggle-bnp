import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr

train = pd.read_csv('../data/train.csv')


dist_num_list = []
for column in train.columns:
    if train[column].dtype == 'object':
        #train[column] = pd.factorize(train[column])[0]
        #cate_col.append(column)
	na_num = np.sum(train[column].isnull())
	dist_num = np.unique(train[column]).shape[0]
	dist_num_list.append(dist_num)
	print '%s:%d:%d' % (column, dist_num, na_num)
	'''
    	print np.max(train[column])
        print np.min(train[column])
 	print 'iiiiiiiiiiiiiiiiiiiiiiiiii'
	'''

print np.sum(dist_num_list) - np.max(dist_num_list)


'''
res = {}
for i in xrange(len(cate_col)):
    for j in xrange(i+1, len(cate_col)):
        p = pearsonr(train[cate_col[i]], train[cate_col[j]]) 
        print '%s_%s: %f' % (cate_col[i], cate_col[j], p[0])
'''



