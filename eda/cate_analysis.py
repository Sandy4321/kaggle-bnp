import pandas as pd
from scipy.stats.stats import pearsonr

train = pd.read_csv('../data/train.csv')

cate_col = []
for column in train.columns:
    if train[column].dtype == 'object':
        train[column] = pd.factorize(train[column])[0]
        cate_col.append(column)
res = {}
for i in xrange(len(cate_col)):
    for j in xrange(i+1, len(cate_col)):
        p = pearsonr(train[cate_col[i]], train[cate_col[j]]) 
        print '%s_%s: %f' % (cate_col[i], cate_col[j], p[0])



