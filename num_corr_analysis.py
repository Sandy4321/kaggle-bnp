import pandas as pd
import cPickle as pk
from scipy.stats.stats import pearsonr

f = open('data/comb_num_0.01.pkl', 'r')

data = pk.load(f)
data.fillna(-1, inplace=True)
print data


'''
cate_col = data.columns 

res = {}
for i in xrange(len(cate_col)):
    for j in xrange(i+1, len(cate_col)):
        p = pearsonr(data[cate_col[i]], data[cate_col[j]]) 
        print '%s_%s: %f' % (cate_col[i], cate_col[j], p[0])
'''


