import pandas as pd
import cPickle as pk
from scipy.stats.stats import pearsonr
from util import calc_MI
import numpy as np

with open('data/comb_num_1.pkl', 'r') as f:
    df = pk.load(f)

cols = []
for column in df.columns:
    cols.append(column)

df.fillna(-1, inplace=True)
# must have np.inf replaced
df.replace(np.inf, -1, inplace=True)

res = {}
for i in xrange(len(cols)):
    for j in xrange(i+1, len(cols)):
        p = calc_MI(df[cols[i]], df[cols[j]], 20) 
        print '%s_%s: %f' % (cols[i], cols[j], p)

