from util import calc_MI_feat_target
import pandas as pd
import cPickle as pk
import numpy as np

train = pd.read_csv('data/train.csv')
with open('data/comb_num_1.pkl', 'r') as f:
    df = pk.load(f)

df.replace(np.inf, -999, inplace=True)

for column in df.columns:
    if column != 'target' and df[column].dtype != 'object':
        mi = calc_MI_feat_target(df[column].iloc[:train.shape[0]], train.target, 10)
        print '%s\t%f' % (column, mi)
