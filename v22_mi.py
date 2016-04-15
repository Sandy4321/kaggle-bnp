from util import calc_MI_cate_feat_target
from preprocess import compute_onehot_feat
import pandas as pd
import cPickle as pk

train = pd.read_csv('data/train.csv')

v22_onehot = compute_onehot_feat(train.v22)

df = pd.DataFrame(v22_onehot)

#with open('data/v22_onehot.pkl', 'w') as f:
#    pk.dump(df, f)

'''
for column in df.columns:
    mi = calc_MI_cate_feat_target(df[column], train.target, 10)
    print '%s\t%f' % (column, mi)
'''
