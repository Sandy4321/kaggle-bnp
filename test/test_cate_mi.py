from util import calc_MI_cate_feat_target
import pandas as pd

train = pd.read_csv('data/train.csv')

column = 'v22'
train[column], tmp_indexer = pd.factorize(train[column], na_sentinel=-1)

mi = calc_MI_cate_feat_target(train[column], train.target, 10)
print '%s\t%f' % (column, mi)
