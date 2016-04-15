from util import calc_MI_cate_feat_target
import pandas as pd

train = pd.read_csv('data/train.csv')

for column in train.columns:
    if train[column].dtype == 'object' and column != 'ID' and column != 'target':
        mi = calc_MI_cate_feat_target(train[column], train.target, 10)
        print '%s\t%f' % (column, mi)
