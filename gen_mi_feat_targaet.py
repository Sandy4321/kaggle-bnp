from util import calc_MI_feat_target
import pandas as pd

train = pd.read_csv('data/train.csv')

for column in train.columns:
    if column != 'target' and train[column].dtype != 'object':
        mi = calc_MI_feat_target(train[column], train.target, 10)
        print '%s\t%f' % (column, mi)
