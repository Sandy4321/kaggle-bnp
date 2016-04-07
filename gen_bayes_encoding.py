import pandas as pd
import numpy as np
import cPickle as pk
from preprocess import bayes_encoding

train = pd.read_csv('./data/train.csv')

res, count_dict, nan_ratio = bayes_encoding(train.v22, train.target)
with open('data/v22_bayes.pkl', 'w') as f:
    pk.dump(res, f)

with open('data/bayes_res.pkl', 'w') as f:
    pk.dump((count_dict, nan_ratio), f)

train.v22 = res
print train.v22

