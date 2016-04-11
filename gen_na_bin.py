import pandas as pd
import numpy as np
import cPickle as pk

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
vs = pd.concat([train, test])


bin_list = []
for dummy_i in xrange(vs.shape[0]):
    if dummy_i % 100 == 0:
        print dummy_i
    row = vs.iloc[dummy_i, :]
    isnull = row.isnull()
    s = np.sum(isnull)
    v = np.var(isnull)
    bin = isnull.apply(lambda x: 1 if x else 0)
    bin_list.append(bin)
    # print bin
    #print s
    #print v

na_binary = pd.DataFrame(bin_list)

with open('data/na_bin.pkl', 'w') as f:
    pk.dump(na_binary, f)
