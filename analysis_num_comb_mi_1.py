import numpy as np
import pandas as pd
from preprocess import compute_onehot_feat
import cPickle as pk

f = open('num_comb_mi.data', 'r')
lines = f.readlines()
mi_list = []
name_list = []
for line in lines:
    feat_name, mi = line.rstrip().split('\t')
    name_list.append(feat_name)
    mi_list.append(float(mi))


mi_arr = np.array(mi_list)
name_arr = np.array(name_list)
print mi_arr
print name_arr

ind = mi_arr > 0.01

#idx_arr = np.asarray(idx_arr, dtype=np.int)
mean_mi = np.mean(mi_arr)
max_mi = np.max(mi_arr)
sum_mi = np.sum(mi_arr[ind])
print '%f\t%f\t%f' % (mean_mi, max_mi, sum_mi)

print np.sum(ind)
print name_arr[ind]


'''
with open('data/comb_num_1.pkl', 'r') as f:
    df = pk.load(f)

#df.replace(np.inf, -1, inplace=True)


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
vs = pd.concat([train, test])

new_df = pd.DataFrame()
for name in name_arr[ind]:
    new_df[name] = df[name]

print new_df
with open('data/comb_num_0.01.pkl', 'w') as f:
    pk.dump(new_df, f)

'''
