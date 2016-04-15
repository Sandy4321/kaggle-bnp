import numpy as np
import pandas as pd
from preprocess import compute_onehot_feat
import cPickle as pk

arr = np.loadtxt('v22_mi.data')
print arr
idx_arr = arr[:,0]
mi_arr = arr[:,1]

idx_arr = np.asarray(idx_arr, dtype=np.int)
mean_mi = np.mean(mi_arr)
max_mi = np.max(mi_arr)
sum_mi = np.sum(mi_arr[mi_arr > 0.00005])
print '%f\t%f\t%f' % (mean_mi, max_mi, sum_mi)

ind = mi_arr > 0.00005
print np.sum(ind)
new_idx_arr = idx_arr[ind]
print new_idx_arr


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
vs = pd.concat([train, test])
v22_onehot = compute_onehot_feat(vs.v22)
df = pd.DataFrame(v22_onehot)

new_df = pd.DataFrame()
for idx in new_idx_arr:
    col_name = 'v22_%d' % idx 
    new_df[col_name] = pd.Series(v22_onehot[:,idx], index=vs.index)

print new_df
with open('data/v22_onehot0.00005.pkl', 'w') as f:
    pk.dump(new_df, f)



