import pandas as pd
import numpy as np
import cPickle as pk

columns_to_drop = ['v8', 'v23', 'v25', 'v36', 'v37', 'v46', 'v51', 'v53', 'v54', 'v63', 'v73', 'v81', 'v82', 'v89', 'v92', 'v95', 'v105', 'v107', 'v108', 'v109', 'v116', 'v117', 'v118', 'v119', 'v123', 'v124', 'v128']
cate_columns_to_drop = ['v107'] 

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

tt = pd.concat([train, test])
tt = tt.drop(cate_columns_to_drop, axis=1)

'''
for column in columns_to_drop:
    if train[column].dtype == 'object':
        print column
'''
to_comb = []
for column in tt.columns:
    if train[column].dtype != 'object' and column != 'ID' and column != 'target':
	to_comb.append(column)

nan_num_list = []
res_df = pd.DataFrame()
for i in xrange(len(to_comb)):
    for j in xrange(i+1, len(to_comb)):
        new_column = '%s_M_%s' % (to_comb[i], to_comb[j])
	print new_column
	vals = []
 	for k in xrange(tt.shape[0]):
	    val = tt[to_comb[i]].iloc[k] * tt[to_comb[j]].iloc[k]
	    vals.append(val)
	res_df[new_column] = pd.Series(vals, index=tt.index)
	nan_num_list.append(np.sum(res_df[new_column].isnull()))

print nan_num_list

with open('data/comb_num.pkl', 'w') as f:
    pk.dump(res_df, f)
