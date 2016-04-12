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
'''
to_comb = []
for column in tt.columns:
    if train[column].dtype != 'object' and column != 'ID' and column != 'target':
	to_comb.append(column)
'''

to_comb = ['v50', 'v12', 'v114', 'v34', 'v40', 'v10', 'v21', 'v14', 'v62', 'v129', 'v72', 'v82', 'v120', 'v36', 'v6', 'v98', 'v124', 'v99', 'v57', 'v28', 'v88', 'v115', 'v68', 'v69', 'v119', 'v81', 'v16', 'v70', 'v78', 'v5']


res_df = pd.DataFrame()
for i in xrange(len(to_comb)):
    for j in xrange(i+1, len(to_comb)):
        mul_column = '%s_M_%s' % (to_comb[i], to_comb[j])
        div_column = '%s_D_%s' % (to_comb[i], to_comb[j])
	print mul_column
	mul_vals = []
	div_vals = []
 	for k in xrange(tt.shape[0]):
	    mul_val = tt[to_comb[i]].iloc[k] * tt[to_comb[j]].iloc[k]
	    mul_val = np.log(1+mul_val)
	    div_val = tt[to_comb[i]].iloc[k] / tt[to_comb[j]].iloc[k]
	    div_val = np.log(1+div_val)

	    mul_vals.append(mul_val)
	    div_vals.append(div_val)
	res_df[mul_column] = pd.Series(mul_vals, index=tt.index)
	res_df[div_column] = pd.Series(div_vals, index=tt.index)


with open('data/comb_num_1.pkl', 'w') as f:
    pk.dump(res_df, f)
