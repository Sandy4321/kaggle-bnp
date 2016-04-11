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
for column in tt.columns:
    if train[column].dtype == 'object':
        print column
'''

# drop v22,v56
#to_comp = ['v110', 'v112', 'v113', 'v125',  'v24', 'v3', 'v30', 'v31', 'v47', 'v52', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91']
to_comp = ['v22', 'v56', 'v110', 'v112', 'v113', 'v125',  'v24', 'v3', 'v30', 'v31', 'v47', 'v52', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91']

dist_num_list = []
res_df = pd.DataFrame()
for i in xrange(len(to_comp)):
    for j in xrange(i+1, len(to_comp)):
#for i in xrange(2):
#    for j in xrange(i+1, 2):
        new_column = '%s_%s' % (to_comp[i], to_comp[j])
	print new_column
	vals = []
 	for k in xrange(tt.shape[0]):
	    val = '%s_%s' % (tt[to_comp[i]].iloc[k], tt[to_comp[j]].iloc[k])
	    vals.append(val)
	res_df[new_column] = pd.Series(vals, index=tt.index)
	dist_num = np.unique(res_df[new_column]).shape[0]
	dist_num_list.append(dist_num)
print dist_num_list

with open('data/comb_cate_v22_v56.pkl', 'w') as f:
    pk.dump(res_df, f)
