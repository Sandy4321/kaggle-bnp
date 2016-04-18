import pandas as pd
import numpy as np
import os
import random

rnd=12
random.seed(rnd)


def gen_run_cmd(run_script, train_file, test_file, predict_file, index_file=''):
    cmd = 'python {run_script} {train_file} {test_file} {predict_file} {index_file}'.format(run_script = run_script,
			train_file = train_file,
			test_file = test_file,
			predict_file = predict_file,
			index_file = index_file)

    return cmd

train_file = 'data/train.csv'
test_file = 'data/test.csv'
train_1_file = 'stack_data/train_1.csv'
train_2_file = 'stack_data/train_2.csv'

ext_predict_1_file = 'stack_data/ext_predict_1.csv'
ext_predict_2_file = 'stack_data/ext_predict_2.csv'
ext_predict_full_file = 'stack_data/ext_predict_full.csv'

nn_predict_1_file = 'stack_data/nn_predict_1.csv'
nn_predict_2_file = 'stack_data/nn_predict_2.csv'
nn_predict_full_file = 'stack_data/nn_predict_full.csv'

xgb_predict_1_file = 'stack_data/xgb_predict_1.csv'
xgb_predict_2_file = 'stack_data/xgb_predict_2.csv'
xgb_predict_full_file = 'stack_data/xgb_predict_full.csv'

train_1_2_idx_file = 'stack_data/train_1_2.txt'
train_2_1_idx_file = 'stack_data/train_2_1.txt'


'''
#######################################################
# prepare data
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
split_prob = 0.5
ind = [np.random.binomial(1,split_prob) for i in xrange(train.shape[0])]
ind = np.array(ind)
train_1 = train[ind == 0]
train_2 = train[ind == 1]
print '%d %d' % (train_1.shape[0], train_2.shape[0])

train_1.to_csv(train_1_file, index=False)
train_2.to_csv(train_2_file, index=False)

ori_ind = np.arange(train.shape[0])
train_1_idx = ori_ind[ind == 0]
train_2_idx = ori_ind[ind == 1]
train_1_train_2_idx = np.hstack([train_1_idx, train_2_idx])
train_2_train_1_idx = np.hstack([train_2_idx, train_1_idx])
np.savetxt(train_1_2_idx_file, train_1_train_2_idx, fmt='%d') 
np.savetxt(train_2_1_idx_file, train_2_train_1_idx, fmt='%d') 
#######################################################
'''

'''
# ext model train feature
cmd = gen_run_cmd('ext_func.py', train_1_file, train_2_file, ext_predict_1_file) 
print cmd
os.system(cmd)
cmd = gen_run_cmd('ext_func.py', train_2_file, train_1_file, ext_predict_2_file) 
print cmd
os.system(cmd)
cmd = gen_run_cmd('ext_func.py', train_file, test_file, ext_predict_full_file) 
print cmd
os.system(cmd)


# nn model train feature
cmd = gen_run_cmd('nn_func.py', train_1_file, train_2_file, nn_predict_1_file) 
print cmd
os.system(cmd)
cmd = gen_run_cmd('nn_func.py', train_2_file, train_1_file, nn_predict_2_file) 
print cmd
os.system(cmd)
cmd = gen_run_cmd('nn_func.py', train_file, test_file, nn_predict_full_file) 
print cmd
os.system(cmd)
'''

# xgb cate comb model train feature
cmd = gen_run_cmd('xgb_cate_comb_func.py', train_1_file, train_2_file, xgb_predict_1_file, train_1_2_idx_file) 
print cmd
os.system(cmd)
cmd = gen_run_cmd('xgb_cate_comb_func.py', train_2_file, train_1_file, xgb_predict_2_file, train_2_1_idx_file) 
print cmd
os.system(cmd)
cmd = gen_run_cmd('xgb_cate_comb_func.py', train_file, test_file, xgb_predict_full_file) 
print cmd
#os.system(cmd)
