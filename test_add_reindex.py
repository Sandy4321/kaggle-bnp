from preprocess import add_cate_comb_reindex
import numpy as np
import pandas as pd


train_1_2_idx_file = 'stack_data/train_1_2.txt'
idx = np.loadtxt(train_1_2_idx_file)
idx = np.asarray(idx, np.int)

train_1 = pd.read_csv('stack_data/train_1.csv')
train_2 = pd.read_csv('stack_data/train_2.csv')
train_test = pd.concat([train_1, train_2])

add_cate_comb_reindex(train_test, train_1_2_idx_file) 


