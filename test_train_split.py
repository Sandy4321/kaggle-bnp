import numpy as np
import pandas as pd

train_2_1_idx = np.loadtxt('stack_data/train_2_1.txt')
train_2_1_idx = np.asarray(train_2_1_idx, np.int) 
train_1_2_idx = np.loadtxt('stack_data/train_1_2.txt')
train_1_2_idx = np.asarray(train_1_2_idx, np.int) 

print train_2_1_idx 

train = pd.read_csv('data/train.csv').fillna(-1)
train_1 = pd.read_csv('stack_data/train_1.csv').fillna(-1)
train_2 = pd.read_csv('stack_data/train_2.csv').fillna(-1)

train_1_2 = pd.concat([train_1, train_2])
train_2_1 = pd.concat([train_2, train_1])

print train.iloc[train_2_1_idx, :].values
print train_2_1.values

print (train.iloc[train_2_1_idx, :].values == train_2_1.values)
print (train.iloc[train_1_2_idx, :].values == train_1_2.values)

print np.sum(train.iloc[train_2_1_idx, :].values == train_2_1.values)
print np.sum(train.iloc[train_1_2_idx, :].values == train_1_2.values)
print train_2_1.size
print train_1_2.size
