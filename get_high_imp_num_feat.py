import pandas as pd
import numpy as np

feat_imp = np.loadtxt('feat_imp.txt')
sort_idx = np.argsort(-feat_imp)

train = pd.read_csv('data/train.csv')

feat_name = ['v'+str(val+1) for val in sort_idx]

num_name = []
for name in feat_name:
    if train[name].dtype != 'object':
        num_name.append(name)

print num_name[:30]

