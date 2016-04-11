import pandas as pd
import numpy as np

train = pd.read_csv('data/train.csv')
feat_imp = np.loadtxt('feat_imp.txt')
sort_idx = np.argsort(-feat_imp)
print feat_imp[sort_idx]
feat_name = ['v'+str(idx+1) for idx in sort_idx]
print feat_name 

num_name = []
for name in feat_name:
    if train[name].dtype != 'object':
        num_name.append(name)

print num_name[:30]


