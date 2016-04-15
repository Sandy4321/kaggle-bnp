import pandas as pd
import numpy as np
from util import calc_MI

def discret(val, bins):
    for idx in xrange(len(bins)):
        if val >= bins[idx] and val <= bins[idx+1] :
            return idx

train = pd.read_csv('data/train.csv')

'''
############################
# compute p_pos, p_neg
p_pos = 0.0
p_neg = 0.0
for val in train.target:
    if val == 0:
        p_neg += 1
    else:
        p_pos += 1

p_neg /= train.shape[0]
p_pos /= train.shape[0]
print '%f - %f' % (p_neg, p_pos)
'''

column = train.v1
target = train.target

p_neg = 0.238801
p_pos = 0.761199 
column = column.round(5)
min_val = np.min(column)
max_val = np.max(column)
column.fillna(-999, inplace=True)
values = column.values
num_bins = 10.0
print max_val
bins = [-999] + np.arange(min_val, max_val+0.0001, (max_val-min_val)/num_bins).tolist()
'''
print discret(-0.1, bins)
print discret(1.9, bins)
print discret(17, bins)
'''
print bins
print values
densitys, bin_edges = np.histogram(values, bins=bins, density=True)
dist_vals = []
print 'discreting.....'
for val in column.values: 
    #print val
    dist_val = discret(val, bins)
    dist_vals.append(dist_val)

print 'computing mi....'
final_mi = 0
dist_vals = np.array(dist_vals)
for level in xrange(len(bins)):
    p_cate_pos = np.sum((dist_vals == level) & (target == 1)) / float(train.shape[0])
    p_cate_neg = np.sum((dist_vals == level) & (target == 0)) / float(train.shape[0])
    p_cate = np.sum((dist_vals == level)) / float(train.shape[0])
    print p_cate_pos
    print p_cate_neg
    if p_cate_pos == 0 or p_cate_neg == 0:
        continue

    final_mi += p_cate_pos * np.log2(p_cate_pos / (p_cate * p_pos))   
    final_mi += p_cate_neg * np.log2(p_cate_neg / (p_cate * p_neg))   

print final_mi
#print calc_MI(column.values, target.values, 10)
