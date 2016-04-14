import pandas as pd
import numpy as np
import cPickle as pk

def third_sep(str):
    c = 0
    for idx,a in enumerate(str):
        if a == '_':
            c += 1
        if c == 3:
            return str[:idx], str[idx+1:-1]

mi_dict = {}
mi_list = []
with open('num_mi.data', 'r') as f:
    lines = f.readlines()
    for line in lines:
        a, b = line.rstrip().split(' ')        
        mi = float(b)
        mi_list.append(mi)
        id1, id2 = third_sep(a)
        if mi_dict.has_key(id1):
            mi_dict[id1].append(mi)
        else:
            mi_dict[id1] = [mi]

        if mi_dict.has_key(id2):
            mi_dict[id2].append(mi)
        else:
            mi_dict[id2] = [mi]

mi_arr = np.array(mi_list)
mi_arr = np.sort(mi_arr)
print mi_arr

with open('mi.dict', 'w') as f:
    pk.dump(mi_dict, f)
