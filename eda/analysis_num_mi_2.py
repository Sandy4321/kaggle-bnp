import cPickle as pk
import numpy as np

with open('mi.dict', 'r') as f:
    mi_dict = pk.load(f)

for key in mi_dict:
    val = np.array(mi_dict[key])
    print '%s - %0.03f - %0.03f' % (key, np.mean(val), np.max(val))
