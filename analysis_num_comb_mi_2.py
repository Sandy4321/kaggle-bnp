import cPickle as pk
import pandas as pd
from util import num_feat_stat 

name = 'comb_num_0.01'
file_name = 'data/' + name +'.pkl'

with open(file_name, 'r') as f:
    df = pk.load(f)

#print df

new_df = pd.DataFrame()
for c in df.columns:
    num_nan, num_inf, num_zero = num_feat_stat(df[c].values)
    if num_nan > 20000 or num_inf > 20000 or num_zero > 20000:
    	print '[filter]%s: %d %d %d' % (c, num_nan, num_inf, num_zero)
    else:
    	print '[pass]%s: %d %d %d' % (c, num_nan, num_inf, num_zero)
    	new_df[c] = df[c]



file_name = 'data/' + name +'_filter.pkl'
with open(file_name, 'w') as f:
    pk.dump(new_df, f)
