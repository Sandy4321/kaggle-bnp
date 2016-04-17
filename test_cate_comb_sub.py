import cPickle as pk
import pandas as pd

train_1 = pd.read_csv('stack_data/train_1.csv')
train_2 = pd.read_csv('stack_data/train_2.csv')

train_test = pd.concat([train_1, train_2])

print train_test.index
print train_test.ID
with open('data/comb_cate_v22_v56.pkl', 'r') as f:
    df = pk.load(f)
print df.columns
print df.ID
