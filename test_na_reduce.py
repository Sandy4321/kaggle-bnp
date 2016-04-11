import cPickle as pk
import pandas as pd

train = pd.read_csv('data/train.csv')

with open('data/na_bin_pca.pkl', 'r') as f:
    na_bin_pca = pk.load(f)
    na_df = pd.DataFrame(na_bin_pca)

for c in na_df.columns:
    train[c] = na_df[c]

print train
