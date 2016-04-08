from preprocess import find_delimiter
import pandas as pd

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
vs = pd.concat([train, test])

delimiter = find_delimiter(vs, 'v1')
print delimiter



