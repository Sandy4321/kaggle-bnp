import pandas as pd
from preprocess import handle_v22

train = pd.read_csv('data/train.csv')

train = handle_v22(train)

print train['v22-1']

