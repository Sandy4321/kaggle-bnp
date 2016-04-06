import pandas as pd
from preprocess import bayes_encoding

train = pd.read_csv('./data/train.csv')
bayes_encoding(train.v22, train.target)

