from sklearn.preprocessing import OneHotEncoder
import pandas as pd

train = pd.read_csv('../data/train.csv')
enc = OneHotEncoder()
df = pd.DataFrame(columns=['v112', 'v113'])
print train.v113
df.v112 = pd.factorize(train.v112, na_sentinel=100000)[0]
df.v113 = pd.factorize(train.v113, na_sentinel=100000)[0]
#df.fillna(100000,inplace=True)
print df
enc.fit(df)
print enc.transform([1,1]).toarray().shape


# print df
#enc.fit

#print train.v107

