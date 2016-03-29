from sklearn.feature_selection import SelectFromModel
import cPickle as pk
import pandas as pd

def factorize_category(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = pd.factorize(df[column])[0]

model_file = open('model/rf_200.pickle')
clf = pk.load(model_file)
model = SelectFromModel(clf, threshold = 0.005, prefit=True)
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

#train_id = train['ID'].values
#train_target = train['target'].values

train_feat = train.drop(['ID', 'target'], axis=1)
test_feat = test.drop('ID', axis=1)

factorize_category(train_feat)
factorize_category(test_feat)
train_feat.fillna(-1,inplace=True)
test_feat.fillna(-1,inplace=True)

train_feat = model.transform(train_feat)

with open('data/train_rf_feat_0.005', 'w') as f:
    pk.dump(train_feat, f)
with open('data/train_target_0.005', 'w') as f:
    pk.dump(train['target'].values, f)


