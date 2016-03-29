import pandas as pd

def factorize_category(train, test):
    for column in train.columns:
        if train[column].dtype == 'object':
            train[column], tmp_indexer = pd.factorize(train[column])
            test[column] = tmp_indexer.get_indexer(test[column])

