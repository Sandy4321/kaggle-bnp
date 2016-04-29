import pandas as pd
import numpy as np
import os
os.sys.path.append(os.path.join( os.path.dirname(__file__), '../'))

item_store_df = pd.read_csv('../data/item_store_feature1.csv')
item_df = pd.read_csv('../data/item_feature1.csv')
sample_df = pd.read_csv('../data/sample_submission.csv')

unique_store = np.unique(item_store_df.store_code)
print 'num of unique store is %d' % len(unique_store)
print('unique stores are ', unique_store)

unique_item = np.unique(item_df.item_id)
print 'num of unique item is %d' % len(unique_item)
#print unique_item

unique_item_store = np.unique(item_store_df.item_id)
print 'num of unique item in store feature is %d' % len(unique_item_store)

print('item in item_feature and item in item_store_feature is equal', np.array_equal(unique_item, unique_item_store))  

unique_item_submission = np.unique(sample_df.item_id)
print 'num of unique item in sample submission is %d' % len(unique_item_submission)

for item in unique_item:
    print('%s\t%d' % (item, item_df[item_df.item_id == item].shape[0]))


print '-' * 80
for item in unique_item_store:
    print '*' * 10
    for store in unique_store:
        print('%s\t%s\t%d' % (item, store, item_store_df[(item_store_df.item_id == item) & (item_store_df.store_code == store)].shape[0]))

