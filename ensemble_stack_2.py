import pandas as pd
import numpy as np


###########################################################
out_train_file = 'stack_data/train_ensemble_1.csv'
out_test_file = 'stack_data/test_ensemble_1.csv'
###########################################################


###########################################################
ext_predict_1_file = 'stack_data/ext_predict_1.csv'
ext_predict_2_file = 'stack_data/ext_predict_2.csv'
ext_predict_full_file = 'stack_data/ext_predict_full.csv'

nn_predict_1_file = 'stack_data/nn_predict_1.csv'
nn_predict_2_file = 'stack_data/nn_predict_2.csv'
nn_predict_full_file = 'stack_data/nn_predict_full.csv'

xgb_predict_1_file = 'stack_data/xgb_predict_1.csv'
xgb_predict_2_file = 'stack_data/xgb_predict_2.csv'
xgb_predict_full_file = 'stack_data/xgb_predict_full.csv'

# list of (train_1_file, train_2_file)
train_file_list = [(ext_predict_1_file, ext_predict_2_file), 
	(nn_predict_1_file, nn_predict_2_file),
	(xgb_predict_1_file, xgb_predict_2_file)]  

# sequece must match with train!!!!!!!!!!!
test_file_list = [ext_predict_full_file, nn_predict_full_file, xgb_predict_full_file]  
###########################################################


###########################################################
# target
train_1_2_idx_file = 'stack_data/train_1_2.txt'
idx = np.loadtxt(train_1_2_idx_file)
idx = np.asarray(idx, np.int)
train_file = 'data/train.csv'
train = pd.read_csv(train_file)
new_train_target = train.target.iloc[idx]
#print train.target
#print new_train_target 
#print new_train_target == train.target
###########################################################



train_df = pd.DataFrame()
for idx, train_file_tuple in enumerate(train_file_list):
    train_1_file, train_2_file = train_file_tuple
    train_1 = pd.read_csv(train_1_file)
    train_2 = pd.read_csv(train_2_file)
    train = pd.concat([train_1, train_2])
    train_df[idx] = train.PredictedProb
# cannot be train_df['target'] = new_train_target!!! sequnce will not be changed in this way!!!
train_df['target'] = new_train_target.values
print train_df

test_df = pd.DataFrame()
for idx, test_file in enumerate(test_file_list):
    test = pd.read_csv(test_file)
    test_df[idx] = test.PredictedProb
print test_df


train_df.to_csv(out_train_file, index=False)
test_df.to_csv(out_test_file, index=False)
