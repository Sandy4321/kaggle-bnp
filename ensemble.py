import pandas as pd

res_list = ['extra_trees.csv', 'xgb_cate_comb.csv', 'xgb_rf_feat.csv'] # ensemble1
res_list = ['extra_trees.csv', 'xgb_cate_comb.csv', 'xgb_delimiter.csv'] # ensemble2
res_list = ['extra_trees.csv', 'xgb_cate_comb.csv'] # ensemble3
res_list = ['extra_trees.csv', 'xgb_cate_comb.csv', 'addNNLinearFt.csv'] # ensemble4
res_list = ['extra_trees.csv', 'xgb_cate_comb.csv', 'addNNLinearFt.csv', 'generic_workflow.csv'] # ensemble6
res_list = ['extra_trees.csv', 'xgb_cate_comb_v22_1150.csv', 'addNNLinearFt.csv'] # ensemble7
res_list = ['extra_trees.csv', 'xgb_cate_comb.csv', 'addNNLinearFt.csv', 'xgbr_cate_comb.csv'] # ensemble8
res_list = ['extra_trees.csv', 'xgb_cate_comb.csv', 'addNNLinearFt_new.csv'] # ensemble9

df = pd.read_csv(res_list[0])
res_pred = df['PredictedProb']
ids = df['ID'].values

for f in res_list[1:]:
    df = pd.read_csv(f)
    res_pred = res_pred + df['PredictedProb']

res_pred /= len(res_list)

preds_out = pd.DataFrame({"ID": ids, "PredictedProb": res_pred.values})
preds_out = preds_out.set_index('ID')
preds_out.to_csv('ensemble9.csv')
print 'finish'



