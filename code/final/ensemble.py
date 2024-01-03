import numpy as np
import pandas as pd

## Hard Coding 방법으로 Ensemble

koelec = pd.read_csv('./output_kfold_koelectra.csv')
diffcse = pd.read_csv('./output_kfold_diffcse.csv')
rlb = pd.read_csv('./output_roberta_large.csv')
rla = pd.read_csv('./output_roberta_large_nospacing.csv')
elec_after = pd.read_csv('./output_electra_best.csv')
fold_electra = pd.read_csv('./after-kfold-electra.csv')

assemble = pd.concat([koelec, elec_after, fold_electra], axis=1).drop(columns='id')
assemble.columns =['koelec','koelec-a','elec-fold']

df_test = pd.read_csv('../data/test.csv')

# 표준편차로 예측 범위가 넓은 값 확인
print(assemble[assemble.std(axis=1) > 0.6])

# ensemble 할 df 확인
sum_df = (koelec  + fold_electra + rlb)
sum_df.target = sum_df.target / 3
sum_df.target = np.where(sum_df.target < 0, 0, sum_df.target)
sum_df.target = np.where(sum_df.target > 5, 5, sum_df.target)
sum_df.target = np.round(sum_df.target, 1)

sub = pd.read_csv('../data/sample_submission.csv')
sub['target'] = sum_df.target
sub.to_csv('./e2r1.csv', index=False)