#! /usr/bin/env python
# -*- coding: utf-8 -*-


import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import pandas as pd
from sklearn.metrics import roc_auc_score

xgb_score = pd.read_csv('./result/xgb_train.csv')
dart_score = pd.read_csv('./result/dart_train.csv')
lgb_dart_score = pd.read_csv('./result/lgb_dart_train.csv')
lgb_gbdt_score = pd.read_csv('./result/lgb_gbdt_train.csv')
# lgb_score_heaf = pd.read_csv('./result/lgb_train_half.csv')
# rf_score = pd.read_csv('./result/rf_train.csv')

# lgb_dart_rank = lgb_dart_score.score.rank()
# lgb_gbdt_rank = lgb_gbdt_score.score.rank()
#
# rank_data = 0.5 * lgb_dart_rank + 0.5 * lgb_gbdt_rank
#
# print rank_data

sub_res = lgb_gbdt_score.score * 0.6 + xgb_train.csv.score * 0.2 + rf_train.score * 0.2

test = pd.read_csv('./data/test_feature_B_V3.csv')

res =pd.DataFrame({'uid':test.uid,'label':sub_res})
res=res.sort_values(by='label',ascending=False)
res.label=res.label.map(lambda x: 1 if x>=0.5 else 0)

res.to_csv('./result/model_merge.csv',index=False,header=False,sep=',',columns=['uid','label'])
