#! /usr/bin/env python
# -*- coding: utf-8 -*-


import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import pandas as pd
import lightgbm as lgb
from sklearn import metrics
import random

train = pd.read_csv('./data/train_feature_B_V3.csv')
test = pd.read_csv('./data/test_feature_B_V3.csv')

# temp_cout = 3999
# for i in range(0, 2000):
#     temp_cout = temp_cout - 1
#     train = train.drop(train.index[random.randint(0, temp_cout)])

dtrain = lgb.Dataset(train.drop(['uid','label'],axis=1),label=train.label)
dtest = lgb.Dataset(test.drop(['uid'],axis=1))



lgb_params =  {
    'boosting_type': 'dart',
    # 'xgboost_dart_mode' : True,
    'objective': 'binary',
    'max_depth': 6,
   # 'metric': ('multi_logloss', 'multi_error'),
   #  'metric_freq': 1,
    'is_training_metric': False,
    'min_data_in_leaf': 64,
    'num_leaves': 64,
    'learning_rate': 0.06,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq' : 1,
    'verbosity':-1,
    'is_unbalance' : 'true',
    'drop_rate': 0.1,
    'skip_drop': 0.9,
}


def evalMetric(preds, dtrain):
    label = dtrain.get_label()

    pre = pd.DataFrame({'preds': preds, 'label': label})
    pre = pre.sort_values(by='preds', ascending=False)

    auc = metrics.roc_auc_score(pre.label, pre.preds)

    pre.preds = pre.preds.map(lambda x: 1 if x >= 0.5 else 0)

    f1 = metrics.f1_score(pre.label, pre.preds)

    res = 0.6 * auc + 0.4 * f1

    return 'res', res, True

lgb.cv(lgb_params,dtrain,feval=evalMetric,early_stopping_rounds=50,verbose_eval=5,num_boost_round=5000,nfold=5,metrics=['evalMetric'])

model =lgb.train(lgb_params,dtrain,feval=evalMetric,verbose_eval=5,num_boost_round=300,valid_sets=[dtrain])

pred=model.predict(test.drop(['uid'],axis=1))
res = pd.DataFrame({'score':pred})
res.to_csv("./result/lgb_dart_train.csv", index=False)