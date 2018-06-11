#! /usr/bin/env python
# -*- coding: utf-8 -*-


import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import pandas as pd
import numpy as np
import xgboost as xgb
import math
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics


train = pd.read_csv('./data/train_feature_B_V3.csv')
test = pd.read_csv('./data/test_feature_B_V3.csv')

X_train = train.drop(['uid', 'label'],axis=1)
X_test = test.drop(['uid'],axis=1)
y_train = train.label

xgb_params = {
    'booster' : 'gbtree',
    'n_estimators': 2000,
    'max_depth': 6,
    'min_child_weight' : 12,
    'gamma' : 0.9,
    'subsample' : 0.8,
    'eval_metric': 'rmse',
    'learning_rate': 0.06,
    'colsample_bytree' : 0.8,
    'objective' : 'binary:logistic',
    'nthread' :  -1
}

# def evalMetric(preds, dtrain):
#     label = dtrain.get_label()
#     pred =
#
#     return 'FSCORE', float(F)
    #
    # pre = pd.DataFrame({'preds': preds, 'label': label})
    # pre = pre.sort_values(by='preds', ascending=False)
    #
    # auc = metrics.roc_auc_score(pre.label, pre.preds)
    #
    # pre.preds = pre.preds.map(lambda x: 1 if x >= 0.5 else 0)
    #
    # f1 = metrics.f1_score(pre.label, pre.preds)
    #
    # res = 0.6 * auc + 0.4 * f1
    #
    # return 'res', res, True

# def evalF1(preds,dtrain):
#     labels = dtrain.get_label()
#     return 'error', math.sqrt(mean_squared_log_error(preds, labels))

dtrain = xgb.DMatrix(X_train,label=y_train)
xgb.cv(xgb_params,dtrain,num_boost_round=10000,nfold=5,verbose_eval=5,early_stopping_rounds=100)

model=xgb.train(xgb_params,dtrain=dtrain,num_boost_round=300,evals=[(dtrain,'train')],verbose_eval=5)

dtest = xgb.DMatrix(X_test)
preds =model.predict(dtest)

print dtest

StackingSubmission = pd.DataFrame({ 'score': preds})

StackingSubmission.to_csv("./result/xgb_train.csv", index=False)