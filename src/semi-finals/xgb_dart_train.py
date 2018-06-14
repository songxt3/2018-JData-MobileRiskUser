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
    'booster' : 'dart',
    'n_estimators': 2000,
    'max_depth': 6,
    'min_child_weight' : 12,
    'eval_metric': 'rmse',
    'gamma' : 0.9,
    'subsample' : 0.8,
    'learning_rate': 0.06,
    'colsample_bytree' : 0.8,
    'eta': 0.02,
    'objective' : 'binary:logistic',
    'sample_type': 'uniform',
    'normalize': 'tree',
    'rate_drop': 0.15,  # 0.1
    'skip_drop': 0.85,  # 0.9
    'nthread' :  -1,
}

dtrain = xgb.DMatrix(X_train,label=y_train)
xgb.cv(xgb_params,dtrain,num_boost_round=10000,nfold=5,verbose_eval=5,early_stopping_rounds=100)

model=xgb.train(xgb_params,dtrain=dtrain,num_boost_round=300,evals=[(dtrain,'train')],verbose_eval=5)

dtest = xgb.DMatrix(X_test)
preds =model.predict(dtest)

print dtest

StackingSubmission = pd.DataFrame({ 'score': preds})

StackingSubmission.to_csv("./result/dart_train.csv", index=False)