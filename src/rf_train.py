#! /usr/bin/env python
# -*- coding: utf-8 -*-


import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
import pandas as pd
import numpy as np

train = pd.read_csv('./data/train_featureV1.csv')
test = pd.read_csv('./data/test_featureV1.csv')

X_train = train.drop(['uid', 'label'],axis=1)
X_test = test.drop(['uid'],axis=1)
y_train = train.label

model = RandomForestRegressor(n_estimators=1000,criterion='mse',max_depth=6,max_features='sqrt',min_samples_leaf=8,n_jobs=12,random_state=0)#min_samples_leaf: 5~10
scores = cross_val_score(model,X_train.values,y_train.values,cv=5,scoring='mean_squared_error')

model.fit(X_train.values,y_train.values)
preds = model.predict(X_test.values)
print preds
res = pd.DataFrame({'score':preds})
res.to_csv("./result/rf_train.csv", index=False)