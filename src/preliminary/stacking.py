#! /usr/bin/env python
# -*- coding: utf-8 -*-


import sys
reload(sys)
sys.setdefaultencoding('utf-8')

# Load in our libraries

import pandas as pd
import numpy as np
import re
import xgboost as xgb
import random
import seaborn as sns
import matplotlib.pyplot as plt
 # %matplotlib inline

import plotly.plotly
import plotly.offline as py
# py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')

# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor,
                              GradientBoostingRegressor, ExtraTreesRegressor)
from sklearn.svm import SVR
from sklearn.cross_validation import KFold
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv('./data/train_featureV1.csv', header = 0)
test = pd.read_csv('./data/test_featureV1.csv', header = 0)


temp_cout = 3999
for i in range(0, 2800):
    temp_cout = temp_cout - 1
    train = train.drop(train.index[random.randint(0, temp_cout)])

train = train.drop(['uid'], axis = 1)
test = test.drop(['uid'], axis = 1)


# Some useful parameters which will come in handy later on
ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds = NFOLDS, random_state=SEED)

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=None, params=None):
        # params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self,x,y):
        return self.clf.fit(x,y)

    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)

# Put in our parameters for said classifiers
# Random Forest parameters -s
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     # 'warm_start': True,
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters -s
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters -s
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters -s
gb_params = {
    'n_estimators': 200,
     #'max_features': 0.2,
    'max_depth': 3,
    'min_samples_leaf': 2,
    'verbose': 1
}

# Support Vector Classifier parameters
svc_params = {
    'kernel' : 'rbf',
    # 'class_weight' : {1: 100},
    'C' : 0.025
    }

# Class to extend XGboost classifer

# Create 5 objects that represent our 4 models
rf = SklearnHelper(clf=RandomForestRegressor, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesRegressor, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostRegressor, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingRegressor, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVR, seed=SEED, params=svc_params)


# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
y_train = train['label'].ravel()
train = train.drop(['label'], axis=1)
x_train = train.values # Creates an array of the train data
x_test = test.values # Creats an array of the test data

def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
print "ef_ok"
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
print "rf_ok"
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost
print "ada_ok"
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
print "gb_ok"
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier
print "svc_ok"

print("Training is complete")

rf_feature = rf.feature_importances(x_train,y_train)
et_feature = et.feature_importances(x_train, y_train)
ada_feature = ada.feature_importances(x_train, y_train)
gb_feature = gb.feature_importances(x_train,y_train)

base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),
     'ExtraTrees': et_oof_train.ravel(),
     'AdaBoost': ada_oof_train.ravel(),
      'GradientBoost': gb_oof_train.ravel()
    })
print base_predictions_train.head()

x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)

gbm = xgb.XGBRegressor(
 booster = 'gbtree',
 n_estimators= 2000,
 max_depth= 6,
 min_child_weight= 12,
 #gamma=1,
 gamma=0.9,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(x_train, y_train)

predictions = gbm.predict(x_test)

print predictions

res = pd.read_csv('./data/TestData.csv', header = 0)

uid = res['uid']

StackingSubmission = pd.DataFrame({ 'uid': uid,
                            'posibility': predictions})
StackingSubmission = StackingSubmission[['uid', 'posibility']]

gbm = xgb.XGBClassifier(
 booster='gbtree',
 n_estimators= 2000,
 max_depth= 6,
 min_child_weight= 12,
 gamma=0.9,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(x_train, y_train)

predictions = gbm.predict(x_test)

uid = StackingSubmission['uid']
pos = StackingSubmission['posibility']

StackingSubmission = pd.DataFrame({ 'uid': uid,
                            'label': predictions, 'posibility': pos})
StackingSubmission = StackingSubmission[['uid', 'label', 'posibility']]

StackingSubmission = StackingSubmission.sort_values(by = ['label', 'posibility'], ascending = False)
StackingSubmission = StackingSubmission.drop(['posibility'], axis=1)

StackingSubmission.to_csv("StackingSubmission.csv", index=False, header=None)

