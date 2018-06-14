#! /usr/bin/env python
# -*- coding: utf-8 -*-


import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import pandas as pd
import lightgbm as lgb
from sklearn import metrics
import random

import seaborn as sns
import matplotlib.pyplot as plt
 # %matplotlib inline

train = pd.read_csv('./data/train_feature_B_V3.csv')
test = pd.read_csv('./data/test_feature_B_V3.csv')

train = train.drop(drop_initial ,axis = 1)
test = test.drop(drop_initial, axis= 1)

# temp_cout = 3999
# for i in range(0, 2000):
#     temp_cout = temp_cout - 1
#     train = train.drop(train.index[random.randint(0, temp_cout)])

dtrain = lgb.Dataset(train.drop(['uid','label'],axis=1),label=train.label)
dtest = lgb.Dataset(test.drop(['uid'],axis=1))



lgb_params =  {
    'boosting_type': 'gbdt',
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

lgb.cv(lgb_params,dtrain,feval=evalMetric,early_stopping_rounds=100,verbose_eval=5,num_boost_round=10000,nfold=5,metrics=['evalMetric'])

model =lgb.train(lgb_params,dtrain,feval=evalMetric,verbose_eval=5,num_boost_round=300,valid_sets=[dtrain])

feature_names = dtrain.feature_name

feature_importance = pd.DataFrame({
        'column': feature_names,
        'importance': model.feature_importance(),
    }).sort_values(by='importance', ascending=False)

drop_column = feature_importance.tail(50).column

drop_column.to_csv("./data/drop_col.csv", index = False, header= False)

print drop_column, feature_importance

plt.xlabel('Importance')
plt.title('Feature Importance')

sns.set_color_codes("muted")
sns.barplot(x = 'importance', y='column', data=feature_importance, color="b")

plt.show()

pred=model.predict(test.drop(['uid'],axis=1))
res = pd.DataFrame({'score':pred})
res.to_csv("./result/lgb_gbdt_train.csv", index=False)