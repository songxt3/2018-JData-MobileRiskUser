# 2018-JData-MobileRiskUser
软件工程综合实训--基于移动网络通讯行为的风险用户识别  
数据链接：https://pan.baidu.com/s/1AYK9efZViaU_du9ra24X1w         
## 代码说明  
**model_select**  
`model_select.py`，进行单模型选择   
**feature_engineering**  
`Handel_data.py`,提取数据特征，进行特征工程  
`feature_drop.py`,特征重要性排名及选择  
**src**   
**preliminary**   
初赛相关代码   
`lgb_gbdt_train.py`,使用gbdt参数训练的LightGBM  
`xgb_gbtree_train.py`,使用gbtree训练的Xgboost  
`rf_train.py`,使用RandomForest训练模型  
`gen_sub.py`,生成提交文件  
`stacking.py`,分类模型stacking  
**semi-finals**  
复赛相关代码  
`lgb_gbdt_train.py`,使用gbdt参数训练的LightGBM  
`lgb_dart_train.py`,使用dart参数训练的LightGBM  
`xgb_gbtree_train.py`,使用gbtree训练的Xgboost  
`xgb_dart_train.py`,使用dart参数训练的Xgboost  
`gen_sub.py`,生成提交文件  
