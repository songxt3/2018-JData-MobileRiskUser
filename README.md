# 2018-JData-MobileRiskUser
软件工程综合实训--基于移动网络通讯行为的风险用户识别  
##代码说明
**model_select**  
`model_select.py`，进行单模型选择   
**feature_engineering**  
`Handel_data.py`,提取数据特征，进行特征工程  
**src**  
`lgb_gbdt_train.py`,使用gbdt参数训练的LightGBM  
`lgb_dart_train.py`,使用dart参数训练的LightGBM
`xgb_gbtree_train.py`,使用gbtree训练的Xgboost  
`xgb_dart_train.py`,使用dart参数训练的Xgboost  
`rf_train.py`,使用RandomForest训练模型  
`gen_sub.py`,生成提交文件
