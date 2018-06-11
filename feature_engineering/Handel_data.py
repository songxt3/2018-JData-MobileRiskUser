#! /usr/bin/env python
# -*- coding: utf-8 -*-


import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import pandas as pd



uid_train = pd.read_csv('./JDATA用户风险识别_train/uid_train.txt',sep='\t',header=None,names=('uid','label'))
voice_train = pd.read_csv('./JDATA用户风险识别_train/voice_train.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':int,'end_time':int})
sms_train = pd.read_csv('./JDATA用户风险识别_train/sms_train.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':int})
wa_train = pd.read_csv('./JDATA用户风险识别_train/wa_train.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'))
# voice_test = pd.read_csv('./JDATA用户风险识别_Test-A/voice_test_a.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':int,'end_time':int})
# sms_test = pd.read_csv('./JDATA用户风险识别_Test-A/sms_test_a.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':int})
# wa_test = pd.read_csv('./JDATA用户风险识别_Test-A/wa_test_a.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'))
voice_test = pd.read_csv('./JDATA用户风险识别_Test-B/voice_test_b.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':int,'end_time':int})
sms_test = pd.read_csv('./JDATA用户风险识别_Test-B/sms_test_b.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':int})
wa_test = pd.read_csv('./JDATA用户风险识别_Test-B/wa_test_b.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'))

'''
voice initial
'''

voice_train['timer'] = voice_train['end_time'] - voice_train['start_time']

voice_train['timer'] = voice_train['timer'].astype(str)

voice_train['timer'] = voice_train['timer'].str.pad(width=8, side='left', fillchar='0')

per_day = voice_train['timer'].str.slice(start=0, stop = 2)
per_hour = voice_train['timer'].str.slice(start=2, stop = 4)
per_min = voice_train['timer'].str.slice(start=4, stop = 6)
per_sec = voice_train['timer'].str.slice(start=6, stop = 8)

voice_train['voice_time'] = per_day.astype(int) * 24 * 60 * 60 + per_hour.astype(int) * 60 * 60 + per_min.astype(int) * 60 + per_sec.astype(int)

voice_test['timer'] = voice_test['end_time'] - voice_test['start_time']

voice_test['timer'] = voice_test['timer'].astype(str)

voice_test['timer'] = voice_test['timer'].str.pad(width=8, side='left', fillchar='0')

per_day1 = voice_test['timer'].str.slice(start=0, stop = 2)
per_hour1 = voice_test['timer'].str.slice(start=2, stop = 4)
per_min1 = voice_test['timer'].str.slice(start=4, stop = 6)
per_sec1 = voice_test['timer'].str.slice(start=6, stop = 8)

voice_test['voice_time'] = per_day1.astype(int) * 24 * 60 * 60 + per_hour1.astype(int) * 60 * 60 + per_min1.astype(int) * 60 + per_sec1.astype(int)


voice_train['day'] = voice_train['start_time'] / 1000000
voice_train['day'] = voice_train['day'].astype(int)
voice_test['day'] = voice_test['start_time'] / 1000000
voice_test['day'] = voice_test['day'].astype(int)

'''
sms initial
'''
sms_train['day'] = sms_train['start_time'] / 1000000
sms_train['day'] = sms_train['day'].astype(int)

sms_test['day'] = sms_test['start_time'] / 1000000
sms_test['day'] = sms_test['day'].astype(int)

'''
wa initial
'''
wa_train['sub_flow'] = wa_train['down_flow'] - wa_train['up_flow']
wa_test['sub_flow'] = wa_test['down_flow'] - wa_test['up_flow']


uid_test = pd.DataFrame({'uid':pd.unique(wa_test['uid'])})
uid_test.to_csv('./JDATA用户风险识别_Test-A/uid_test_a.txt',index=None)

voice = pd.concat([voice_train,voice_test],axis=0)
sms = pd.concat([sms_train,sms_test],axis=0)
wa = pd.concat([wa_train,wa_test],axis=0)

wa['date'].fillna(0)
# wa['date'] = wa['date'].astype(int)

'''
voice data
'''
# gp = voice_train.groupby('uid')['timer']
# x = gp.apply(lambda x: x.mean())
# voice['voice_time_mean'] = x.values
# voice_time = ['voice_time_mean']

voice_time = voice.groupby('uid')['voice_time'].agg(['std','median','max','min','mean','sum']).add_prefix('agg_time').reset_index()

voice_opp_num = voice.groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('voice_opp_num_').reset_index()

voice_opp_head = voice.groupby(['uid'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('voice_opp_head_').reset_index()

voice_opp_len=voice.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('voice_opp_len_').reset_index().fillna(0)

voice_call_type = voice.groupby(['uid','call_type'])['uid'].count().unstack().add_prefix('voice_call_type_').reset_index().fillna(0)

voice_in_out = voice.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('voice_in_out_').reset_index().fillna(0)

voice_day_count = voice.groupby(['uid', 'day'])['uid'].count().unstack().add_prefix('voice_day_').reset_index().fillna(0)

'''
sms data
'''
sms_opp_num = sms.groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('sms_opp_num_').reset_index()

sms_opp_head=sms.groupby(['uid'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('sms_opp_head_').reset_index()

sms_opp_len=sms.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('sms_opp_len_').reset_index().fillna(0)

sms_in_out = sms.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('sms_in_out_').reset_index().fillna(0)

sms_day_count = sms.groupby(['uid', 'day'])['uid'].count().unstack().add_prefix('sms_day_').reset_index().fillna(0)

sms_day_view = sms.groupby(['uid'])['day'].agg(['std','median']).add_prefix('sms_day_view_').reset_index()

'''
app data
'''
wa_name = wa.groupby(['uid'])['wa_name'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('wa_name_').reset_index()

visit_cnt = wa.groupby(['uid'])['visit_cnt'].agg(['std','median','max','min','mean','sum']).add_prefix('wa_visit_cnt_').reset_index()

visit_dura = wa.groupby(['uid'])['visit_dura'].agg(['std','median','max','min','mean','sum']).add_prefix('wa_visit_dura_').reset_index()

up_flow = wa.groupby(['uid'])['up_flow'].agg(['std','median','max','min','mean','sum']).add_prefix('wa_up_flow_').reset_index()

down_flow = wa.groupby(['uid'])['down_flow'].agg(['std','median','max','min','mean','sum']).add_prefix('wa_down_flow_').reset_index()

wa_day_count = wa.groupby(['uid', 'date'])['uid'].count().unstack().add_prefix('wa_day_').reset_index().fillna(0)

sub_flow = wa.groupby(['uid'])['sub_flow'].agg(['std','median','max','min','mean','sum']).add_prefix('wa_sub_flow_').reset_index()

view_data = wa.groupby(['uid'])['date'].agg(['std','median']).add_prefix('view_date_').reset_index()

'''
marge
'''
feature = [voice_opp_num,voice_opp_head,voice_opp_len,voice_call_type,voice_in_out,voice_day_count,voice_time,sms_opp_num,sms_opp_head,sms_opp_len,sms_in_out,sms_day_count,wa_name,visit_cnt,visit_dura,up_flow,
           down_flow, wa_day_count]
# feature = [voice_opp_num, voice_opp_head,voice_opp_len,voice_call_type,voice_in_out,voice_time,voice_day_count,sms_opp_num,sms_opp_head,sms_opp_len,sms_in_out,sms_day_count, wa_name,visit_cnt,visit_dura,up_flow,
#            down_flow,wa_day_count]
train_feature = uid_train
for feat in feature:
    train_feature=pd.merge(train_feature,feat,how='left',on='uid')
train_feature = train_feature.fillna(0)

test_feature = uid_test
for feat in feature:
    test_feature=pd.merge(test_feature,feat,how='left',on='uid')
test_feature = test_feature.fillna(0)

train_feature.to_csv('./data/train_feature_B_V3.csv',index=None)
test_feature.to_csv('./data/test_feature_B_V3.csv',index=None)