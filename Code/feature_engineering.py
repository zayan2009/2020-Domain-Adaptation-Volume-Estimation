#!/usr/bin/env python
# coding: utf-8

# # Libs

# In[ ]:


import os
import time
import glob
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder

import lightgbm as lgb


# # Load Data

# In[ ]:


avi_train_seg = np.load('../Data/avi_train_seg.npy')
trj_train_seg = np.load('../Data/trj_train_seg.npy')
lane_functions = np.load('../Data/lane_functions.npy')


# In[ ]:


num_lanes = lane_functions.sum(axis=1)
lane_functions = pd.DataFrame(lane_functions,columns=['through','left','right',
                                                      'thr_left','thr_right','u_turn'])
lane_functions['linkIdx'] = np.arange(1,25)
lane_functions['num_lanes'] = num_lanes


# In[ ]:


def cap(x):
    base = x['through'] * 1 + x['left'] * 0.85 + x['right'] * 1
    mix = x['thr_left'] * 0.8 + x['thr_right'] * 1 + x['u_turn'] * 0.7
    return base + mix

lane_functions['capacity'] = lane_functions.apply(lambda x:cap(x),axis=1)


# # Data Preparation

# ## Sample Preparation

# In[ ]:


def get_sample(trj_train_seg):
    data = pd.DataFrame(columns=['linkIdx','datetime','volume'])
    for approach in range(24):
        table = pd.DataFrame(np.zeros((30 * 144,3)),columns=['linkIdx','datetime','volume'])
        table.iloc[:,0] = approach + 1
        for day in range(30):
            if day < 9:
                datetime = [pd.to_datetime('2018010' + str(day + 1)) + pd.Timedelta(i * 10,unit='m') for i in range(144)]
            else:
                datetime = [pd.to_datetime('201801' + str(day + 1)) + pd.Timedelta(i * 10,unit='m') for i in range(144)]
            table.iloc[144 * day:144 * (day + 1),2] = trj_train_seg[day,:,approach]
            table.iloc[144 * day:144 * (day + 1),1] = datetime
        data = pd.concat((data,table))
    data['linkIdx'] = data['linkIdx'].astype('int')
    data['datetime'] = pd.to_datetime(data['datetime'])
    data['volume'] = data['volume'].astype('float')
    return data


# In[ ]:


X, Y = get_sample(trj_train_seg), get_sample(avi_train_seg)
print("[Basic Stats] num. of samples: {:d}".format(X.shape[0]))


# ## Feature Engineering

# In[ ]:


def holiday(x):
    if x.month == 1 and x.day == 1:
        return 1
    else:
        return 0

    
def peak(x):
    if 7 < x.hour < 9:
        return 1
    elif 11 < x.hour < 13:
        return 2
    elif 17 < x.hour < 19:
        return 3
    else:
        return 0

    
def itv_cnt(x, itv_length=600):
    return (x.hour * 3600 + x.minute * 60) // itv_length


def exponential_smoothing(alpha, s):
    s2 = np.zeros(s.shape)
    s2[0] = s[0]
    for i in range(1, len(s2)):
        s2[i] = alpha * s[i] + (1 - alpha) * s2[i - 1]
    return s2


def get_es_volume(trj_train_seg, alpha):
    trj_train_seg_es = trj_train_seg.copy()
    for day_idx in range(30):
        for seg_idx in range(24):
            seq = trj_train_seg[day_idx, :, seg_idx]
            seq_es = exponential_smoothing(alpha, seq)
            trj_train_seg_es[day_idx, :, seg_idx] = seq_es
    return get_sample(trj_train_seg_es)['volume']


def merge_volume_features(X, feature_cols, target_col, aggfuncs=['mean', 'median', 'std']):
    
    for feature in feature_cols:
        for fn in aggfuncs:
            df = X.pivot_table(index='linkIdx',
                               columns=feature,
                               values=target_col,
                               aggfunc=fn).reset_index()
            df.columns = ['linkIdx'] + list(df.columns[1:])
            df = df.melt(id_vars=['linkIdx'],
                         value_vars=list(df.columns[1:]),
                         var_name=feature,
                         value_name=feature + '_' + fn + '_' + target_col)
            df[feature] = pd.to_numeric(df[feature])
            X = pd.merge(X, df, on=['linkIdx',feature])
    return X


# In[ ]:


# =============================
# Feature Engineering
# =============================

# static attributes
X = pd.merge(X,lane_functions)
X['weekday'] = X['datetime'].map(lambda x:x.weekday())
X['interval'] = X['datetime'].map(lambda x:itv_cnt(x))
X['holiday'] = X['datetime'].map(lambda x:holiday(x))
X['peak'] = X['datetime'].map(lambda x:peak(x))
X['linkIdx'] = X['linkIdx'].astype(int)

# exponenrially smoothed volume
X['volume_es_p7'] = get_es_volume(trj_train_seg, alpha=0.7).values
X['volume_es_p6'] = get_es_volume(trj_train_seg, alpha=0.6).values
X['volume_es_p5'] = get_es_volume(trj_train_seg, alpha=0.5).values


# In[ ]:


# cross volume features
feature_cols = ['through', 'left', 'right', 'thr_left',
                'thr_right', 'u_turn', 'num_lanes',
                'weekday', 'interval', 'holiday', 'peak']
X = merge_volume_features(X, feature_cols, target_col='volume')
X = merge_volume_features(X, feature_cols, target_col='volume_es_p7')
X = merge_volume_features(X, feature_cols, target_col='volume_es_p6')
X = merge_volume_features(X, feature_cols, target_col='volume_es_p5')

# penetration rates
X['tmp'] = Y['volume'].values
interval_volume = X.pivot_table(index='interval',
                                values=['volume', 'tmp', 'volume_es_p7',
                                        'volume_es_p6', 'volume_es_p5'],
                                aggfunc='sum').reset_index()
interval_volume['penetration'] = interval_volume['volume'] / interval_volume['tmp']
interval_volume['penetration_p7'] = interval_volume['volume_es_p7'] / interval_volume['tmp']
interval_volume['penetration_p6'] = interval_volume['volume_es_p6'] / interval_volume['tmp']
interval_volume['penetration_p5'] = interval_volume['volume_es_p5'] / interval_volume['tmp']
interval_volume.drop(['tmp','volume','volume_es_p7','volume_es_p5','volume_es_p6'],axis=1,inplace=True)
X = pd.merge(X, interval_volume, on='interval')

# scaled volume
for up_idx, up_col in enumerate(['volume','volume_es_p7',
                                 'volume_es_p6','volume_es_p5']):
    for down_idx, down_col in enumerate(['penetration','penetration_p7',
                                         'penetration_p6','penetration_p5']):
        X['scaled_volume_' + str(up_idx) + '_' + str(down_idx)] = X[up_col] / X[down_col]

# correlations & feature count
X.drop('tmp',axis=1,inplace=True)


# In[ ]:


# cat & num features
categorical_features = ['interval', 'weekday', 'holiday', 'peak']
numeric_features = list(set(X.drop(['datetime', 'linkIdx'], axis=1).columns) - set(categorical_features))
print("[Basic Stats] dim. of features: {:d}".format(X.shape[1]))


# In[ ]:
type_features = [categorical_features, numeric_features]
with open('../Data/feature_types.json','w') as f:
    json.dump(type_features, f)

# save data
X.to_csv('../Data/X.csv')
Y.to_csv('../Data/Y.csv')

