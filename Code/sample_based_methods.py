#!/usr/bin/env python
# coding: utf-8

# # Libs

# In[ ]:


import os
import time
import glob
import json
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD

import lightgbm as lgb

from utils import *


# load Data
X = pd.read_csv('../Data/X.csv')
Y = pd.read_csv('../Data/Y.csv')

# load feature type columns
with open('../Data/feature_types.json', 'r') as f:
    categorical_features, numeric_features = json.load(f)

# one-hot encoding cat features
X_onehot = one_hot_encoding(X, categorical_features)

# split the dataset
X_train, X_test, y_train, y_test = dataset_split(X, Y,
                                                 split_mode='fix_transfer',
                                                 use_simple_features=False,
                                                 use_best_features=False,
                                                 num_features=0)

sp = time.time()

# build dataset
X_data = pd.concat((X_train, X_test),axis=0)
y_data = np.concatenate((np.zeros((X_train.shape[0],1)),np.ones((X_test.shape[0],1))))

# define classifier
lr = LogisticRegression(penalty='l2')
lr = lr.fit(X_data, y_data)

# weight samples
sample_weights = lr.predict_proba(X_train)[:,1]

# train Regressor
train_data = lgb.Dataset(X_train, y_train,
                         categorical_feature=categorical_features)
train_data.set_weight(sample_weights.reshape(-1))
test_data = lgb.Dataset(X_test, y_test, reference=train_data)

params = {
    'objective':'regression',
    'boosting':'gbdt',
    'metric':'mae',
    'num_rounds':20000,
    'learning_rate':0.001,
    'max_depth':8,
    'num_leaves':120,
    'feature_fraction':0.8,
    'bagging_fraction':0.8,
    'bagging_freq':200,
    'verbose':0
}

gbm = lgb.train(params, train_data,
                valid_sets=[test_data, train_data],
                valid_names=['test','train'],
                verbose_eval=1000,
                early_stopping_rounds=100)
print("[Duration] {:.2f}".format(time.time() - sp))

# evalulate
y_pred = gbm.predict(X_test,num_iteration=gbm.best_iteration)

print("[LightGBM] mae: {:.2f} | mape: {:.2f}% | mspe: {:.2f}%\n\n\n".format(
    mae(y_pred, y_test),
    100 * mape(y_pred,y_test), 100 * mspe(y_pred, y_test)))

