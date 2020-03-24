#!/usr/bin/env python
# coding: utf-8

# # Libs

# In[ ]:


import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.feature_selection import SelectFromModel

import lightgbm as lgb

from utils import *


# # Load Data

# In[ ]:


X = pd.read_csv('../Data/X.csv')
Y = pd.read_csv('../Data/Y.csv')

X_data = X.drop(['datetime','linkIdx'],axis=1)
y_data = Y['volume']


# # Feature Selection

# ## LGB

# In[ ]:


params = {
    'objective':'regression',
    'boosting':'gbdt',
    'num_rounds':10000,
    'learning_rate':0.01,
    'max_depth':8,
    'num_leaves':100,
    'bagging_fraction':0.9,
    'bagging_freq':100,
    'verbose':2
}

train_data = lgb.Dataset(X_data, y_data,
                         categorical_feature=['weekday','interval','holiday','peak'])
gbm = lgb.train(params, train_data)


# In[ ]:


y_pred = gbm.predict(X_data)

print("[LightGBM] mae: {:.2f} | mape: {:.2f}% | mspe: {:.2f}%".format(
    mae(y_pred, y_data),
    100 * mape(y_pred,y_data), 100 * mspe(y_pred, y_data)))


# In[ ]:


gb = GradientBoostingRegressor(n_estimators=10000, learning_rate=0.01,
                               max_depth=10, subsample=0.8)
selector = SelectFromModel(estimator=gb,max_features=50)
selector = selector.fit(X_data, y_data)
best_features = X_data.columns[np.where(selector.get_support() == True)[0]]

with open('./best_50_features_gb.json','r') as f:
    json.dump(best_features, f)
print("[Best Features] ",best_features)
