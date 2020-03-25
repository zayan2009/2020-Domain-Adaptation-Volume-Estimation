#!/usr/bin/env python
# coding: utf-8

# # Libs

# In[ ]:


import os
import time
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder




def dataset_split(X, Y, split_mode, use_best_features, num_features, use_features=[]):

    if use_best_features:
        with open('../Data/best_50_features_gb.json','r') as f:
            best_features = json.load(f)
        assert num_features < len(best_features)

        X_data = X[best_features[:num_features]]
    elif len(use_features) > 0:
        X_data = X.drop(['linkIdx', 'datetime'], axis=1)[use_features]
    else:
    	X_data = X.drop(['linkIdx', 'datetime'], axis=1)
    Y_data = Y['volume']

    if split_mode is 'random':
        X_train, X_test, y_train, y_test = train_test_split(X_data,
                                                            Y_data,
                                                            test_size=0.2,
                                                            random_state=np.random.randint(0,10000))
    elif split_mode is 'fix_transfer':
        train_ls = [1,2,3,4,5,6,7,8,24,23,22,21,20,19,18,17,15,16]
        test_ls = [9,10,11,12,13,14]

        train_idx = X[X['linkIdx'].isin(train_ls)].index
        test_idx = X[X['linkIdx'].isin(test_ls)].index

        X_train, X_test = X_data.iloc[train_idx, :], X_data.iloc[test_idx, :]
        y_train, y_test = Y_data.iloc[train_idx], Y_data.iloc[test_idx]

    elif split_mode is 'random_transfer':
        test_ls = np.unique(np.random.choice(np.arange(1,25),6))
        train_ls = list(set(np.arange(1,25)) - set(test_ls))

        train_idx = X[X['linkIdx'].isin(train_ls)].index
        test_idx = X[X['linkIdx'].isin(test_ls)].index

        X_train, X_test = X_data.iloc[train_idx, :], X_data.iloc[test_idx, :]
        y_train, y_test = Y_data.iloc[train_idx], Y_data.iloc[test_idx]

        print("Train links: {:s} | Test links: {:s}".format(str(train_ls), str(test_ls)))
    else:
        print("split mode is wrong !")

    return X_train, X_test, y_train, y_test



def one_hot_encoding(X, categorical_features):
    one_hot_enc = OneHotEncoder()
    for feat in categorical_features:
        encoded_matrix = one_hot_enc.fit_transform(X[feat].values.reshape(-1,1)).toarray()
        num_class = encoded_matrix.shape[1]
        one_hot_features = pd.DataFrame()
        for class_ in range(num_class):
            one_hot_features[feat + '_' + str(class_)] = encoded_matrix[:,class_]
        X = pd.concat((X, one_hot_features),axis=1)
    return X



def mae(y_pred, y_test):
    return np.sum(np.abs(y_pred - y_test) * y_test) / np.sum(y_test)



def mape(y_pred, y_test):
    return np.sum(np.abs(y_pred - y_test)) / np.sum(y_test)



def mspe(y_pred, y_test):
    return np.sum(np.square(y_pred - y_test)) / np.sum(np.square(y_test))

