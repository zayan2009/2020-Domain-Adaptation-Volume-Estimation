#!/usr/bin/env python
# coding: utf-8

# # Libs

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
import numpy as np
import os
import time
import json
import warnings
warnings.filterwarnings("ignore")


def dataset_split(X, Y, split_mode, use_best_features, num_features, use_simple_features):

    if use_best_features:
        with open('../Data/best_50_features_gb.json', 'r') as f:
            best_features = json.load(f)
        assert num_features < len(best_features)

        X_data = X[best_features[:num_features]]
    
    elif use_simple_feature:
        simple_features = ['volume', 'volume_es_p6', 'volume_es_p5', 'volume_es_p7',
                           'through', 'left', 'right', 'thr_left',
                           'thr_right', 'u_turn', 'num_lanes',
                           'weekday', 'interval', 'holiday', 'peak']
        X_data = X.drop(['linkIdx', 'datetime'], axis=1)[use_features]
    
    else:  # use all features
        X_data = X.drop(['linkIdx', 'datetime'], axis=1)
    Y_data = Y['volume']
    
    
    
    if split_mode is 'random':
        X_train, X_test, y_train, y_test = train_test_split(X_data,
                                                            Y_data,
                                                            test_size=0.2,
                                                            random_state=np.random.randint(0, 10000))
    elif split_mode is 'fix_transfer':

        train_ls = np.arange(1, 19)
        test_ls = np.arange(19, 25)

        train_idx = X[X['linkIdx'].isin(train_ls)].index
        test_idx = X[X['linkIdx'].isin(test_ls)].index

        X_train, X_test = X_data.iloc[train_idx, :], X_data.iloc[test_idx, :]
        y_train, y_test = Y_data.iloc[train_idx], Y_data.iloc[test_idx]

    elif split_mode is 'random_transfer':
        test_ls = np.unique(np.random.choice(np.arange(1, 25), 6))
        train_ls = list(set(np.arange(1, 25)) - set(test_ls))

        train_idx = X[X['linkIdx'].isin(train_ls)].index
        test_idx = X[X['linkIdx'].isin(test_ls)].index

        X_train, X_test = X_data.iloc[train_idx, :], X_data.iloc[test_idx, :]
        y_train, y_test = Y_data.iloc[train_idx], Y_data.iloc[test_idx]

        print("Train links: {:s} | Test links: {:s}".format(
            str(train_ls), str(test_ls)))
    else:
        print("split mode is wrong !")

    return X_train, X_test, y_train, y_test


def one_hot_encoding(X, categorical_features):
    one_hot_enc = OneHotEncoder()
    for feat in categorical_features:
        encoded_matrix = one_hot_enc.fit_transform(
            X[feat].values.reshape(-1, 1)).toarray()
        num_class = encoded_matrix.shape[1]
        one_hot_features = pd.DataFrame()
        for class_ in range(num_class):
            one_hot_features[feat + '_' +
                             str(class_)] = encoded_matrix[:, class_]
        X = pd.concat((X, one_hot_features), axis=1)
    return X


def mae(y_pred, y_test):
    return np.sum(np.abs(y_pred - y_test) * y_test) / np.sum(y_test)


def mape(y_pred, y_test):
    return np.sum(np.abs(y_pred - y_test)) / np.sum(y_test)


def mspe(y_pred, y_test):
    return np.sum(np.square(y_pred - y_test)) / np.sum(np.square(y_test))


def get_gbm_best_k_features(booster, k=15, return_score=True):
    featimp = pd.DataFrame(columns=['feature_name', 'importance'])
    featimp['feature_name'] = booster.feature_name()
    featimp['importance'] = booster.feature_importance()
    featimp = featimp.sort_values('importance', ascending=False)
    if return_score:
        return featimp.iloc[:k, :].values.tolist()
    else:
        return featimp.iloc[:k, 0].values.tolist()


def get_logit_best_k_features(logiter, feature_names, k=15, return_score=True):
    featimp = pd.DataFrame(columns=['feature_name', 'importance'])
    featimp['feature_name'] = feature_names
    featimp['importance'] = logiter.coef_
    featimp = featimp.sort_values('importance', ascending=False)
    if return_score:
        return featimp.iloc[:k, :].values.tolist()
    else:
        return featimp.iloc[:k, 0].values.tolist()
