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

import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LeakyReLU, BatchNormalization
from tensorflow.keras.layers import Dense, Reshape, Activation, Dropout, Flatten
from tensorflow.keras.layers import Embedding, Concatenate, Add, Conv1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import he_normal, constant
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical, normalize

from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor

import lightgbm as lgb

from utils import *
from feat_eng import engineering


X = pd.read_csv('../Data/X.csv')
Y = pd.read_csv('../Data/Y.csv')

# load feature type columns
with open('../Data/feature_types.json', 'r') as f:
    categorical_features, numeric_features = json.load(f)

# one-hot encoding cat features
X_onehot = one_hot_encoding(X, categorical_features)

X_train, X_test, y_train, y_test = dataset_split(X, Y,
                                                 split_mode='fix_transfer',
                                                 use_features=[],
                                                 use_best_features=False,
                                                 num_features=0)
current_cat_feats = list(set(X_train.columns).intersection(set(categorical_features)))

for num_rbst_feat in np.arange(10, 50):
    X_data = pd.concat((X_train, X_test),axis=0)
    y_data = np.concatenate((np.zeros((X_train.shape[0],1)), np.ones((X_test.shape[0],1))),axis=0)

    params = {
        'objective':'binary',
        'boosting':'gbdt',
        'metric':'binary_logloss',
        'num_rounds':2000,
        'learning_rate':0.01,
        'max_depth':3,
        'num_leaves':5,
        'subsample':0.1,
        'bagging_fraction':0.1,
        'bagging_freq':100,
        'verbose':0
    }

    train_data = lgb.Dataset(X_data, y_data.reshape(-1),
                             categorical_feature=current_cat_feats)
    gbm = lgb.train(params, train_data,
                    valid_sets=[train_data],
                    valid_names=['train'],
                    verbose_eval=500)
    clf_features = get_gbm_best_k_features(gbm, k = 5,return_score=False)


    robust_features = set(X_data.columns.tolist()) - set(clf_features)
    X_data = X_data[robust_features]

    def build_mlp_clf(input_shape):
        x_in = Input(shape=(input_shape,))

        def dense_block(h, units):
            h = Dense(units=units, use_bias=True,
                      activation=None,
                      kernel_initializer=he_normal(),
                      bias_initializer=constant(0.0))(h)
            h = BatchNormalization()(h)
            h = LeakyReLU(0.2)(h)
            h = Dropout(rate=0.5)(h)
            return h

        h = dense_block(x_in, units=32)
        h = dense_block(h, units=16)
        h = Dense(units=1, use_bias=False,
                  activation='sigmoid',
                  kernel_initializer='normal',
                  bias_initializer=constant(0.0))(h)

        mlp_clf = Model(inputs=x_in, outputs=h)
        mlp_clf.compile(loss='binary_crossentropy', optimizer=Adam(5e-4), metrics=['accuracy'])

        return mlp_clf

    mlp_clf = build_mlp_clf(input_shape=X_data.shape[1])
    hist = mlp_clf.fit(X_data, y_data, batch_size=512, epochs=20, shuffle=True, verbose=0)

    mlp_weights = mlp_clf.predict(X_train[robust_features].values)
    np.save('../Data/fix_nn_weights.npy',mlp_weights)

    current_cat_feats = list(set(current_cat_feats).intersection(set(robust_features)))
    sp = time.time()
    train_data = lgb.Dataset(X_train[robust_features], y_train,
                             categorical_feature=current_cat_feats)
    train_data.set_weight(mlp_weights.reshape(-1))
    test_data = lgb.Dataset(X_test[robust_features], y_test, reference=train_data)

    params = {
        'objective':'regression',
        'boosting':'gbdt',
        'metric':'mae',
        'num_rounds':20000,
        'learning_rate':0.001,
        'max_depth':8,
        'num_leaves':100,
        'feature_fraction':0.5,
        'bagging_fraction':0.5,
        'extra_trees':True,
        'bagging_freq':200,
        'verbose':0
    }

    gbm = lgb.train(params, train_data,
                    valid_sets=[test_data, train_data],
                    valid_names=['test','train'],
                    verbose_eval=500,
                    early_stopping_rounds=100)
    print("[Duration] {:.2f}".format(time.time() - sp))

    y_pred = gbm.predict(X_test[robust_features],num_iteration=gbm.best_iteration)

    print("[LightGBM] mae: {:.2f} | mape: {:.2f}% | mspe: {:.2f}%".format(
        mae(y_pred, y_test),
        100 * mape(y_pred,y_test), 100 * mspe(y_pred, y_test)))

    print("\n\n\n")