# -*- coding: utf-8 -*-
from tqdm import tqdm
from collections import Counter
import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from xgboost import plot_importance
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import confusion_matrix, cohen_kappa_score, mean_squared_error, f1_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
import json
import pandas as pd
import numpy as np
import warnings
import random
import time
import os
import matplotlib.pyplot as plt
import multiprocessing
import seaborn as sns
from multiprocessing import Lock, Process, Queue, current_process
import scipy as sp
from functools import partial
from numba import jit
from sklearn.preprocessing import OneHotEncoder
import gc
pd.set_option('display.max_columns', 1000)
warnings.filterwarnings('ignore')


DATA_PATH = ""#"../input/liverpool-ion-switching"

train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
submission_df = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))

#Feature engineering

from tqdm import tqdm_notebook
for batch_size in tqdm_notebook([10000, 25000, 50000]):
    train['batch'] = ((train['time'] * 10000) - 1) // batch_size
    train['batch_index'] = ((train['time'] * 10000) - 1)  - (train['batch'] * batch_size) #10_000 is same as 10000.
    train['batch_slices'] = train['batch_index']  // (batch_size / 10)
    train['batch_slices2'] = train['batch'].astype(str).str.zfill(3) + '_' + train['batch_slices'].astype(str).str.zfill(3) #zfill() is zero patting function
    
    for agg_feature in ['batch', 'batch_slices2']:
        train[f"min_{agg_feature}_{batch_size}"] = train.groupby(agg_feature)['signal'].transform('min')
        train[f"max_{agg_feature}_{batch_size}"] = train.groupby(agg_feature)['signal'].transform('max')
        train[f"std_{agg_feature}_{batch_size}"] = train.groupby(agg_feature)['signal'].transform('std')
        train[f"mean_{agg_feature}_{batch_size}"] = train.groupby(agg_feature)['signal'].transform('mean')
        
        train[f"min_{agg_feature}_{batch_size}_diff"] = train[f"min_{agg_feature}_{batch_size}"] - train['signal']
        train[f"max_{agg_feature}_{batch_size}_diff"] = train[f"max_{agg_feature}_{batch_size}"] - train['signal']
        train[f"std_{agg_feature}_{batch_size}_diff"] = train[f"std_{agg_feature}_{batch_size}"] - train['signal']
        train[f"mean_{agg_feature}_{batch_size}_diff"] = train[f"mean_{agg_feature}_{batch_size}"] - train['signal']
        
        train[f'signal_shift+1_{agg_feature}_{batch_size}'] = train.groupby([agg_feature]).shift(1)['signal']
        train[f'signal_shift-1_{agg_feature}_{batch_size}'] = train.groupby([agg_feature]).shift(-1)['signal']
        #train[f'signal_shift+2_{agg_feature}_{batch_size}'] = train.groupby([agg_feature]).shift(2)['signal']
        train[f'signal_shift-2_{agg_feature}_{batch_size}'] = train.groupby([agg_feature]).shift(-2)['signal']
        train[f'signal_shift-3_{agg_feature}_{batch_size}'] = train.groupby([agg_feature]).shift(-3)['signal'] #add
        train[f'signal_shift-4_{agg_feature}_{batch_size}'] = train.groupby([agg_feature]).shift(-4)['signal'] #add
        
        gc.collect()

    window_sizes = [5, 10, 1000, 5000]#100, 500,10, 25, 50, 25000, 30000]
    for window in window_sizes:
        train["rolling_maen_" + str(window) + '_batch_' + str(batch_size)] = train.groupby('batch')['signal'].rolling(window=window).mean().reset_index()['signal']
        train["rolling_std_" + str(window) + '_batch_' + str(batch_size)] = train.groupby('batch')['signal'].rolling(window=window).std().reset_index()['signal']
        #train["rolling_var_" + str(window) + '_batch_' + str(batch_size)] = train.groupby('batch')['signal'].rolling(window=window).var().reset_index()['signal']
        train["rolling_min_" + str(window) + '_batch_' + str(batch_size)] = train.groupby('batch')['signal'].rolling(window=window).min().reset_index()['signal']
        train["rolling_max_" + str(window) + '_batch_' + str(batch_size)] = train.groupby('batch')['signal'].rolling(window=window).max().reset_index()['signal']
        
        train["rolling_std_" + str(window) + '_batch_' + str(batch_size)+'_diff'] = train.groupby('batch')['signal'].rolling(window=window).std().reset_index()['signal'] - train['signal'] #add
        train["rolling_max_" + str(window) + '_batch_' + str(batch_size)+'_diff'] = train.groupby('batch')['signal'].rolling(window=window).max().reset_index()['signal'] - train['signal'] #add
        #train["rolling_median_" + str(window) + '_batch_' + str(batch_size)] = train.groupby('batch')['signal'].rolling(window=window).median().reset_index()['signal']
        #train["rolling_min_max_ratio_" + str(window)+ '_batch_' + str(batch_size)] = train["rolling_min_" + str(window)+ '_batch_' + str(batch_size)] / train["rolling_max_" + str(window)+ '_batch_' + str(batch_size)]
        #train["rolling_min_max_diff_" + str(window)+ '_batch_' + str(batch_size)] = train["rolling_max_" + str(window)+ '_batch_' + str(batch_size)] - train["rolling_min_" + str(window)+ '_batch_' + str(batch_size)]
        #train["rolling_min_max_mean_" + str(window)+ '_batch_' + str(batch_size)] = (train["rolling_max_" + str(window)+ '_batch_' + str(batch_size)] + train["rolling_min_" + str(window)+ '_batch_' + str(batch_size)])/2
        #a = (train['signal'] - train['rolling_min_' + str(window)+ '_batch_' + str(batch_size)]) / (train['rolling_max_' + str(window)+ '_batch_' + str(batch_size)] - train['rolling_min_' + str(window)+ '_batch_' + str(batch_size)])
        #train["norm_" + str(window)+ '_batch_' + str(batch_size)] = a * (np.floor(train['rolling_max_' + str(window)+ '_batch_' + str(batch_size)]) - np.ceil(train['rolling_min_' + str(window)+ '_batch_' + str(batch_size)]))
        #del a
        gc.collect()

        ewma = pd.Series.ewm

        #train[f'exp_Moving__{window}_{batch_size}'] = train.groupby('batch')['signal'].apply(lambda x: x.ewm(alpha=0.5, adjust=False).mean())
train.fillna(0, inplace=True)

train=train.drop(["batch_slices2"],axis=1) #drop categoricaldata
gc.collect()

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

train=reduce_mem_usage(train)


from tqdm import tqdm_notebook
for batch_size in tqdm_notebook([10000, 25000, 50000]):
    test['batch'] = ((test['time'] * 10000) - 1) // batch_size
    test['batch_index'] = ((test['time'] * 10000) - 1)  - (test['batch'] * batch_size) #10_000 is same as 10000.
    test['batch_slices'] = test['batch_index']  // (batch_size / 10)
    test['batch_slices2'] = test['batch'].astype(str).str.zfill(3) + '_' + test['batch_slices'].astype(str).str.zfill(3) #zfill() is zero patting function
    
    for agg_feature in ['batch', 'batch_slices2']:
        test[f"min_{agg_feature}_{batch_size}"] = test.groupby(agg_feature)['signal'].transform('min')
        test[f"max_{agg_feature}_{batch_size}"] = test.groupby(agg_feature)['signal'].transform('max')
        test[f"std_{agg_feature}_{batch_size}"] = test.groupby(agg_feature)['signal'].transform('std')
        test[f"mean_{agg_feature}_{batch_size}"] = test.groupby(agg_feature)['signal'].transform('mean')
        
        test[f"min_{agg_feature}_{batch_size}_diff"] = test[f"min_{agg_feature}_{batch_size}"] - test['signal']
        test[f"max_{agg_feature}_{batch_size}_diff"] = test[f"max_{agg_feature}_{batch_size}"] - test['signal']
        test[f"std_{agg_feature}_{batch_size}_diff"] = test[f"std_{agg_feature}_{batch_size}"] - test['signal']
        test[f"mean_{agg_feature}_{batch_size}_diff"] = test[f"mean_{agg_feature}_{batch_size}"] - test['signal']
        
        test[f'signal_shift+1_{agg_feature}_{batch_size}'] = test.groupby([agg_feature]).shift(1)['signal']
        test[f'signal_shift-1_{agg_feature}_{batch_size}'] = test.groupby([agg_feature]).shift(-1)['signal']
        #test[f'signal_shift+2_{agg_feature}_{batch_size}'] = test.groupby([agg_feature]).shift(2)['signal']
        test[f'signal_shift-2_{agg_feature}_{batch_size}'] = test.groupby([agg_feature]).shift(-2)['signal']
        test[f'signal_shift-3_{agg_feature}_{batch_size}'] = test.groupby([agg_feature]).shift(-3)['signal'] #add
        test[f'signal_shift-4_{agg_feature}_{batch_size}'] = test.groupby([agg_feature]).shift(-4)['signal'] #add
        gc.collect()

    window_sizes = [5, 10, 1000, 5000]#10, 25, 50, 100, 500, , 25000, 30000
    for window in window_sizes:
        test["rolling_maen_" + str(window) + '_batch_' + str(batch_size)] = test.groupby('batch')['signal'].rolling(window=window).mean().reset_index()['signal']
        test["rolling_std_" + str(window) + '_batch_' + str(batch_size)] = test.groupby('batch')['signal'].rolling(window=window).std().reset_index()['signal']
        #test["rolling_var_" + str(window) + '_batch_' + str(batch_size)] = test.groupby('batch')['signal'].rolling(window=window).var().reset_index()['signal']
        test["rolling_min_" + str(window) + '_batch_' + str(batch_size)] = test.groupby('batch')['signal'].rolling(window=window).min().reset_index()['signal']
        test["rolling_max_" + str(window) + '_batch_' + str(batch_size)] = test.groupby('batch')['signal'].rolling(window=window).max().reset_index()['signal']
        
        test["rolling_std_" + str(window) + '_batch_' + str(batch_size)+'_diff'] = test.groupby('batch')['signal'].rolling(window=window).std().reset_index()['signal'] - test['signal']#add
        test["rolling_max_" + str(window) + '_batch_' + str(batch_size)+'_diff'] = test.groupby('batch')['signal'].rolling(window=window).max().reset_index()['signal'] - test['signal']#add
        #test["rolling_median_" + str(window) + '_batch_' + str(batch_size)] = test.groupby('batch')['signal'].rolling(window=window).median().reset_index()['signal']
        #test["rolling_min_max_ratio_" + str(window)+ '_batch_' + str(batch_size)] = test["rolling_min_" + str(window)+ '_batch_' + str(batch_size)] / test["rolling_max_" + str(window)+ '_batch_' + str(batch_size)]
        #test["rolling_min_max_diff_" + str(window)+ '_batch_' + str(batch_size)] = test["rolling_max_" + str(window)+ '_batch_' + str(batch_size)] - test["rolling_min_" + str(window)+ '_batch_' + str(batch_size)]
        #test["rolling_min_max_mean_" + str(window)+ '_batch_' + str(batch_size)] = (test["rolling_max_" + str(window)+ '_batch_' + str(batch_size)] + test["rolling_min_" + str(window)+ '_batch_' + str(batch_size)])/2
        #a = (test['signal'] - test['rolling_min_' + str(window)+ '_batch_' + str(batch_size)]) / (test['rolling_max_' + str(window)+ '_batch_' + str(batch_size)] - test['rolling_min_' + str(window)+ '_batch_' + str(batch_size)])
        #test["norm_" + str(window)+ '_batch_' + str(batch_size)] = a * (np.floor(test['rolling_max_' + str(window)+ '_batch_' + str(batch_size)]) - np.ceil(test['rolling_min_' + str(window)+ '_batch_' + str(batch_size)]))

        #del a
        gc.collect()
        ewma = pd.Series.ewm

        #test[f'exp_Moving__{window}_{batch_size}'] = test.groupby('batch')['signal'].apply(lambda x: x.ewm(alpha=0.5, adjust=False).mean())
test.fillna(0, inplace=True)

test=test.drop(["batch_slices2"],axis=1) #drop categoricaldata
gc.collect()
test=reduce_mem_usage(test)

train=train.drop(columns=['time'])
test=test.drop(columns=['time'])

#helper function
def lgb_Metric(y_true, y_pred):
    #labels = dtrain.get_label()
    #print(preds.shape)
    #print(preds)
    y_pred = np.round(np.clip(y_pred, 0, 10)).astype(int)#np.argmax(y_pred, axis=0)
    y_pred = np.array(y_pred).reshape(y_true.shape)
#     score = metrics.cohen_kappa_score(labels, preds, weights = 'quadratic')
    score = f1_score(y_true=y_true ,y_pred=y_pred, average='macro')
    return ('KaggleMetric', score, True)

#define model
class Base_Model(object):
    
    def __init__(self, train_df, test_df, features, categoricals=[], n_splits=5, verbose=True,ps={}):
        self.train_df = train_df
        self.test_df = test_df
        self.features = features
        self.n_splits = n_splits
        self.categoricals = categoricals
        self.target = 'open_channels' #set your objective variable here
        self.cv = self.get_cv()
        self.verbose = verbose
#         self.params = self.get_params()
        self.params = self.set_params(ps)
        self.y_pred, self.score, self.model = self.fit()
        
    def train_model(self, train_set, val_set):
        raise NotImplementedError
        
    #split data x and y, and kfold
    def get_cv(self):
        #cv = GroupKFold(n_splits=self.n_splits)
        cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        #cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        return cv.split(self.train_df, self.train_df[self.target])
    
    def get_params(self):
        raise NotImplementedError
        
    def convert_dataset(self, x_train, y_train, x_val, y_val):
        raise NotImplementedError
        
    def convert_x(self, x):
        return x
        
    def fit(self):
        oof_pred = np.zeros((len(train_df), ))
        y_pred = np.zeros((len(test_df), ))
        for fold, (train_idx, val_idx) in enumerate(self.cv):
            x_train, x_val = self.train_df[self.features].iloc[train_idx], self.train_df[self.features].iloc[val_idx]
            y_train, y_val = self.train_df[self.target][train_idx], self.train_df[self.target][val_idx]
            train_set, val_set = self.convert_dataset(x_train, y_train, x_val, y_val)
            model = self.train_model(train_set, val_set)
            conv_x_val = self.convert_x(x_val)
            oof_pred[val_idx] = model.predict(conv_x_val).reshape(oof_pred[val_idx].shape)
            x_test = self.convert_x(self.test_df[self.features])
            y_pred += model.predict(x_test).reshape(y_pred.shape) / self.n_splits
            print('Partial score of fold {} is: {}'.format(fold, lgb_Metric(y_val, oof_pred[val_idx])[1]))
        _, loss_score, _ = lgb_Metric(self.train_df[self.target], oof_pred)
        if self.verbose:
            print('Our macroF1 is: ', loss_score)
        return y_pred, loss_score, model


import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

class Nn_Model(Base_Model):
    
    def __init__(self, train_df, test_df, features, categoricals=[], n_splits=4, verbose=True):
        features = features.copy()
        if len(categoricals) > 0:
            for cat in categoricals:
                enc = OneHotEncoder()
                train_cats = enc.fit_transform(train_df[[cat]])
                test_cats = enc.transform(test_df[[cat]])
                cat_cols = ['{}_{}'.format(cat, str(col)) for col in enc.active_features_]
                features += cat_cols
                train_cats = pd.DataFrame(train_cats.toarray(), columns=cat_cols)
                test_cats = pd.DataFrame(test_cats.toarray(), columns=cat_cols)
                train_df = pd.concat([train_df, train_cats], axis=1)
                test_df = pd.concat([test_df, test_cats], axis=1)
        scalar = MinMaxScaler()
        train_df[features] = scalar.fit_transform(train_df[features])
        test_df[features] = scalar.transform(test_df[features])
        print(train_df[features].shape)
        super().__init__(train_df, test_df, features, categoricals, n_splits, verbose)
        
    def train_model(self, train_set, val_set):
        verbosity = 100 if self.verbose else 0
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(train_set['X'].shape[1],)),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(25, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='relu')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=4e-4), loss='mse')
        print(model.summary())
        save_best = tf.keras.callbacks.ModelCheckpoint('nn_model.w8', save_weights_only=True, save_best_only=True, verbose=1)
        early_stop = tf.keras.callbacks.EarlyStopping(patience=20)
        model.fit(train_set['X'], 
                train_set['y'], 
                validation_data=(val_set['X'], val_set['y']),
                epochs=10,
                 callbacks=[save_best, early_stop])
        model.load_weights('nn_model.w8')
        return model
        
    def convert_dataset(self, x_train, y_train, x_val, y_val):
        train_set = {'X': x_train, 'y': y_train}
        val_set = {'X': x_val, 'y': y_val}
        return train_set, val_set
        
    def get_params(self):
        return None
    def set_params(self, ps={}):
        return self.get_params()

train_df=train
test_df=test
del train
del test

categoricals = [] #In this time, there were no categorical feature
features = test_df.columns

#train NN model
nn_model = Nn_Model(train_df, test_df, features, categoricals=categoricals)

#Define opt rounder
#ref:https://www.kaggle.com/artgor/eda-and-model-f1-optimization
import numpy as np
import scipy as sp
from functools import partial
from sklearn import metrics

class OptimizedRounderF1(object):
    """
    An optimizer for rounding thresholds
    to maximize f1 score
    """
    def __init__(self):
        self.coef_ = 0

    def _f1_loss(self, coef, X, y):
        """
        Get loss according to
        using current coefficients
        
        :param coef: A list of coefficients that will be used for rounding
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        return -metrics.f1_score(y, X_p, average='macro')

    def fit(self, X, y):
        """
        Optimize rounding thresholds
        
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        loss_partial = partial(self._f1_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        """
        Make predictions with specified thresholds
        
        :param X: The raw predictions
        :param coef: A list of coefficients that will be used for rounding
        """
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


    def coefficients(self):
        """
        Return the optimized coefficients
        """
        return self.coef_['x']

#Hevy code. In kaggle submission, this code should be comment-out
y_train_pred=nn_model.model.predict(train_df[features])#, num_iteration=lgb_model.model.best_iteration) #default =100
y_train_true=train_df["open_channels"]
optR = OptimizedRounderF1()
optR.fit(y_train_pred.reshape(-1,), y_train_true)
coefficients=optR.coefficients()
print(coefficients)

opt_pred = optR.predict(y_train_pred.reshape(-1,), coefficients)
print('f1', metrics.f1_score(y_train_true, opt_pred, average = 'macro'))


# Showing Confusion Matrix
# Thanks to https://www.kaggle.com/marcovasquez/basic-nlp-with-tensorflow-and-wordcloud
def plot_cm(y_train_true, y_train_pred, title):
    figsize=(14,14)
    y_train_pred = y_train_pred.astype(int)
    cm = confusion_matrix(y_train_true, y_train_pred, labels=np.unique(y_train_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_train_true), columns=np.unique(y_train_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)

# Showing Confusion Matrix for LGB model
#plot_cm(y_train_true, y_train_pred, 'Confusion matrix for LGB model')
plot_cm(y_train_true, opt_pred, 'Confusion matrix for NN model')
plt.savefig("confusion_matrix_Low_overfitting.png")

#model evaluation
test_preds=nn_model.y_pred
opt_test_preds = optR.predict(test_preds.reshape(-1,), coefficients)
submission_df['open_channels'] = opt_test_preds
submission_df.to_csv('submission_low_overfitting_nn.csv', index=False, float_format='%.4f')

#validate test prediction
test = pd.read_csv('test.csv')
test["open_channels"]=submission_df["open_channels"]
test_after=test[1000000:]
test_after["signal_shift"]=test_after["signal"].shift(1).fillna(0)

#Plot data
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

my_cmap = ListedColormap(['blue','red','green',"black"])
#label=["open_channel=0", "open_channel=1","open_channel=2","open_channel=3"]
plt.scatter(x=test_after["signal"], y=test_after["signal_shift"], c=test_after["open_channels"], cmap=my_cmap)
plt.xlabel("signal",fontsize=20)
plt.ylabel("signal_shift(1)", fontsize=20)
plt.legend()
plt.show()
plt.savefig("test_validation_low_overfit_nn.png")

print(test_after["open_channels"].value_counts())


