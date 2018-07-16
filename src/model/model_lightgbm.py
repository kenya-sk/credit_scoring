#! /usr/bin/env python

import numpy as np 
import pandas as pd 
import lightgbm as lgb

# For model estimation
from sklearn.preprocessing import LabelEncoder,MinMaxScaler, Imputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def lightgbm_main(train_X_path, train_y_path, test_path):
    X = np.load(train_X_path)
    y = np.load(train_y_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val)

    params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'metric': {'multi_logloss'},
            'num_class': 3,
            'learning_rate': 0.1,
            'num_leaves': 23,
            'min_data_in_leaf': 1,
            'num_iteration': 100,
            'verbose': 1,
            'n_jobs': -1
    }

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=200,
                )

    lgb_val_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)[:, 1]
    val_auc= roc_auc_score(np.array(y_val), lgb_val_pred)
    print("validation ROC AUC score: {}".format(val_auc))

    # submission data
    test_data = np.load(test_path)
    print("test data shape: {}".format(test_data.shape))
    test_pred = gbm.predict(test_data, num_iteration=gbm.best_iteration)[:, 1]
    print("pred data shape: {}".format(test_pred.shape))
    return test_pred


if __name__ == "__main__":
    train_X_path = "../../all/train_X.npy"
    train_y_path = "../../all/train_target.npy"
    test_path = "../../all/test.npy"
    lightgbm_main(train_X_path, train_y_path, test_path)