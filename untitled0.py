# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 01:38:08 2020

@author: fredh
"""

cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

from sklearn.metrics import mean_absolute_error

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)  

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)