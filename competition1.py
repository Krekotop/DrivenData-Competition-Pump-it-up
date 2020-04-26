# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:00:33 2020

@author: fredh
"""


import pandas as pd 

def import_data():
    
    X_train = pd.read_csv(r'C:\Users\fredh\Documents\Data Driven platform competition - Pump it up\data\training-set-values.csv', 
                          index_col = [0],
                          dtype = {'region_code':'object', 'district_code': 'object'},
                          parse_dates = ['date_recorded'],
                          infer_datetime_format = True)

    y_train = pd.read_csv(r'C:\Users\fredh\Documents\Data Driven platform competition - Pump it up\data\training-set-labels.csv', 
                          index_col = [0])
    y_train = y_train.status_group

    X_test = pd.read_csv(r'C:\Users\fredh\Documents\Data Driven platform competition - Pump it up\data\test-set-values.csv', 
                         index_col = [0],
                         )
    return X_train, y_train, X_test

X_train, y_train, X_test = import_data()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def make_submission(model, X_test):
    y_test_pred = model.predict(X_test)
    predictions = pd.Series(data = y_test_pred,
                            index = X_test.index,
                            name = 'status_group')
    date = pd.Timestamp.now().strftime(format = '%Y-%m-%d_%H-%M_')
    predictions.to_csv(f'C:/Users/fredh/Documents/Data Driven platform competition - Pump it up/predictions/{date}submission.csv',
                       index = True, header = True)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
import scipy.stats as stats
from scipy.stats import loguniform
from time import time
import numpy as np
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

categorical_feat = X_train.select_dtypes(include = 'object').columns.to_list()
num_feat = X_train.select_dtypes(include='number').columns.to_list()
num_pipe_7 = Pipeline([
        ('imputer', SimpleImputer(missing_values=0,strategy = 'mean')),
        ('scaler', StandardScaler())
                    ])
cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('encoder',OneHotEncoder(handle_unknown = 'ignore'))
        ])
ct_7 = ColumnTransformer(remainder = 'drop',
                         transformers = [('numerical',num_pipe_7, num_feat),
                                        ('categorical', cat_pipe, categorical_feat)]
                       )

model_7 = Pipeline([
        ('ct',ct_7),
        ('pca', TruncatedSVD(n_components=15)),
        ('classifier', SVC())
        ])

param_dist = {'classifier__C': [1,10,100],
              'classifier__kernel': ['poly', 'rbf', 'sigmoid'],
              'classifier__gamma': [0.1,0.01, 0.001]}
#[0.0001, 0.001,0.01,0.1]
# run randomized search
n_iter_search = 20
grid_search = GridSearchCV(model_7, param_dist )
start = time()
grid_search.fit(X_train, y_train)
print("GridSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(grid_search.cv_results_)
