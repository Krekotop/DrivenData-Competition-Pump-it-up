# -*- coding: utf-8 -*-
"""
Data driven competition - Pump it up. 
Description taken from "www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/"

Can you predict which water pumps are faulty?
Using data from Taarifa and the Tanzanian Ministry of Water, can you predict which pumps are functional,
which need some repairs, and which don't work at all? This is an intermediate-level practice competition. 
Predict one of these three classes based on a number of variables about what kind of pump is operating, 
when it was installed, and how it is managed. A smart understanding of which waterpoints will fail can 
improve maintenance operations and ensure that clean, potable water is available to communities across Tanzania.
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

# Create Submission Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
#let's try to train the model based on one feature only, 'gps_height'
#columns = ['gps_height', 'installer','water_quality','quality_group','quantity','amount_tsh']
columns = ['gps_height']
ct_1 = ColumnTransformer(remainder = 'drop',
                       transformers = [('select','passthrough', columns)]
                       )
model_1 = Pipeline([('selector', ct_1),
                    ('predictor', DecisionTreeClassifier())]
                   )
model_1.fit(X_train, y_train)

model_1_score = model_1.score(X_train,y_train)
# this will save answers in csv file 
def make_submission(model, X_test):
    y_test_pred = model.predict(X_test)
    predictions = pd.Series(data = y_test_pred,
                            index = X_test.index,
                            name = 'status_group')
    date = pd.Timestamp.now().strftime(format = '%Y-%m-%d_%H-%M_')
    predictions.to_csv(f'C:/Users/fredh/Documents/Data Driven platform competition - Pump it up/predictions/{date}submission.csv',
                       index = True, header = True)

#Let's try to include only numerical features 

num_feat = X_train.select_dtypes(include='number').columns.to_list()
from sklearn.impute import SimpleImputer
num_pipe = Pipeline([
        ('imputer', SimpleImputer(missing_values=0,strategy = 'mean'))
                    ])
ct_2 = ColumnTransformer(remainder = 'drop',
                         transformers = [('numerical',num_pipe, num_feat)]
                       )
model_2 = Pipeline([
        ('ct',ct_2),
        ('classifier', DecisionTreeClassifier())
        ])

model_2.fit(X_train,y_train)

model_2_score = model_2.score(X_train,y_train)

make_submission(model_2,X_test)

# Let's try to include both numerical and categorical features
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
categorical_feat = X_train.select_dtypes(include = 'object').columns.to_list()
num_pipe_3 = Pipeline([
        ('imputer', SimpleImputer(missing_values=0,strategy = 'mean')),
        ('scaler', StandardScaler())
                    ])
cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('encoder',OneHotEncoder(handle_unknown = 'ignore'))
        ])
ct_3 = ColumnTransformer(remainder = 'drop',
                         transformers = [('numerical',num_pipe_3, num_feat),
                                        ('categorical', cat_pipe, categorical_feat)]
                       )
model_3 = Pipeline([
        ('ct',ct_3),
        ('classifier', DecisionTreeClassifier())
        ])

model_3.fit(X_train, y_train)

model_3_score = model_3.score(X_train,y_train)

make_submission(model_3,X_test)

#Trying different classifier

categorical_feat = X_train.select_dtypes(include = 'object').columns.to_list()
num_pipe_4 = Pipeline([
        ('imputer', SimpleImputer(missing_values=0,strategy = 'mean')),
        ('scaler', StandardScaler())
                    ])
cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('encoder',OneHotEncoder(handle_unknown = 'ignore'))
        ])
ct_4 = ColumnTransformer(remainder = 'drop',
                         transformers = [('numerical',num_pipe_4, num_feat),
                                        ('categorical', cat_pipe, categorical_feat)]
                       )
model_4 = Pipeline([
        ('ct',ct_4),
        ('classifier', RandomForestClassifier())
        ])

model_4.fit(X_train, y_train)

model_4_score = model_4.score(X_train,y_train)

make_submission(model_4,X_test)

#include PCA? 
from sklearn.decomposition import TruncatedSVD
categorical_feat = X_train.select_dtypes(include = 'object').columns.to_list()
num_pipe_5 = Pipeline([
        ('imputer', SimpleImputer(missing_values=0,strategy = 'mean')),
        ('scaler', StandardScaler())
                    ])
cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('encoder',OneHotEncoder(handle_unknown = 'ignore'))
        ])
ct_5 = ColumnTransformer(remainder = 'drop',
                         transformers = [('numerical',num_pipe_5, num_feat),
                                        ('categorical', cat_pipe, categorical_feat)]
                       )
model_5 = Pipeline([
        ('ct',ct_4),
        ('pca', TruncatedSVD(n_components=10)),
        ('classifier', RandomForestClassifier(n_jobs = -1))
        ])

model_5.fit(X_train, y_train)

model_5_score = model_5.score(X_train,y_train)

make_submission(model_5,X_test)

#different classifier?
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
categorical_feat = X_train.select_dtypes(include = 'object').columns.to_list()
num_pipe_6 = Pipeline([
        ('imputer', SimpleImputer(missing_values=0,strategy = 'mean')),
        ('scaler', StandardScaler())
                    ])
cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('encoder',OneHotEncoder(handle_unknown = 'ignore'))
        ])
ct_6 = ColumnTransformer(remainder = 'drop',
                         transformers = [('numerical',num_pipe_6, num_feat),
                                        ('categorical', cat_pipe, categorical_feat)]
                       )

model_6 = Pipeline([
        ('ct',ct_6),
        ('pca', TruncatedSVD(n_components=15)),
        ('classifier', SVC(gamma='auto'))
        ])

model_6.fit(X_train, y_train)

model_6_score = model_6.score(X_train,y_train)

make_submission(model_6,X_test)
#different classifier




"""
#Trying different classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
categorical_feat = X_train.select_dtypes(include = 'object').columns.to_list()
num_pipe_5 = Pipeline([
        ('imputer', SimpleImputer(missing_values=0,strategy = 'mean')),
        ('scaler', StandardScaler())
                    ])
cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('encoder',OneHotEncoder(handle_unknown = 'ignore'))
        ])
ct_5 = ColumnTransformer(remainder = 'drop',
                         transformers = [('numerical',num_pipe_5, num_feat),
                                        ('categorical', cat_pipe, categorical_feat)]
                       )
model_5 = Pipeline([
        ('ct',ct_4),
        ('classifier', GradientBoostingClassifier(n_estimators = 300))
        ])


model_5.fit(X_train, y_train)

model_5_score = model_5.score(X_train,y_train)

make_submission(model_5,X_test)
"""
