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
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
categorical_feat = X_train.select_dtypes(include = 'object').columns.to_list()
num_feat = X_train.select_dtypes(include='number').columns.to_list()
num_pipe_7 = Pipeline([
        ('imputer', IterativeImputer(max_iter=10, random_state=0)),
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
from xgboost import XGBClassifier
"""
space = [
Real(0.6, 0.7, name="colsample_bylevel"),
Real(0.6, 0.7, name="colsample_bytree"),
Real(0.01, 1, name="gamma"),
Real(0.0001, 1, name="learning_rate"),
Real(0.1, 10, name="max_delta_step"),
Integer(6, 15, name="max_depth"),
Real(10, 500, name="min_child_weight"),
Integer(10, 100, name="n_estimators"),
Real(0.1, 100, name="reg_alpha"),
Real(0.1, 100, name="reg_lambda"),
Real(0.4, 0.7, name="subsample")
]
"""
pt = []
i = 3
while i<11:
    model_7 = Pipeline([
                        ('ct',ct_7),
                        ('pca', TruncatedSVD(n_components=15)),
                        ('classifier', XGBClassifier(
                        learning_rate =0.1,
                        n_estimators=150,
                        max_depth=i,
                        min_child_weight=1,
                        gamma=0,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective= 'binary:logistic',
                        nthread=4,
                        scale_pos_weight=1,
                        seed=27))
        ])
        
    model_7.fit(X_train, y_train)
    model_7_score = model_7.score(X_train,y_train)
    pt.append(model_7_score)
    i+=2
#make_submission(model_7,X_test)