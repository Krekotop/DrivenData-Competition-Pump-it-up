# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 20:34:29 2020

@author: fredh
"""# -*- coding: utf-8 -*-
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

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#let's try to train the model based on one feature only, 'gps_height'

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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from mlxtend.classifier import EnsembleVoteClassifier

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

categorical_feat = X_train.select_dtypes(include = 'object').columns.to_list()
num_pipe_4 = Pipeline([
        ('imputer', IterativeImputer(missing_values = np.nan,max_iter=15, random_state=0)),
        ('scaler', StandardScaler())
                    ])

cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('encoder',OneHotEncoder(handle_unknown = 'ignore'))
        ])
preprocessor = ColumnTransformer(remainder = 'drop',
                         transformers = [('numerical',num_pipe_4, num_feat),
                                        ('categorical', cat_pipe, categorical_feat)]
                       )

model_4 = Pipeline([
        ('preprocessor',preprocessor),
        ('classifier', AdaBoostClassifier())
        ])

model_7 = Pipeline([
        ('preprocessor',preprocessor),
        ('pca', TruncatedSVD(n_components=5)),
        ('classifier', XGBClassifier(objective = 'multi:softmax', booster = 'gbtree', 
                                     nrounds = 'min.error.idx', 
                                     num_class = 4, maximize = False, eval_metric = 'merror', 
                                     eta = .2,
                                     max_depth = 14, colsample_bytree = .4))
        ])
model_3 = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier())
        ])

        
eclf = EnsembleVoteClassifier(clfs=[model_4, model_7, model_3], voting='hard')
eclf.fit(X_train,y_train)
eclf_score = eclf.score(X_train,y_train)

make_submission(eclf,X_test)
