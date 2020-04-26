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
import numpy as np
categorical_feat = X_train.select_dtypes(include = 'object').columns.to_list()
num_pipe_4 = Pipeline([
        ('imputer', IterativeImputer(missing_values = np.nan,max_iter=15, random_state=0)),
        ('scaler', StandardScaler())
                    ])

cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy = 'constant')),
        ('encoder',OneHotEncoder(handle_unknown = 'ignore'))
        ])
preprocessor = ColumnTransformer(remainder = 'drop',
                         transformers = [('numerical',num_pipe_4, num_feat),
                                        ('categorical', cat_pipe, categorical_feat)]
                       )
model_4 = Pipeline([
        ('preprocessor',preprocessor),
        ('classifier', RandomForestClassifier(n_jobs = -1, n_estimators = 150))])
"""min_samples_split = 5, min_samples_leaf = 1, 
max_features = 'auto',
max_depth = None, bootstrap = True"""
model_4.fit(X_train, y_train)

model_4_score = model_4.score(X_train,y_train)

make_submission(model_4,X_test)
"""
import eli5
onehot_columns = list(model_4.named_steps['preprocessor'].named_transformers_['categorical'].named_steps['encoder'].get_feature_names(input_features=categorical_feat))
numeric_features_list = list(num_feat)
numeric_features_list.extend(onehot_columns)
p = eli5.explain_weights(model_4.named_steps['classifier'], top=50, feature_names=numeric_features_list)


ohe = (model_4.named_steps['preprocessor']
         .named_transformers_['categorical']
         .named_steps['encoder'])
feature_names = ohe.get_feature_names(input_features=categorical_feat)
feature_names = np.r_[feature_names, num_feat]

tree_feature_importances = (
    model_4.named_steps['classifier'].feature_importances_)
sorted_idx = tree_feature_importances.argsort()

y_ticks = np.arange(0, len(feature_names))
fig, ax = plt.subplots()
ax.barh(y_ticks, tree_feature_importances[sorted_idx])
ax.set_yticklabels(feature_names[sorted_idx])
ax.set_yticks(y_ticks)
ax.set_title("Random Forest Feature Importances (MDI)")
fig.tight_layout()
plt.show()


"""
