
import pandas as pd 
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
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
import numpy as np 
# Let's try to include both numerical and categorical features
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
categorical_feat = X_train.select_dtypes(include = 'object').columns.to_list()
num_pipe_3 = Pipeline([
        ('imputer', IterativeImputer(missing_values = np.nan,max_iter=15, random_state=0)),
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
kt = [0.000015,0.000023,0.000027]
pt = []
for i in kt:
    model_3 = Pipeline([('ct',ct_3),
                        ('classifier', DecisionTreeClassifier(ccp_alpha = i))
                        ])    
    model_3.fit(X_train, y_train)

    model_3_score = model_3.score(X_train,y_train)
    pt.append(model_3_score)
    make_submission(model_3, X_test)
