
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
from sklearn.decomposition import TruncatedSVD
# this will save answers in csv file 
def make_submission(model, X_test):
    y_test_pred = model.predict(X_test)
    predictions = pd.Series(data = y_test_pred,
                            index = X_test.index,
                            name = 'status_group')
    date = pd.Timestamp.now().strftime(format = '%Y-%m-%d_%H-%M_')
    predictions.to_csv(f'C:/Users/fredh/Documents/Data Driven platform competition - Pump it up/predictions/{date}submission.csv',
                       index = True, header = True)

from sklearn.neighbors import KNeighborsClassifier
num_feat = X_train.select_dtypes(include='number').columns.to_list()
from sklearn.impute import SimpleImputer
import numpy as np 
# Let's try to include both numerical and categorical features
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
categorical_feat = X_train.select_dtypes(include = 'object').columns.to_list()
num_pipe = Pipeline([
        ('imputer', IterativeImputer(missing_values = np.nan,max_iter=15, random_state=0)),
        ('scaler', StandardScaler())
                    ])
cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('encoder',OneHotEncoder(handle_unknown = 'ignore'))
        ])
ct = ColumnTransformer(remainder = 'drop',
                         transformers = [('numerical',num_pipe, num_feat),
                                        ('categorical', cat_pipe, categorical_feat)]
                       )

model = Pipeline([('ct',ct),
                  ('pca', TruncatedSVD(n_components=15)),
                  ('classifier', KNeighborsClassifier(n_neighbors = 15))
                    ])    

model.fit(X_train, y_train)

score = model.score(X_train,y_train)

