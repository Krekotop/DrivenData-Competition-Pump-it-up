"""
Created on Thu Apr 16 15:00:33 2020

@author: fredh
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.core import Dense,Activation
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
from keras.wrappers.scikit_learn import KerasClassifier
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
num_pipe = Pipeline([
        ('imputer', IterativeImputer(max_iter=10, random_state=0)),
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


import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve

from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

from sklearn import  datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline


def create_model():

    model = Sequential()
    model.add(Dense(100, input_dim=64))
    model.add(Activation('tanh'))

    """
    #other layer
    model.add(Dense(500))
    model.add(Activation('tanh'))
    """

    model.add(Dense(10))
    model.add(Activation('softmax'))
    # Compile model
    model.compile(loss = 'binary_crossentropy', optimizer = 'adadelta', metrics=['accuracy'])
    return model

rbm = BernoulliRBM(random_state=0, verbose=True)

#This is the model you want. it is in sklearn format
clf = KerasClassifier(build_fn=create_model, verbose=0)

classifier = Pipeline(steps=[('rbm', rbm), ('VNN', clf)])

#%%
# Training

# Hyper-parameters. These were set by cross-validation,
# using a GridSearchCV. Here we are not performing cross-validation to
# save time.
rbm.learning_rate = 0.06
rbm.n_iter = 20
# More components tend to give better prediction performance, but larger
# fitting time
rbm.n_components = 64

#adapt targets to hot matrix
yTrain = np_utils.to_categorical(y_train, 10)
# Training RBM-Logistic Pipeline
classifier.fit(X_train, yTrain)

