import pandas as pd
import numpy as np
import util
from datetime import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# load data sets
test = pd.read_csv('../input/test.csv', index_col=0)
train = pd.read_csv('../input/train.csv', index_col=0)


def drop_feature(df, column_name):
    df.drop([column_name], axis=1, inplace=True)


# drop mostly incomplete 'Cabin' feature
drop_feature(test, 'Cabin')
drop_feature(train, 'Cabin')

# drop 'Ticket' feature as not in consistent format
drop_feature(test, 'Ticket')
drop_feature(train, 'Ticket')

# drop 'Name' feature as it most likely has no effect on Survived label
drop_feature(test, 'Name')
drop_feature(train, 'Name')

# update observations missing Age values with mean of train set Ages
mean_train_age = np.mean(train['Age'])
train['Age'].fillna(mean_train_age, inplace=True)
test['Age'].fillna(mean_train_age, inplace=True)

# update observations missing Fare costs with mean of train set Fares
mean_train_fare = np.mean(train['Fare'])
train['Fare'].fillna(mean_train_fare, inplace=True)
test['Fare'].fillna(mean_train_fare, inplace=True)

# fill in missing Embarked values with most common value:
train['Embarked'].fillna('S', inplace=True)
test['Embarked'].fillna('S', inplace=True)


# one hot encode category features
def one_hot_encode(df, feature):
    dummies = pd.get_dummies(df[feature],
                             prefix=feature,
                             prefix_sep='_')
    df = pd.concat([df, dummies], axis=1)
    drop_feature(df, feature)
    return df


train = one_hot_encode(train, 'Embarked')
test = one_hot_encode(test, 'Embarked')
train = one_hot_encode(train, 'Sex')
test = one_hot_encode(test, 'Sex')
train = one_hot_encode(train, 'Pclass')
test = one_hot_encode(test, 'Pclass')

# extract labels from train set
# (test set has no labels)
train_labels = train['Survived'].copy()
drop_feature(train, 'Survived')

# set all dtypes to be same so we can scale number features
train = train.astype(np.float32)
test = test.astype(np.float32)

# normalize number features
scaler = StandardScaler(copy=False)
scaler.fit_transform(train['Fare'].values.reshape(-1, 1))
scaler.transform(test['Fare'].values.reshape(-1, 1))
scaler.fit_transform(train['Age'].values.reshape(-1, 1))
scaler.transform(test['Age'].values.reshape(-1, 1))
scaler.fit_transform(train['Parch'].values.reshape(-1, 1))
scaler.transform(test['Parch'].values.reshape(-1, 1))
scaler.fit_transform(train['SibSp'].values.reshape(-1, 1))
scaler.transform(test['SibSp'].values.reshape(-1, 1))


# train model, print results, pickle grid, return grid
def train_model(estimator, param_grid):
    grid = GridSearchCV(estimator, param_grid, cv=None, scoring=None)
    start_time = dt.now()
    print('Training', type(estimator).__name__, 'model...')
    grid.fit(train, train_labels)
    end_time = dt.now()
    util.print_results(estimator, start_time, end_time, grid, list(train))
    util.save_grid(grid)
    return grid


# build models
best_grid = None

# Best score: 0.830527497194
estimator = SVC()
param_grid = [{
              'kernel': ['poly'],
              'C': [2.0],
              'degree': [3],
              'coef0': [0.9],
              'class_weight': [None]
              }]
grid = train_model(estimator, param_grid)
best_grid = grid

# Best score: 0.83277216610
estimator = SVC()
param_grid = [{
              'kernel': ['rbf'],
              'C': [2.0],
              'gamma': [0.111],
              'class_weight': [None]
              }]
grid = train_model(estimator, param_grid)
if grid.best_score_ > best_grid.best_score_:
        best_grid = grid

# Best score: 0.822671156004
estimator = MLPClassifier()
param_grid = [{
              'activation': ['relu'],
              'solver': ['lbfgs'],
              'alpha': [1e-6],
              'max_iter': [100],
              'batch_size': ['auto'],
              'tol': [1e-4],
              'verbose': [False],
              'hidden_layer_sizes': [(3, 3)],
              'early_stopping': [False]
              }]
grid = train_model(estimator, param_grid)
if grid.best_score_ > best_grid.best_score_:
        best_grid = grid

# Best score: 0.830527497194
estimator = RandomForestClassifier()
param_grid = [{
              'n_estimators': [100],
              'criterion': ['gini'],
              'max_depth': [5, 7],
              'min_samples_split': [2],
              'min_samples_leaf': [1],
              'min_weight_fraction_leaf': [0.0],
              'max_features': ['auto'],
              'max_leaf_nodes': [None],
              'min_impurity_decrease': [0.0],
              'bootstrap': [False],
              'oob_score': [False],
              'verbose': [False],
              'class_weight': [None]
              }]
grid = train_model(estimator, param_grid)
if grid.best_score_ > best_grid.best_score_:
        best_grid = grid

print('Best estimator was:', type(best_grid.best_estimator_).__name__)
print('With best params:', best_grid.best_params_)
print('With best score:', best_grid.best_score_)
