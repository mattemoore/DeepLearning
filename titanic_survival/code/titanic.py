import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


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

# update rows missing Age values with mean of train Ages
mean_train_age = np.mean(train['Age'])
train['Age'].fillna(mean_train_age, inplace=True)
test['Age'].fillna(mean_train_age, inplace=True)

# update rows missing Fare costs with mean of train Fares
mean_train_fare = np.mean(train['Fare'])
train['Fare'].fillna(mean_train_fare, inplace=True)
test['Fare'].fillna(mean_train_fare, inplace=True)

# fill in missing Embarked values with most common value:
# S=914, C=270, Q=123, NaN=2
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
ss = StandardScaler(copy=False)
ss.fit_transform(train['Fare'].values.reshape(-1, 1))
ss.transform(test['Fare'].values.reshape(-1, 1))
ss.fit_transform(train['Age'].values.reshape(-1, 1))
ss.transform(test['Age'].values.reshape(-1, 1))
ss.fit_transform(train['Parch'].values.reshape(-1, 1))
ss.transform(test['Parch'].values.reshape(-1, 1))
ss.fit_transform(train['SibSp'].values.reshape(-1, 1))
ss.transform(test['SibSp'].values.reshape(-1, 1))

print(train.info())
print(train_labels)
print(train.info())
print(train.loc[1])
print(test.loc[892])


# TODO: balance out the classes!!!!
print(train_labels.value_counts())


nn = MLPClassifier((1000, 1000, 1000, 1000, 1000), activation='relu',
                   solver='sgd', alpha=1e-4, batch_size=20,
                   learning_rate='adaptive', learning_rate_init=1e-1,
                   power_t=0.5, max_iter=200, shuffle=True,
                   tol=1e-4, verbose=True, momentum=0.9,
                   nesterovs_momentum=True, early_stopping=True)

lr = LogisticRegression()
# scores = cross_val_score(nn, train,
#                    train_labels, cv=3)
# print(scores)
