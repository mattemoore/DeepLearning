import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer, StandardScaler
import matplotlib.pyplot as plt


# load data sets
test = pd.read_csv('../input/test.csv', index_col=0)
train = pd.read_csv('../input/train.csv', index_col=0)

# concatenate test and train sets
test_and_train = pd.concat([train, test])


def drop_feature(df, column_name):
    df.drop([column_name], axis=1, inplace=True)


# drop mostly incomplete 'Cabin' feature
drop_feature(test_and_train, 'Cabin')

# drop 'Ticket' feature as it is not orred and has different prefixes
drop_feature(test_and_train, 'Ticket')

# drop 'Name' feature as it most likely has no effect on survivorability
drop_feature(test_and_train, 'Name')

# update rows missing Age values with mean of all Ages
mean_age = np.mean(test_and_train['Age'])
test_and_train['Age'].fillna(mean_age, inplace=True)

# update rows missing Fare costs with mean of all Fare costs
mean_fare = np.mean(test_and_train['Fare'])
test_and_train['Fare'].fillna(mean_fare, inplace=True)

# normalize number features
StandardScaler(copy=False).fit_transform(test_and_train['Fare'])
StandardScaler(copy=False).fit_transform(test_and_train['Age'])
# StandardScaler(copy=False).fit_transform(test_and_train['Parch'])
# StandardScaler(copy=False).fit_transform(test_and_train['SibSp'])

# fill in missing Embarked values with most common value:
# S      914
# C      270
# Q      123
# NaN      2
test_and_train['Embarked'].fillna('S', inplace=True)


# one hot encode category features
def one_hot_encode(df, feature):
    dummies = pd.get_dummies(df[feature],
                             prefix=feature,
                             prefix_sep='_')
    df = pd.concat([df, dummies], axis=1)
    drop_feature(df, feature)
    return df


test_and_train = one_hot_encode(test_and_train, 'Embarked')
test_and_train = one_hot_encode(test_and_train, 'Sex')
test_and_train = one_hot_encode(test_and_train, 'Pclass')

# debug data
print(test_and_train.info())
print(test_and_train.loc[1281])
print(test_and_train.iloc[-1:])

# sanity checks
assert(len(test_and_train) == len(test) + len(train))
assert(test_and_train['Survived'].count() == len(train))

