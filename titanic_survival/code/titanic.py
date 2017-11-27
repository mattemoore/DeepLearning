import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt


test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')


# drop df column inplace
def drop_feature(df, column_name):
    df.drop([column_name], axis=1, inplace=True)


# drop mostly incomplete 'Cabin' feature
drop_feature(train, 'Cabin')
drop_feature(test, 'Cabin')

# drop 'Ticket' feature as it is not ordered and has different prefixes
drop_feature(train, 'Ticket')
drop_feature(test, 'Ticket')

# combine train and test sets for mean calculations
test_and_train = test.append(train, ignore_index=True)

# update rows missing Age values with mean of all Ages
mean_age = np.mean(test_and_train['Age'])
train['Age'].fillna(mean_age, inplace=True)
test['Age'].fillna(mean_age, inplace=True)

# update test set rows missing Fare costs
# with mean of all Fare costs
# (train set not missing any)
mean_fare = np.mean(test_and_train['Fare'])
test['Fare'].fillna(mean_fare, inplace=True)

# 2 rows in train set are missing Embarked values, drop them
# (test set not missing any)
train = train.dropna()


def one_hot_encode_feature(df, col_to_enc, enc_col_names):
    one_hot = LabelBinarizer().fit_transform(df[col_to_enc])
    one_hot_df = pd.DataFrame(one_hot, columns=enc_col_names)
    df.drop([col_to_enc], axis=1, inplace=True)
    return pd.concat([df, one_hot_df], axis=1)


# one hot encode the Embarked feature in test and train sets
embarked_values = ['Embark_Cherbourg',
                   'Embark_Queenstown',
                   'Embark_Southampton']

train = one_hot_encode_feature(train, 'Embarked', embarked_values)
test = one_hot_encode_feature(test, 'Embarked', embarked_values)


# normalize fare column

# print dataframe info (check for missing values)
# print(test.info())
# print(train.info())



