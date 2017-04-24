'''Predict stock market prices, make billions.'''

# pylint: disable=invalid-name

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data in numpy array
STOCK_SYMBOL = 'MSFT'
ALL_PRICES = pd.read_csv('../input/prices.csv')
STOCK_PRICES = np.array(ALL_PRICES[ALL_PRICES['symbol'] == STOCK_SYMBOL])
'''
ALL_FUNDAMENTALS = pd.read_csv('../input/fundamentals.csv')
COMPANY_FUNDAMENTALS = \
    ALL_FUNDAMENTALS[ALL_FUNDAMENTALS['Ticker Symbol'] == STOCK_SYMBOL]
'''

# csv column indexes
DATE_COL = 0
SYMBOL_COL = 1
OPEN_COL = 2
CLOSE_COL = 3
LOW_COL = 4
HIGH_COL = 5
VOLUME_COL = 6

# hyper-parameters
WINDOW_SIZE = 30

# X is matrix of features and bias term
X = np.array(
    STOCK_PRICES[WINDOW_SIZE:, [OPEN_COL, LOW_COL, HIGH_COL, VOLUME_COL]],
    dtype='float'
)
X = np.concatenate((X, np.ones((len(X), 1))), axis=1)
num_orig_cols = X.shape[1]

# Y is matrix of actual output values
Y = np.array(
    STOCK_PRICES[WINDOW_SIZE:, CLOSE_COL],
    dtype='float'
)

# Dates are not features but we want to save them for plotting later
dates = np.array(
    STOCK_PRICES[WINDOW_SIZE:, [0]],
    dtype='datetime64'
)

# add historical closing prices to X for 'Rolling Window Linear Regression'
X = np.concatenate(
    (X, np.zeros((len(X), WINDOW_SIZE))),
    axis=1
)
for row in range(len(X)):
    for day in range(1, WINDOW_SIZE + 1):
        col_offset = num_orig_cols - 1 + day
        row_offset = WINDOW_SIZE + row - day
        X[row, col_offset] = STOCK_PRICES[row_offset, CLOSE_COL]

# assert X.shape[1] == (WINDOW_SIZE + num_orig_cols)
# pd.DataFrame(X).to_csv('X.csv')
# pd.DataFrame(X).to_csv('Y.csv')

# seperate training and test sets
# np.random.randn()

# solve for w (weights)
w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
print('w is:', w)

# calculate predicted values based on our model (weights)
Y_hat = X.dot(w)

# convert numpy dates to pandas dates
pd_dates = []
for d in range(len(dates)):
    pd_dates.append(pd.Timestamp(dates[d, 0]))

# plot predicted closing prices against actual
# closing prices across time
plt.scatter(pd_dates, Y)
plt.plot(pd_dates, Y_hat, color='red')
plt.show()


def get_r_squared(actuals, predicted):
    '''Calculate r_squared'''
    d1 = actuals - predicted
    d2 = actuals - actuals.mean()
    r_2 = 1 - d1.dot(d1) / d2.dot(d2)
    print('r_squared is:', r_2)


# calculate r_squared
get_r_squared(Y, Y_hat)
