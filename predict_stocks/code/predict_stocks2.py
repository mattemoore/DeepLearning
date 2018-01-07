# * Predict price of WMT stock a number of days after
#   its fundamentals are released
# * Data available free from Quandl.com


import datetime as dt
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import utils


NUM_DAYS_OUT = 15

# Load quarterly fundamentals, save last as that is the one we want to predict
fund = utils.load_wmt_fund()

# Load stock ticker info
eod = utils.load_wmt_eod()

# Load pre-announce earning info
pre = utils.load_wmt_pre()

# add closest eod[adj_close] to fund[reportperiod] as a feature
for i in range(len(fund)):
    fund_date = fund.loc[i, 'reportperiod']
    close_on_fund_date = eod.iloc[eod.index.get_loc(fund_date, method='ffill')]['Adj_Close']
    fund.loc[i, 'close_reportperiod'] = close_on_fund_date

# add matching sample from pre to fund
# skip last sample in pre as it won't have matching sample in fund
# TODO: why are we missing 'Q' record in pre records for 2015-1-31 and 2017-1-31?
# TODO: add columns of matching pre to fund
'''
for i in range(len(pre) - 1):
    pre_date = pre.iloc[i]['per_end_date']
    mask = fund['reportperiod'] == pre_date
    fund.loc[mask, 'announce_date'] = pre.iloc[i + 1]['announce_date']
print(fund['announce_date'])
print(fund[pd.notnull(fund['announce_date'])])
'''

# remove last sample in fund and save it for prediction
fund_to_predict = fund.iloc[-1]
fund.drop(fund.index[-1], axis=0, inplace=True)
predict_date = fund_to_predict['reportperiod'] + dt.timedelta(days=NUM_DAYS_OUT)

# Targets = eod[adj_close] at fund[reportperiod] + X days (or next available)
targets = pd.DataFrame(np.zeros((len(fund), 1)))
for i in range(len(fund)):
    eoq_45 = fund.loc[i, 'reportperiod'] + dt.timedelta(days=NUM_DAYS_OUT)
    close_45 = eod.iloc[eod.index.get_loc(eoq_45, method='ffill')]['Adj_Close']
    targets.iloc[i] = close_45

# Remove all date columns before scaling
fund_cleaned = fund.drop(['calendardate', 'datekey',
                          'reportperiod', 'lastupdated',
                          'ticker', 'dimension'], axis=1)
features = list(fund_cleaned)
X_train, X_test, y_train, y_test = train_test_split(fund_cleaned, targets)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def clean_sample_for_prediction(dirty_sample):
    sample_no_dates = dirty_sample[6:]
    sample_reshaped = sample_no_dates.values.reshape(1, -1)
    return sample_reshaped.astype(np.float64)


fund_to_predict_cleaned = clean_sample_for_prediction(fund_to_predict)
fund_to_predict_scaled = scaler.transform(fund_to_predict_cleaned)


def create_model(model, X_train, X_test, y_train, y_test,
                 fund_to_predict_scaled, param_grid):
    print('\n--------' + type(model).__name__ + '--------')
    grid_search = GridSearchCV(model, param_grid, scoring='r2')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print('Best score on train set:', grid_search.best_score_)
    print('Best test score:', best_model.score(X_test, y_test))
    print('Prediction:', best_model.predict(fund_to_predict_scaled), predict_date)
    print('Best params:', grid_search.best_params_)
    print('Best estimator:', best_model)
    return grid_search


# create and test models
ridge_reg = Ridge()
param_grid = {'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0]}
ridge_grid = create_model(ridge_reg, X_train, X_test,
                          y_train, y_test,
                          fund_to_predict_scaled, param_grid)

lasso_reg = Lasso()
param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0, 3.0]}
lasso_grid = create_model(lasso_reg, X_train, X_test,
                          y_train, y_test,
                          fund_to_predict_scaled, param_grid)

elastic_reg = ElasticNet()
param_grid = {'alpha': [0.5, 0.75, 1.0, 2.0, 3.0],
              'l1_ratio': [0.1, 0.25, 0.5, 0.75, 1.0]}
elastic_grid = create_model(elastic_reg, X_train, X_test,
                            y_train, y_test,
                            fund_to_predict_scaled, param_grid)

neural_reg = MLPRegressor()
t = ()
for i in range(100):
    t = t + (100,)
param_grid = {'hidden_layer_sizes': [t],
              'solver': ['lbfgs'],
              'max_iter': [10000]}
neural_grid = create_model(neural_reg, X_train, X_test,
                           y_train.values.ravel(), y_test.values.ravel(),
                           fund_to_predict_scaled, param_grid)
