import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas

#load data
X = []
Y = []
X = pandas.read_csv('data_1d.csv', header=None, usecols=[0], squeeze=True)
Y = pandas.read_csv('data_1d.csv', header=None, usecols=[1], squeeze=True)

# convert to numpy arrays
X = np.array(X)
Y = np.array(Y)
print(X.shape)
print(Y.shape)

# find a and b for line of best fit (yhat = ax + b)

denonimator = X.dot(X) - X.mean() * X.sum()
a = (X.dot(Y) - Y.mean() * X.sum()) / denonimator
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denonimator

# calculate predicted Y
yhat = a * X + b

# plot line of best fit
plt.scatter(X, Y)
plt.plot(X, yhat)
plt.show()

# check our work with R-squared
# Rsquared = 1 - Sum of square residuals / Sum of Square total
# SSresiduals = Sum(yi - yhat_i)^2
# SStotal = Sum(yi - y_mean)^2
