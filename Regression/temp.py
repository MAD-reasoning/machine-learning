# -*- coding: utf-8 -*-

# libraries to import
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Read files
data = pd.read_csv('Salary_Data.csv')
x = data['YearsExperience'].values
y = data['Salary'].values

# data transform
x = x.reshape((len(x),1))

# split data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# predicting y
y_predc = regressor.predict(x_test)

# score
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(y_true=y_test, y_pred=y_predc))

# plot
plt.scatter(x, y, color='orange')
plt.plot(x_test, y_predc)
plt.show()