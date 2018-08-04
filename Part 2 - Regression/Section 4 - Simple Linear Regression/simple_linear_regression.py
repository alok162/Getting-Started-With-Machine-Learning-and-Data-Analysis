#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 08:21:37 2018

@author: alok
"""

# Simple Linear Regression

# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc()[:,:-1].values
y = dataset.iloc()[:,1].values


# splitting the dataset into training set and test set.
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

'''# feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)'''

#VU250919

# fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# predicting the test set results
y_pred = regressor.predict(x_test)

# visualizing the training set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('salary vs experience (training set) ')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

# visualizing the test set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('salary vs experience (training set) ')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()








