#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 08:21:37 2018
@author: alok
"""

# Polynomial Regression

# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc()[:,1:2].values 
y = dataset.iloc()[:,2].values


# splitting the dataset into training set and test set.
'''from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)'''

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 3, random_state = 0)
regressor.fit(x, y)

# predicting a new result
y_pred = regressor.predict(6.5)

# visualising the decision tree regression results(for higher resolutionand smoother  curve)
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('truth or bluff (decision for regresson)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()
