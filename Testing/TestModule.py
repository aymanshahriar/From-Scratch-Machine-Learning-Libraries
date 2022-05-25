"""
This module contains various test cases that test the MyLinearRegression class,
which is inside the MyLinearModel module
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

import Old
from MyLinearModel import MyLinearRegression

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('expand_frame_repr', False)

def test_model(x, y, alpha, diff):
    # Get m and b of sklearn model
    model = LinearRegression()
    sklearn_m = model.coef_
    sklearn_b = model.intercept_

    # Get m and b of old model
    old_m, old_b = Old.gradient_descent(x, y, alpha, diff)

    # Get m and b of new model
    model = MyLinearRegression()
    model.fit(x, y, learning_rate=alpha, diff=diff, iter=100)
    new_m = model.m
    new_b = model.b

    print('sklearn m:', sklearn_m, '   old m:', old_m, '    new m:', new_m)
    print('sklearn b:', sklearn_b, '   old b:', old_b, '    new b:', new_b)
    print()

alpha = 0.01
diff = 0.0000001
num_iter = 1000

# Test 1   Expected values are m=2, b=0
x = list(range(1, 11))
y = [2 * i for i in x]
test_model(x, y, alpha, diff)

# Test 2  Expected values are m = 10.388111888111885, b = 50.22727272727275
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
y = [52, 74, 79, 95, 115, 110, 129, 126, 147, 146, 156, 184]
test_model(x, y, alpha, diff)

# Test 3
dataframe = pd.read_csv('heights.csv')
x = dataframe['height'].values
y = dataframe['weight'].values
test_model(x, y, alpha, diff)

# Test 4
# Import the dataset as a dataframe
dataframe = pd.read_csv('https://raw.githubusercontent.com/Codecademy/datasets/master/streeteasy/manhattan.csv')
# Split the dataset into features (x) and target (y)
X = dataframe[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs']]
y = dataframe['rent']

# For each feature and the target, create a simple linear regression model using sklearn's LinearRegression class and my own linear regression class
# Then for each feature and target, compare the slope (m) and intercept (b) of the two models
for featureName in X.columns:
    x = X[featureName].values
    test_model(x, y.values, alpha=0.01, diff=0.001)

# Test 5
# Testing the total loss function
x = [1, 2, 3]
y = [5, 1, 3]
# y = x
m1 = 1
b1 = 0
# y = 0.5x + 1
m2 = 0.5
b2 = 1
total_loss1 = MyLinearRegression.total_loss(x, y, m1, b1)
total_loss2 = MyLinearRegression.total_loss(x, y, m2, b2)
print(total_loss1, total_loss2)

# Test 6
# Test the step gradient function
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
y = [52, 74, 79, 95, 115, 110, 129, 126, 147, 146, 156, 184]
m=0
b=0
model = MyLinearRegression()
m, b = model.step_gradient(x, y, m, b, 0.01)
print(m, b)

# Test 7
# Tests Multiple Linear Regression
# Got this test from: https://medium.com/analytics-vidhya/implementing-gradient-descent-for-multi-linear-regression-from-scratch-3e31c114ae12
from sklearn.datasets import load_boston
boston = load_boston()
X = boston.data
Y = boston.target
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_transform=sc.fit_transform(X)
model = MyLinearRegression()
model.fit(X_transform, Y, iter=2000)
print("m (coefficients):\n", model.m)
print("b (intercept/bias):\n", model.b)
print()
y_pred = model.predict(X_transform)
df_pred = pd.DataFrame()
df_pred["y_actual"] = Y
df_pred['y_pred'] = y_pred
print(df_pred.tail(5))
