# To test this linear regression library, we will use three datasets. And for each of the three datasets, we will use a single feature each time

# Maybe later we can expand this into a test class where the user can test my LinearRegression class with scikit's LinearRegression class by either
# using a default dataset (which I will provide) or by using their own dataset

# TODO: Implement max iteration

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('expand_frame_repr', False)


def test_model(x, y, alpha):
    model = LinearRegression()
    model.fit(np.reshape(x, (-1, 1)), y)
    expected_m = model.coef_[0]
    expected_b = model.intercept_
    m, b = gradient_descent(x, y, alpha)

    print('Expected m:', expected_m, '   Actual m:', m)
    print('Expected b:', expected_b, '   Actual b:', b)
    print()


alpha = 0.0001
# num_iter = 1000

# Test 1
x = list(range(1, 11))  # checks out. Expected values are m=2, b=0
y = [2 * i for i in x]
test_model(x, y, alpha)

# Test 2
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
y = [52, 74, 79, 95, 115, 110, 129, 126, 147, 146, 156, 184]
test_model(x, y, alpha)

# Test 3
dataframe = pd.read_csv('heights.csv')
x = dataframe['height'].values
y = dataframe['weight'].values
test_model(x, y, alpha)
'''
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
    test_model(x, y.values)
'''