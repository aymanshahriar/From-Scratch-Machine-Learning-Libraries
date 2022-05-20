# Implementing gradient descent based linear regression from scratch
# (That optimizes the coefficients (m1, ..., mn) and slope (b) by minimizing the total loss function using gradient descent)

# This class should have two attributes/variables: coef_ and intercept_
# later I need to incorporate a .fit() method, which will just use gradient_descent
# later look into best default alpha value, allow users to specify their own alpha value
# later look into when to stop doing iteration of gradient descent

# How many iterations of gradient descent should we perform?
# Let us set a precision variable in our algorithm which calculates the difference between two consecutive “x” values .
# If the difference between x values from 2 consecutive iterations is lesser than the precision we set, stop the algorithm !

import warnings
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
np.set_printoptions(linewidth=200)


class MyLinearRegression:

    # The __init__ method is used as the constructor method in python
    def __init__(self, m=[], b=0):
        self.m = []  ## m is now an array
        self.b = b

    # X is now a 2d array, where each row contains the features of a single datapoint
    def total_loss(self, X, y):
        loss_lst = []
        for datapoint, y_value in zip(X, y):  # datapoint is a list that contains the features of a single datapoint. Datapoint can also be called x
            # Find the y-value predicted by the n-dimensional plane
            y_plane = sum([m_i * x_i for m_i, x_i in zip(self.m, datapoint)]) + self.b
            # Calculate the difference between the actual y-value and the y-value predicted by the n-dimensional plane
            difference = y_value - y_plane
            # We square the difference so that points above and below the plane affect the total loss in the same way
            loss = difference ** 2
            loss_lst.append(loss)
        total_loss = sum(loss_lst)
        return total_loss

    @staticmethod
    # X is now a 2d array, where each row contains the features of a single datapoint
    def total_loss(self, X, y, m, b):
        loss_lst = []
        for datapoint, y_value in zip(X, y):  # datapoint is a list that contains the features of a single datapoint. Datapoint can also be called x
            # Find the y-value predicted by the n-dimensional plane
            y_plane = sum([m_i * x_i for m_i, x_i in zip(m, datapoint)]) + b
            # Calculate the difference between the actual y-value and the y-value predicted by the n-dimensional plane
            difference = y_value - y_plane
            # We square the difference so that points above and below the plane affect the total loss in the same way
            loss = difference ** 2
            loss_lst.append(loss)
        total_loss = sum(loss_lst)
        return total_loss

    # This function will find the gradient of the total loss function with respect to b (we will assume that (m_1, ..., m_n) are constants)
    def get_gradient_at_b(self, X, y, m, b):
        summation = sum([y_point-sum([m_i*x_i for m_i, x_i in zip(m, x)])-b for x, y_point in zip(X, y)])
        N = len(y)
        gradient_slope_derivative = (-2/N) * summation
        return gradient_slope_derivative

    # This function will find the gradient of the total loss function with respect to m_i (we will assume that b and (m_1,...m_n-m_i) are constants)
    # i is the index of m where m_i is located
    def get_gradient_at_m(self, X, y, m, i, b):
        summation = sum([ x[i] * (y_point-sum([m_i*x_i for m_i, x_i in zip(m, x)])-b) for x, y_point in zip(X, y)])
        N = len(y)
        gradient_slope_derivative = (-2/N) * summation
        return gradient_slope_derivative

    #TODO: Fixed issue with b, now test?

    # This function will find the gradients at b_current and m_current (remember, m_current is a list), and then return new b and m values that have been moved in that direction
    # The step_gradient funcion will take the data (x and y values), the current value of m, b and alpha, and perform a single iteration of gradient descent
    #    (for both m and b)
    def step_gradient(self, x, y, m, b, learning_rate):
        #print("old b:", b)
        # Find the gradients at the current values of m and b
        m_gradients = []
        for i in range(len(m)):
            m_gradients.append(self.get_gradient_at_m(x, y, m, i, b))
        b_gradient = self.get_gradient_at_b(x, y, m, b)
        # Update the m and b values, moving them in the opposite direction of the slope
        new_m = []
        for i in range(len(m)):
            new_m.append(m[i] - (m_gradients[i] * learning_rate))
        new_b = b - (b_gradient * learning_rate)
        #print("new b:", new_b)
        return new_m, new_b

    # To know when to stop doing iterations of gradient descent, we have a precision value. We compare the values of m (All the values) and b from the previous iteration. If their difference
    #  is less than or equal to the percision value, it means that the values of m and b are berely changing anymore, which means that m and b have reached the bottom of the curve
    #  (ie, m and b are values are optimized to give the minimum loss value)
    def gradient_descent(self, x, y, m, b, learning_rate=0.01, iter=2000, diff=0.0000001):
        current_iter = 1
        current_diff = diff + 1
        while (current_iter <= iter) and (current_diff > diff):
            #print(current_iter)
            old_m = m
            old_b = b
            #print('before',m, b)
            m, b = self.step_gradient(x, y, m, b, learning_rate)
            #print('after',m, b)
            current_iter += 1

            diff_m = abs(np.array(old_m) - np.array(m))
            diff_b = abs(old_b - b)

            diff_1 = np.max(diff_m)
            current_diff = max(diff_1, diff_b)
        # if number of iterations has been exceeded, throw a warning
        if current_iter > iter:
            warnings.warn('Number of itrations exceeded without convergence being reached')
        return m, b

    def fit(self, x, y, learning_rate=0.01, iter=2000, diff=0.0000001):
        num_features = len(x[0])
        m = [0] * num_features
        b = 0
        self.m, self.b = self.gradient_descent(x, y, m, b, learning_rate, iter, diff)

    def predict(self, X):
        # Assume x is a numpy arr
        #return [(self.m * x[0]) + self.b for x in X]

        return [sum([m_i*x_i for m_i,x_i in zip(self.m, x)]) + self.b           for x in X ]

    def set_params(self, m=None, b=None):
        if m is not None:
            self.m = m
        if b is not None:
            self.b = b

    def compare_parameters_with_sklearn(self, x, y):
        sklearn_model = LinearRegression()
        sklearn_model.fit(x, y)
        sklearn_m = sklearn_model.coef_[0]
        sklearn_b = sklearn_model.intercept_
        return {'MyLinearRegression model parameters:': [self.m, self.b],
                'Sklearn model parameters:': [sklearn_m, sklearn_b]}


# Got this test from: https://medium.com/analytics-vidhya/implementing-gradient-descent-for-multi-linear-regression-from-scratch-3e31c114ae12
'''from sklearn.datasets import load_boston
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

y_pred = model.predict(X_transform)
df_pred = pd.DataFrame()
df_pred["y_actual"] = Y
df_pred['y_pred'] = y_pred
print(df_pred.tail(5))
#print("Y_pred:", y_pred)
# [-0.9155399781125949, 1.0595478243808099, 0.0740346154648282, 0.6912684806788304, -2.042436031571363, 2.6869087270532956, 0.008781186222241914, -3.1069660406054305, 2.4933486628329105, -1.8859109682475264, -2.0538173689458428, 0.8482275227360613, -3.737055038596798]
'''