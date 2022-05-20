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

class MyLinearRegression:

    # The __init__ method is used as the constructor method in python
    def __init__(self, m=0, b=0):
        self.m = m
        self.b = b

    def total_loss(self, X, y):
        loss_lst = []
        for x_value, y_value in zip(X, y):
            y_line = (self.m * x_value) + self.b
            difference = y_value - y_line
            loss = difference ** 2  # We square the difference so that points above and below the line affect the total loss in the same way
            loss_lst.append(loss)
        total_loss = sum(loss_lst)
        return total_loss

    @staticmethod
    def total_loss(X, y, m, b):
        loss_lst = []
        for x_value, y_value in zip(X, y):
            y_line = (m * x_value) + b
            difference = y_value - y_line
            loss = difference ** 2  # We square the difference so that points above and below the line affect the total loss in the same way
            loss_lst.append(loss)
        total_loss = sum(loss_lst)
        return total_loss

    @staticmethod
    def total_loss_concise(X, y, m, b):
        total_loss = sum([(y_value - ((m * x_value) + b)) ** 2 for x_value, y_value in zip(X, y)])
        return total_loss

    # This function will find the gradient of the total loss function with respect to b (we will assume that m is a constant)
    def get_gradient_at_b(self, x, y, m, b):
        sum_of_differences = sum([y_point - (m * x_point + b) for x_point, y_point in zip(x, y)])
        N = len(x)
        b_gradient_slope_derivative = (-2 / N) * sum_of_differences
        return b_gradient_slope_derivative

    # This function will find the gradient of the total loss function with respect to m (we will assume that b is a constant)
    def get_gradient_at_m(self, x, y, m, b):
        sum_of_differences = sum([x_point * (y_point - (m * x_point + b)) for x_point, y_point in zip(x, y)])
        N = len(x)
        m_gradient_slope_derivative = (-2 / N) * sum_of_differences
        return m_gradient_slope_derivative

    # This function will find the gradients at b_current and m_current, and then return new b and m values that have been moved in that direction
    # The step_gradient funcion will take the data (x and y values), the current value of m, b and alpha, and perform a single iteration of gradient descent
    #    (for both m and b)
    def step_gradient(self, x, y, m_current, b_current, learning_rate):
        # Find the gradients at the current values of m and b
        m_gradient = self.get_gradient_at_m(x, y, m_current, b_current)
        b_gradient = self.get_gradient_at_b(x, y, m_current, b_current)

        # Update the m and b values, moving them in the opposite direction of the slope
        new_m = m_current - (m_gradient * learning_rate)
        new_b = b_current - (b_gradient * learning_rate)

        return new_m, new_b

    # To know when to stop doing iterations of gradient descent, we have a precision value. We compare the values of m and b from the previous iteration. If their difference
    #  is less than or equal to the percision value, it means that the values of m and b are berely changing anymore, which means that m and b have reached the bottom of the curve
    #  (ie, m and b are values are optimized to give the minimum loss value)
    def gradient_descent(self, x, y, m, b, learning_rate=0.01, iter=1000, diff=0.0000001):
        current_iter = 1
        current_diff = diff+1
        while (current_iter <= iter) and (current_diff > diff):
            old_m = m
            old_b = b
            m, b = self.step_gradient(x, y, m, b, learning_rate)
            current_iter += 1
            diff_m = abs(old_m - m)
            diff_b = abs(old_b - b)
            current_diff = max(diff_m, diff_b)
        # if number of iterations has been exceeded, throw a warning
        if current_iter > iter:
            warnings.warn('Number of itrations exceeded without convergence being reached')

        return m, b

    def fit(self, x, y, learning_rate=0.01, iter=1000, diff=0.0000001):
        self.m, self.b = self.gradient_descent(x, y, 0, 0, learning_rate, iter, diff)

    def predict(self, X):
        # Assume x is a numpy arr
        return [(self.m * x[0]) + self.b for x in X]

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
        return {'MyLinearRegression model parameters:': [self.m, self.b], 'Sklearn model parameters:': [sklearn_m, sklearn_b]}


'''# Test 1
x = list(range(1, 11))  # checks out. Expected values are m=2, b=0
y = [2 * i for i in x]

model = MyLinearRegression()
model.fit(x, y)
print(model.m, model.b)'''







