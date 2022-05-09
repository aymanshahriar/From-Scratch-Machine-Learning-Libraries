# Implementing gradient descent based linear regression from scratch
# (That optimizes the coefficients (m1, ..., mn) and slope (b) by minimizing the total loss function using gradient descent)

# This class should have two attributes/variables: coef_ and intercept_
# later I need to incorporate a .fit() method, which will just use gradient_descent
# later look into best default alpha value, allow users to specify their own alpha value
# later look into when to stop doing iteration of gradient descent

# How many iterations of gradient descent should we perform?
# Let us set a precision variable in our algorithm which calculates the difference between two consecutive “x” values .
# If the difference between x values from 2 consecutive iterations is lesser than the precision we set, stop the algorithm !
from sklearn.linear_model import LinearRegression
import pandas as pd


# The get_gradient_m function will take the data (x and y values), the current values of m,b and return the gradient/slope/tangent at that value of m
def get_gradient_m(x, y, m, b):
    N = len(x)
    gradient_m = ((-2) / N) * sum([x_point * (y_point - (m * x_point) - b) for x_point, y_point in zip(x, y)])
    return gradient_m


# The get_gradient_b function will take the data (x and y values), the current values of m,b and return the gradient/slope/tangent at that value of b
def get_gradient_b(x, y, m, b):
    N = len(x)
    gradient_b = ((-2) / N) * sum([y_point - (m * x_point) - b for x_point, y_point in zip(x, y)])
    return gradient_b


# The step_gradient funcion will take the data (x and y values), the current value of m, b and alpha, and perform a single iteration of gradient descent
#    (for both m and b)
# To know when to stop doing iterations of gradient descent, we have a precision value. We compare the values of m and b from the previous iteration. If their difference
#  is less than or equal to the percision value, it means that the values of m and b are berely changing anymore, which means that m and b have reached the bottom of the curve
#  (ie, m and b are values are optimized to give the minimum loss value)
def step_gradient_descent(x, y, m, b, alpha, precision=0.000001):
    precision = 0.000001
    diff_m = 1
    diff_b = 1
    iter = 0
    while (diff_m > precision and diff_b > precision):
        iter += 1
        prev_m = m
        prev_b = b

        gradient_m = get_gradient_m(x, y, m, b)
        gradient_b = get_gradient_b(x, y, m, b)

        m = m - (gradient_m * alpha)
        b = b - (gradient_b * alpha)

        diff_m = abs(m - prev_m)
        diff_b = abs(b - prev_b)
    print("Num iterations:", iter)
    return m, b


# The gradient_descent function will take the datapoints (x, y), alpha, number of iterations and finds the "best" line
# ie. the values of m and b that minimizes the total loss of the dataset
def gradient_descent(x, y, alpha):
    m = 0
    b = 0

    m, b = step_gradient_descent(x, y, m, b, alpha)

    return m, b
