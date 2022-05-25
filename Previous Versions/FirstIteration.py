"""
Gradient Descent Based Simple Linear Regression Model
"""

# Author: Ayman Shahriar <ayman.shahriar@ucalgary.ca>

from sklearn.linear_model import LinearRegression
import pandas as pd


def get_gradient_m(x, y, m, b):
    """
    Finds the gradient/slope/derivative of the Total Loss function with respect to b (we will assume m is a constant)

    Parameters
    ----------
    X : 2d matrix of features
    y : 1d array of target
    m : Current value of coefficient m
    b : Current value of intercept/bias b

    Returns
    -------
    gradient_slope_derivative : The gradient/slope/derivative of the Total Loss function with respect to b (we will assume that m is a constant)
    """

    N = len(x)
    gradient_m = ((-2) / N) * sum([x_point * (y_point - (m * x_point) - b) for x_point, y_point in zip(x, y)])
    return gradient_m

def get_gradient_b(x, y, m, b):
    """
    Finds the gradient/slope/derivative of the Total Loss function with respect to m (we will assume b is a constant)

    Parameters
    ----------
    X : 2d matrix of features (n_datapoints, n_features)
    y : 1d array of target
    m : Current value of coefficient m
    b : Current value of intercept/bias b

    Returns
    -------
    gradient_slope_derivative : The gradient/slope/derivative of the Total Loss function with respect to m (we will assume that b is a constant)
    """

    N = len(x)
    gradient_b = ((-2) / N) * sum([y_point - (m * x_point) - b for x_point, y_point in zip(x, y)])
    return gradient_b

def step_gradient_descent(x, y, m, b, alpha, precision=0.000001):
    """
    Uses the given coefficient m and intercept/bias (b) to perform a single iteration of Gradient Descent.

    Parameters
    ----------
    X : 2d matrix of features (n_datapoints, n_features)
    y : 1d array of target
    m : Current values of coefficient m
    b : Current value of intercept/bias b
    learning_rate : The amount used to change the coefficients or intercept in gradient Descent

    Returns
    -------
    m : Value of coefficient m that minimizes the Total Loss function
    b : Value of intercept/bias b that minimizes the Total Loss function
    """

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

def gradient_descent(x, y, alpha, precision):
    """
    Performs Gradient Descent to find the values of m and b that minimize Gradient Descent.

    Parameters
    ----------
    X : 2d matrix of features (n_datapoints, n_features)
    y : 1d array of target
    m : Current value of coefficient m
    b : Current value of intercept/bias b
    learning_rate : The amount used to change the coefficients or intercept in gradient Descent
    iter : Maximum number of iterations of Gradient Descent
    diff : To know when to stop doing iterations of gradient descent, we have a precision value. We compare
                the values of m and b from the previous iteration. If their difference is less than or
                equal to the precision value, it means that the values of m and b are barely changing anymore,
                which means that m and b have reached the bottom of the curve.
                (ie, m and b are set to values that give the minimum total loss)

    Returns
    -------
    m : Values of the coefficient (m) that minimizes the Total Loss function
    b : Value of the intercept/bias (b) that minimizes the Total Loss function
    """
    m = 0
    b = 0

    m, b = step_gradient_descent(x, y, m, b, alpha, precision)

    return m, b
