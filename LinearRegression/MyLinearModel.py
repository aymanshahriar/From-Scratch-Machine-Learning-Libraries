"""
Gradient Descent Based Multiple Linear Regression Model
"""

# Author: Ayman Shahriar <ayman.shahriar@ucalgary.ca>

import warnings
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
np.set_printoptions(linewidth=200)


class MyLinearRegression:
    '''
    Implements Gradient Descent based Linear Regression from scratch
    (ie. It finds the optimal coefficients (m1, ..., mn) and intercept/bias (b) by minimizing the Total Loss Function
    using Gradient Descent)

    This class has two attributes/variables: coef_ (m1, ..., mn) and intercept_ (b, also known as bias)

    How many iterations of gradient descent does the class perform?
    A precision variable is set in the algorithm, which calculates the difference between the coefficients and slopes
    generated in two consecutive iterations of Gradient Descent.
    If the difference between 2 consecutive iterations is less than the precision, then the Gradient
    Descent algorithm is stopped

    Attributes
    ----------
    m : Coefficients (m1,...,mn), default is 0,...,0
    b : Intercept (b), default is 0
    '''

    def __init__(self, m=[], b=0):
        # Note: The __init__ method is used as the constructor method in python
        self.m = []  ## m is now an array
        self.b = b

    # X is now a 2d array, where each row contains the features of a single datapoint
    def total_loss(self, X, y):
        """
        Calculates Total Loss (a measure of how well an n-dimensional plane fits a set of data)

        Parameters
        ----------
        X : 2d matrix of features (n_datapoints, n_features)
        y : 1d array of target
        m : Current values of coefficients m1,...mn
        b : Current value of intercept/bias b

        Returns
        -------
        total_loss: The Total Loss
        """

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
        """
        Calculates Total Loss (a measure of how well an n-dimensional plane fits a set of data)

        Parameters
        ----------
        X : 2d matrix of features (n_datapoints, n_features)
        y : 1d array of target
        m : Current values of coefficients m1,...mn
        b : Current value of intercept/bias b

        Returns
        -------
        total_loss: The Total Loss
        """

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

    def get_gradient_at_b(self, X, y, m, b):
        """
        Finds the gradient/slope/derivative of the Total Loss function with respect to b (we will assume that (m_1, ..., m_n) are constants)

        Parameters
        ----------
        X : 2d matrix of features (n_datapoints, n_features)
        y : 1d array of target
        m : Current values of coefficients m1,...mn
        b : Current value of intercept/bias b

        Returns
        -------
        gradient_slope_derivative : The gradient/slope/derivative of the Total Loss function with respect to b (we will assume that (m_1, ..., m_n) are constants)
        """

        summation = sum([y_point-sum([m_i*x_i for m_i, x_i in zip(m, x)])-b for x, y_point in zip(X, y)])
        N = len(y)
        gradient_slope_derivative = (-2/N) * summation
        return gradient_slope_derivative

    def get_gradient_at_m(self, X, y, m, i, b):
        """
        Finds the gradient/slope/derivative of the Total Loss function with respect to m_i (we will assume that b and (m_1,...m_n-m_i) are constants)

        Parameters
        ----------
        X : 2d matrix of features (n_datapoints, n_features)
        y : 1d array of target
        m : Current values of coefficients m1,...mn
        i : The index of m where m_i is located
        b : Current value of intercept/bias b

        Returns
        -------
        gradient_slope_derivative : The gradient/slope/derivative of the Total Loss function with respect to m_i (we will assume that b and (m_1,...m_n-m_i) are constants)
        """

        summation = sum([ x[i] * (y_point-sum([m_i*x_i for m_i, x_i in zip(m, x)])-b) for x, y_point in zip(X, y)])
        N = len(y)
        gradient_slope_derivative = (-2/N) * summation
        return gradient_slope_derivative



    # This function will find the gradients at b_current and m_current (remember, m_current is a list), and then return new b and m values that have been moved in that direction
    # The step_gradient funcion will take the data (x and y values), the current value of m, b and alpha, and perform a single iteration of gradient descent
    #    (for both m and b)
    def step_gradient(self, X, y, m, b, learning_rate):
        """
        Uses the given coefficients (m1,...,mn) and intercept/bias (b) to perform a single iteration of Gradient Descent.

        Parameters
        ----------
        X : 2d matrix of features (n_datapoints, n_features)
        y : 1d array of target
        m : Current values of coefficients m1,...mn
        b : Current value of intercept/bias b
        learning_rate : The amount used to change the coefficients or intercept in gradient Descent

        Returns
        -------
        m : Values of coefficients (m1,...,mn) minimizes the Total Loss function
        b : Value of intercept/bias (b) that minimizes the Total Loss function
        """

        # Find the gradients at the current values of m and b
        m_gradients = []
        for i in range(len(m)):
            m_gradients.append(self.get_gradient_at_m(X, y, m, i, b))
        b_gradient = self.get_gradient_at_b(X, y, m, b)
        # Update the m and b values, moving them in the opposite direction of the slope
        new_m = []
        for i in range(len(m)):
            new_m.append(m[i] - (m_gradients[i] * learning_rate))
        new_b = b - (b_gradient * learning_rate)
        return new_m, new_b

    def gradient_descent(self, X, y, m, b, learning_rate=0.01, iter=2000, diff=0.0000001):
        """
        Performs Gradient Descent to find the values of (m1, ..., mn) and b that minimize Gradient Descent.

        Parameters
        ----------
        X : 2d matrix of features (n_datapoints, n_features)
        y : 1d array of target
        m : Current values of coefficients m1,...mn
        b : Current value of intercept/bias b
        learning_rate : The amount used to change the coefficients or intercept in gradient Descent
        iter : Maximum number of iterations of Gradient Descent
        diff : To know when to stop doing iterations of gradient descent, we have a precision value. We compare
                     the values of m1, ..., mn and b from the previous iteration. If their difference is less than or
                     equal to the precision value, it means that the values of m1,...mn and b are barely changing anymore,
                     which means that m1,...mn and b have reached the bottom of the curve.
                     (ie, m1,...,m_n and b are set to values that give the minimum total loss)

        Returns
        -------
        m : Values of coefficients (m1,...,mn) minimizes the Total Loss function
        b : Value of intercept/bias (b) that minimizes the Total Loss function
        """

        current_iter = 1
        current_diff = diff + 1
        while (current_iter <= iter) and (current_diff > diff):
            old_m = m
            old_b = b
            m, b = self.step_gradient(X, y, m, b, learning_rate)
            current_iter += 1
            diff_m = abs(np.array(old_m) - np.array(m))
            diff_b = abs(old_b - b)
            diff_1 = np.max(diff_m)
            current_diff = max(diff_1, diff_b)
        # If number of iterations has been exceeded, throw a warning
        if current_iter > iter:
            warnings.warn('Number of itrations exceeded without convergence being reached')
        return m, b

    def fit(self, X, y, learning_rate=0.01, iter=2000, diff=0.0000001):
        """
        Fit linear model.

        Parameters
        ----------
        X : 2d matrix of features (n_datapoints, n_features)
        y : 1d array of target
        learning_rate : The amount used to change the coefficients or intercept in gradient Descent
        iter : Maximum number of iterations of Gradient Descent
        diff : Difference between coefficients/intercept from two consecutive iterations, used to decide when
                     Gradient Descent should be stopped
        """

        num_features = len(X[0])
        m = [0] * num_features
        b = 0
        self.m, self.b = self.gradient_descent(X, y, m, b, learning_rate, iter, diff)

    def predict(self, X):
        """
        Given a 2d array of features (where each row contains the features of a single datapoint),
        predict the target of each datapoint

        Parameters
        ----------
        X : 2d matrix of features (n_datapoints, n_features)

        Return
        ------
        y : 1d array of targets predicted for each datapoint
        """

        return [sum([m_i*x_i for m_i,x_i in zip(self.m, x)]) + self.b for x in X ]

    def set_params(self, m=None, b=None):
        """
        Set the coefficients (m1,...mn) and intercept/bias (b) of the model

        Parameters
        ----------
        m : The new values of the coefficients (m1,...mn)
        b : The new value of the slope/intercept (b)
        """

        if m is not None:
            self.m = m
        if b is not None:
            self.b = b

    def compare_parameters_with_sklearn(self, X, y):
        """
        Compares the values of the coefficients (m1,...,mn) and intercept/bias (b) with those generated by a default
        sklearn LinearRegression model

        Parameters
        __________
        X : 2d matrix of features (n_datapoints, n_features)
        y : 1d array of target

        Return
        ------
        comparison_dict : A dictionary whose first key-value pair contains the values of the coefficients (m1,...,mn)
                            and intercept/bias (b), and whose second key-value pair contains the values generated by
                            a default sklearn LinearRegression model
        """

        sklearn_model = LinearRegression()
        sklearn_model.fit(X, y)
        sklearn_m = sklearn_model.coef_[0]
        sklearn_b = sklearn_model.intercept_
        comparison_dict = {'MyLinearRegression model parameters:': [self.m, self.b],
                           'Sklearn model parameters:': [sklearn_m, sklearn_b]}
        return comparison_dict
