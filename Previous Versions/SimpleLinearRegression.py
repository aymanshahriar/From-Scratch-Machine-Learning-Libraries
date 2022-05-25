"""
Gradient Descent Based Simple Linear Regression Model
"""

# Author: Ayman Shahriar <ayman.shahriar@ucalgary.ca>

import warnings
from sklearn.linear_model import LinearRegression


class MyLinearRegression:
    '''
    Implements Gradient Descent based Linear Regression from scratch
    (ie. It finds the optimal coefficient (m) and slope (b) by minimizing the Total Loss Function using Gradient Descent)

    This class has two attributes/variables: coef_ (m) and intercept_ (b, also known as bias)

    How many iterations of gradient descent does the class perform?
    A precision variable is set in the algorithm, which calculates the difference between the coefficients and slopes
    generated in two consecutive iterations of Gradient Descent.
    If the difference between 2 consecutive iterations is less than the precision, then the Gradient
    Descent algorithm is stopped

    Attributes
    ----------
    m : Coefficient, default is 0
    b : Intercept, default is 0
    '''

    def __init__(self, m=0, b=0):
        # The __init__ method is used as the constructor method in python
        self.m = m
        self.b = b

    def total_loss(self, X, y):
        """
        Calculates Total Loss (a measure of how well a line fits a set of data)

        Parameters
        ----------
        X : 2d matrix of features (n_datapoints, n_features)
        y : 1d array of target
        m : Current value of coefficient
        b : Current value of intercept/bias

        Returns
        -------
        total_loss: The Total Loss
        """

        loss_lst = []
        for x_value, y_value in zip(X, y):
            y_line = (self.m * x_value) + self.b
            difference = y_value - y_line
            # We square the difference so that points above and below the line affect the total loss in the same way
            loss = difference ** 2
            loss_lst.append(loss)
        total_loss = sum(loss_lst)
        return total_loss

    @staticmethod
    def total_loss(X, y, m, b):
        """
        Calculates Total Loss (a measure of how well an n-dimensional plane fits a set of data)

        Parameters
        ----------
        X : 2d matrix of features (n_datapoints, n_features)
        y : 1d array of target
        m : Current value of coefficient
        b : Current value of intercept/bias

        Returns
        -------
        total_loss: The Total Loss
        """

        loss_lst = []
        for x_value, y_value in zip(X, y):
            y_line = (m * x_value) + b
            difference = y_value - y_line
            # We square the difference so that points above and below the line affect the total loss in the same way
            loss = difference ** 2
            loss_lst.append(loss)
        total_loss = sum(loss_lst)
        return total_loss

    @staticmethod
    def total_loss_concise(X, y, m, b):
        total_loss = sum([(y_value - ((m * x_value) + b)) ** 2 for x_value, y_value in zip(X, y)])
        return total_loss

    def get_gradient_at_b(self, x, y, m, b):
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

        sum_of_differences = sum([y_point - (m * x_point + b) for x_point, y_point in zip(x, y)])
        N = len(x)
        b_gradient_slope_derivative = (-2 / N) * sum_of_differences
        return b_gradient_slope_derivative

    def get_gradient_at_m(self, x, y, m, b):
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

        sum_of_differences = sum([x_point * (y_point - (m * x_point + b)) for x_point, y_point in zip(x, y)])
        N = len(x)
        m_gradient_slope_derivative = (-2 / N) * sum_of_differences
        return m_gradient_slope_derivative

    def step_gradient(self, x, y, m_current, b_current, learning_rate):
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

        # Find the gradients at the current values of m and b
        m_gradient = self.get_gradient_at_m(x, y, m_current, b_current)
        b_gradient = self.get_gradient_at_b(x, y, m_current, b_current)

        # Update the m and b values, moving them in the opposite direction of the slope
        new_m = m_current - (m_gradient * learning_rate)
        new_b = b_current - (b_gradient * learning_rate)

        return new_m, new_b

    def gradient_descent(self, x, y, m, b, learning_rate=0.01, iter=1000, diff=0.0000001):
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
        """
        Fit linear model.

        Parameters
        ----------
        x : 2d matrix of features (n_datapoints, n_features)
        y : 1d array of target
        learning_rate : The amount used to change the coefficients or intercept in gradient Descent
        iter : Maximum number of iterations of Gradient Descent
        diff : Difference between coefficients/intercept from two consecutive iterations, used to decide when
                     Gradient Descent should be stopped
        """

        self.m, self.b = self.gradient_descent(x, y, 0, 0, learning_rate, iter, diff)

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

        return [(self.m * x[0]) + self.b for x in X]

    def set_params(self, m=None, b=None):
        """
        Set the coefficient m and intercept/bias b of the model

        Parameters
        ----------
        m : The new values of the coefficient
        b : The new value of the slope/intercept
        """

        if m is not None:
            self.m = m
        if b is not None:
            self.b = b

    def compare_parameters_with_sklearn(self, x, y):
        """
        Compares the values of the coefficient m and intercept/bias b with those generated by a default
        sklearn LinearRegression model

        Parameters
        __________
        X : 2d matrix of features (n_datapoints, n_features)
        y : 1d array of target

        Return
        ------
        comparison_dict : A dictionary whose first key-value pair contains the values of the coefficient (m)
                            and intercept/bias (b), and whose second key-value pair contains the values generated by
                            a default sklearn LinearRegression model
        """

        sklearn_model = LinearRegression()
        sklearn_model.fit(x, y)
        sklearn_m = sklearn_model.coef_[0]
        sklearn_b = sklearn_model.intercept_
        return {'MyLinearRegression model parameters:': [self.m, self.b],
                'Sklearn model parameters:': [sklearn_m, sklearn_b]}







