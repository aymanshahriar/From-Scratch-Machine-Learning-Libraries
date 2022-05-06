import warnings


class MyLinearRegression:

    # The __init__ method is used as the constructor method in python
    def __init__(self, m=0, b=0):
        self.m = m
        self.b = b

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
    def step_gradient(self, x, y, m_current, b_current, learning_rate):
        # Find the gradients at the current values of m and b
        m_gradient = get_gradient_at_m(self, x, y, m_current, b_current)
        b_gradient = get_gradient_at_b(self, x, y, m_current, b_current)

        # Update the m and b values, moving them in the opposite direction of the slope
        new_m = m_current - (m_gradient * learning_rate)
        new_b = b_current - (b_gradient * learning_rate)

        return new_m, new_b
