import warnings


# Total Loss function
# Input: datapoints and a line (m and b)
# Output: Calculates the loss for each datapoint and returns the total loss
def total_loss_concise(self, X, y, m, b):
    total_loss = sum([(y_value - ((m * x_value) + b)) ** 2 for x_value, y_value in zip(X, y)])
    return total_loss


def total_loss(self, X, y, m, b):
    loss_lst = []
    for x_value, y_value in zip(X, y):
        y_line = (m * x_value) + b
        difference = y_value - y_line
        loss = difference ** 2  # We square the difference so that points above and below the line affect the total loss in the same way
        loss_lst.append(loss)
    total_loss = sum(loss_lst)
    return total_loss


'''# Testing the total loss function
x = [1, 2, 3]
y = [5, 1, 3]

# y = x
m1 = 1
b1 = 0

# y = 0.5x + 1
m2 = 0.5
b2 = 1

total_loss1 = total_loss(x, y, m1, b1)
total_loss2 = total_loss(x, y, m2, b2)

print(total_loss1, total_loss2)
'''

#######################################################################################################################
# This function will find the gradient of the total loss function with respect to b (we will assume that m is a constant)
def get_gradient_at_b(x, y, m, b):
    sum_of_differences = sum( [y_point - (m*x_point+b) for x_point, y_point in zip(x, y)] )
    N = len(x)
    b_gradient_slope_derivative = (-2/N) * sum_of_differences
    return b_gradient_slope_derivative

# This function will find the gradient of the total loss function with respect to m (we will assume that b is a constant)
def get_gradient_at_m(x, y, m, b):
    sum_of_differences = sum( [x_point*(y_point - (m*x_point+b)) for x_point, y_point in zip(x, y)] )
    N = len(x)
    m_gradient_slope_derivative = (-2/N) * sum_of_differences
    return m_gradient_slope_derivative

# This function will find the gradients at b_current and m_current, and then return new b and m values that have been moved in that direction
def step_gradient(x, y, m_current, b_current, learning_rate):
    # Find the gradients at the current values of m and b
    m_gradient = get_gradient_at_m(x, y, m_current, b_current)
    b_gradient = get_gradient_at_b(x, y, m_current, b_current)

    # Update the m and b values, moving them in the opposite direction of the slope
    new_m = m_current - (m_gradient * learning_rate)
    new_b = b_current - (b_gradient * learning_rate)

    return new_m, new_b

def gradient_descent(x, y, m, b, learning_rate=0.01, iter=2000, diff=0.00001):
    current_iter = 1
    current_diff = diff
    m,b = 0, 0
    while (current_iter <= iter) and (current_diff > diff):
        old_m = m
        old_b = b
        m, b = step_gradient(x, y, m, b, learning_rate)
        current_iter += 1
        diff_m = abs(old_m-m)
        diff_b = abs(old_b-b)
        current_diff = max(diff_m, diff_b)

    # if number of iterations has been exceeded, throw a warning
    if current_iter > iter:
        warnings.warn('Number of itrations exceeded without convergence being reached')

    return m,b

def fit(self, x, y, learning_rate=0.01, iter=2000, diff=0.00001):
    self.m, self.b = gradient_descent(x, y, 0, 0, learning_rate, iter, diff)

'''
# Test the step gradient function
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
y = [52, 74, 79, 95, 115, 110, 129, 126, 147, 146, 156, 184]

m=0
b=0

m, b = step_gradient(x, y, m, b, 0.01)

print(m, b)
'''










