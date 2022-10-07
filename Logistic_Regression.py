"""

"""
# Import Librrary
import numpy as np


# Training functin of logistic regression
def training(x, y):
    xT = x.T
    yT = np.reshape(y, (1, 4))
    # Parameters for the modeling
    b = 0
    # The learning Rate
    alpha = 3
    # THe number of iterations to prefore gradient descent
    epochs = 100
    # NUmber of elements in X
    m = len(x)
    W = np.array([(0, 0, 0, 0)])
    # Perfoming Gradient Descent
    for t in range(epochs):
        D_W = np.array([(0, 0)])
        D_b = 0
        J = 0
        Z = np.dot(W, xT) + b
        A = 1 / (1 + np.exp(-Z))
        J = -np.sum(np.dot(y, np.log(A).T)) + \
            np.dot((1 - y), np.log((1 - A)).T) / m
        D_W = np.dot((A - yT), x) / m
        D_b = np.sum(A - yT) / m

        W = W - alpha * D_W
        b = b - alpha * D_b
        print (t, J, W[0, 0], W[0, 1], b)
    # Return hte model parameters
    return W, b


# Test the logstic regression
def test_function(new_x, W, b):
    Z = np.dot(W, new_x.T) + b
    A = 1 / (1 + np.exp(-Z))
    return A


# Input for logstic regression
x = np.array(
    [(0.3, 0.3, 0.2, 0.1),
     (0.3, 0.4, 0.5, 0.7),
     (0.6, 0.6, 0.4, 0.2),
     (0.6, 0.7, 0.8, 0.9)])
y = np.array([(0), (1), (0), (1)])
# Training for logstic regression
w, b = training(x, y)

# New input and prediction
new_x = np.array([(0.3, 0.35, 0.4, 0.45)])
test_result = test_function(new_x, w, b)
print(test_result)