#!/usr/bin/env python
# coding=utf-8
from __future__ import division

import matplotlib  as mpl
mpl.use('TkAgg')
from pylab import scatter, show, title, xlabel, ylabel, plot, contour

import numpy as np
import pandas as pd

def computeCost(X, y, theta1):
    m = len(X)
    inner = np.power(((X * theta1.T) - y), 2)
    J = np.sum(inner) / (2 * m)

    return J

def gradient_descent(X, y, theta, alpha, iterations):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iterations)

    for iteration in range(iterations):
        error = (X * theta.T) - y
        for param in range(parameters):
            term = np.multiply(error, X[:, param]) # we are calculating hθ(x(i)) - y(i) * x(i)
            temp[0, param] = temp[0, param] - (alpha/len(X)) * np.sum(term)

        theta = temp
        cost[iteration] = computeCost(X, y, theta)
    print 'Cost after gradient descent: ' + str(cost[-1])
    return theta, cost

if __name__ == "__main__":

    print "Loading data file containing population in a city, and profit of a truck "
    data = pd.read_csv('../data/ex1data1.txt', header=None, names=['Population', 'Profit'])

    # append a ones column to the front of the data set
    data.insert(0, 'Ones', 1)
    cols = data.shape[1]
    X = data.iloc[:, 0:cols-1]

    y = data.iloc[:, cols-1:cols]

    #Plot the data
    # scatter(data.iloc[:, 0], data.iloc[:, 1], marker='o', c='b')
    # title('Profits distribution')
    # xlabel('Population of City in 10,000s')
    # ylabel('Profit in $10,000s')
    # show()

    #The cost function is expecting numpy matrices so we need to convert X and y before we can use them. We also need to initialize theta.
    X = np.matrix(X.values)
    y = np.matrix(y.values)
    theta1 = np.matrix(np.array([0,0]))

    # gradient descent settings
    alpha = 0.01
    iterations = 1500

    # compute Cost for our initial solution
    computeCost(X, y, theta1)
    # run the gradient descent algorithm to fit our parameters theta to the training set
    g, cost = gradient_descent(X, y, theta1, alpha, iterations)
    # Finally we can compute the cost (error) of the trained model using our fitted parameters.
    computeCost(X, y, g)