#!/usr/bin/env python
from __future__ import division

import matplotlib as mpl

mpl.use('TkAgg')
import numpy as np
import pandas as pd


def computeCost(X, y, theta1):
    m = len(X)
    inner = np.power(((X * theta1.T) - y), 2)
    J =  np.sum(inner) / (2 * m)
    print "Cost is ", J
    return J

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    J_history = np.zeros(iterations)
    for numbers in range(iterations):
        a = theta[0][0] - alpha*(1/m)*sum((X * theta.flatten() - y)*x[:,0])
        b = theta[1][0] - alpha*(1/m)*sum((X.dot(theta).flatten() - y)*x[:,1])
        theta[0][0],theta[1][0]=a,b
        print theta[0][0]
        print theta[1][0]
        J.append(cost(x,y,theta))
        print 'Cost: ' + str(J[-1])
    return 0

if __name__ == "__main__":

    print "Loading data file containing population in a city, and profit of a truck "
    data = pd.read_csv('/data/ex1data1.txt', header=None, names=['Population', 'Profit'])

    # append a ones column to the front of the data set
    data.insert(0, 'Ones', 1)
    cols = data.shape[1]
    X = data.iloc[:, 0:cols-1]

    y = data.iloc[:, cols-1:cols]

    #Plot the data
    # scatter(data[:, 0], data[:, 1], marker='o', c='b')
    # title('Profits distribution')
    # xlabel('Population of City in 10,000s')
    # ylabel('Profit in $10,000s')
    # show()

    #convert our data frames to numpy matrices.
    X = np.matrix(X.values)
    y = np.matrix(y.values)
    # parameter matrix
    theta1 = np.matrix(np.array([0,0]))

    # gradient descent settings
    alpha = .01
    iterations = 1500

    # compute Cost
    computeCost(X, y, theta1)

    gradient_descent(X, y, theta1, alpha, iterations)