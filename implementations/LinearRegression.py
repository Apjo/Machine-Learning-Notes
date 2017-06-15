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
if __name__ == "__main__":

    print "Loading data file containing population in a city, and profit of a truck "
    data = pd.read_csv('ex1data1.txt', header=None, names=['Population', 'Profit'])

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

    # compute Cost
    computeCost(X, y, theta1)