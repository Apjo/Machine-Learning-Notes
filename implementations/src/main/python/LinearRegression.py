from numpy import *
import math

def hypothesis(theta0, theta1, x):
    return (theta0 + theta1 * x)

def squaredError(theta0, theta1, x, y):
    return math.pow(hypothesis(theta0, theta1, x) - y, 2)

def costFunction(theta0, theta1):
    return 0

#Evaluating a numerical deerivative
def derivative(withRespectTo, t0, t1):
    h = 1./1000.
    if (withRespectTo == "theta0"):
        rise = costFunction(t0 + h, t1) - costFunction(t0,t1)
    else: #has to be wrt to theta1
        rise = costFunction(t0 , t1 + h) - costFunction(t0,t1)
    run = h
    slope = rise/run

    return slope

if __name__ == "__main__":
    dataFile = open('../resources/ex1data1.txt', 'r')

    print """
    This program is just a rough reimplementation of linear regression in Python. It's not particularly
    optimized in any way but it does give a sense of backpropagation, computing the loss function, and
    updating the weights. The derivatives are taken numerically, instead of analytically. Numeric
    derivatives are a bit slower and less intuitive, but do still work in this case.
    """
    print "Loading data file containing population in a city, and profit of a truck "
    data = loadtxt(dataFile, delimiter=',')
    profit, population = data[:, 0], data[:, 1]

    # training examples
    m = len(population)

    # set up data for linear regression

    profit = ones((m, 1)), data[:, 0]

    # Initialize fitting parameters
    theta = zeros((2, 1))

    # Gradient descent settings
    iterations = 1500
    alpha = 0.01

    for i in range(0, iterations):
        i + 10


