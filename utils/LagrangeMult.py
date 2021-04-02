import numpy as np 
from scipy.optimize import fsolve


def func(X):
    x = X[0]
    y = X[1]
    z = X[2]
    L = X[3]
    return x + y + z + L * (x**2 + y**2 + z**2 - 1)

def dfunc(X):
    dLambda = np.zeros(len(X))
    h = 1e-3
    for i in range(len(X)):
        dX = np.zeros(len(X))
        dX[i] = h
        dLambda[i] = (func(X+dX)-func(X-dX))/(2*h)
    return dLambda

X1 = fsolve(dfunc, [1,1,1,0])
print(X1,func(X1))