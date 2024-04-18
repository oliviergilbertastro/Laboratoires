import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

x = [0, .1, .499, .5, .6, 1.0, 1.4, 1.5, 1.899, 1.9, 2.0]
y = [0, .06, .17, .19, .21, .26, .28, .29, .30, .31, .32]



def cubic_spline_interpolation(x, y):
    """
    x: array-like of N values
    y: array-like of N values

    We will have (N-1) segments of 3rd degree polynomials, which means we'll have 4(N-1) coefficients to store

    returns a Spline object which is a piecewise function (e.g. Spline(x)=y)
    """
    #Check if x and y are the same length:
    if len(x) != len(y):
        raise IndexError("x and y are not the same size")
    N = len(x)
    for i in range(1,N):
        if x[i-1] > x[i]:
            raise ValueError("x is not increasing")

    #We will have N-1 piecewise functions:
    #q[i] = (1-t)*y[i-1]+t*y[i]+t*(1-t)*((1-t)*a[i]+t*b[i])

    #We'll try to find the coefficients using *linear algebra*
    X_vec = np.array(x)
    Y_vec = np.array(y)
    coeffs_matrix = np.zeros((4,N))

    for i in range(N-1):
        c1 = x[i+1]-x[i]
        c2 = y[i+1]-y[i]
        #a[i+1] = k[i]*c1-c2
        #b[i+1] = -k[i]*c1+c2

    pass

class LinearPiecewise():
    def __init__(self, a, b, exes):
        self.a =  a
        self.b = b
        self.exes = exes
        pass

    def line(self, x, a, b):
        return a*x+b

    def __call__(self, x):
        i = 0
        if x < self.exes[0]:
            return self.line(x, self.a[0], self.b[0])
        while self.exes[i] <= x and i < len(self.a):
            i += 1
        return self.line(x, self.a[i-1], self.b[i-1])


def linear_interpolation(x, y):
    """
    x: array-like of x values
    y: array-like of y values

    returns a LinearPiecewise object which is a piecewise function
    """
    #Check if x and y are the same length:
    if len(x) != len(y):
        raise IndexError("x and y are not the same size")
    N = len(x)
    for i in range(1,N):
        if x[i-1] > x[i]:
            raise ValueError("x is not increasing")

    a = []
    b = []
    for i in range(N-1):
        a.append((y[i+1]-y[i])/(x[i+1]-x[i]))
        b.append(y[i]-a[-1]*x[i])
    return LinearPiecewise(a, b, x)



spline = sp.interpolate.make_interp_spline(x, y, k=1)
x_range = np.linspace(x[0], x[-1], 10000)

linepiece = linear_interpolation(x, y)
y_line = []
for i in range(len(x_range)):
    y_line.append(linepiece(x_range[i]))
plt.plot(x, y, 'o')
plt.plot(x_range, spline(x_range), linewidth=3)
plt.plot(x_range, y_line)
plt.show()
