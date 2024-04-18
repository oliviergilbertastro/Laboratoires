import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

x = [0, .1, .499, .5, .6, 1.0, 1.4, 1.5, 1.899, 1.9, 2.0]
y = [0, .06, .17, .19, .21, .26, .28, .29, .30, .31, .32]

class SplinePiecewise():
    def __init__(self, a, b, c, d, exes):
        self.a =  a
        self.b = b
        self.c = c
        self.d = d
        self.exes = exes
        pass

    def polynomial_3(self, x, a, b, c, d):
        return a*x**3+b*x**2+c*x+d

    def __call__(self, x):
        i = 0
        if x < self.exes[0]:
            return self.line(x, self.a[0], self.b[0])
        while self.exes[i] <= x and i < len(self.a):
            i += 1
        return self.polynomial_3(x, self.a[i-1], self.b[i-1], self.c[i-1], self.d[i-1])


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
    #S[i] = a[i]*x**3+b[i]*x**2+c[i]*x+d[i]

    #We'll try to find the coefficients using *linear algebra*
    A = np.zeros((4*(N-1),4*(N-1)))
    B = np.zeros((4*(N-1),))

    for i in range(N-1):
        B[i] = y[i]
        A[i,4*i:4*(i+1)] = [x[i]**3, x[i]**2, x[i], 1]
    for i in range(N-1):
        B[N-1+i] = y[i+1]
        A[N-1+i,4*i:4*(i+1)] = [x[i+1]**3, x[i+1]**2, x[i+1], 1]
    for i in range(N-2):
        B[2*(N-1)+i] = 0
        A[2*(N-1)+i, 4*i:4*(i+2)] = [3*x[i+1]**2, 2*x[i+1], 1, 0, -3*x[i+1]**2, -2*x[i+1], -1, 0]
    for i in range(N-2):
        B[2*(N-1)+N-2+i] = 0
        A[2*(N-1)+N-2+i, 4*i:4*(i+2)] = [6*x[i+1], 2, 0, 0, -6*x[i+1], -2, 0, 0]
    #Conditions frontières
    B[-2:] = 0
    A[-2, :4] = [6*x[0], 2, 0, 0]
    A[-1, -4:] = [6*x[-1], 2, 0, 0]

    #Calculate the coefficient matrix
    X = np.linalg.solve(A, B)
    a = []
    b = []
    c = []
    d = []
    for i in range(int(len(X)/4)):
        a.append(X[i*4])
        b.append(X[i*4+1])
        c.append(X[i*4+2])
        d.append(X[i*4+3])

    return SplinePiecewise(a, b, c, d, x)



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

if True:
    mySpline = cubic_spline_interpolation(x, y)
    spline = sp.interpolate.CubicSpline(x, y)
    x_range = np.linspace(x[0], x[-1], 10000)
    linepiece = linear_interpolation(x, y)
    y_line = []
    for i in range(len(x_range)):
        y_line.append(mySpline(x_range[i]))
    plt.plot(x, y, 'o')
    plt.plot(x_range, spline(x_range), linewidth=3, label='SciPy')
    plt.plot(x_range, y_line, label='Nous')
    plt.legend()
    plt.show()
