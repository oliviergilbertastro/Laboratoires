import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


red_data = np.loadtxt('PHY-2006/Projet1/data/stars_pictures/red_QE_curve.csv', skiprows=1, delimiter=',')
red_data = red_data[red_data[:, 0].argsort()]
red_x = red_data[:,0]
red_y = red_data[:,1]

blue_data = np.loadtxt('PHY-2006/Projet1/data/stars_pictures/blue_QE_curve.csv', skiprows=1, delimiter=',')
blue_data = blue_data[blue_data[:, 0].argsort()]
blue_x = blue_data[:,0]
blue_y = blue_data[:,1]

green_data = np.loadtxt('PHY-2006/Projet1/data/stars_pictures/green_QE_curve.csv', skiprows=1, delimiter=',')
green_data = green_data[green_data[:, 0].argsort()]
green_x = green_data[:,0]
green_y = green_data[:,1]

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
            return self.polynomial_3(x, self.a[0], self.b[0], self.c[0], self.d[0])
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

if False:
    if True:
        redSpline = cubic_spline_interpolation(red_x, red_y)
        blueSpline = cubic_spline_interpolation(blue_x, blue_y)
        greenSpline = cubic_spline_interpolation(green_x, green_y)
    else:
        redSpline = linear_interpolation(red_x, red_y)
        blueSpline = linear_interpolation(blue_x, blue_y)
        greenSpline = linear_interpolation(green_x, green_y)
    #spline = sp.interpolate.CubicSpline(red_x, red_y)
    x_range = np.linspace(red_x[0], red_x[-1], 10000)
    red_line = []
    blue_line = []
    green_line = []
    for i in range(len(x_range)):
        red_line.append(redSpline(x_range[i]))
        blue_line.append(blueSpline(x_range[i]))
        green_line.append(greenSpline(x_range[i]))
    #plt.plot(red_x, red_y, 'o')
    #plt.plot(x_range, spline(x_range), linewidth=3, label='SciPy')
    ax1 = plt.subplot(111)
    ticklabels = ax1.get_xticklabels()
    ticklabels.extend( ax1.get_yticklabels() )
    for label in ticklabels:
        label.set_fontsize(14)
    plt.plot(x_range, red_line, label='Nous', color='red')
    plt.plot(x_range, blue_line, label='Nous', color='blue')
    plt.plot(x_range, green_line, label='Nous', color='green')
    #plt.legend()
    plt.xlabel("Wavelength $\lambda$ [nm]", fontsize=15)
    plt.ylabel("Relative response [%]", fontsize=15)
    plt.show()







if True:
    x = [0, .1, .499, .5, .6, 1.0, 1.4, 1.5, 1.899, 1.9, 2.0]
    y = [0, .06, .17, .19, .21, .26, .28, .29, .30, .31, .32]

    x_range = np.linspace(x[0], x[-1], 1000)

    mySpline = cubic_spline_interpolation(x, y)
    spSpline = make_interp_spline(x,y, k=3)
    print(spSpline.t)
    ax1 = plt.subplot(111)
    ticklabels = ax1.get_xticklabels()
    ticklabels.extend( ax1.get_yticklabels() )
    for label in ticklabels:
        label.set_fontsize(14)
    mySpline_y = []
    for i in range(len(x_range)):
        mySpline_y.append(mySpline(x_range[i]))
    plt.plot(x, y, 'o')
    #plt.plot(x_range, mySpline_y, label='Nous')
    plt.plot(x_range, spSpline(x_range), label='SciPy')
    plt.legend()
    plt.show()