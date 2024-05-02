import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, CubicSpline

class CubicSplinePiecewise():
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

    return CubicSplinePiecewise(a, b, c, d, x)



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






if __name__ == "__main__":
    if input('Comparer spline cubique? [y/n]') == 'y':
        #x = [0, .1, .499, .5, .6, 1.0, 1.4, 1.5, 1.899, 1.9, 2.0]
        #y = [0, .06, .17, .19, .21, .26, .28, .29, .30, .31, .32]
        x = [0.0, 1.0, 1.5, 2.5, 4.0, 4.5, 5.5, 6.0, 8.0, 10.0]
        y = [10, 8, 5, 4, 3.5, 3.4, 6, 7.1, 8, 8.5]
        
        x = [0, 1, 3, 5]
        y = [0, 6, 5, 1]

        x_range = np.linspace(x[0], x[-1], 1000)

        mySpline = cubic_spline_interpolation(x, y)
        spSpline = CubicSpline(x,y, bc_type='natural') #Mêmes conditions frontières que nous (2e dérivées = 0 aux bouts)
        ax1 = plt.subplot(111)
        ticklabels = ax1.get_xticklabels()
        ticklabels.extend( ax1.get_yticklabels() )
        for label in ticklabels:
            label.set_fontsize(14)
        mySpline_y = []
        for i in range(len(x_range)):
            mySpline_y.append(mySpline(x_range[i]))
        plt.plot(x, y, 'o', color='red')
        #plt.plot(x_range, mySpline_y, linewidth=2, label='Nous', color='blue')
        #plt.plot(x_range, spSpline(x_range), linewidth=3, linestyle='dashed', label='SciPy', color='black')
        #plt.legend()
        plt.show()

    if input('Powerpoint? [y/n]') == 'y':
        x = [0.0, 1.0, 1.5, 2.5, 4.0, 4.5, 5.5, 6.0, 8.0, 10.0]
        y = [10, 8, 5, 4, 3.5, 3.4, 6, 7.1, 8, 8.5]
        
        x = [0, 1, 3, 5]
        y = [0, 6, 5, 1]

        x_range = np.linspace(x[0], x[-1], 1000)

        mySpline = cubic_spline_interpolation(x, y)
        polynoms_y = []
        for k in range(len(x)-1):
            polynoms_y.append([])
            for i in x_range:
                polynoms_y[k].append(mySpline.polynomial_3(i, mySpline.a[k], mySpline.b[k], mySpline.c[k], mySpline.d[k]))
        for ik in range(len(polynoms_y)):
            ax1 = plt.subplot(111)
            ticklabels = ax1.get_xticklabels()
            ticklabels.extend( ax1.get_yticklabels() )
            for label in ticklabels:
                label.set_fontsize(14)
            mySpline_y = []
            for i in range(len(x_range)):
                mySpline_y.append(mySpline(x_range[i]))
            #plt.plot(x_range, mySpline_y, linewidth=2, label='Nous', color='blue')
            
            plt.plot(x_range, polynoms_y[ik], linestyle='dashed', linewidth=2)
            plt.plot(x, y, 'o', color='red')
            plt.show()

















    if input('Comparer spline avec polynome? [y/n]') == 'y':
        #x = [0, .1, .499, .5, .6, 1.0, 1.4, 1.5, 1.899, 1.9, 2.0]
        #y = [0, .06, .17, .19, .21, .26, .28, .29, .30, .31, .32]
        x = [0.0, 1.0, 1.5, 2.5, 4.0, 4.5, 5.5, 6.0, 8.0, 10.0]
        y = [10, 8, 5, 4, 3.5, 3.4, 6, 7.1, 8, 8.5]
        
        x = np.linspace(-1, 1, 30)
        y = []
        def func(x):
            return 1/(1+25*x**2)
        for i in range(len(x)):
            y.append(func(x[i]))

        x_range = np.linspace(x[0], x[-1], 1000)
        y_range = []
        for i in range(len(x_range)):
            y_range.append(func(x_range[i]))
        mySpline = cubic_spline_interpolation(x, y)
        
        ax1 = plt.subplot(111)
        ticklabels = ax1.get_xticklabels()
        ticklabels.extend( ax1.get_yticklabels() )
        for label in ticklabels:
            label.set_fontsize(14)
        mySpline_y = []
        for i in range(len(x_range)):
            mySpline_y.append(mySpline(x_range[i]))
        #plt.plot(x, y, 'o', color='red')
        plt.plot(x_range, y_range, label='$f(x)$', linewidth=2, color='black')
        #plt.plot(x_range, mySpline_y, linewidth=2, label='Spline Cubique', color='blue')
        for i in range(2):
            polynomial = np.polyfit(x,y, deg=5+i*4) #Mêmes conditions frontières que nous (2e dérivées = 0 aux bouts)
            plt.plot(x_range, np.polyval(polynomial, x_range), linewidth=2, linestyle='dashed', label=f'Polynôme deg {5+4*i}', color=["blue","red"][i])
        plt.xlabel(r"$x$", fontsize=15)
        plt.ylabel(r"$f(x)$", fontsize=15)
        plt.title(r"$f(x)=\left(1+25x^2\right)^{-1}$", fontsize=15)
        plt.legend()
        plt.show()
    

    if input('Comparer spline linéaire? [y/n]') == 'y':
        x = [0.0, 1.0, 1.5, 2.5, 4.0, 4.5, 5.5, 6.0, 8.0, 10.0]
        y = [10, 8, 5, 4, 3.5, 3.4, 6, 7.1, 8, 8.5]
        x = [0, 1, 3, 5]
        y = [0, 6, 5, 1]

        x_range = np.linspace(x[0], x[-1], 1000)

        mySpline = cubic_spline_interpolation(x, y)
        myLinear = linear_interpolation(x,y)
        ax1 = plt.subplot(111)
        ticklabels = ax1.get_xticklabels()
        ticklabels.extend( ax1.get_yticklabels() )
        for label in ticklabels:
            label.set_fontsize(14)
        mySpline_y = []
        myLinear_y = []
        for i in range(len(x_range)):
            mySpline_y.append(mySpline(x_range[i]))
            myLinear_y.append(myLinear(x_range[i]))
        plt.plot(x, y, 'o', color='red')
        plt.plot(x_range, mySpline_y, linewidth=2, label='Spline cubique', color='purple')
        plt.plot(x_range, myLinear_y, linewidth=2, linestyle='dashed', label='Spline linéaire', color='blue')
        plt.legend()
        plt.show()