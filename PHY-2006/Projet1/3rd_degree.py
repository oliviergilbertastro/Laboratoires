import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def polynomial(x,a,b,c,d):
    return a*x**3+b*x**2+c*x+d

x_data = [200, 400, 599, 600, 800]
y_data = [140, 90, 41, 41, 203]
res = curve_fit(polynomial, x_data, y_data)[0]

x_fit = np.linspace(200, 900, 1000)
y_fit = polynomial(x_fit, res[0], res[1], res[2], res[3])
print(res)
plt.plot(x_data, y_data, 'o')
plt.plot(x_fit, y_fit)
plt.show()