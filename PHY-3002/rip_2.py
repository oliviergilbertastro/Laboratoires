import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

maxima = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
pos_maxima_241_Hz = [-34.39,-28.77,-22.73,-16.98,-10.59,-4.21,0,8.85,15.23,21.55,28.43,35.66,43.94]
pos_maxima_341_Hz = [-41.73,-35.82,-29.73,-22.67,-15.17,-7.31,0,8.75,17.40,26.13,36.02,46.87,58.70]
pos_maxima_441_Hz = [ -40.81,-34.05,-26.62,-18.36,-9.60,0,9.93,19.70,31.22,43.90,58.34]

flipped_pos_maxima_441_Hz = []
for i in range(len(pos_maxima_441_Hz)):
    flipped_pos_maxima_441_Hz.append(-pos_maxima_441_Hz[len(pos_maxima_441_Hz)-1-i])
pos_maxima_441_Hz = flipped_pos_maxima_441_Hz

m0_to_plan = 176
L = 1855
theta = np.arctan(m0_to_plan/L)

def alpha_n(pos_max):
    pos_max = np.array(pos_max)+m0_to_plan #mm
    angles = np.arctan(pos_max/L)
    return angles

lam = 632.8E-6

def line(n, Lambda):
    return np.cos(theta)-n*lam/Lambda


alpha_ns = alpha_n(pos_maxima_441_Hz)
maxima = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
#maxima = [5,4,3,2,1,0,-1,-2,-3,-4,-5]
res = curve_fit(line, maxima, np.cos(alpha_ns))[0]


print(res)
x_fit = np.linspace(-6,6,100)
y_fit = line(x_fit, Lambda=res[0])
plt.plot(maxima, np.cos(alpha_ns), "o")
plt.plot(x_fit, y_fit, "--")
plt.xlabel(r"Ordre $m$", fontsize=17)
plt.ylabel(r"$\cos(\alpha_n)$", fontsize=17)
plt.show()


x_fit = np.linspace(-6,6,100)
y_fit = line(x_fit, Lambda=res[0])
plt.plot(maxima, alpha_ns, "o")
plt.plot(x_fit, np.arccos(y_fit), "--")
plt.xlabel(r"Ordre $m$", fontsize=17)
plt.ylabel(r"$\alpha_n$", fontsize=17)
plt.show()