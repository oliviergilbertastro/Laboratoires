import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt


data = np.array([(0, 0), (0.1, .06),  (.499, .17), (.5, .19), (.6, .21),
(1, .26), (1.4, .28), (1.5, .29), (1.899, .30), (1.9, .31), (2.0, .32)])

x = data[:,0]
y = data[:,1]

x = [0.0, 1.0, 1.5, 2.5, 4.0, 4.5, 5.5, 6.0, 8.0, 10.0]
y = [10, 8, 5, 4, 3.5, 3.4, 6, 7.1, 8, 8.5]
x_range = np.linspace(x[0],x[-1], 1000)


ax1 = plt.subplot(111)
ticklabels = ax1.get_xticklabels()
ticklabels.extend( ax1.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(14)
plt.plot(x,y, "o")
for i in range(3):
    bspline = make_interp_spline(x,y,k=i+1, bc_type=[None,None,'natural'][i])
    plt.plot(x_range, bspline(x_range), label=f'deg {i+1}')
plt.xlabel(r'$x$', fontsize=15)
plt.ylabel(r'$y$', fontsize=15)
plt.legend()
plt.show()