import matplotlib.pyplot as plt
import numpy as np

b=1
def E_y(x1,x2):
    return np.abs((-1j/b)*np.sin((4*np.pi*x1)/(2*b))*np.cos((2*np.pi*x2)/(b)))

x1_min = -5.0
x1_max = 5.0
x2_min = -5.0
x2_max = 5.0

x1, x2 = np.meshgrid(np.arange(x1_min,x1_max, 0.005), np.arange(x2_min,x2_max, 0.005))

y = E_y(x1,x2)

ax1 = plt.subplot(111)

plt.imshow(y,extent=[x1_min,x1_max,x2_min,x2_max], cmap="gray", origin='lower')

ticklabels = ax1.get_xticklabels()
ticklabels.extend( ax1.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(10)

cbar = plt.colorbar(ticks=[0,1])

cbar.ax.set_yticklabels(['0', 'maxima'])
plt.ylabel(r'$y$', size=17)
plt.xlabel(r'$x$', size=17)
plt.title(r"Norme de la composante $E_y$ du mode TE42" , fontsize=13)

plt.show()