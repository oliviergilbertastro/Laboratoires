import numpy as np
import matplotlib.pyplot as plt

k_0 = 1
gamma = 1
w_0 = 2
n = 1


def alpha(w):
    return 2*k_0*(1/(2*n)*(0.1*gamma*w)/((w_0**2)**2+(gamma*w)**2))



x = np.linspace(0, 100, 1000)
y = alpha(x)

plt.plot(x, y)
plt.xlabel(r'$\omega$', fontsize=17)
plt.ylabel(r'$\alpha$', fontsize=17)
plt.show()