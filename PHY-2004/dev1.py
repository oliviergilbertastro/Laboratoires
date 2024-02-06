import numpy as np
import matplotlib.pyplot as plt

c=299792458
w_0 = 1
F = 0.1

def d(w):
    return w_0**2-w**2

def n(w, gamma):
    delta=d(w)
    B=gamma*w
    return np.sqrt(1/2*(1+F*delta/(delta**2+B**2)+np.sqrt(1+2*F*delta/(delta**2+B**2)*(delta+F/2))))

def alpha(w, gamma):
    delta=d(w)
    B=gamma*w
    k_0 = w/c
    return k_0/n(w, gamma)*(F*B/(delta**2+B**2))



w_sim = np.linspace(0, 2*w_0, 1000000)

ax1 = plt.subplot(121)
ticklabels = ax1.get_xticklabels()
ticklabels.extend(ax1.get_yticklabels())
for label in ticklabels:
    label.set_fontsize(14)
ax2 = plt.subplot(122)
ticklabels = ax2.get_xticklabels()
ticklabels.extend(ax2.get_yticklabels())
for label in ticklabels:
    label.set_fontsize(14)
ax1.plot(w_sim, n(w_sim, 0.1), label='$\gamma=0.1$')
ax1.plot(w_sim, n(w_sim, 0.01), label='$\gamma=0.01$')
ax1.set_xlabel(r'$\omega$', fontsize=17)
ax1.set_ylabel(r'$n$', fontsize=17)
ax2.plot(w_sim, alpha(w_sim, 0.1), label='$\gamma=0.1$')
ax2.plot(w_sim, alpha(w_sim, 0.01), label='$\gamma=0.01$')
ax2.set_xlabel(r'$\omega$', fontsize=17)
ax2.set_ylabel(r'$\alpha$', fontsize=17)
ax1.legend()
ax2.legend()
plt.show()