import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

x = [-1, -0.9300341296928327, -0.8395904436860069, -0.7337883959044369, -0.6569965870307167, -0.5972696245733788, -0.5392491467576792, -0.4744027303754266, -0.3976109215017065, -0.30375426621160406, -0.23549488054607504, -0.14846416382252559, -0.06996587030716728, 0.017064846416382284, 0.11774744027303763, 0.19965870307167233, 0.2849829351535835, 0.32423208191126274, 0.39419795221843, 0.453924914675768, 0.5221843003412969, 0.6023890784982935, 0.6638225255972696, 0.7252559726962458, 0.7832764505119454, 0.8447098976109215, 0.9078498293515358, 0.9436860068259385, 0.9778156996587031, 1]
y = [0, 0.009375000000000001, 0.0421875, 0.1046875, 0.1640625, 0.22031250000000002, 0.2765625, 0.3453125, 0.4296875, 0.5328125, 0.6109375, 0.7015625000000001, 0.78125, 0.8578125000000001, 0.93125, 0.9734375000000001, 0.9968750000000001, 1, 0.99375, 0.9750000000000001, 0.9359375000000001, 0.8640625000000001, 0.7875000000000001, 0.6953125, 0.58125, 0.453125, 0.284375, 0.184375, 0.078125,  -0.0015625]


cs = CubicSpline(x,y)

k_autre = np.linspace(-1,1,10000)
k = np.linspace(-1,0.333,10000)
E = cs(k)

from scipy.special import gamma
def g(k, n=3):
    return 2/gamma(n/2)*(1/np.sqrt(np.pi))**n*(k/2)**(n-1)

# Calcul de la dérivée dE/dk
dE_dk = cs.derivative()(k)

# Calcul de g(E) = g(k)/|dE/dk| = g(k)*dk/dE
g_E = g(k, n=1) / np.abs(dE_dk)

from scipy.constants import hbar, m_e
def dispersion(k):
    return hbar**2*k**2/(2*m_e)

#plt.plot(k, E, label="$E(k)$")
plt.plot(k, dispersion(k), label="$dispersion$")
plt.xlabel("$k$", fontsize=16)
plt.ylabel("$E$", fontsize=16)
plt.show()
# Tracé de E(k)
fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# First panel
ax.plot(k_autre, cs(k_autre), label="$E(k)$")
ax.set_xlabel("$k$", fontsize=16)
ax.set_ylabel("$E$", fontsize=16)
ax.legend()
labels = ax.get_xticks().tolist()
for i in range(len(labels)):
    labels[i] = ""
labels[1] = r"$-\pi/a$"
labels[-2] = r"$\pi/a$"
ax.set_xticklabels(labels)
labels = ax.get_yticks().tolist()
for i in range(len(labels)):
    labels[i] = ""
ax.set_yticklabels(labels)

print(np.min(g_E))
# Second panel
ax2.plot(g_E, E, label="$g(E)$")
ax2.fill_betweenx(E, 0.16635908732952864, g_E, color="blue", alpha=0.5)
ax2.set_xlabel("$g(E)$", fontsize=16)
ax2.set_ylabel("$E$", fontsize=16)
ax2.legend()
labels = ax.get_xticks().tolist()
for i in range(len(labels)):
    labels[i] = ""
ax2.set_xticklabels(labels)
labels = ax.get_yticks().tolist()
for i in range(len(labels)):
    labels[i] = ""
ax2.set_yticklabels(labels)
plt.tight_layout()
plt.show()
