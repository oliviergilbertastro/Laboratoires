import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

x = [-1, -0.9300341296928327, -0.8395904436860069, -0.7337883959044369, -0.6569965870307167, -0.5972696245733788, -0.5392491467576792, -0.4744027303754266, -0.3976109215017065, -0.30375426621160406, -0.23549488054607504, -0.14846416382252559, -0.06996587030716728, 0.017064846416382284, 0.11774744027303763, 0.19965870307167233, 0.2849829351535835, 0.32423208191126274, 0.39419795221843, 0.453924914675768, 0.5221843003412969, 0.6023890784982935, 0.6638225255972696, 0.7252559726962458, 0.7832764505119454, 0.8447098976109215, 0.9078498293515358, 0.9436860068259385, 0.9778156996587031, 1]
y = [0, 0.009375000000000001, 0.0421875, 0.1046875, 0.1640625, 0.22031250000000002, 0.2765625, 0.3453125, 0.4296875, 0.5328125, 0.6109375, 0.7015625000000001, 0.78125, 0.8578125000000001, 0.93125, 0.9734375000000001, 0.9968750000000001, 1, 0.99375, 0.9750000000000001, 0.9359375000000001, 0.8640625000000001, 0.7875000000000001, 0.6953125, 0.58125, 0.453125, 0.284375, 0.184375, 0.078125,  -0.0015625]


cs = CubicSpline(x,y)

k = np.linspace(-1,1,10000)
k = np.linspace(-1,0.333,10000)
E = cs(k)

from scipy.special import gamma
def g(k, n=3):
    return 2/gamma(n/2)*(1/np.sqrt(np.pi))**n*(k/2)**(n-1)

# Calcul de la dérivée dE/dk
dE_dk = cs.derivative()(k)

# Calcul de g(E) = 1/|dE/dk|
g_E = g(k) / np.abs(dE_dk)

# Tracé de E(k)
plt.figure(figsize=(6,4))
plt.plot(k, E, label="$E(k)$")
plt.xlabel("$k$")
plt.ylabel("$E$")
plt.legend()
plt.show()

# Tracé de dE/dk
plt.figure(figsize=(6,4))
plt.plot(k, dE_dk, label="$dE/dk$")
plt.xlabel("$E$")
plt.ylabel("$dE/dk$")
plt.legend()
plt.show()

# Tracé de g(E)
plt.figure(figsize=(6,4))
plt.plot(E, g_E, label="$g(E) = g(k)dk/dE$")
plt.xlabel("$E$")
plt.ylabel("$g(E)$")
plt.legend()
plt.show()

# Tracé de g(E)
#frame1 = plt.gca()
plt.figure(figsize=(6,4))
plt.plot(E, g_E, label="$g(E) = g(k)dk/dE$")
plt.xlabel("$E$")
plt.ylabel("$g(E)$")
#plt.legend()
plt.xticks([])
plt.yticks([])
plt.show()