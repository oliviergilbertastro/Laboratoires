"""
#4e du devoir 5 de relgen:

Pour un photon, montrer numériquement à partir de l'intégrale obtenue en
(4c) que le photon orbite entre un et deux tours autour du trou noir avant de s'échapper
vers l'infini pour r_min ∈ (1.502, 1.545)R_S.

"""


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import quad

R_S = 1 # Rayon Schwarzchild
r_mins = np.linspace(1.502, 1.545, 100)*R_S

def function_to_integrate(x, r_min, L, m):
    return 1/np.sqrt(1/r_min**2*(1-x**2)+R_S/r_min**3*(x**3-1)+R_S*m**2/(L**2*r_min)*(x-1))

trajs = []
nb_tours = []
for r_min in r_mins:
    delta = quad(function_to_integrate, 0, 1, args=(r_min, 1, 0))[0]-np.pi
    trajs.append(delta)
    nb_tours.append(delta/(2*np.pi))

plt.plot(r_mins, trajs)
plt.xlabel(r"$r_\mathrm{min}$ [$R_S$]", fontsize=17)
plt.ylabel(r"$\delta$ [rad]", fontsize=17)
plt.show()

plt.plot(r_mins, nb_tours)
plt.xlabel(r"$r_\mathrm{min}$ [$R_S$]", fontsize=17)
plt.ylabel(r"Nombre de tours", fontsize=17)
plt.show()