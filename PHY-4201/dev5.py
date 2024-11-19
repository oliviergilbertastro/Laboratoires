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

R_S = 1 # En unités de Rayon de Schwarzchild
r_mins = np.linspace(1.502, 1.545, 100)*R_S

def function_to_integrate(x, r_min, L, m):
    return 1/np.sqrt((1-x**2)+R_S/r_min*(x**3-1)+R_S*r_min*m**2/(L**2)*(x-1))

trajs = []
nb_tours = []
for r_min in r_mins:
    delta = 2*np.abs(quad(function_to_integrate, 0, 1, args=(r_min, 1, 0))[0])-np.pi
    trajs.append(delta)
    nb_tours.append(delta/(2*np.pi))

fig = plt.figure(figsize=(5.5,4))
ax1 = plt.subplot(111)
plt.plot(r_mins, trajs)
plt.xlabel(r"$r_\mathrm{min}$ [$R_S$]", fontsize=17)
plt.ylabel(r"$\delta$ [rad]", fontsize=17)
plt.subplots_adjust(0.125,0.150,0.974,0.974)
ax1.xaxis.set_tick_params(labelsize=14)
ax1.yaxis.set_tick_params(labelsize=14)
plt.savefig("PHY-4201/dev5_fig1.pdf")
plt.show()

fig = plt.figure(figsize=(5.5,4))
ax1 = plt.subplot(111)
plt.plot(r_mins, nb_tours)
plt.xlabel(r"$r_\mathrm{min}$ [$R_S$]", fontsize=17)
plt.ylabel(r"Nombre de tours", fontsize=17)
plt.subplots_adjust(0.125,0.150,0.974,0.974)
ax1.xaxis.set_tick_params(labelsize=15)
ax1.yaxis.set_tick_params(labelsize=15)
plt.savefig("PHY-4201/dev5_fig2.pdf")
plt.show()