import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import G

J = G # moment angulaire du trou noir
M = np.linspace(1,10,1000) # masse du trou noir
a = J/M # jsp trop c'est quoi que ça represente mais c'est ça

r_plus_sol1 = G*M+np.sqrt(G**2*M**2-a**2) # horizon des evenements externe
r_plus_sol2 = G*M-np.sqrt(G**2*M**2-a**2) # horizon des evenements externe

Omega_H_sol1 = a/(r_plus_sol1**2+a**2) # vitesse angulaire a l'horizon externe
Omega_H_sol2 = a/(r_plus_sol2**2+a**2) # vitesse angulaire a l'horizon externe

plt.plot(M, r_plus_sol1, color="blue")
plt.plot(M, r_plus_sol2, color="blue")
plt.xlabel(r"$M$", fontsize=15)
plt.ylabel(r"$r_H$", fontsize=15)
plt.show()

plt.plot(M, Omega_H_sol1, color="blue")
plt.plot(M, Omega_H_sol2, color="blue")
plt.xlabel(r"$M$", fontsize=15)
plt.ylabel(r"$\Omega_H$", fontsize=15)
plt.show()

plt.plot(M, G*M**2)
plt.xlabel(r"$M$", fontsize=15)
plt.ylabel(r"$GM^2$", fontsize=15)
plt.show()