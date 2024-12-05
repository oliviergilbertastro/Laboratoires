import numpy as np
import matplotlib.pyplot as plt

from scipy.constants import G, hbar, m_n
from astropy.constants import M_sun

M = M_sun.value
N = M/m_n


def mean_energy(R):
    return M**(5/3)*hbar**2/(5*R**2)*(3**7*np.pi**2/(2**7*m_n**8))**(1/3)
def grav_energy(R):
    return -3*G*M**2/(5*R)
def energy(R):
    return np.array(mean_energy(R)) + np.array(grav_energy(R))


radius = np.linspace(1E3, 3*1E4, 100000)


R = 2*hbar**2/(3*G)*((3/2)**7*np.pi**2/(M*m_n**8))**(1/3)


E_mean = mean_energy(radius)
E_grav = grav_energy(radius)
E_tot = energy(radius)


r = radius[(list(E_tot).index(np.min(E_tot)))] # meters
print(f"Rayon qui minimise théorique: {R/1000} km")
print(f"Rayon qui minimise pratique: {r/1000} km")


ax1 = plt.subplot(111)
plt.plot(radius/1000, E_mean, color="blue", label=r"$\overline{E}$")
plt.plot(radius/1000, E_grav, color="red", label=r"$E_\mathrm{grav}$")
plt.plot(radius/1000, E_tot, color="purple", label=r"$E_\mathrm{tot}$", linewidth=2)
plt.vlines(r/1000, np.min(E_grav), np.max(E_mean), linestyle="--", color="black", linewidth=2)
plt.xlabel(r"$R$ [km]", fontsize=15)
plt.ylabel(r"$E$ [J]", fontsize=15)
#plt.xscale("log")
ax1.xaxis.set_tick_params(labelsize=13)
ax1.yaxis.set_tick_params(labelsize=13)
plt.legend(fontsize=14)
plt.subplots_adjust(0.150, 0.14, 0.96, 0.93)
plt.show()