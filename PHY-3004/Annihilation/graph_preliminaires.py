import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

R_DETECTOR = 5.08/2 #cm
DISTANCE_SOURCE = 13 # cm

def A_normal_theorique(phi, ratio, offset):
    # ratio = dis_source/r_detector
    phi = np.abs(phi-offset)
    epsilon = 2*np.arccos(ratio*np.sin((phi)/2))
    return np.nan_to_num((epsilon-np.sin(epsilon))/np.pi)

def rad_to_deg(rads):
    return np.array(rads)*180/np.pi
def deg_to_rad(degs):
    return np.array(degs)*np.pi/180

angles_theo = np.linspace(-30,30,1000)
ax1 = plt.subplot(111)
plt.plot(angles_theo, A_normal_theorique(deg_to_rad(angles_theo), DISTANCE_SOURCE/R_DETECTOR, 0), "-", linewidth=3, color="red", alpha=0.5, label="$d=$13cm")
plt.plot(angles_theo, A_normal_theorique(deg_to_rad(angles_theo), 15/R_DETECTOR, 0), "--", linewidth=2, color="blue", alpha=0.5, label="$d=$15cm")
plt.hlines(0.5, -30, 30, linestyles="dashed", colors="black")

plt.plot([-22.53, 22.53], [0,0], "o", color="red")
plt.plot([-9.05,9.05], [0.5,0.5], "*", markersize=10, color="red")

plt.plot([-19.5,19.5], [0,0], "o", color="blue")
plt.plot([-7.84,7.84], [0.5,0.5], "*", markersize=10, color="blue")

plt.text(-30, 0.51, r"$N/N_0=50\%$", fontsize=16)
plt.legend(fontsize=15)
plt.xlabel(r"$\phi \, \mathrm{[^\circ]}$", fontsize=16)
plt.ylabel(r"$N/N_0$", fontsize=16)
ax1.xaxis.set_tick_params(labelsize=15)
ax1.yaxis.set_tick_params(labelsize=15)
plt.tight_layout()
plt.savefig("PHY-3004/Annihilation/Figures/prelim.pdf")
plt.show()