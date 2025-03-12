import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

R_DETECTOR = 5.08 #cm
DISTANCE_SOURCE = 13 # cm

def read_spec(detecteur="fixe", deg=0):
    return np.loadtxt(f"PHY-3004/Annihilation/Data/angles/{detecteur}_Na22_{deg}.txt", skiprows=15, max_rows=4096, dtype=float)

data_coincidences = np.loadtxt("PHY-3004/Annihilation/Data/angles/coincidences.txt", skiprows=2, dtype=float)
angles = data_coincidences[:-1,0]
N_coincidences = data_coincidences[:-1,1]
N_coincidences = np.nan_to_num(N_coincidences)

bkg_coincidences = data_coincidences[-1,1]
N_coincidences -= bkg_coincidences
uncertainty_coincidences = np.sqrt(0**2) # need to add the error from the poisson uncertainty

def A_normal_theorique(phi, ratio, offset):
    # ratio = dis_source/r_detector
    phi = np.abs(phi-offset)
    epsilon = 2*np.arccos(ratio*np.sin((phi)/2))
    return np.nan_to_num((epsilon-np.sin(epsilon))/np.pi)


def rad_to_deg(rads):
    return np.array(rads)*180/np.pi
def deg_to_rad(degs):
    return np.array(degs)*np.pi/180

from scipy.optimize import curve_fit
# Fitting the curve to the data
res, sigmas = curve_fit(A_normal_theorique, deg_to_rad(angles), N_coincidences/np.max(N_coincidences), p0=[DISTANCE_SOURCE/R_DETECTOR, 0])

print(res, sigmas)
angles_theo = np.linspace(-40,40,1000)
plt.plot(angles_theo, A_normal_theorique(deg_to_rad(angles_theo), DISTANCE_SOURCE/R_DETECTOR, 0), "--", color="red", label="théorique")
plt.errorbar(angles, N_coincidences/np.max(N_coincidences), yerr=uncertainty_coincidences/np.max(N_coincidences), fmt="o", color="black", label="données")
plt.plot(angles_theo, A_normal_theorique(deg_to_rad(angles_theo), res[0], res[1]), "-", color="blue", label="fit")
plt.plot(angles_theo, A_normal_theorique(deg_to_rad(angles_theo), 5, -0.01178981), "-", color="orange", label="manual fit")
plt.legend()
plt.xlabel(r"$\phi \, \mathrm{[^\circ]}$", fontsize=15)
plt.ylabel(r"$\%$ max", fontsize=15)
plt.show()
