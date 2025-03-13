import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

R_DETECTOR = 5.08/2 #cm
DISTANCE_SOURCE = 13 # cm

def read_spec(detecteur="fixe", deg=0):
    return np.loadtxt(f"PHY-3004/Annihilation/Data/angles/{detecteur}_Na22_{deg}.txt", skiprows=15, max_rows=4096, dtype=float)

data_coincidences = np.loadtxt("PHY-3004/Annihilation/Data/angles/coincidences.txt", skiprows=2, dtype=float)
angles = data_coincidences[:-1,0]
N_coincidences = data_coincidences[:-1,1]
N_coincidences = np.nan_to_num(N_coincidences)

bkg_coincidences = data_coincidences[-1,1]
uncertainty_coincidences = np.sqrt(N_coincidences+bkg_coincidences)
N_coincidences -= bkg_coincidences
uncertainty_angles = np.ones_like(angles)*0.5

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
# Fitting the theoretical curve to the data
uncertainty_coincidences /= np.max(N_coincidences)
N_coincidences /= np.max(N_coincidences)
res, cov_matrix = curve_fit(
    A_normal_theorique, 
    deg_to_rad(angles), 
    N_coincidences, 
    sigma=uncertainty_coincidences,
    p0=[DISTANCE_SOURCE / R_DETECTOR, 0]  # Initial guess
)
sigmas = np.sqrt(np.diag(cov_matrix))
print(res, sigmas)
angles_theo = np.linspace(-40,40,1000)
plt.plot(angles_theo, A_normal_theorique(deg_to_rad(angles_theo), DISTANCE_SOURCE/R_DETECTOR, 0), "--", color="red", label="théorique")
plt.errorbar(angles, N_coincidences, xerr=uncertainty_angles, yerr=uncertainty_coincidences, fmt="o", color="black", label="données")
plt.plot(angles_theo, A_normal_theorique(deg_to_rad(angles_theo), res[0], res[1]), "-", color="blue", label="fit")
plt.legend()
plt.xlabel(r"$\phi \, \mathrm{[^\circ]}$", fontsize=15)
plt.ylabel(r"$\%$ max", fontsize=15)
plt.show()



# Mont-Carlo to trace the uncertainties on the model:
from tqdm import tqdm
from random import gauss

def resample_array(current, current_std):
    new = []
    for s in range(0, len(current)):
        if current_std[s] > 0:
            new.append(gauss(current[s], current_std[s]))
        else:
            new.append(current[s])
    return np.asarray(new)


fits = []
for i in tqdm(range(10000)):
    params = resample_array(res,sigmas) # Resample the fitted parameters in their uncertainties
    fits.append(A_normal_theorique(deg_to_rad(angles_theo), *params))

ax1 = plt.subplot(111)
plt.plot(angles_theo, A_normal_theorique(deg_to_rad(angles_theo), DISTANCE_SOURCE/R_DETECTOR, 0), "--", color="red", label="théorique")
plt.fill_between(angles_theo, np.quantile(fits, 0.0015, axis=0), np.quantile(fits, 0.9985, axis=0), color="orange", edgecolor="none", alpha=0.2)
plt.fill_between(angles_theo, np.quantile(fits, 0.0225, axis=0), np.quantile(fits, 0.9775, axis=0), color="orange", edgecolor="none", alpha=0.4)
plt.fill_between(angles_theo, np.quantile(fits, 0.1585, axis=0), np.quantile(fits, 0.8415, axis=0), color="orange", edgecolor="none", alpha=0.6)
plt.errorbar(angles, N_coincidences, xerr=uncertainty_angles, yerr=uncertainty_coincidences, fmt="o", color="black", label="données")
plt.plot(angles_theo, A_normal_theorique(deg_to_rad(angles_theo), res[0], res[1]), "-", color="red", label="fit")
plt.legend(fontsize=15)
plt.xlabel(r"$\phi \, \mathrm{[^\circ]}$", fontsize=16)
plt.ylabel(r"$N/N_0$", fontsize=16)
ax1.xaxis.set_tick_params(labelsize=15)
ax1.yaxis.set_tick_params(labelsize=15)
plt.tight_layout()
plt.savefig("PHY-3004/Annihilation/Figures/theoretical_fit.pdf")
plt.show()


# Fitting a gaussian to the data
def gaussian(x, sig, b, mu, c):
    return (
        b / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x-mu) / sig, 2.0) / 2) + c
    )

res, cov_matrix = curve_fit(
    gaussian, 
    deg_to_rad(angles), 
    N_coincidences, 
    sigma=uncertainty_coincidences,
    p0=[1, 1, 0, 0]  # Initial guess
)
sigmas = np.sqrt(np.diag(cov_matrix))
print(res, sigmas)

fits = []
for i in tqdm(range(10000)):
    params = resample_array(res,sigmas) # Resample the fitted parameters in their uncertainties
    fits.append(gaussian(deg_to_rad(angles_theo), *params))

ax1 = plt.subplot(111)
plt.plot(angles_theo, A_normal_theorique(deg_to_rad(angles_theo), DISTANCE_SOURCE/R_DETECTOR, 0), "--", color="red", label="théorique")
plt.fill_between(angles_theo, np.quantile(fits, 0.0015, axis=0), np.quantile(fits, 0.9985, axis=0), color="blue", edgecolor="none", alpha=0.2)
plt.fill_between(angles_theo, np.quantile(fits, 0.0225, axis=0), np.quantile(fits, 0.9775, axis=0), color="blue", edgecolor="none", alpha=0.4)
plt.fill_between(angles_theo, np.quantile(fits, 0.1585, axis=0), np.quantile(fits, 0.8415, axis=0), color="blue", edgecolor="none", alpha=0.6)
plt.errorbar(angles, N_coincidences, xerr=uncertainty_angles, yerr=uncertainty_coincidences, fmt="o", color="black", label="données")
plt.plot(angles_theo, gaussian(deg_to_rad(angles_theo), *res), "-", color="blue", label="fit gaussien")
plt.legend(fontsize=15)
plt.xlabel(r"$\phi \, \mathrm{[^\circ]}$", fontsize=16)
plt.ylabel(r"$N/N_0$", fontsize=16)
ax1.xaxis.set_tick_params(labelsize=15)
ax1.yaxis.set_tick_params(labelsize=15)
plt.tight_layout()
plt.savefig("PHY-3004/Annihilation/Figures/gaussian_fit.pdf")
plt.show()