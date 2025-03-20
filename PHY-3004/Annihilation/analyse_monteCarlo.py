import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import gauss
from scipy.optimize import curve_fit

def resample_array(current, current_std):
    new = []
    for s in range(0, len(current)):
        if current_std[s] > 0:
            new.append(gauss(current[s], current_std[s]))
        else:
            new.append(current[s])
    return np.asarray(new)

# Fitting a gaussian to the data
def gaussian(x, sig, b, mu, c):
    return (
        b / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x-mu) / sig, 2.0) / 2) + c
    )

R_DETECTOR = 5.08/2 #cm
DISTANCE_SOURCE = 15 # cm

def read_spec(detecteur="fixe", deg=0, res=""):
    if res != "":
        res = "_"+res
    return np.loadtxt(f"PHY-3004/Annihilation/Data/angles/{detecteur}_Na22_{deg}{res}.txt", skiprows=15, max_rows=4096, dtype=float)

data_coincidences = np.loadtxt("PHY-3004/Annihilation/Data/angles/coincidences_resolution.txt", skiprows=2, dtype=float)

RESOLUTIONS = [100,75,50,25]

def A_normal_theorique(phi, ratio, offset):
    # ratio = dis_source/r_detector
    phi = np.abs(phi-offset)
    epsilon = 2*np.arccos(ratio*np.sin((phi)/2))
    return np.nan_to_num((epsilon-np.sin(epsilon))/np.pi)

def rad_to_deg(rads):
    return np.array(rads)*180/np.pi
def deg_to_rad(degs):
    return np.array(degs)*np.pi/180

for resolution in RESOLUTIONS:
    angles = data_coincidences[:-1,0]
    N_coincidences = data_coincidences[:-1,RESOLUTIONS.index(resolution)+1]
    N_coincidences = np.nan_to_num(N_coincidences)

    bkg_coincidences = data_coincidences[-1,RESOLUTIONS.index(resolution)+1]
    uncertainty_coincidences = np.sqrt(N_coincidences+bkg_coincidences)
    N_coincidences -= bkg_coincidences
    uncertainty_angles = np.ones_like(angles)*0.5
    plt.errorbar(angles, N_coincidences, yerr=uncertainty_coincidences, xerr=uncertainty_angles, label=f"resolution = {resolution}")
plt.legend()
plt.show()
plt.savefig(f"PHY-3004/Annihilation/Figures/diff_resolutions.pdf")

liste_sigma = []
liste_sigma_std = []
for resolution in RESOLUTIONS:
    angles = data_coincidences[:-1,0]
    N_coincidences = data_coincidences[:-1,RESOLUTIONS.index(resolution)+1]
    N_coincidences = np.nan_to_num(N_coincidences)

    bkg_coincidences = data_coincidences[-1,RESOLUTIONS.index(resolution)+1]
    uncertainty_coincidences = np.sqrt(N_coincidences+bkg_coincidences)
    N_coincidences -= bkg_coincidences
    uncertainty_angles = np.ones_like(angles)*0.5

    print(angles)
    print(N_coincidences)
    # Fitting the theoretical curve to the data
    uncertainty_coincidences /= np.max(N_coincidences)
    N_coincidences /= np.max(N_coincidences)


    # Monte-Carlo to resample all values in their uncertainties, which gives us lists of the params to evaluate quantiles
    mcmc_iterations = 1000
    res_list = []
    sigmas_list = []
    for i in tqdm(range(mcmc_iterations)):
        resampled_angles = resample_array(angles, uncertainty_angles)
        resampled_N_coincidences = resample_array(N_coincidences, uncertainty_coincidences)
        resampled_N_coincidences = np.where(resampled_N_coincidences < 0, 0, resampled_N_coincidences)
        res, cov_matrix = curve_fit(
            gaussian, 
            deg_to_rad(resampled_angles), 
            resampled_N_coincidences,
            p0=[1,1,0,0],  # Initial guess
            bounds=[[0,0,0,0],[np.inf, np.inf,np.inf,np.inf]]
        )
        sigmas = np.sqrt(np.diag(cov_matrix))
        res_list.append(res)
        sigmas_list.append(sigmas)
    angles_theo = np.linspace(-40,40,1000)
    res_list = np.array(res_list)
    np.savetxt(f"PHY-3004/Annihilation/params_{resolution}.txt", res_list)

    liste_sigma.append(np.median(res_list[:,0]))
    liste_sigma_std.append(np.std(res_list[:,0]))
    print(np.median(res_list, axis=0),np.std(res_list, axis=0))
    fits = []
    for i in tqdm(range(mcmc_iterations)):
        params = resample_array(np.median(res_list, axis=0),np.std(res_list, axis=0)) # Resample the fitted parameters in their uncertainties
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
    plt.title(f"Résolution = {resolution}", fontsize=16)
    ax1.xaxis.set_tick_params(labelsize=15)
    ax1.yaxis.set_tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig(f"PHY-3004/Annihilation/Figures/gaussian_fit_{resolution}.pdf")
    plt.show()

ax1 = plt.subplot(111)
#plt.plot(RESOLUTIONS, rad_to_deg(liste_sigma), "o")
plt.errorbar(RESOLUTIONS, rad_to_deg(liste_sigma), yerr=rad_to_deg(liste_sigma_std), xerr=np.ones_like(RESOLUTIONS)*3, fmt="o")
plt.xlabel(r"Résolution", fontsize=16)
plt.ylabel(r"$\sigma$ [$^\circ$]", fontsize=16)
ax1.xaxis.set_tick_params(labelsize=15)
ax1.yaxis.set_tick_params(labelsize=15)
plt.tight_layout()
plt.show()