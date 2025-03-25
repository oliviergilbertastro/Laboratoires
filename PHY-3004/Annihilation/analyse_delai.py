import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

R_DETECTOR = 5.08/2 #cm
DISTANCE_SOURCE = 13 # cm

data_delais = np.loadtxt("PHY-3004/Annihilation/Data/delai/delai.txt", skiprows=2, dtype=float)
delais = data_delais[:,0]
N_coincidences = data_delais[:,1]
N_coincidences = np.nan_to_num(N_coincidences)

uncertainty_coincidences = np.sqrt(N_coincidences)
uncertainty_delais = np.ones_like(delais)*0.02

#delais -= 0.01988732
#delais = np.abs(delais)
plt.plot(delais, N_coincidences, "o")
plt.show()

from scipy.optimize import curve_fit
from tqdm import tqdm
from random import gauss
# Fitting the gaussian curve to the data

def resample_array(current, current_std):
    new = []
    for s in range(0, len(current)):
        if current_std[s] > 0:
            new.append(gauss(current[s], current_std[s]))
        else:
            new.append(current[s])
    return np.asarray(new)

def gaussian(x, sig, b, mu, c):
    return (
        b / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x-mu) / sig, 2.0) / 2) + c
    )
res, cov_matrix = curve_fit(
        gaussian, 
        delais, 
        N_coincidences,
        p0=[1,np.max(N_coincidences),0,0],  # Initial guess
        bounds=[[0,0,0,0],[np.inf, np.inf,np.inf,np.inf]]
    )
sigmas = np.sqrt(np.diag(cov_matrix))
print(res, sigmas)

#uncertainty_coincidences /= np.max(N_coincidences)
#N_coincidences /= np.max(N_coincidences)
# Monte-Carlo to resample all values in their uncertainties, which gives us lists of the params to evaluate quantiles
mcmc_iterations = 10000
res_list = []
sigmas_list = []
for i in tqdm(range(mcmc_iterations)):
    resampled_delais = resample_array(delais, uncertainty_delais)
    resampled_N_coincidences = resample_array(N_coincidences, uncertainty_coincidences)
    resampled_N_coincidences = np.where(resampled_N_coincidences < 0, 0, resampled_N_coincidences)
    try:
        res, cov_matrix = curve_fit(
            gaussian, 
            resampled_delais, 
            resampled_N_coincidences,
            p0=[1,np.max(N_coincidences),0,0],  # Initial guess
            bounds=[[0,0,0,0],[np.inf, np.inf,np.inf,np.inf]]
        )
    except:
        pass
        #plt.plot(resampled_delais, resampled_N_coincidences)
        #plt.show()
    sigmas = np.sqrt(np.diag(cov_matrix))
    res_list.append(res)
    sigmas_list.append(sigmas)
print(res, sigmas)


delais_theo = np.linspace(-0.2,0.3,1000)
fits = []
for i in tqdm(range(mcmc_iterations)):
    params = res_list[int(np.random.randint(0,len(res_list)))] # Resample the fitted parameters in their uncertainties
    fits.append(gaussian(delais_theo, *params))

res = np.median(res_list, axis=0)
ax1 = plt.subplot(111)
plt.plot(delais_theo, gaussian(delais_theo, *res), "--", color="red", label="fit")
plt.fill_between(delais_theo, np.quantile(fits, 0.0015, axis=0), np.quantile(fits, 0.9985, axis=0), color="blue", edgecolor="none", alpha=0.2)
plt.fill_between(delais_theo, np.quantile(fits, 0.0225, axis=0), np.quantile(fits, 0.9775, axis=0), color="blue", edgecolor="none", alpha=0.4)
plt.fill_between(delais_theo, np.quantile(fits, 0.1585, axis=0), np.quantile(fits, 0.8415, axis=0), color="blue", edgecolor="none", alpha=0.6)
plt.errorbar(delais, N_coincidences, xerr=uncertainty_delais, yerr=uncertainty_coincidences, fmt="o", color="black", label="données")
plt.legend(fontsize=15)
plt.xlabel(r"$\Delta t\, \mathrm{[\mu s]}$", fontsize=16)
plt.ylabel(r"$N_c$", fontsize=16)
ax1.xaxis.set_tick_params(labelsize=15)
ax1.yaxis.set_tick_params(labelsize=15)
plt.tight_layout()
plt.savefig("PHY-3004/Annihilation/Figures/delai_fit_2.pdf")
plt.show()


import sys
sys.exit()

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


ax1 = plt.subplot(111)
plt.legend(fontsize=15)
plt.xlabel(r"$\Delta t\, \mathrm{[\mu s]}$", fontsize=16)
plt.ylabel(r"$N_c$", fontsize=16)
ax1.xaxis.set_tick_params(labelsize=15)
ax1.yaxis.set_tick_params(labelsize=15)
plt.tight_layout()
plt.savefig("PHY-3004/Annihilation/Figures/delai_fit2.pdf")
plt.show()