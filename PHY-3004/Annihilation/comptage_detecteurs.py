import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

R_DETECTOR = 5.08/2 #cm
DISTANCE_SOURCE = 13 # cm

#data_coincidences = np.loadtxt("PHY-3004/Annihilation/Data/angles/coincidences.txt", skiprows=2, dtype=float)
#angles = data_coincidences[:-1,0]
#N_coincidences = data_coincidences[:-1,1]
#N_coincidences = np.nan_to_num(N_coincidences)

def read_spec(deg=0, detecteur="fixe"):
    return np.loadtxt(f"PHY-3004/Annihilation/Data/angles/{detecteur}_Na22_{deg}.Spe", skiprows=12, max_rows=4096, dtype=float)


desintegrations_par_s_par_detecteur = 116.88275917159761


Activity = 0.331E-6 * 3.7E10 # /s
spec = read_spec(0)

from scipy.integrate import quad
def gaussienne(x, A, mu, sigma, B, C):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + B * x + C
params_fixe = [0, 439, 165.94214010233696, 225.91540225175675, 50.82879428348806, 1.157130594959765, 0.4045817620286838, 0.46691357851332144]
params_mobile = [0, 439, 156.1533060312751, 205.71907092059504, 49.73216362969077, 1.07992364101181, 0.39196032062595415, 0.4517528852761061]

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

efficacities = []
for params in [params_fixe, params_mobile]:
    x0, x_1, A, mu, sigma, A_err, mu_Err, sigma_Err = params # Choose here wether params_fixe or params_mobile
    efficacity_list = []
    for i in tqdm(range(1000)):
        # Resample everything to do mcmc
        A_resampled, mu_resampled, sigma_resampled, DISTANCE_SOURCE_resampled = resample_array([A, mu, sigma, DISTANCE_SOURCE], [A_err, mu_Err, sigma_Err, 0.5])
        #DISTANCE_SOURCE_resampled = resample_array(DISTANCE_SOURCE, 0.5)


        background_subtracted = quad(gaussienne, x0, x_1, args=(A_resampled, mu_resampled, sigma_resampled,0,0))[0]
        G = (np.pi*R_DETECTOR**2)/(4*np.pi*DISTANCE_SOURCE_resampled**2)
        f = 1.78
        efficacity = (background_subtracted)/300*1/(G*f*Activity)
        efficacity_list.append(efficacity)


    print(np.median(efficacity_list), np.std(efficacity_list))
    efficacities.append([np.median(efficacity_list), np.std(efficacity_list)])


efficacity_final = (efficacities[0][0]+efficacities[1][0])/2
efficacity_final_err = np.sqrt(efficacities[0][1]**2+efficacities[1][1]**2)
print(efficacity_final, efficacity_final_err)

efficacity_theorique = 0.34057509399566865 # from plotdigitizer