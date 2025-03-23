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
param_pic1275_fixe = [0, 599, 17.22200702175016, 328.5997793873903, 92.30544684333118, 0.37208571808126334, 2.2314191710734614, 2.873405399316031]
param_pic1275_mob =[0, 599, 16.486394742368812, 267.50302762583095, 82.03752185849137, 0.3593121109065612, 2.0305880625008847, 2.4973164627420195]

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
areas = []
for params in [param_pic1275_fixe, param_pic1275_mob]:
    x0, x_1, A, mu, sigma, A_err, mu_Err, sigma_Err = params # Choose here wether params_fixe or params_mobile
    efficacity_list = []
    area_list = []

    pic = '1275'


    for i in tqdm(range(1000)):
        # Resample everything to do mcmc
        A_resampled, mu_resampled, sigma_resampled, DISTANCE_SOURCE_resampled = resample_array([A, mu, sigma, DISTANCE_SOURCE], [A_err, mu_Err, sigma_Err, 0.5])
        #DISTANCE_SOURCE_resampled = resample_array(DISTANCE_SOURCE, 0.5)


        background_subtracted = quad(gaussienne, x0, x_1, args=(A_resampled, mu_resampled, sigma_resampled,0,0))[0]
        G = (np.pi*R_DETECTOR**2)/(4*np.pi*DISTANCE_SOURCE_resampled**2)
        if pic=='511':
            f = 1.78

        if pic=='1275':
            f = 0.994 
        efficacity = (background_subtracted)/300*1/(G*f*Activity)
        efficacity_list.append(efficacity)
        area_list.append(background_subtracted)


    print(np.median(efficacity_list), np.std(efficacity_list))
    efficacities.append([np.median(efficacity_list), np.std(efficacity_list)])
    areas.append([np.median(area_list), np.std(area_list)])

efficacity_final = (efficacities[0][0]+efficacities[1][0])/2
efficacity_final_err = np.sqrt(efficacities[0][1]**2+efficacities[1][1]**2)
print("Efficacité", efficacity_final, efficacity_final_err)
area_final = (areas[0][0]+areas[1][0])/2
area_final_err = np.sqrt(areas[0][1]**2+areas[1][1]**2)
print("Aire", area_final, area_final_err)

efficacity_theorique = 0.34057509399566865 # from plotdigitizer




A_1275, A_pm_1275, eff_1275, eff_pm_1275 = 3681.8177971981627, 191.725309326626, 0.10588515085434425, 0.0129668263820589
A_511, A_pm_511, eff_511, eff_pm_511 = 20309.515088486252, 332.07718842439675, 0.3254740990993334, 0.03591170733802908


# Calcul du ratio corrigé en prenant en compte l'efficacité
A_ratio = (A_511 ) / (A_1275)
eff_ratio =  (eff_511 ) / (eff_1275)
print('A_ratio, eff_ratio', A_ratio, eff_ratio)

corrected_ratio = A_ratio/eff_ratio
# Calcul de l'incertitude pour le ratio corrigé en tenant compte des incertitudes sur A et eff
uncertainty_corrected_ratio = corrected_ratio * np.sqrt(
    (A_pm_511 / A_511)**2 + (A_pm_1275 / A_1275)**2 + (eff_pm_511 / eff_511)**2 + (eff_pm_1275 / eff_1275)**2
)


print('ratio_pic, uncertainty_ratio_pic', corrected_ratio, uncertainty_corrected_ratio)


f_1275 = 0.994
f_511 = 1.78

ratio = f_511/f_1275
print('ratio_théorique', ratio)
