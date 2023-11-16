import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit


def get_values_from_file(filename):
    values = pd.read_csv(filename, delimiter="\t", decimal=",", skiprows=21)
    return values



#On créé une liste des résistances mesurées et une liste des tensions mesurées (AVEC CONDENSATEUR):

resistance_avecC = []
resistance_avecC_stdev = []
tension_avecC = []
tension_avecC_stdev = []

for i in range(14):
    res = np.array(get_values_from_file(f"lab5/AVEC_C/resistance_{i}.lvm").iloc[:, 1])
    ten = np.array(get_values_from_file(f"lab5/AVEC_C/tension_efficace_{i}.lvm").iloc[:, 1])

    #On calcule la val. médianne et l'incertitude (stdev):



    resistance_avecC.append(np.median(res))
    tension_avecC.append(np.median(ten))
    resistance_avecC_stdev.append(np.std(res))
    tension_avecC_stdev.append(np.std(ten))



puissance_moy_avecC = []
puissance_moy_avecC_stdev = []
for i in range(len(resistance_avecC)):
    puissance_moy_avecC.append(tension_avecC[i]**2/resistance_avecC[i])
    #Propagate the uncertainty to the power:
    puissance_moy_avecC_stdev.append(puissance_moy_avecC[i]*np.sqrt((resistance_avecC_stdev[i]/resistance_avecC[i])**2+(tension_avecC_stdev[i]/tension_avecC[i])**2))



ax1 = plt.subplot(111)
ticklabels = ax1.get_xticklabels()
ticklabels.extend( ax1.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(14)
plt.errorbar(resistance_avecC, puissance_moy_avecC, puissance_moy_avecC_stdev, resistance_avecC_stdev, ".")
plt.xscale('log')
plt.ylabel(r'P$_\mathrm{moy}$ [W]', size=17)
plt.xlabel(r'Résistance [$\Omega$]', size=17)
plt.show()

ax1 = plt.subplot(111)
ticklabels = ax1.get_xticklabels()
ticklabels.extend( ax1.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(14)
plt.plot(resistance_avecC, tension_avecC, ".")
plt.xscale('log')
plt.ylabel(r'Tension [V]', size=17)
plt.xlabel(r'Résistance [$\Omega$]', size=17)
plt.show()




x_sim = np.linspace(10,250, 100)



#même fct sans fixer V_S à 1/sqrt(2)
w = 1000*2*np.pi #2pi*1000Hz
c = 4E-6 #Farads
v_S = 1/np.sqrt(2)
x_S = 0
def puissance_tension_var(R, R_S):
    r_ch = (R/(1+(R*w*c)**2))
    x_ch = (-R**2*w*c/(1+(R*w*c)**2))
    return r_ch*v_S**2*(1/((r_ch+R_S)**2+(x_ch+x_S)**2))

#Fitting power:


#MCMC:
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

resistance_fit = []
tension_efficace_fit = []
x_s_fit = []
monte_carlo_iterations = input("How many iterations? [1000]")
if monte_carlo_iterations == "":
    monte_carlo_iterations = 1000
for i in tqdm(range(int(monte_carlo_iterations))):

    #Resample the capacity each iteration to account for its 20% uncertainty
    #Using c=3.31E-6 gives a good fit
    c = gauss(4E-6, (4E-6)*0.2)

    res = curve_fit(puissance_tension_var, resample_array(resistance_avecC, resistance_avecC_stdev), resample_array(puissance_moy_avecC, puissance_moy_avecC_stdev))[0]
    #tension_efficace_fit.append(res[0])
    resistance_fit.append(res[0])

#Make histograms of fitted params
ax1 = plt.subplot(111)
counts, bins = np.histogram(resistance_fit, bins=200)
ax1.stairs(counts, bins)
ax1.set_ylabel(r'Counts', size=17)
ax1.set_xlabel(r'Résistance [$\Omega$]', size=17)
plt.axvline(x = np.quantile(resistance_fit, 0.1585), color = 'blue', linestyle = '-', label=r'1$\sigma$')
plt.axvline(x = np.quantile(resistance_fit, 0.8415), color = 'blue', linestyle = '-')
plt.axvline(x = np.quantile(resistance_fit, 0.0225), color = 'green', linestyle = '-', label=r'2$\sigma$')
plt.axvline(x = np.quantile(resistance_fit, 0.9775), color = 'green', linestyle = '-')
plt.axvline(x = np.quantile(resistance_fit, 0.0015), color = 'orange', linestyle = '-', label=r'3$\sigma$')
plt.axvline(x = np.quantile(resistance_fit, 0.9985), color = 'orange', linestyle = '-')
plt.legend()
plt.show()

median_res = np.median(resistance_fit)

found_sim = []
for i in range(len(x_sim)):
    found_sim.append(puissance_tension_var(x_sim[i], median_res))

#print(np.array(puissance_moy_sansC)/np.array(pow_2))

ax1 = plt.subplot(111)
ticklabels = ax1.get_xticklabels()
ticklabels.extend( ax1.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(14)
plt.errorbar(resistance_avecC, puissance_moy_avecC, puissance_moy_avecC_stdev, resistance_avecC_stdev, ".", label="data")
plt.plot(x_sim, found_sim, label="fit")
plt.legend()
plt.xscale('log')
plt.ylabel(r'P$_\mathrm{moy}$ [W]', size=17)
plt.xlabel(r'Résistance [$\Omega$]', size=17)
plt.show()

sigma_1 = ((np.quantile(resistance_fit, 0.8415)-np.quantile(resistance_fit, 0.50))+(np.quantile(resistance_fit, 0.5)-np.quantile(resistance_fit, 0.1585)))/2
sigma_2 = ((np.quantile(resistance_fit, 0.9775)-np.quantile(resistance_fit, 0.50))+(np.quantile(resistance_fit, 0.5)-np.quantile(resistance_fit, 0.0225)))/2
sigma_3 = ((np.quantile(resistance_fit, 0.9985)-np.quantile(resistance_fit, 0.50))+(np.quantile(resistance_fit, 0.5)-np.quantile(resistance_fit, 0.0015)))/2

print(f"R_S:     {median_res}")
print(f"        1 sigma: {sigma_1}")
print(f"        2 sigma: {sigma_2}")
print(f"        3 sigma: {sigma_3}")