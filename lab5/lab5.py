import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit


def get_values_from_file(filename):
    values = pd.read_csv(filename, delimiter="\t", decimal=",", skiprows=21)
    return values




def create__osci_plot(filename, name):
    values = pd.read_csv(filename, delimiter=",", decimal=".", skiprows=1)
    x = np.array(values.iloc[:, 0])
    y = np.array(values.iloc[:, 1])
    z = np.array(values.iloc[:, 2])



    ax1 = plt.subplot(111)
    ticklabels = ax1.get_xticklabels()
    ticklabels.extend( ax1.get_yticklabels() )
    for label in ticklabels:
        label.set_fontsize(10)
    plt.plot(x, y, label="Canal 1")
    plt.plot(x, z, label="Canal 2")
    plt.legend()
    #plt.suptitle(f'Oscilloscope avec impulsion de durée {name}', size=17)
    plt.ylabel(r'Tension [V]', size=17)
    plt.xlabel(r'Temps [s]', size=17)
    plt.savefig(r'C:\Users\olivi\Desktop\Devoirs\PhysElectronique\figures\lab5'+f"\oscilloscope_{name}.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    return True



#create__osci_plot("lab5/usb/scope_i5.csv", "5W")
#create__osci_plot("lab5/usb/scope_i6.csv", "W")
#create__osci_plot("lab5/usb/scope_i7.csv", "W (court-circuité)")
#create__osci_plot("lab5/usb/scope_i8.csv", "5W (court-circuité)")


#On créé une liste des résistances mesurées et une liste des tensions mesurées (SANS CONDENSATEUR):

resistance_sansC = []
resistance_sansC_stdev = []
tension_sansC = []
tension_sansC_stdev = []

for i in range(14):
    res = np.array(get_values_from_file(f"lab5/SANS_C/resistance_{i}.lvm").iloc[:, 1])
    ten = np.array(get_values_from_file(f"lab5/SANS_C/tension_efficace_{i}.lvm").iloc[:, 1])

    #On calcule la val. médianne et l'incertitude (stdev):



    resistance_sansC.append(np.median(res))
    tension_sansC.append(np.median(ten))
    resistance_sansC_stdev.append(np.std(res)+0.00002*np.median(res)+0.000005*1000)
    tension_sansC_stdev.append(5E-3)



puissance_moy_sansC = []
puissance_moy_sansC_stdev = []
for i in range(len(resistance_sansC)):
    puissance_moy_sansC.append(tension_sansC[i]**2/resistance_sansC[i])
    #Propagate the uncertainty to the power:
    puissance_moy_sansC_stdev.append(puissance_moy_sansC[i]*np.sqrt((resistance_sansC_stdev[i]/resistance_sansC[i])**2+(tension_sansC_stdev[i]/tension_sansC[i])**2))



ax1 = plt.subplot(111)
ticklabels = ax1.get_xticklabels()
ticklabels.extend( ax1.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(14)
plt.errorbar(resistance_sansC, puissance_moy_sansC, puissance_moy_sansC_stdev, resistance_sansC_stdev, ".")
plt.xscale('log')
plt.ylabel(r'P$_\mathrm{moy}$ [W]', size=17)
plt.xlabel(r'Résistance [$\Omega$]', size=17)
plt.show()


ax1 = plt.subplot(111)
ticklabels = ax1.get_xticklabels()
ticklabels.extend( ax1.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(14)
plt.plot(resistance_sansC, tension_sansC, ".")
plt.xscale('log')
plt.ylabel(r'Tension [V]', size=17)
plt.xlabel(r'Résistance [$\Omega$]', size=17)
plt.show()




x_sim = np.linspace(10,250, 100)


#Model of P(R,V):

def puissance(known_params, R_S):
    return known_params[0]*np.abs(known_params[1])**2*(1/(known_params[0]+R_S)**2)

def puissance(R_ch, R_S):
    return R_ch*np.abs(1/np.sqrt(2))**2*(1/(R_ch+R_S)**2)

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
monte_carlo_iterations = input("How many iterations? [1000]")
if monte_carlo_iterations == "":
    monte_carlo_iterations = 1000
for i in tqdm(range(int(monte_carlo_iterations))):
    res = curve_fit(puissance, resample_array(resistance_sansC, resistance_sansC_stdev), resample_array(puissance_moy_sansC, puissance_moy_sansC_stdev))[0]
    #tension_efficace_fit.append(res[0])
    resistance_fit.append(res[0])

#Make histograms of fitted params
ax1 = plt.subplot(111)
counts, bins = np.histogram(resistance_fit, bins=200)
ax1.stairs(counts, bins)
ax1.set_ylabel(r"Nombre d'instances", size=17)
ax1.set_xlabel(r'Résistance [$\Omega$]', size=17)
plt.axvline(x = np.quantile(resistance_fit, 0.1585), color = 'blue', linestyle = '-', label=r'1$\sigma$')
plt.axvline(x = np.quantile(resistance_fit, 0.8415), color = 'blue', linestyle = '-')
plt.axvline(x = np.quantile(resistance_fit, 0.0225), color = 'green', linestyle = '-', label=r'2$\sigma$')
plt.axvline(x = np.quantile(resistance_fit, 0.9775), color = 'green', linestyle = '-')
plt.axvline(x = np.quantile(resistance_fit, 0.0015), color = 'orange', linestyle = '-', label=r'3$\sigma$')
plt.axvline(x = np.quantile(resistance_fit, 0.9985), color = 'orange', linestyle = '-')
plt.legend()
#plt.savefig(r'C:\Users\olivi\Desktop\Devoirs\PhysElectronique\figures\lab5'+f"\hist_sansC.pdf", format="pdf", bbox_inches="tight")
plt.show()

median_res = np.median(resistance_fit)


#Get models for median, 1sigma+ and 1sigma- to eventually fillbetween uncertainties

found_sim = []
sim_hi1 = []
sim_lo1 = []
sim_hi2 = []
sim_lo2 = []
sim_hi3 = []
sim_lo3 = []
for i in range(len(x_sim)):
    found_sim.append(puissance(x_sim[i], median_res))
    sim_hi1.append(puissance(x_sim[i], np.quantile(resistance_fit, 0.8415)))
    sim_lo1.append(puissance(x_sim[i], np.quantile(resistance_fit, 0.1585)))
    sim_hi2.append(puissance(x_sim[i], np.quantile(resistance_fit, 0.9775)))
    sim_lo2.append(puissance(x_sim[i], np.quantile(resistance_fit, 0.0225)))
    sim_hi3.append(puissance(x_sim[i], np.quantile(resistance_fit, 0.9985)))
    sim_lo3.append(puissance(x_sim[i], np.quantile(resistance_fit, 0.0015)))

#print(np.array(puissance_moy_sansC)/np.array(pow_2))

ax1 = plt.subplot(111)
ticklabels = ax1.get_xticklabels()
ticklabels.extend( ax1.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(14)
plt.errorbar(resistance_sansC, puissance_moy_sansC, puissance_moy_sansC_stdev, resistance_sansC_stdev, ".", label="données")
plt.plot(x_sim, found_sim, color="red", label="modèle ajusté")
plt.fill_between(x_sim, sim_lo3, sim_hi1, color="orange", alpha=0.2)
plt.fill_between(x_sim, sim_lo2, sim_hi1, color="orange", alpha=0.4)
plt.fill_between(x_sim, sim_lo1, sim_hi1, color="orange", alpha=0.6)
plt.legend()
plt.xscale('log')
plt.ylabel(r'P$_\mathrm{moy}$ [W]', size=17)
plt.xlabel(r'Résistance [$\Omega$]', size=17)
#plt.savefig(r'C:\Users\olivi\Desktop\Devoirs\PhysElectronique\figures\lab5\resistance_sansC.pdf', format="pdf", bbox_inches="tight")
plt.show()

sigma_1 = ((np.quantile(resistance_fit, 0.8415)-np.quantile(resistance_fit, 0.50))+(np.quantile(resistance_fit, 0.5)-np.quantile(resistance_fit, 0.1585)))/2
sigma_2 = ((np.quantile(resistance_fit, 0.9775)-np.quantile(resistance_fit, 0.50))+(np.quantile(resistance_fit, 0.5)-np.quantile(resistance_fit, 0.0225)))/2
sigma_3 = ((np.quantile(resistance_fit, 0.9985)-np.quantile(resistance_fit, 0.50))+(np.quantile(resistance_fit, 0.5)-np.quantile(resistance_fit, 0.0015)))/2

print(f"R_S:     {np.median(resistance_fit)}")
print(f"        1 sigma: {sigma_1}")
print(f"        2 sigma: {sigma_2}")
print(f"        3 sigma: {sigma_3}")
