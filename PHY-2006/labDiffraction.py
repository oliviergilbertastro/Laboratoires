import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from random import gauss

def resample_array(current, current_std):
    new = []
    for s in range(0, len(current)):
        if current_std[s] > 0:
            new.append(gauss(current[s], current_std[s]))
        else:
            new.append(current[s])
    return np.asarray(new)

def get_values_from_file(filename):
    values = pd.read_csv(filename, delimiter="\t", decimal=",", skiprows=3)
    return values




#Quality plot:
ax1 = plt.subplot(111)
ticklabels = ax1.get_xticklabels()
ticklabels.extend( ax1.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(14)
#plt.errorbar(resistance_avecC, puissance_moy_avecC, puissance_moy_avecC_stdev, resistance_avecC_stdev, ".", label="données")
#plt.plot(x_sim, found_sim, color="red", label="modèle ajusté")
#plt.plot(x_sim, empirique, color="blue", label=r"modèle 50$\Omega$")
#plt.fill_between(x_sim, sim_lo3, sim_hi3, color="orange", alpha=0.2)
#plt.fill_between(x_sim, sim_lo2, sim_hi2, color="orange", alpha=0.4)
#plt.fill_between(x_sim, sim_lo1, sim_hi1, color="orange", alpha=0.6)
#plt.fill_betweenx([0, 0.003], x_sim[sim_lo3.index(max(sim_lo3))], x_sim[sim_hi3.index(max(sim_hi3))], color="green", alpha=0.3, label="transfert de puissance maximal")
plt.legend()
plt.xscale('log')
plt.ylabel(r'P$_\mathrm{moy}$ [W]', size=17)
plt.xlabel(r'Résistance [$\Omega$]', size=17)
#plt.savefig(r'C:\Users\olivi\Desktop\Devoirs\PhysElectronique\figures\lab5\resistance_avecC.pdf', format="pdf", bbox_inches="tight")
plt.show()