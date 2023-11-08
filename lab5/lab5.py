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
tension_sansC = []

for i in range(14):
    res = np.array(get_values_from_file(f"lab5/SANS_C/resistance_{i}.lvm").iloc[:, 1])
    ten = np.array(get_values_from_file(f"lab5/SANS_C/tension_efficace_{i}.lvm").iloc[:, 1])

    resistance_sansC.extend(res)
    tension_sansC.extend(ten)

puissance_moy_sansC = []
for i in range(len(resistance_sansC)):
    puissance_moy_sansC.append(tension_sansC[i]**2/resistance_sansC[i])

ax1 = plt.subplot(111)
ticklabels = ax1.get_xticklabels()
ticklabels.extend( ax1.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(14)
plt.plot(resistance_sansC, puissance_moy_sansC, ".")
plt.xscale('log')
plt.ylabel(r'P$_\mathrm{moy}$ [W]', size=17)
plt.xlabel(r'Résistance [$\Omega$]', size=17)
plt.show()




x_sim = np.linspace(10,250)

#Model of V(R):

def tension(R,I):
    return R*I

res=  curve_fit(tension, resistance_sansC, tension_sansC)
ax1 = plt.subplot(111)
ticklabels = ax1.get_xticklabels()
ticklabels.extend( ax1.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(14)
plt.plot(resistance_sansC, tension_sansC, ".", label="data")
plt.plot(x_sim, tension(x_sim, res[0]), "--", label="fit")
#plt.xscale('log')
plt.ylabel(r'Tension [V]', size=17)
plt.xlabel(r'Résistance [$\Omega$]', size=17)
plt.show()


#Model of P(R,V):

def puissance(known_params, R_S):
    return known_params[0]*np.abs(known_params[1])**2*(1/(known_params[0]+R_S)**2)

#Fitting power:


res = curve_fit(puissance, [resistance_sansC, tension_sansC], puissance_moy_sansC)
print(res[0])

ax1 = plt.subplot(111)
ticklabels = ax1.get_xticklabels()
ticklabels.extend( ax1.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(14)
plt.plot(resistance_sansC, puissance_moy_sansC, ".", label="data")
plt.plot(x_sim, puissance(x_sim, res[0]), "--", label="fit")
plt.xscale('log')
plt.ylabel(r'P$_\mathrm{moy}$ [W]', size=17)
plt.xlabel(r'Résistance [$\Omega$]', size=17)
plt.show()




print(res)