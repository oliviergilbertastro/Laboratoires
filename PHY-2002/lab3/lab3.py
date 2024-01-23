import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_values_from_file(filename):
    values = pd.read_csv(filename, delimiter="\t", decimal=",", skiprows=21)
    return values



def create_scatter(x_values, y_values, title="Title"):
    ax1 = plt.subplot(111)
    ticklabels = ax1.get_xticklabels()
    ticklabels.extend( ax1.get_yticklabels() )
    for label in ticklabels:
        label.set_fontsize(10)
    plt.plot(x_values, y_values, "o", color='blue')
    plt.suptitle(f'{title}', size=17)
    plt.ylabel(r'Intensité du courant [A]', size=17)
    plt.xlabel(r'Différence de potentiel [V]', size=17)
    plt.savefig(r'C:\Users\olivi\Desktop\Devoirs\PhysElectronique\figures\lab3'+f"\{title}.pdf", format="pdf", bbox_inches="tight")
    plt.show()

def create_multiple(x_values, y_values, title="Title"):
    ax1 = plt.subplot(111)
    ticklabels = ax1.get_xticklabels()
    ticklabels.extend( ax1.get_yticklabels() )
    for label in ticklabels:
        label.set_fontsize(10)
    for i in range(len(x_values)):
        plt.plot(x_values[i], y_values[i], "-", label=f"Tension base: {str(i*0.2)[:3]} V")
        print(f"Tension base: {str(i*0.2)[:3]} V")
        print(np.median(y_values[i][11:]))
        sigma_lo = np.median(y_values[i][11:]) - np.quantile(y_values[i][11:], 0.16)
        sigma_hi = np.quantile(y_values[i][11:], 0.84) - np.median(y_values[i][11:])
        uncertainty = sigma_hi if sigma_hi>sigma_lo else sigma_lo
        print("uncertainty:", uncertainty)
    #plt.axvline(1)
    #plt.text(2, 0.45, r"$i_\mathrm{sat}=0.5$A", fontsize=16)
    plt.legend()
    plt.suptitle(f'{title}', size=17)
    plt.ylabel(r'Intensité du courant [A]', size=17)
    plt.xlabel(r'Différence de potentiel [V]', size=17)
    plt.savefig(r'C:\Users\olivi\Desktop\Devoirs\PhysElectronique\figures\lab3'+f"\{title}.pdf", format="pdf", bbox_inches="tight")
    plt.show()




val = get_values_from_file("lab3/lab3_3epartie_condensateur.lvm")
tension = []
courant = []
for i in range(np.array(val).shape[0]):
    tension.append(np.array(val)[:, 1:2][i][0])
    courant.append(np.array(val)[:, 2:3][i][0])
#create_scatter(tension, courant, "Courbe $i-v$ du condensateur")

val = get_values_from_file("lab3/lab3_3epartie_bobine.lvm")
tension = []
courant = []
for i in range(np.array(val).shape[0]):
    tension.append(np.array(val)[:, 1:2][i][0])
    courant.append(np.array(val)[:, 2:3][i][0])
#create_scatter(tension, courant, "Courbe $i-v$ de la bobine d'inductance")

val = get_values_from_file("lab3/transitor_pas_brule.lvm")
tensions = []
courants = []
for i in range(np.array(val).shape[0]):
    tensions.append(np.array(val)[:, 1:2][i][0])
    courants.append(np.array(val)[:, 2:3][i][0])

tension = []
courant = []

for i in range(6):
    tension.append(tensions[i*20:i*20+20])
    courant.append(courants[i*20:i*20+20])
create_multiple(tension, courant, "Courbes $i-v$ du transistor")









#diodes:
val = get_values_from_file("lab3/diode_polarise_directe.lvm")
tension1 = []
courant1 = []
for i in range(np.array(val).shape[0]):
    tension1.append(np.array(val)[:, 1:2][i][0])
    courant1.append(np.array(val)[:, 2:3][i][0])
#create_scatter(tension1, courant1, "Courbe $i-v$ de la diode en polarisation directe")

val = get_values_from_file("lab3/diode_polarite_inverse.lvm")
tension2 = []
courant2 = []
for i in range(np.array(val).shape[0]):
    tension2.append(np.array(val)[:, 1:2][i][0])
    courant2.append(np.array(val)[:, 2:3][i][0])
create_scatter(tension2, courant2, "Courbe $i-v$ de la diode en polarisation inverse")

val = get_values_from_file("lab3/diode_zener.lvm")
tension = []
courant = []
for i in range(np.array(val).shape[0]):
    tension.append(np.array(val)[:, 1:2][i][0])
    courant.append(np.array(val)[:, 2:3][i][0])
#create_scatter(tension, courant, "Courbe $i-v$ de la diode Zener")


ax1 = plt.subplot(111)
ticklabels = ax1.get_xticklabels()
ticklabels.extend( ax1.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(10)
plt.plot(tension1, courant1, ".", label=f"Standard polarisation directe")
plt.plot(tension, courant, ".", label=f"Zener polarisation inverse")
plt.plot(tension2, courant2, ".", label=f"Standard polarisation inverse")
plt.legend()
plt.suptitle(f'Courbe $i-v$ de la diode Zener', size=17)
plt.ylabel(r'Intensité du courant [A]', size=17)
plt.xlabel(r'Différence de potentiel [V]', size=17)
plt.savefig(r'C:\Users\olivi\Desktop\Devoirs\PhysElectronique\figures\lab3'+f"\Courbe $i-v$ de la diode Zener.pdf", format="pdf", bbox_inches="tight")
plt.show()




