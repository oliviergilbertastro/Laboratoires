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




val = get_values_from_file("lab3/lab3_3epartie_condensateur.lvm")
tension = []
courant = []
for i in range(np.array(val).shape[0]):
    tension.append(np.array(val)[:, 1:2][i][0])
    courant.append(np.array(val)[:, 2:3][i][0])
create_scatter(tension, courant, "Courbe $i-v$ du condensateur")

val = get_values_from_file("lab3/lab3_3epartie_bobine.lvm")
tension = []
courant = []
for i in range(np.array(val).shape[0]):
    tension.append(np.array(val)[:, 1:2][i][0])
    courant.append(np.array(val)[:, 2:3][i][0])
create_scatter(tension, courant, "Courbe $i-v$ de la bobine d'inductance")