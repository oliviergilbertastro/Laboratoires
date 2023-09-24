import numpy as np
import matplotlib.pyplot as plt
from pylab import genfromtxt
import pandas as pd

def conv(x):
    return x.replace(',', '.').encode()

def get_values_from_file(filename):
    values = pd.read_csv(filename, delimiter=";", decimal=",", skiprows=22)
    return values

def create_hist(values, combinaison="Inox-Alu", bins=200):
    space=0
    counts, bins = np.histogram(values, bins=bins)
    for i, ct in enumerate(counts):
        if ct>0:
            print(bins[i]-space)
            space = bins[i]
    ax1 = plt.subplot(111)
    ticklabels = ax1.get_xticklabels()
    ticklabels.extend( ax1.get_yticklabels() )
    for label in ticklabels:
        label.set_fontsize(10)
    #print(counts)
    #print(bins)
    plt.stairs(counts, bins)
    plt.suptitle(f'Histogramme pour la combinaison {combinaison}', size=17)
    plt.xlabel(r'Amplitude [V]', size=17)
    plt.ylabel(r'Nombre de mesures', size=17)
    plt.savefig(r'C:\Users\olivi\Desktop\Devoirs\PhysElectronique\figures\lab1'+f"\histogram_{combinaison}.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def create_scatter(values, combinaison="Inox-Alu", bins=200):
    ax1 = plt.subplot(111)
    ticklabels = ax1.get_xticklabels()
    ticklabels.extend( ax1.get_yticklabels() )
    for label in ticklabels:
        label.set_fontsize(10)
    plt.plot(values, "o", color='blue')
    plt.suptitle(f'Nuage de points pour la combinaison {combinaison}', size=17)
    plt.ylabel(r'Amplitude [V]', size=17)
    plt.xlabel(r'n$^\mathrm{ième}$ mesure', size=17)
    plt.savefig(r'C:\Users\olivi\Desktop\Devoirs\PhysElectronique\figures\lab1'+f"\scatter_{combinaison}.pdf", format="pdf", bbox_inches="tight")
    plt.show()

#Choices:
#NOTHING:           0
#HISTOGRAMS ONLY:   1
#SCATTER PLOTS:     2
#BOTH PLOTS:        3
choice = int(input("Choice? [0,1,2,3]"))

#Start data analysis
val = get_values_from_file("lab1/lab1_InoxAlu_1010_09_19_2023.lvm")
if choice in (1,3):
    create_hist(val, "Inox-Alu -10V à 10V")
if choice in (2,3):
    create_scatter(val, "Inox-Alu -10V à 10V")

val = get_values_from_file("lab1/lab1_InoxAlu_19_09_2023.lvm")
if choice in (1,3):
    create_hist(val, "Inox-Alu -1V à 1V", bins=1000)
if choice in (2,3):
    create_scatter(val, "Inox-Alu -1V à 1V", bins=1000)

val = get_values_from_file("lab1/lab1_InoxZinc_19_09_2023.lvm")
if choice in (1,3):
    create_hist(val, "Inox-Zinc -1V à 1V", bins=1000)
if choice in (2,3):
    create_scatter(val, "Inox-Zinc -1V à 1V", bins=1000)

val = get_values_from_file("lab1/lab1_AcierAlu_19_09_2023.lvm")
if choice in (1,3):
    create_hist(val, "Acier-Alu -1V à 1V")
if choice in (2,3):
    create_scatter(val, "Acier-Alu -1V à 1V", bins=1000)

val = get_values_from_file("lab1/lab1_ZincAlu_19_09_2023.lvm")
if choice in (1,3):
    create_hist(val, "Zinc-Alu -1V à 1V", bins=1000)
if choice in (2,3):
    create_scatter(val, "Zinc-Alu -1V à 1V", bins=1000)
