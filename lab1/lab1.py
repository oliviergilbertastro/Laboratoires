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
    #print(counts)
    #print(bins)
    plt.stairs(counts, bins)
    plt.suptitle(f'Histogramme pour la combinaison {combinaison}', size=17)
    plt.xlabel(r'Amplitude [V]', size=17)
    plt.show()


def create_scatter(values, combinaison="Inox-Alu", bins=200):
    plt.plot(values, "o", color='blue')
    plt.suptitle(f'Histogramme pour la combinaison {combinaison}', size=17)
    plt.xlabel(r'Amplitude [V]', size=17)
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
    create_hist(val, "Inox-Alu -10 10")
if choice in (2,3):
    create_scatter(val, "Inox-Alu -10 10")

val = get_values_from_file("lab1/lab1_InoxAlu_19_09_2023.lvm")
if choice in (1,3):
    create_hist(val, "Inox-Alu", bins=1000)
if choice in (2,3):
    create_scatter(val, "Inox-Alu", bins=1000)

val = get_values_from_file("lab1/lab1_InoxZinc_19_09_2023.lvm")
if choice in (1,3):
    create_hist(val, "Inox-Zinc", bins=1000)
if choice in (2,3):
    create_scatter(val, "Inox-Zinc", bins=1000)

val = get_values_from_file("lab1/lab1_AcierAlu_19_09_2023.lvm")
if choice in (1,3):
    create_hist(val, "Acier-Alu")
if choice in (2,3):
    create_scatter(val, "Acier-Alu", bins=1000)

val = get_values_from_file("lab1/lab1_ZincAlu_19_09_2023.lvm")
if choice in (1,3):
    create_hist(val, "Zinc-Alu", bins=1000)
if choice in (2,3):
    create_scatter(val, "Zinc-Alu", bins=1000)
