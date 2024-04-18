import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



#PHOTON COUNTS: FIG 77-78
#H ALPHA E-WIDTH: FIG 83-84
#COLORS (B-V)+(V-R): FIG 57-60

def open_spectra(fig, Z=0.001):
    file = pd.read_csv(f"PHY-2200/data/fig{fig}_z{str(Z)[-2:]}.dat", delimiter="\t", decimal=".", skiprows=3, encoding='latin-1', engine='python')
    file = np.array(file)
    wav = []
    spectra_cube = [] #Chaque colonne est un moment, chaque ligne est une longueur d'onde
    for line in file:
        linedata = line[0].split()
        wav.append(float(linedata[0]))
        spectra_cube.append(np.array(linedata[1:], float))
    return wav, np.array(spectra_cube)

def open_photon(fig, Z=0.001):
    """
    Returns tuple of (time, #photons) and #photons is itself a 2D array in which each column is for a different IMF (2.35, 3.30, useless)
    """
    file = pd.read_csv(f"PHY-2200/data/fig{fig}_z{str(Z)[-2:]}.dat", delimiter="\t", decimal=".", skiprows=3, encoding='latin-1', engine='python')
    file = np.array(file)
    time = []
    nb_photons = []
    for line in file:
        linedata = line[0].split()
        time.append(float(linedata[0]))
        nb_photons.append(np.array(linedata[1:], float))
    return time, 10**np.array(nb_photons)

time, photons = open_photon(fig=77, Z=0.040)
for i in range(2):
    plt.plot(time, photons[:, i])
plt.xlabel("Temps [années]")
plt.ylabel("Flux de photons ionisants [$\mathrm{s}^{-1}$]")
plt.yscale('log')
plt.xscale('log')
plt.show()