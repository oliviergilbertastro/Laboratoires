import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

wav, spectra_cube = open_spectra(fig=8, Z=0.001)
for i in range(30):
    plt.plot(wav, spectra_cube[:, i])
plt.xlim((-1000, 12000))
plt.show()