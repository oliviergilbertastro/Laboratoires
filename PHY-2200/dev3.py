import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def open_figure(fig, Z=0.001):
    file = pd.read_csv(f"PHY-2200/data/fig{fig}_z{str(Z)[-2:]}.dat", delimiter="\t", decimal=".", skiprows=3, encoding='latin-1', engine='python')
    file = np.array(file)
    wav = []
    for line in file:
        linedata = line[0].split()
        wav.append(float(linedata[0]))
    return wav

wav = open_figure(fig=7, Z=0.040)
plt.plot(wav)
plt.show()