import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq



data_HeNe = pd.read_csv('PHY-2006/Michelson/Data/jnterferogrammeHeNe.txt', delimiter="\t", decimal=".", skiprows=17, encoding='latin-1', engine='python')

pos_HeNe, sig_HeNe = data_HeNe.iloc[:, 1], data_HeNe.iloc[:, 2]
marche_HeNe = np.array(pos_HeNe)*0.2



plt.plot(marche_HeNe, sig_HeNe)
plt.xlabel(r'$\delta$ [$\mathrm{\mu m}$]')
plt.ylabel(r'Tension [mV]')
plt.show()

fourier = np.fft.fft(sig_HeNe)
freq = np.fft.fftfreq(len(marche_HeNe), d=np.abs(marche_HeNe[1]-np.abs(marche_HeNe[0]))*1E-6)

indice_max_amplitude = np.argmax(np.abs(fourier.imag))

freq_res = freq[indice_max_amplitude]

print(f'Fréq résonance: {np.abs(freq_res)} [1/m]')
print(f'Longueur donde: {(1/np.abs(freq))} [m]')
plt.plot((1/freq)*1E9, np.abs(fourier))
plt.show()
