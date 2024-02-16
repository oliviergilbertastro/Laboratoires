import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq



data_HeNe = pd.read_csv('PHY-2006/Michelson/Data/jnterferogrammeHeNe.txt', delimiter="\t", decimal=".", skiprows=17, encoding='latin-1', engine='python')

pos_HeNe, sig_HeNe = data_HeNe.iloc[:, 1], data_HeNe.iloc[:, 2]
marche_HeNe = np.array(pos_HeNe)*2



plt.plot(marche_HeNe, sig_HeNe)
plt.xlabel(r'$\delta$ [$\mathrm{\mu m}$]')
plt.ylabel(r'Tension [mV]')
plt.show()

fourier = np.fft.fft(sig_HeNe)
freq = np.fft.fftfreq(len(marche_HeNe), d=np.abs(marche_HeNe[1]-np.abs(marche_HeNe[0]))*1E-6)

indice_max_amplitude = np.argmax(np.abs(fourier.imag))

freq_res = freq[indice_max_amplitude]


print(indice_max_amplitude)
print(f'Fréq résonance: {np.abs(freq_res)} [1/m]')
print(f'Longueur donde: {(1/np.abs(freq[indice_max_amplitude]))} [m]')
plt.plot((1/freq)*1E9, np.abs(fourier))
plt.xlabel(r'$\lambda$ [nm]')
plt.show()


#On fit un sinus:
from scipy.optimize import curve_fit

def sinus(x, freq, phi, c):
    return c*np.sin(2*np.pi*freq*(x)-phi)-645.2327517

res = curve_fit(sinus, marche_HeNe, sig_HeNe)[0]
print(res)
print('Longueur onde:', 299792458/np.abs(res[0]))

x_sim = np.linspace(59, 70, 1000)
y_sim = sinus(x_sim, res[0], res[1], res[2])

plt.plot(marche_HeNe, sig_HeNe, label='data')
plt.plot(x_sim, y_sim, label='fit')
plt.xlabel(r'$\delta$ [$\mathrm{\mu m}$]')
plt.ylabel(r'Tension [mV]')
plt.legend()
plt.show()