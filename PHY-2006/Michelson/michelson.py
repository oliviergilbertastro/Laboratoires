import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def get_values_from_file(filename):
    values = pd.read_csv(filename, delimiter="\t", lineterminator='\n', decimal=",", skiprows=0, encoding='latin-1')
    return values


data_HeNe = pd.read_csv('PHY-2006/Michelson/Data/jnterferogrammeHeNe.txt', delimiter="\t", decimal=".", skiprows=17, encoding='latin-1', engine='python')

pos_HeNe, sig_HeNe = data_HeNe.iloc[:, 1], data_HeNe.iloc[:, 2]
marche_HeNe = np.array(pos_HeNe)*2*10**(-6)

# Number of sample points
N = len(sig_HeNe)
# sample spacing
T = 0.0001

yf = np.fft.rfft(np.array(sig_HeNe))
xf = np.fft.rfftfreq(len(marche_HeNe), 0.1*10**(-6) )
yf = fft(np.array(sig_HeNe))
xf = fftfreq(N, T)[:N//2]

#Si xf est k, alors lambda = 2pi/k donc lambda = 2pi/xf
la_HeNe = 2*np.pi/np.array(xf)

plt.plot(marche_HeNe, sig_HeNe)
plt.xlabel(r'$\delta$ [$\mathrm{\mu m}$]')
plt.ylabel(r'Intensité [V]')
plt.show()

print(len(sig_HeNe))
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.show()