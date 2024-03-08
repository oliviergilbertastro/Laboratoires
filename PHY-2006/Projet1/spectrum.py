import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def open_spectrum(filepath):
    file = pd.read_csv(filepath, delimiter="\t", decimal=",", skiprows=14, encoding='latin-1', engine='python')
    wavelengths = np.array(file.iloc[:, 0])
    counts = np.array(file.iloc[:, 1])
    density = counts/np.max(counts)
    return wavelengths, counts, density



#Start with the sun

sun_wav, sun_counts, sun_energydensity = open_spectrum('PHY-2006/Projet1/data/sun/sun_USB4F104151__0__16-39-31-696.txt')




h = 6.62607015E-34 #m^2.kg.s^-1
c = 299792458 #m/s
k_B = 1.380649E-23 #m^2.kg.s^-2.K^-1

def planckslaw(wav, temp):
    wav = wav*10**(-9)
    return 8*np.pi*h*c/wav**5*(1/(np.exp(h*c/(wav*k_B*temp))-1))/865000 #865000 est le max de la fonction pour un corps noir de T=5500K

wav_sim = np.linspace(200, 900, 1000)
sed_sim = planckslaw(wav_sim, 5500)



#Fit
from scipy.optimize import curve_fit
plt.plot(sun_wav)
plt.show()
temp_experimentale = curve_fit(planckslaw, sun_wav[1000:3000], sun_energydensity[1000:3000], p0=[5500])[0]

sed_fit = planckslaw(wav_sim, temp_experimentale)

ax1 = plt.subplot(111)
ticklabels = ax1.get_xticklabels()
ticklabels.extend( ax1.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(14)
ax1.plot(sun_wav, sun_energydensity, label='Données')
ax1.plot(wav_sim, sed_sim, label='Corps noir de $T=5500$K')
ax1.plot(wav_sim, sed_fit, label=f'Corps noir de $T={np.round(temp_experimentale)}$K')
plt.xlabel('$\lambda$ [nm]', fontsize=17)
plt.ylabel("Intensité renormalisée", fontsize=17)
plt.legend(fontsize=14)
plt.show()