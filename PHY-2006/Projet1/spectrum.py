import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def open_spectrum(filepath):
    file = pd.read_csv(filepath, delimiter="\t", decimal=",", skiprows=14, encoding='latin-1', engine='python')
    wavelengths = np.array(file.iloc[:, 0])
    counts = np.array(file.iloc[:, 1])
    density = counts/np.max(counts)
    return wavelengths, counts, density



#Start with the atmosphere
atm_wav_list, atm_counts_list, atm_energydensity_list = [], [], []
for i in range(1, 5):
    w, c, d = open_spectrum(f'PHY-2006/Projet1/data/atmosphere/atmo{i}.txt')
    atm_wav_list.append(w)
    atm_counts_list.append(c)
    atm_energydensity_list.append(d)

atm_wav = np.mean(atm_wav_list, axis=0)
atm_counts = np.mean(atm_counts_list, axis=0)/100
atm_energydensity = np.mean(atm_energydensity_list, axis=0)

#Start with the sun
sun_wav_list, sun_counts_list, sun_energydensity_list = [], [], []

for i in range(1, 11):
    w, c, d = open_spectrum(f'PHY-2006/Projet1/data/sun/sun{i}.txt')
    sun_wav_list.append(w)
    sun_counts_list.append(c)
    sun_energydensity_list.append(d)


sun_wav = np.mean(sun_wav_list, axis=0)
sun_counts = np.mean(sun_counts_list, axis=0)
sun_energydensity = np.mean(sun_energydensity_list, axis=0)

ax1 = plt.subplot(111)
ticklabels = ax1.get_xticklabels()
ticklabels.extend( ax1.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(14)
ax1.plot(sun_wav, sun_counts, label='Données')
ax1.plot(atm_wav, atm_counts, label=f'Atmosphère')
plt.xlabel('$\lambda$ [nm]', fontsize=17)
plt.ylabel("Counts", fontsize=17)
plt.legend(fontsize=14)
plt.show()










h = 6.62607015E-34 #m^2.kg.s^-1
c = 299792458 #m/s
k_B = 1.380649E-23 #m^2.kg.s^-2.K^-1
b = 2.897771955E-3 #m.K

def planckslaw(wav, temp):
    wav = wav*10**(-9)
    return 8*np.pi*h*c/wav**5*(1/(np.exp(h*c/(wav*k_B*temp))-1))/865000 #865000 est le max de la fonction pour un corps noir de T=5500K



#Fit Wien's law
wav_peak = sun_wav[np.where(sun_counts == np.max(sun_counts))]
temp_wien = b/(wav_peak*10**(-9))
print('Wavelenght peak [nm]:', wav_peak)
print('Temperature:', temp_wien)
print(b/5778)

wav_sim = np.linspace(200, 900, 1000)
sed_sim = planckslaw(wav_sim, 5772)

#Fit planck's law
from scipy.optimize import curve_fit
if False:
    plt.plot(sun_wav)
    plt.xlabel('Index')
    plt.ylabel('$\lambda$ [nm]')
    plt.show()
temp_experimentale = curve_fit(planckslaw, sun_wav[1600:2500], sun_energydensity[1600:2500], p0=[6500])[0]

sed_fit = planckslaw(wav_sim, temp_experimentale)

ax1 = plt.subplot(111)
ticklabels = ax1.get_xticklabels()
ticklabels.extend( ax1.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(14)
ax1.plot(sun_wav, sun_energydensity, label='Données')
ax1.plot(wav_sim, sed_sim, label=f'Corps noir de $T={np.round(temp_wien)}$K')
ax1.plot(wav_sim, sed_fit, label=f'Corps noir de $T={np.round(temp_experimentale)}$K')
plt.xlabel('$\lambda$ [nm]', fontsize=17)
plt.ylabel("$I/I_\mathrm{max}$", fontsize=17)
plt.legend(fontsize=14)
plt.show()