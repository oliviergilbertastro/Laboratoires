import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

h = 6.62607015E-34 #m^2.kg.s^-1
c = 299792458 #m/s
k_B = 1.380649E-23 #m^2.kg.s^-2.K^-1
b = 2.897771955E-3 #m.K


def open_spectrum(filepath):
    file = pd.read_csv(filepath, delimiter="\t", decimal=",", skiprows=14, encoding='latin-1', engine='python')
    wavelengths = np.array(file.iloc[:, 0])
    counts = np.array(file.iloc[:, 1])
    density = counts/np.max(counts)
    return wavelengths, counts, density

def sensitivity(wav):
    '''Returns the photons/count for a wavelength'''
    if False:
        return 4.37173854E-6*wav**3-5.23293797E-3*wav**2+1.66567753*wav-18.7919391
    return 86

#Start with the atmosphere
atm_wav_list, atm_counts_list, atm_energydensity_list = [], [], []
for i in range(1, 5):
    w, cts, d = open_spectrum(f'PHY-2006/Projet1/data/atmosphere/atmo{i}.txt')
    atm_wav_list.append(w)
    atm_counts_list.append(cts)
    atm_energydensity_list.append(d)

atm_wav = np.mean(atm_wav_list, axis=0)
atm_counts = np.mean(atm_counts_list, axis=0)/100
atm_energydensity = np.mean(atm_energydensity_list, axis=0)

#Start with the sun
sun_wav_list, sun_counts_list, sun_energydensity_list = [], [], []

for i in range(11, 16):
    w, cts, d = open_spectrum(f'PHY-2006/Projet1/data/sun/sun{i}.txt')
    sun_wav_list.append(w)
    sun_counts_list.append(cts)
    sun_energydensity_list.append(d)


sun_wav = np.mean(sun_wav_list, axis=0)
sun_counts = np.mean(sun_counts_list, axis=0)
sun_radiance = []
sun_radiance_uncorr = []
for i in range(len(sun_wav)):
    #Integration time of 2ms, so we calculate the radiance in W/m^2/nm
    photon_count = sun_counts[i]*sensitivity(sun_wav[i])
    photon_energy = h*c/(sun_wav[i]*10**(-9))
    exp_time = 0.0005
    sun_radiance.append((photon_count*photon_energy/exp_time/(np.pi*(20E-6)**2))/0.56)
    sun_radiance_uncorr.append(photon_count*photon_energy/exp_time/(np.pi*(20E-6)**2))
sun_energydensity = np.mean(sun_energydensity_list, axis=0)




ax1 = plt.subplot(122)
ax2 = plt.subplot(121, sharex=ax1)
ticklabels = ax1.get_xticklabels()
ticklabels.extend( ax1.get_yticklabels() )
ticklabels.extend( ax2.get_xticklabels() )
ticklabels.extend( ax2.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(14)
ax1.plot(sun_wav, sun_radiance_uncorr, label='Données')
#ax1.plot(atm_wav, atm_counts, label=f'Atmosphère')
ax1.set_xlabel('$\lambda$ [nm]', fontsize=17)
ax1.set_ylabel("Radiance [W/m$^2$/nm]", fontsize=17)
ax1.legend(fontsize=14)
ax2.plot(sun_wav, sun_counts, label='Données')
#ax1.plot(atm_wav, atm_counts, label=f'Atmosphère')
ax2.set_xlabel('$\lambda$ [nm]', fontsize=17)
ax2.set_ylabel("Counts", fontsize=17)
ax2.legend(fontsize=14)
plt.show()


plt.plot(sun_wav, sun_radiance_uncorr/sun_counts)
plt.xlabel('$\lambda$ [nm]', fontsize=17)
plt.ylabel("Radiance/Counts", fontsize=17)
plt.show()





distance_to_sun = 1.4852E11 #m
solid_angle = (np.pi*(20E-6)**2)/(4*np.pi*distance_to_sun**2)

def planckslaw_radiance(wav, temp, scale):
    wav = wav*10**(-9)
    return scale*2*h*c**2/wav**5*(1/(np.exp(h*c/(wav*k_B*temp))-1))

#def planckslaw_radiance(wav, temp, scale):
#    wav = wav*10**(-9)
#    return scale*8*np.pi*h*c/wav**5*(1/(np.exp(h*c/(wav*k_B*temp))-1))

def scale_factor(y, scale):
    return y*scale

from scipy.optimize import curve_fit

#Fit Wien's law
wav_peak = sun_wav[np.where(sun_radiance == np.max(sun_radiance))]
temp_wien = b/(wav_peak*10**(-9))
print('Wavelenght peak [nm]:', wav_peak)
print('Temperature:', temp_wien)
print('Theoretical wavelenght peak:', round(b/5778*10**9), 'nm')

sed_sim = planckslaw_radiance(sun_wav, temp_wien, 1)

sed_sim_scale = curve_fit(scale_factor, sed_sim[1000:1500], sun_radiance[1000:1500])[0]
print('Sed_sim_scale', sed_sim_scale)
wav_sim = np.linspace(200, 1000, 1000)
sed_sim = planckslaw_radiance(wav_sim, temp_wien, sed_sim_scale)

#Fit planck's law

if True:
    plt.plot(sun_wav)
    plt.xlabel('Index')
    plt.ylabel('$\lambda$ [nm]')
    plt.show()
#temp_experimentale, scale = curve_fit(planckslaw_radiance, sun_wav[1600:2500], sun_radiance[1600:2500], p0=[5500, 1E-13])[0]
#temp_experimentale, scale = curve_fit(planckslaw_radiance, sun_wav, sun_radiance, p0=[5500, 1E-13])[0]
temp_experimentale, scale = curve_fit(planckslaw_radiance, sun_wav[1300:2000], sun_radiance[1300:2000], p0=[5500, 1E-13])[0]
sed_fit = planckslaw_radiance(wav_sim, temp_experimentale, scale)
print('*************FIT***************')
print('T', temp_experimentale)
print('Scale', scale)

ax1 = plt.subplot(111)
ticklabels = ax1.get_xticklabels()
ticklabels.extend( ax1.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(14)
#sed_sim = sed_sim/np.max(sed_sim)
#sed_fit = sed_fit/np.max(sed_fit)
#sun_radiance = sun_radiance/np.max(sun_radiance)
#for i in range(len(sun_wav_list)):
#    ax1.plot(sun_wav_list[i], sun_energydensity_list[i], color='purple')
ax1.plot(sun_wav, sun_radiance, label='Données corrigées')
ax1.plot(sun_wav, sun_radiance_uncorr, label='Données')

ax1.plot(wav_sim, sed_sim, label=f'Corps noir de $T={round(temp_wien[0])}$K')
ax1.plot(wav_sim, sed_fit, label=f'Corps noir de $T={round(temp_experimentale)}$K')
plt.xlabel('$\lambda$ [nm]', fontsize=17)
plt.ylabel("Radiance [W/m$^2$/nm]", fontsize=17)
plt.legend(fontsize=14)
plt.show()