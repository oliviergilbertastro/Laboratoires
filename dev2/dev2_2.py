import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_values_from_file(filename):
    values = pd.read_csv(filename, delimiter="\t", decimal=",", skiprows=21)
    return values


print(np.angle(1j))

#freq = np.linspace(1, 15E6, 1000)

freq = np.logspace(np.log10(1), np.log10(15E6), 1000, 10)

gain = []
ph = []
hh = []

def h(f):
    w = 2*np.pi*f
    return 12*((1-12*((((1/(-2j/(w*1E-6)+12))+1/270+1j*w*1E-6))/(1+12*((1/(-2j/(w*1E-6)+12))+1/270+1j*w*1E-6))))/(-2j/(w*1E-6)+12))

def g(f):
    return np.abs(h(f))

def phase(f):
    return np.angle(h(f))

for i in freq:
    gain.append(g(i))
    ph.append(phase(i))
    hh.append(h(i))

ax1 = plt.subplot(111)
ticklabels = ax1.get_xticklabels()
ticklabels.extend( ax1.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(14)
plt.plot(freq, gain, '.', color='purple')
plt.xscale('log')
plt.ylabel(r'Gain [dB]', size=17)
plt.xlabel(r'Fréquence [Hz]', size=17)
plt.savefig(r'dev2/bode_gain_2.pdf', format="pdf", bbox_inches="tight")
plt.show()

ax1 = plt.subplot(111)
ticklabels = ax1.get_xticklabels()
ticklabels.extend( ax1.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(14)
plt.plot(freq, ph, '.', color='purple')
plt.xscale('log')
plt.ylabel(r'Phase $\varphi$ [rad]', size=17)
plt.xlabel(r'Fréquence [Hz]', size=17)
plt.savefig(r'dev2/bode_phase_2.pdf', format="pdf", bbox_inches="tight")
plt.show()