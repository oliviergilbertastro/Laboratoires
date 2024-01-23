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
    return 20*np.log10(np.abs(h(f)))

def phase(f):
    return np.angle(h(f))

wlist = []
for i in freq:
    gain.append(g(i))
    ph.append(phase(i))
    hh.append(h(i))
    wlist.append(i*2*np.pi)

def index_closest_value(list, value, number=0):
    value_list = np.ones(np.array(list).shape)*value
    clist = np.sort(np.abs(np.copy(list)-value_list))
    unsorted_list = (np.abs(np.copy(list)-value_list)).tolist()
    return list.index(list[unsorted_list.index(clist[number])])


w_c = []
for i in range(2):
    w_c.append(wlist[index_closest_value(gain, np.max(gain)-3, i)])

w_0 = np.sqrt(w_c[0]*w_c[1])
print('f_0', freq[gain.index(np.max(gain))])
print('w_0', 2*np.pi*freq[gain.index(np.max(gain))])
print('w_0', wlist[gain.index(np.max(gain))])
print('w_0', np.sqrt(w_c[0]*w_c[1]))

ax1 = plt.subplot(111)
ticklabels = ax1.get_xticklabels()
ticklabels.extend( ax1.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(14)
plt.plot(wlist, gain, '-', color='purple')
plt.axvline(w_c[0], 0, 1, color='red', label=r'$\omega_c$')
plt.axvline(w_c[1], 0, 1, color='red')
plt.axvline(w_0, 0, 1, color='blue', label=r'$\omega_0$')
plt.legend()
plt.xscale('log')
plt.ylabel(r'Gain [dB]', size=17)
plt.xlabel(r'Fréquence angulaire $\omega$ [rad/s]', size=17)
plt.savefig(r'dev2/bode_gain_2.pdf', format="pdf", bbox_inches="tight")
plt.show()

ax1 = plt.subplot(111)
ticklabels = ax1.get_xticklabels()
ticklabels.extend( ax1.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(14)
plt.plot(wlist, ph, '-', color='purple')
plt.xscale('log')
plt.ylabel(r'Phase $\varphi$ [rad]', size=17)
plt.xlabel(r'Fréquence angulaire $\omega$ [rad/s]', size=17)
plt.savefig(r'dev2/bode_phase_2.pdf', format="pdf", bbox_inches="tight")
plt.show()