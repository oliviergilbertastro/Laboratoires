import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_values_from_file(filename):
    values = pd.read_csv(filename, delimiter="\t", decimal=",", skiprows=21)
    return values




#freq = np.linspace(1, 15E6, 1000)

freq = np.logspace(np.log10(1), np.log10(15E6), 1000, 10)

hh = []
gain = []
ph = []

def h(f):
    w = 2*np.pi*f
    return (1E6-0.001*w**2)*(1.27E6-0.00127*w**2-270j*w)/(72900*w**2+(1.27E6-0.00127*w**2)**2)


def h(f):
    w = 2*np.pi*f
    return ((270j*w/(w**2*0.001-1E6))+1.27)/((-270*w/(w**2*0.001-1E6))**2+1.27**2)

#def h(f):
#    w = 2*np.pi*f
#    s = -1j*w
#    return 1/(1-270*(1000/s+s*1E-6+1/1000))

#def g(f):
#    w = 2*np.pi*f
#    return np.sqrt((1E6-0.001*w**2)**2*(1.27E6-0.00127*w**2)**2+72900*w**2*(1E6-0.001*w**2)**2)/(72900*w**2+(1.27E6-0.00127*w**2)**2)

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


f_c = [w_c[0]/(2*np.pi), w_c[1]/(2*np.pi)]
print('w_c', w_c)
print('f_c', f_c)
print('w_0', w_0)
print('f_0', w_0/(2*np.pi))

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
plt.savefig(r'dev2/bode_gain_1.pdf', format="pdf", bbox_inches="tight")
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
plt.savefig(r'dev2/bode_phase_1.pdf', format="pdf", bbox_inches="tight")
plt.show()