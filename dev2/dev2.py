import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_values_from_file(filename):
    values = pd.read_csv(filename, delimiter="\t", decimal=",", skiprows=21)
    return values




#freq = np.linspace(1, 15E6, 1000)

freq = np.logspace(np.log10(1), np.log10(15E6), 1000, 10)

gain = []

def g(f):
    w = 2*np.pi*f
    return np.sqrt((1E6-0.001*w**2)**2*(1.27E6-0.00127*w**2)**2+72900*w**2*(1E6-0.001*w**2)**2)/(72900*w**2+(1.27E6-0.00127*w**2)**2)

for i in freq:
    gain.append(g(i))


ax1 = plt.subplot(111)
ticklabels = ax1.get_xticklabels()
ticklabels.extend( ax1.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(14)
plt.plot(freq, gain, '.', color='purple')
plt.xscale('log')
plt.ylabel(r'Gain [dB]', size=17)
plt.xlabel(r'Fréquence [Hz]', size=17)
plt.savefig(r'dev2/bode.pdf', format="pdf", bbox_inches="tight")
plt.show()