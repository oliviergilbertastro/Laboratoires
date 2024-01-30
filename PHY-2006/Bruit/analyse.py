import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


data = np.loadtxt('PHY-2006/Bruit/signalSimu.txt')

cycles = [[],]
last_dim = 0

for i in range(len(data)):
    dim = int(i // 100)
    if dim != last_dim:
        cycles.append([])
    cycles[dim].append(data[i])
    last_dim = dim



ax1 = plt.subplot(131)
ax2 = plt.subplot(132, sharex=ax1, sharey=ax1)
ax3 = plt.subplot(133, sharex=ax1, sharey=ax1)

ax1.plot(data[:100])
ax1.set_title('Données brutes')
ax2.plot(np.mean(cycles, axis=0))
ax2.set_title(f'Données moyennées sur {last_dim+1} cycles')
ax3.plot(np.median(cycles, axis=0))
ax3.set_title(f'Données "médiannées" sur {last_dim+1} cycles')
plt.show()