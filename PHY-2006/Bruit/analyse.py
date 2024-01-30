import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


data = np.loadtxt('PHY-2006/Bruit/signalSimu.txt')

plt.plot(data)
plt.show()

#On se créé un array de chaque point:
modif = []
for i in range(len(data)):
    if i > 100:
        modif[i % 100].append(data[i])
    else:
        modif.append([])
        modif[i % 100].append(data[i])

cycles = [[],]
last_dim = 0

for i in range(len(data)):
    dim = int(i // 100)
    if dim != last_dim:
        cycles.append([])
    cycles[dim].append(data[i])
    last_dim = dim

plt.plot(np.mean(cycles, axis=0))
plt.show()
print(cycles)