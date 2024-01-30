import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_values_from_file(filename):
    values = pd.read_csv(filename, delimiter="\t", decimal=",", skiprows=0)
    return values

data = []
clusters_of_data = []
data_highs = []
data_lows = []
for k in range(3,7):
    file = get_values_from_file(f'PHY-2006/Bruit/labBruit_{k}.lvm')
    for n in range(6):
        print(len(data))
        for i, val in enumerate(file.iloc[:, n]):
            if np.isnan(val) or i < 2500 or i > 22500 or val > 0.76:
                pass
            else:
                data.append(val)
for i in range(len(data)):
    if i % 5000 == 0:
        clusters_of_data.append([])
    clusters_of_data[i//5000].append(data[i])

for i in clusters_of_data:
    if np.mean(i) > 0.71:
        data_highs.append(i)
    else:
        data_lows.append(i)

#for i in range(len(clusters_of_data)-1):
#    plt.plot(range(i*5000, (i+1)*5000), clusters_of_data[i])
#plt.plot(range(len(data_highs)), data_highs)
#plt.plot(range(len(data_highs), len(data_highs)+len(data_lows)), data_lows)


#print(np.mean(data_highs), np.mean(data_lows))


analyse = []

for i in range(int(len(clusters_of_data)/2)):
    analyse.extend(data_lows[i])
    analyse.extend(data_highs[i])






cycles = [[],]
last_dim = 0

data = analyse[160000:]
plt.plot(data)
plt.show()

for i in range(len(data)):
    dim = int(i // 10000)
    if dim != last_dim:
        cycles.append([])
    cycles[dim].append(data[i])
    last_dim = dim

cycles = cycles[:-1]

ax1 = plt.subplot(131)
ax2 = plt.subplot(132, sharex=ax1, sharey=ax1)
ax3 = plt.subplot(133, sharex=ax1, sharey=ax1)

ax1.plot(data[:10000])
ax1.set_title('Données brutes')
ax2.plot(np.mean(cycles, axis=0))
ax2.set_title(f'Données moyennées sur {last_dim+1} cycles')
ax3.plot(np.median(cycles, axis=0))
ax3.set_title(f'Données "médiannées" sur {last_dim+1} cycles')
plt.show()