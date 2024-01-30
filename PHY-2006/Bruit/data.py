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
            if np.isnan(val) or i < 2500 or i > 22500:
                pass
            else:
                data.append(val)
for i in range(len(data)):
    if i % 5000 == 0:
        clusters_of_data.append([])
    clusters_of_data[i//5000].append(data[i])

for i in clusters_of_data:
    if np.mean(i) > 0.71:
        data_highs.extend(i)
    else:
        data_lows.extend(i)

#for i in range(len(clusters_of_data)-1):
#    plt.plot(range(i*5000, (i+1)*5000), clusters_of_data[i])
plt.plot(range(len(data_highs)), data_highs)
plt.plot(range(len(data_highs), len(data_highs)+len(data_lows)), data_lows)
plt.show()