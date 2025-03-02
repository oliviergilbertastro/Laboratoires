import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = ["blue", "red", "green", "orange"]

def loadFile(source="Am", detector="XR100T_CdTe", iteration=0, res=256):
    return np.loadtxt(f"PHY-3004/SPEC_X/{source}_{res}_{detector}_{iteration}.mca", skiprows=12, max_rows=res, dtype=float)

data = []
resolutions = [1024]
index_max_res = resolutions.index(np.max(resolutions))
index_min_res = resolutions.index(np.min(resolutions))

for res in resolutions:
    data.append(loadFile(res=res))

for n in range(len(resolutions)):
    new_data = []
    for i in range(len(data[n])):
        for k in range(int(np.max(resolutions)/resolutions[n])):
            new_data.append(data[n][i])
    data[n] = np.array(new_data)


for i in range(len(data)):
    data[i] = np.abs(data[i])

for n in range(len(resolutions)):
    plt.plot(data[n]*(np.max(data[index_min_res])/np.max(data[n])), label=resolutions[n])
plt.legend()
plt.xlabel("Channel")
plt.ylabel("Count")
plt.show()





def gaussian(x, sig, b, mu, c):
    return (
        b / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x-mu) / sig, 2.0) / 2) + c
    )

from scipy.optimize import curve_fit

FWHMs = []
for m, pics in enumerate([(155,170),(202,215),(628,707)]):

    gaussiennes = []
    for n in range(len(resolutions)):
        gaussiennes.append(data[n][pics[0]:pics[1]]*(np.max(data[index_min_res])/np.max(data[n])))
        #plt.plot(range(len(gaussiennes[n])), gaussiennes[n], label=resolutions[n])
    #plt.legend()
    #plt.xlabel("Channel")
    #plt.ylabel("Count")
    #plt.show()

    res = []
    FWHMs.append([])
    for n in range(len(resolutions)):
        res.append(curve_fit(gaussian, range(len(gaussiennes[n])), gaussiennes[n], p0=[len(gaussiennes[n]), np.max(gaussiennes[n])*(np.sqrt(2.0 * np.pi) * 1000), len(gaussiennes[n])/2, np.min(gaussiennes[n])], bounds=[(0,0,0,0),(np.inf,np.inf,np.inf,np.inf)])[0])
        FWHMs[m].append(2*np.sqrt(2*np.log(2))*res[n][0])
        print(f"FWHM = {FWHMs[m][n]}")

    x_sim = np.linspace(0, len(gaussiennes[0]), 1000)
    for n in range(len(resolutions)):
        plt.bar(range(len(gaussiennes[n])), gaussiennes[n], width=1, color=COLORS[n], alpha=0.4, label=resolutions[n])
        plt.plot(x_sim, gaussian(x_sim, res[n][0], res[n][1], res[n][2], res[n][3]), color=COLORS[n])
    plt.legend()
    plt.xlabel("Channel")
    plt.ylabel("Count")
    plt.show()