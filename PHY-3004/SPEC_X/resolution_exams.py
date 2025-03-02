import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = ["blue", "red", "green", "orange"]

def loadFile(source="Am", detector="XR100T_CdTe", iteration=0, res=256):
    return np.loadtxt(f"PHY-3004/SPEC_X/{source}_{res}_{detector}_{iteration}.mca", skiprows=12, max_rows=res, dtype=float)

data = []
detectors = ["XR100T_CdTe", "XR100CR_Si"]

for detector in detectors:
    data.append(list(loadFile(res=1024, detector=detector)))
    for i in range(1,5):
        data[-1]+=(loadFile(res=1024, detector=detector, iteration=i))
for n in range(len(detectors)):
    plt.plot(data[n], label=detectors[n])
plt.legend()
plt.xlabel("Channel")
plt.ylabel("Count")
plt.show()

# Étalonnage de professionnel:
from scipy.optimize import curve_fit
x_data = []
etalonnages = []
for i in range(len(detectors)):
    x_data.append(np.array(range(1024)))
    peaks_indices = [[163,208,692],[184,234,785]][i]
    etalonnages.append(curve_fit(lambda x,a,b: x*a+b, peaks_indices, [13.95,17.74,59.54])[0])
    x_data[i] = x_data[i]*etalonnages[i][0]+etalonnages[i][1]

for n in range(len(detectors)):
    plt.plot(x_data[n], data[n], label=detectors[n])
plt.legend()
plt.xlabel("keV")
plt.ylabel("Count")
plt.show()

def gaussian(x, sig, b, mu, c):
    return (
        b / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x-mu) / sig, 2.0) / 2) + c
    )


FWHMs = []
for m, pics in enumerate([(155,170),(202,215),(628,707)]):

    gaussiennes = []
    for n in range(len(detectors)):
        new_pics = (int(pics[0]*etalonnages[0][0]/etalonnages[n][0]), int(pics[1]*etalonnages[0][0]/etalonnages[n][0]))
        gaussiennes.append(data[n][new_pics[0]:new_pics[1]])


    res = []
    FWHMs.append([])
    for n in range(len(detectors)):
        res.append(curve_fit(gaussian, range(len(gaussiennes[n])), gaussiennes[n], p0=[len(gaussiennes[n]), np.max(gaussiennes[n])*(np.sqrt(2.0 * np.pi) * 10), len(gaussiennes[n])/2, np.min(gaussiennes[n])], bounds=[(0,0,0,0),(np.inf,np.inf,np.inf,np.inf)])[0])
        #print(res)
        FWHMs[m].append(2*np.sqrt(2*np.log(2))*res[n][0])
        print(f"FWHM = {FWHMs[m][n]}")

    x_sim = np.linspace(0, len(gaussiennes[0]), 1000)
    for n in range(len(detectors)):
        plt.bar(range(len(gaussiennes[n])), gaussiennes[n], width=1, color=COLORS[n], alpha=0.4, label=detectors[n])
        plt.plot(x_sim, gaussian(x_sim, res[n][0], res[n][1], res[n][2], res[n][3]), color=COLORS[n])
    plt.legend()
    plt.xlabel("Channel")
    plt.ylabel("Count")
    plt.show()