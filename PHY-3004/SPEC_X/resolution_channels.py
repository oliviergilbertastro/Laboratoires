import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def loadFile(source="Am", detector="CdTe", gain=7670, res=256):
    return np.loadtxt(f"PHY-3004/SPEC_X/Am_CdTe_7670_{res}.mca", skiprows=15, max_rows=res, dtype=float)

data = []
for res in [256, 4096]:
    data.append(loadFile(res=res))

new_data = []
for i in range(len(data[0])):
    for k in range(16):
        new_data.append(data[0][i])
data[0] = np.array(new_data)


for i in range(len(data)):
    data[i] = np.abs(data[i])
    print(np.min(data[i]))
print(np.max(data[0])/np.max(data[1]))

print(data[1][3500:3550])

print((data[1]*np.max(data[0])/np.max(data[1]))[3500:3550])

plt.plot(data[0], label="256")
plt.plot(data[1]*np.max(data[0])/np.max(data[1]), label="4096")
plt.legend()
plt.xlabel("Channel")
plt.ylabel("Count")
plt.show()





def gaussian(x, sig, b, mu, c):
    return (
        b / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x-mu) / sig, 2.0) / 2) + c
    )

from scipy.optimize import curve_fit

for pics in [(770,902),(1045,1124),(1519,1638),(3271,3700)]:

    gaussiennes = [data[0][pics[0]:pics[1]], data[1][pics[0]:pics[1]]]
    plt.plot(range(len(gaussiennes[0])), gaussiennes[0], label="256")
    plt.plot(range(len(gaussiennes[1])), gaussiennes[1]*np.max(data[0])/np.max(data[1]), label="4096")
    plt.legend()
    plt.xlabel("Channel")
    plt.ylabel("Count")
    plt.show()



    res0 = curve_fit(gaussian, range(len(gaussiennes[0])), gaussiennes[0], p0=[1.46726127e+01, 2.59850085e+06, len(gaussiennes[0])/2, 5.4e+3], bounds=[(0,0,0,0),(np.inf,np.inf,np.inf,np.inf)])[0]
    res1 = curve_fit(gaussian, range(len(gaussiennes[1])), gaussiennes[1]*np.max(data[0])/np.max(data[1]), p0=[1.22854407e+01, 2.44366868e+06, len(gaussiennes[1])/2, 5.4e+3], bounds=[(0,0,0,0),(np.inf,np.inf,np.inf,np.inf)])[0]

    print(res0)
    print(res1)

    for res in [res0, res1]:
        print(f"FWHM = {2*np.sqrt(2*np.log(2))*res[0]}")

    x_sim = np.linspace(0, len(gaussiennes[0]), 1000)
    #plt.plot(range(len(gaussiennes[0])), gaussiennes[0], ".", label="256", color="blue")
    #plt.plot(range(len(gaussiennes[1])), gaussiennes[1]*np.max(data[0])/np.max(data[1]), ".", label="4096", color="orange")
    plt.bar(range(len(gaussiennes[0])), gaussiennes[0], width=1, color="blue", alpha=0.4, label="256")
    plt.bar(range(len(gaussiennes[1])), gaussiennes[1]*np.max(data[0])/np.max(data[1]), width=1, color="red", alpha=0.4, label="4096")
    plt.plot(x_sim, gaussian(x_sim, res0[0], res0[1], res0[2], res0[3]), color="blue")
    plt.plot(x_sim, gaussian(x_sim, res1[0], res1[1], res1[2], res1[3]), color="red")
    plt.legend()
    plt.xlabel("Channel")
    plt.ylabel("Count")
    plt.show()
