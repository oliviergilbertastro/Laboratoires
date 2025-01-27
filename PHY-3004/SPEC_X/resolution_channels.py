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



gaussiennes = [data[0][770:902], data[1][770:902]]
plt.plot(range(len(gaussiennes[0])), gaussiennes[0], label="256")
plt.plot(range(len(gaussiennes[1])), gaussiennes[1]*np.max(data[0])/np.max(data[1]), label="4096")
plt.legend()
plt.xlabel("Channel")
plt.ylabel("Count")
plt.show()

def gaussian(x, sig, b, mu):
    return (
        b / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x-mu) / sig, 2.0) / 2)
    )


from scipy.optimize import curve_fit

res0 = curve_fit(gaussian, range(len(gaussiennes[0])), gaussiennes[0])[0]
res1 = curve_fit(gaussian, range(len(gaussiennes[1])), gaussiennes[1]*np.max(data[0])/np.max(data[1]))[0]

print(res0)
print(res1)

for res in [res0, res1]:
    print(f"FWHM = {2*np.sqrt(2*np.log(2))*res[0]}")

x_sim = np.linspace(0, len(gaussiennes[0]), 1000)
plt.plot(range(len(gaussiennes[0])), gaussiennes[0], label="256")
plt.plot(range(len(gaussiennes[1])), gaussiennes[1]*np.max(data[0])/np.max(data[1]), label="4096")
plt.plot(x_sim, gaussian(x_sim, res0[0], res0[1], res0[2]), label="256")
plt.plot(x_sim, gaussian(x_sim, res1[0], res1[1], res1[2]), label="4096")
plt.legend()
plt.xlabel("Channel")
plt.ylabel("Count")
plt.show()
