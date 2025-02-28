import numpy as np
import matplotlib.pyplot as plt

# Charger les données
filename = "PHY-3004/SPEC_X/Am_CdTe_7670_256.mca"
data = np.loadtxt(filename, skiprows=25, max_rows=1040, dtype=float)
print(data)
# Tracer le spectre
plt.figure(figsize=(8, 5))
plt.plot(range(256), data, drawstyle="steps-mid")
plt.xlabel("Canaux")
plt.ylabel("Intensité")
plt.title("Spectre MCA - Am/CdTe (Gain 7670)")
plt.grid()
plt.show()
