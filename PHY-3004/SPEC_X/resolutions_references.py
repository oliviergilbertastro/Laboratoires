import numpy as np
import matplotlib.pyplot as plt

def FWHM(E):
    return np.sqrt(145**2-120**2+2440*E)


sources = [59.5409, 33.196, 14.4128, 136.4737, 88.0341]

for E in sources:
    print(f"{FWHM(E):.2f} eV")


E = np.linspace(30, 500, 1000)
plt.plot(E, FWHM(E))
plt.show()