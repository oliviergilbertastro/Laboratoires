import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from random import gauss

def resample_array(current, current_std):
    new = []
    for s in range(0, len(current)):
        if current_std[s] > 0:
            new.append(gauss(current[s], current_std[s]))
        else:
            new.append(current[s])
    return np.asarray(new)

def get_values_from_file(filename):
    values = pd.read_csv(filename, delimiter=",", decimal=".", skiprows=1)
    return values



#Distance écran-slit:
l = 0.37
l_sigma = 0.002
l = l
a_th=0.04E-3
d_th=0.50E-3

x_values = np.array(get_values_from_file('PHY-2006/Diffraction/mesures2.csv').iloc[:, 0])
y_values = np.array(get_values_from_file('PHY-2006/Diffraction/mesures2.csv').iloc[:, 1])


plt.plot(x_values, y_values)
plt.show()

x_values = (x_values-622*np.ones(x_values.shape))/8620
theta_values = []
for i in x_values:
    theta_values.append(np.arctan(i/l))
theta_values = np.array(theta_values)
lama = 650E-9
I_0 = np.max(y_values)


def fonc_fit(theta, a, d):
    k1 = (np.pi*a*np.sin(theta))/lama
    k2 = (np.pi*d*np.sin(theta))/lama
    return ((np.sin(k1)/k1)**2)*((np.sin(2*k2)/(2*np.sin(k2)))**2)


res = curve_fit(fonc_fit, theta_values, y_values/I_0, p0=[a_th, d_th])[0]
print(res)


x_fit = np.linspace(-0.2, 0.2, 10000)
y_fit = fonc_fit(x_fit, res[0], res[1])


#Quality plot:
ax1 = plt.subplot(111)
ticklabels = ax1.get_xticklabels()
ticklabels.extend( ax1.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(14)
plt.plot(theta_values, y_values/I_0, '.', color="red", label="données")
plt.plot(x_fit, y_fit, '-', color="purple", label="fit")
plt.legend()
plt.xlabel(r'$\theta$ (rad)', size=17)
plt.ylabel(r'Intensité ($I_0$)', size=17)
#plt.savefig(r'C:\Users\olivi\Desktop\Devoirs\PhysElectronique\figures\lab5\resistance_avecC.pdf', format="pdf", bbox_inches="tight")
plt.show()