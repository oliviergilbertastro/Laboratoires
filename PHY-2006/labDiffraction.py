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

x_values = np.array(get_values_from_file('PHY-2006/mesures.csv').iloc[:, 0])
y_values = np.array(get_values_from_file('PHY-2006/mesures.csv').iloc[:, 1])


x_values = (x_values-679*np.ones(x_values.shape))/11500
theta_values = []
for i in x_values:
    theta_values.append(np.arctan(i/l))
theta_values = np.array(theta_values)
lama = 650E-9
I_0 = np.max(y_values)

def gaussian(x, sig, b, c):
    return (
        b / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x) / sig, 2.0) / 2) +c
    )

def fonc_fit(theta, a):
    k = (np.pi*a*np.sin(theta))/lama
    return (np.sin(k)/k)**2
#On fit une gaussienne, on soustrait, puis on fit la fonction d'intensité:
x_fit = np.linspace(-0.2, 0.2, 10000)

g_fit = curve_fit(gaussian, np.concatenate((theta_values[:500], theta_values[900:])), np.concatenate((y_values[:500], y_values[900:]))/I_0)[0]
print('Standard deviation:', g_fit)
data_gy = []
g_y = []
for i in range(len(x_values)):
    data_gy.append(y_values[i]-gaussian(x_values[i], g_fit[0], g_fit[1], g_fit[2]))
for i in range(len(x_fit)):
    g_y.append(gaussian(x_fit[i], g_fit[0], g_fit[1], g_fit[2]))
res = curve_fit(fonc_fit, theta_values, y_values/I_0, p0=[0.00004])[0]




print(res)

#Quality plot:
ax1 = plt.subplot(111)
ticklabels = ax1.get_xticklabels()
ticklabels.extend( ax1.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(14)
plt.plot(theta_values, y_values/I_0, color="red", label="données")
plt.plot(theta_values, data_gy/I_0, label="données-gaussian")
plt.xlabel(r'Position (m)', size=17)
plt.ylabel(r'Intensité (grayscale)', size=17)
plt.plot(x_fit, fonc_fit(x_fit, res[0]), color="blue", label="fit")
plt.plot(x_fit, gaussian(x_fit, g_fit[0], g_fit[1], g_fit[2]), '--', color='orange', label='gaussian fit')
#plt.plot(x_fit, gaussian(x_fit, 4), '--', color='green', label='gaussian manual')
#plt.plot(x_fit, fonc_fit(x_fit, 0.00004), color="orange", label="théorique")
plt.legend()
plt.xlabel(r'$\theta$ (rad)', size=17)
plt.ylabel(r'Intensité ($I_0$)', size=17)
#plt.savefig(r'C:\Users\olivi\Desktop\Devoirs\PhysElectronique\figures\lab5\resistance_avecC.pdf', format="pdf", bbox_inches="tight")
plt.show()