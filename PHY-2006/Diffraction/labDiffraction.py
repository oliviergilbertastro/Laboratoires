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

x_values = np.array(get_values_from_file('PHY-2006/Diffraction/mesures.csv').iloc[:, 0])
y_values = np.array(get_values_from_file('PHY-2006/Diffraction/mesures.csv').iloc[:, 1])


x_values = (x_values-679*np.ones(x_values.shape))/11500
#x_values = (x_values-679*np.ones(x_values.shape))/11300
theta_values = []
for i in x_values:
    theta_values.append(np.arctan(i/l))
theta_values = np.array(theta_values)
lama = 650E-9
I_0 = np.max(y_values)

def gaussian(x, sig, b):
    return (
        b / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x) / sig, 2.0) / 2)
    )

def fonc_fit(theta, a, b):
    k = (np.pi*a*np.sin(theta))/lama
    return ((np.sin(k)/k)**2)**b

def fonc_plus_gauss_fit(theta, a, b, sig, c):
    k = (np.pi*a*np.sin(theta))/lama
    return (np.sin(k)/k)**(2*c)+gaussian(theta, sig, b)


#On fit une gaussienne, on soustrait, puis on fit la fonction d'intensité:
x_fit = np.linspace(-0.2, 0.2, 10000)
y_gauss_fit = []
y_total_fit = []
y_simple_fit = []
y_real_gauss_fit = []
res_gauss = curve_fit(gaussian, np.concatenate((theta_values[:500], theta_values[900:])), np.concatenate((y_values[:500], y_values[900:]))/I_0)[0]
#res_total = curve_fit(fonc_plus_gauss_fit, theta_values, y_values/I_0, p0=[0.00004,-0.05075882,-0.05648042,0.02295134])[0]
res_simple = curve_fit(fonc_fit, theta_values, y_values/I_0, p0=[[0.00004, 0.5]])[0]
print(res_simple)
print(res_gauss)
for i in (x_fit):
    #y_gauss_fit.append(gaussian(i, res_gauss[0], res_gauss[1], res_gauss[2]))
    y_gauss_fit.append(gaussian(i, res_gauss[0], res_gauss[1]))
    y_simple_fit.append(fonc_fit(i, res_simple[0], res_simple[1]))

y_gauss_fit = np.array(y_gauss_fit)*0.7
y_total_fit = np.array(y_gauss_fit) + np.array(y_simple_fit)


#Quality plot:
ax1 = plt.subplot(111)
ticklabels = ax1.get_xticklabels()
ticklabels.extend( ax1.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(14)
plt.plot(theta_values, y_values/I_0, color="red", label="données")
#plt.plot(theta_values, data_gy/I_0, label="données-gaussian")
plt.xlabel(r'Position (m)', size=17)
plt.ylabel(r'Intensité (grayscale)', size=17)
plt.plot(x_fit, fonc_fit(x_fit, res_simple[0], res_simple[1]), color="purple", label="fit")
#plt.plot(x_fit, fonc_fit(x_fit, 0.00004, 1), color="blue", label="théorique")
plt.plot(x_fit, y_gauss_fit, '--', color='orange', label='gaussienne')
plt.plot(x_fit, y_total_fit, '-', color='green', label='fit+gaussienne')
plt.legend()
plt.xlabel(r'$\theta$ (rad)', size=17)
plt.ylabel(r'Intensité ($I_0$)', size=17)
#plt.savefig(r'C:\Users\olivi\Desktop\Devoirs\PhysElectronique\figures\lab5\resistance_avecC.pdf', format="pdf", bbox_inches="tight")
plt.show()