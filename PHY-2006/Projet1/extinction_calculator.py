import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

extinction_file = pd.read_csv("PHY-2006/Projet1/data/extinction_data.csv", delimiter=",", decimal=".", skiprows=1, encoding='latin-1', engine='python')
wav = np.array(extinction_file.iloc[:, 0])/10
kappa = np.array(extinction_file.iloc[:, 1])



def polynomial(x,c1,c2,c3,c4,c5):
    return c1*x**3+c2*x**2+c3*x+c4+c5*x**4

res = curve_fit(polynomial, wav, kappa)[0]

x_fit = np.linspace(200, 900, 1000)
y_fit = polynomial(x_fit, res[0], res[1], res[2], res[3], res[4])




def mag_to_percentage(extinction_coeff, observation_angle=0):
    airmass = 1/np.cos(observation_angle)
    return 1-10**(-0.4*extinction_coeff*airmass)

def correct_intensity(uncorr_intensity, extinction_perc):
    """
    I_corr = I_0/correction_percentage
    """
    return uncorr_intensity/(1-extinction_perc)


if __name__ == "__main__":
    print(res)
    plt.plot(wav, kappa, 'o')
    plt.plot(x_fit, y_fit)
    plt.xlabel("$\lambda$ [nm]")
    plt.ylabel("$\kappa(\lambda)$")
    plt.show()

    extinction_percentages = mag_to_percentage(y_fit, observation_angle=73/180*np.pi)
    plt.plot(x_fit, correct_intensity(np.ones(np.shape(y_fit)), extinction_percentages))
    plt.xlabel("$\lambda$ [nm]")
    plt.ylabel("$I/I_0$")
    plt.show()
    for i in range(8):
        print(mag_to_percentage(i/10, observation_angle=45/180*np.pi))

