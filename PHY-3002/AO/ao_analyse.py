import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


lambdas = [0.473,0.532,0.632] # bleu,vert,rouge micromètres
parcours_optique = 113 #cm

def get_angles_peaks(filename):
    data = np.loadtxt(filename)
    pos = data[0] # pour les ordres m = -2 , -1 , 0 , +1 , +2
    ang = np.arctan(pos/parcours_optique)
    incertitude_sur_pos = np.ones_like(ang)*0.05
    incertitude_sur_parcour = np.ones_like(ang)*2
    les_parcours = np.ones_like(ang)*parcours_optique
    inc = ang*np.sqrt((incertitude_sur_pos/pos)**2+(incertitude_sur_parcour/les_parcours))
    return ang, inc

frequencies = [34,36,38,40,42,44,46] # MHz
for k, couleur in enumerate(["bleu","vert","rouge"]):
    angles = []
    uncertainties = []
    for f in frequencies:
        ang, inc = get_angles_peaks(f"PHY-3002/AO/data/{couleur}_{f}.txt")
        print(inc)
        angles.append(ang)
        uncertainties.append(inc)
    angles = np.array(angles)*1000
    uncertainties = np.array(np.abs(uncertainties))*1000

    fig = plt.figure(figsize=(5.25,5))
    ax1 = plt.subplot(111)
    for i in range(5):
        plt.errorbar(frequencies, angles[:,i], uncertainties[:,i], fmt="o", color=["blue","green","red"][k])
        #plt.plot(frequencies, angles[:,i], "o", color=["blue","green","red"][k])
        plt.plot(frequencies, angles[:,i], "-", color=["blue","green","red"][k])
        plt.ylabel(r"Angle de déviation $\alpha$ [mrad]", fontsize=17)
        plt.xlabel(r"Fréquence $f$ [MHz]", fontsize=17)
        plt.title(f"$\lambda$={lambdas[k]} $\mu$m ({couleur})", fontsize=17)
        plt.subplots_adjust(0.2,0.13,0.98,0.93)
        ax1.xaxis.set_tick_params(labelsize=15)
        ax1.yaxis.set_tick_params(labelsize=15)
    plt.show()
