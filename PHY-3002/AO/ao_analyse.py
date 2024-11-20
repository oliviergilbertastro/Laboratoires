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

def find_line(x_data, y_data, sigma=None):
    """
    gives intercept and slope
    """
    res = curve_fit(lambda x,a: x*a, x_data, y_data, sigma=sigma)
    return res[0]

frequencies = [34,36,38,40,42,44,46] # MHz
mes_vs = []
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

    fig = plt.figure(figsize=(6.25,5))
    ax1 = plt.subplot(111)
    slopes = []
    for i in range(5):
        slope = find_line(frequencies, angles[:,i], uncertainties[:,i])[0]
        print(f"alpha = {slope}f")
        slopes.append(slope)
        plt.errorbar(frequencies, angles[:,i], uncertainties[:,i], fmt="o", color=["blue","green","red"][k])
        #plt.plot(frequencies, angles[:,i], "o", color=["blue","green","red"][k])
        plt.plot(frequencies, angles[:,i], "-", color=["blue","green","red"][k])
        plt.plot(frequencies, slope*np.array(frequencies), "--", color="black")
    mes_vs.append(lambdas[k]*1E6/(slopes[3]/1000*1000000))
    col_labels=[r'$m$',r'Pente [$f/\alpha$]']
    table_vals=np.array([np.array([-2,-1,0,1,2], dtype=int)[::-1],np.around(slopes, decimals=2)[::-1]]).T
    # the rectangle is where I want to place the table
    the_table = plt.table(cellText=table_vals,
                    colWidths = [0.05,0.12],
                    cellLoc = "center",
                    #rowLabels=row_labels,
                    colLabels=col_labels,
                    loc='upper right')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1.5, 3)
    left, right = plt.xlim()
    plt.xlim(right=right+5)


    plt.ylabel(r"Angle de déviation $\alpha$ [mrad]", fontsize=17)
    plt.xlabel(r"Fréquence $f$ [MHz]", fontsize=17)
    plt.title(f"$\lambda$={lambdas[k]} $\mu$m ({couleur})", fontsize=17)
    plt.subplots_adjust(0.2,0.13,0.98,0.93)
    ax1.xaxis.set_tick_params(labelsize=15)
    ax1.yaxis.set_tick_params(labelsize=15)
    plt.savefig(f"PHY-3002/AO/graph/final_{couleur}.png")
    plt.show()





import os, parse ,sys
code_cours = "3002\AO"
parent_dir = parse.parse("{}\PHY-"+code_cours, os.path.dirname(os.path.realpath(__file__)))[0]
sys.path.append(parent_dir)
from utils import *


print_color("Avec bleu:", color="blue")

v_s = np.mean(mes_vs), np.std(mes_vs)

print_color(f"v_s = {v_s[0]} \pm {v_s[1]}")

print_color("Sans bleu:", color="blue")

mes_vs = mes_vs[1:]
v_s = np.mean(mes_vs), np.std(mes_vs)

print_color(f"v_s = {v_s[0]} \pm {v_s[1]}")