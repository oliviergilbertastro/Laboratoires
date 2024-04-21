import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



#PHOTON COUNTS: FIG 77-78
#H ALPHA E-WIDTH: FIG 83-84
#COLORS (B-V)+(V-R): FIG 57-60

def open_spectra(fig, Z=0.001):
    file = pd.read_csv(f"PHY-2200/data/fig{fig}_z{str(Z)[-2:]}.dat", delimiter="\t", decimal=".", skiprows=3, encoding='latin-1', engine='python')
    file = np.array(file)
    wav = []
    spectra_cube = [] #Chaque colonne est un moment, chaque ligne est une longueur d'onde
    for line in file:
        linedata = line[0].split()
        wav.append(float(linedata[0]))
        spectra_cube.append(np.array(linedata[1:], float))
    return wav, np.array(spectra_cube)

def open_photon(fig, Z=0.001):
    """
    Returns tuple of (time, #photons) and #photons is itself a 2D array in which each column is for a different IMF (2.35, 3.30, useless)
    """
    file = pd.read_csv(f"PHY-2200/data/fig{fig}_z{str(Z)[-2:]}.dat", delimiter="\t", decimal=".", skiprows=3, encoding='latin-1', engine='python')
    file = np.array(file)
    time = []
    nb_photons = []
    for line in file:
        linedata = line[0].split()
        time.append(float(linedata[0]))
        nb_photons.append(np.array(linedata[1:], float))
    return time, 10**np.array(nb_photons)

def open_ewidth(fig, Z=0.001):
    """
    Returns tuple of (time, equivalent width) and the equivalent width is itself a 2D array in which each column is for a different IMF (2.35, 3.30, useless)
    """
    file = pd.read_csv(f"PHY-2200/data/fig{fig}_z{str(Z)[-2:]}.dat", delimiter="\t", decimal=".", skiprows=3, encoding='latin-1', engine='python')
    file = np.array(file)
    time = []
    e_width = []
    for line in file:
        linedata = line[0].split()
        time.append(float(linedata[0]))
        e_width.append(np.array(linedata[1:], float))
    return time, 10**np.array(e_width)

def open_color(fig, Z=0.001):
    """
    Returns tuple of (time, color index) and the color index is itself a 2D array in which each column is for a different IMF (2.35, 3.30, useless)
    """
    file = pd.read_csv(f"PHY-2200/data/fig{fig}_z{str(Z)[-2:]}.dat", delimiter="\t", decimal=".", skiprows=3, encoding='latin-1', engine='python')
    file = np.array(file)
    time = []
    color_index = []
    for line in file:
        linedata = line[0].split()
        time.append(float(linedata[0]))
        color_index.append(np.array(linedata[1:], float))
    return time, np.array(color_index)

if input("Photons counts? [y/n]") == "y":
    time_z0001, photons_z0001 = open_photon(fig=77, Z='0.001')
    time_z0004, photons_z0004 = open_photon(fig=77, Z='0.004')
    time_z0008, photons_z0008 = open_photon(fig=77, Z='0.008')
    time_z0020, photons_z0020 = open_photon(fig=77, Z='0.020')
    time_z0040, photons_z0040 = open_photon(fig=77, Z='0.040')
    ax1 = plt.subplot(111)
    ticklabels = ax1.get_xticklabels()
    ticklabels.extend( ax1.get_yticklabels() )
    for label in ticklabels:
        label.set_fontsize(14)
    imf = 1
    plt.title(r"$\alpha=$"+['2.35','3.30'][imf], fontsize=15)
    plt.plot(time_z0001, photons_z0001[:, imf], label=r"$z=0.001$")
    plt.plot(time_z0004, photons_z0004[:, imf], label=r"$z=0.004$")
    plt.plot(time_z0008, photons_z0008[:, imf], label=r"$z=0.008$")
    plt.plot(time_z0020, photons_z0020[:, imf], label=r"$z=0.020$")
    plt.plot(time_z0040, photons_z0040[:, imf], label=r"$z=0.040$")
    plt.xlabel("Temps [années]", fontsize=15)
    plt.ylabel("Flux de photons ionisants [$\mathrm{s}^{-1}$]", fontsize=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(fontsize=15)
    plt.show()

    ax1 = plt.subplot(111)
    ticklabels = ax1.get_xticklabels()
    ticklabels.extend( ax1.get_yticklabels() )
    for label in ticklabels:
        label.set_fontsize(14)
    plt.title(r"$z=0.040$", fontsize=15)
    plt.plot(time_z0040, photons_z0040[:, 0], label=r"$\alpha=2.35$")
    plt.plot(time_z0040, photons_z0040[:, 1], label=r"$\alpha=3.30$")
    plt.xlabel("Temps [années]", fontsize=15)
    plt.ylabel("Flux de photons ionisants [$\mathrm{s}^{-1}$]", fontsize=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(fontsize=15)
    plt.show()



if input("E-Width? [y/n]") == "y":
    time_z0001, ewidth_z0001 = open_ewidth(fig=83, Z='0.001')
    time_z0004, ewidth_z0004 = open_ewidth(fig=83, Z='0.004')
    time_z0008, ewidth_z0008 = open_ewidth(fig=83, Z='0.008')
    time_z0020, ewidth_z0020 = open_ewidth(fig=83, Z='0.020')
    time_z0040, ewidth_z0040 = open_ewidth(fig=83, Z='0.040')
    ax1 = plt.subplot(111)
    ticklabels = ax1.get_xticklabels()
    ticklabels.extend( ax1.get_yticklabels() )
    for label in ticklabels:
        label.set_fontsize(14)
    imf = 1
    plt.title(r"$\alpha=$"+['2.35','3.30'][imf], fontsize=15)
    plt.plot(time_z0001, ewidth_z0001[:, imf], label=r"$z=0.001$")
    plt.plot(time_z0004, ewidth_z0004[:, imf], label=r"$z=0.004$")
    plt.plot(time_z0008, ewidth_z0008[:, imf], label=r"$z=0.008$")
    plt.plot(time_z0020, ewidth_z0020[:, imf], label=r"$z=0.020$")
    plt.plot(time_z0040, ewidth_z0040[:, imf], label=r"$z=0.040$")
    plt.xlabel("Temps [années]", fontsize=15)
    plt.ylabel(r"Largeur équivalente de H$_\alpha$ [$\mathrm{s}^{-1}$]", fontsize=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(fontsize=15)
    plt.show()

    ax1 = plt.subplot(111)
    ticklabels = ax1.get_xticklabels()
    ticklabels.extend( ax1.get_yticklabels() )
    for label in ticklabels:
        label.set_fontsize(14)
    plt.title(r"$z=0.040$", fontsize=15)
    plt.plot(time_z0040, ewidth_z0040[:, 0], label=r"$\alpha=2.35$")
    plt.plot(time_z0040, ewidth_z0040[:, 1], label=r"$\alpha=3.30$")
    plt.xlabel("Temps [années]", fontsize=15)
    plt.ylabel(r"Largeur équivalente de H$_\alpha$ [$\mathrm{s}^{-1}$]", fontsize=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(fontsize=15)
    plt.show()


if input("Color? [y/n]") == "y":
    time_z0001, color_z0001 = open_color(fig=57, Z='0.001')
    time_z0004, color_z0004 = open_color(fig=57, Z='0.004')
    time_z0008, color_z0008 = open_color(fig=57, Z='0.008')
    time_z0020, color_z0020 = open_color(fig=57, Z='0.020')
    time_z0040, color_z0040 = open_color(fig=57, Z='0.040')
    ax1 = plt.subplot(111)
    ticklabels = ax1.get_xticklabels()
    ticklabels.extend( ax1.get_yticklabels() )
    for label in ticklabels:
        label.set_fontsize(14)
    imf = 1
    plt.title(r"$\alpha=$"+['2.35','3.30'][imf], fontsize=15)
    plt.plot(time_z0001, color_z0001[:, imf], label=r"$z=0.001$")
    plt.plot(time_z0004, color_z0004[:, imf], label=r"$z=0.004$")
    plt.plot(time_z0008, color_z0008[:, imf], label=r"$z=0.008$")
    plt.plot(time_z0020, color_z0020[:, imf], label=r"$z=0.020$")
    plt.plot(time_z0040, color_z0040[:, imf], label=r"$z=0.040$")
    plt.xlabel("Temps [années]", fontsize=15)
    plt.ylabel("(B-V)", fontsize=15)
    plt.xscale('log')
    plt.legend(fontsize=15)
    plt.show()

    ax1 = plt.subplot(111)
    ticklabels = ax1.get_xticklabels()
    ticklabels.extend( ax1.get_yticklabels() )
    for label in ticklabels:
        label.set_fontsize(14)
    plt.title(r"$z=0.040$", fontsize=15)
    plt.plot(time_z0040, color_z0040[:, 0], label=r"$\alpha=2.35$")
    plt.plot(time_z0040, color_z0040[:, 1], label=r"$\alpha=3.30$")
    plt.xlabel("Temps [années]", fontsize=15)
    plt.ylabel("(B-V)", fontsize=15)
    plt.xscale('log')
    plt.legend(fontsize=15)
    plt.show()


    time_z0001, color_z0001 = open_color(fig=59, Z='0.001')
    time_z0004, color_z0004 = open_color(fig=59, Z='0.004')
    time_z0008, color_z0008 = open_color(fig=59, Z='0.008')
    time_z0020, color_z0020 = open_color(fig=59, Z='0.020')
    time_z0040, color_z0040 = open_color(fig=59, Z='0.040')
    ax1 = plt.subplot(111)
    ticklabels = ax1.get_xticklabels()
    ticklabels.extend( ax1.get_yticklabels() )
    for label in ticklabels:
        label.set_fontsize(14)
    plt.title(r"$\alpha=$"+['2.35','3.30'][imf], fontsize=15)
    plt.plot(time_z0001, color_z0001[:, imf], label=r"$z=0.001$")
    plt.plot(time_z0004, color_z0004[:, imf], label=r"$z=0.004$")
    plt.plot(time_z0008, color_z0008[:, imf], label=r"$z=0.008$")
    plt.plot(time_z0020, color_z0020[:, imf], label=r"$z=0.020$")
    plt.plot(time_z0040, color_z0040[:, imf], label=r"$z=0.040$")
    plt.xlabel("Temps [années]", fontsize=15)
    plt.ylabel("(V-R)", fontsize=15)
    plt.xscale('log')
    plt.legend(fontsize=15)
    plt.show()

    ax1 = plt.subplot(111)
    ticklabels = ax1.get_xticklabels()
    ticklabels.extend( ax1.get_yticklabels() )
    for label in ticklabels:
        label.set_fontsize(14)
    plt.title(r"$z=0.040$", fontsize=15)
    plt.plot(time_z0040, color_z0040[:, 0], label=r"$\alpha=2.35$")
    plt.plot(time_z0040, color_z0040[:, 1], label=r"$\alpha=3.30$")
    plt.xlabel("Temps [années]", fontsize=15)
    plt.ylabel("(V-R)", fontsize=15)
    plt.xscale('log')
    plt.legend(fontsize=15)
    plt.show()