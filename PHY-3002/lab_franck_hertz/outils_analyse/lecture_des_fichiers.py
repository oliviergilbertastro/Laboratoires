import csv
from os import PathLike

import numpy as np


def read_csv(path_complet_vers_csv: str or PathLike,
                          nb_de_lignes_a_retirer: int,
                          delimier: str = ",") -> np.array:
    """
    Cette méthode extrait les valeurs d'un tableau csv avec 3 colonnes.

    :param path_complet_vers_csv: chemin vers le ficher à lire
    :param nb_de_lignes_a_retirer: nombre de ligne à ne pas lire au début du fichier
    :param delimier: caractère séparant les valeurs d'une même rangée

    :return: Un array numpy ayant comme dimensions (n_rangés, 3 colonnes)
    """
    empty_array = []
    with open(path_complet_vers_csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimier)
        for row in list(csv_reader)[nb_de_lignes_a_retirer:]:
            empty_array.append([row[0], row[1], row[2]])

    array_as_numpy_float = np.asarray(empty_array, dtype=np.float32)

    return array_as_numpy_float


def crop_ramp(valeurs: np.array, indice_colonne_rampe: int, zero_threshold: float,
                                                 infinity_threshold: float) -> np.array:
    """
    Cette méthode permet de retirer tous les éléments de l'array où le gradient de valeurs de tension de la rampe
    associée est nul* ou inifni**.

    *Ici on considère que le gradient est nul que lorsque qu'il est - zero_threshold < gradient < zero_threshold.
    ***Ici on considère que le gradient est infini que lorsque qu'il est -infinity_threshold > gradient ou
     infinity_threshold < gradient.

    :param valeurs: array numpy contenant au moins la colonne du temps et la colonne du potentiel de la rampe
    :param indice_colonne_rampe: indice associé à la colonne du potentiel de la rampe dans l'array numpy
    :param zero_threshold: valeurs limites supérieures pour considérer un gradient comme étant nul
                           cette valeur devrait être bien plus petite que la pente qu'on s'attend à calculer.
                           Dans le cadre de ce laboratoire, ça devrait être en-dessous de 0.1
    :param infinity_threshold: valeurs limites inférieures pour considérer un gradient comme étant infini
                            cette valeur devrait être bien plus grande que la pente qu'on s'attend à calculer.
                           Dans le cadre de ce laboratoire, ça devrait être suprieur  ou égal à 0.1

    :return:
    """

    # ici on prend delta_V_i = V_i+1 - V_i-1. Nous perdons donc les 2 éléments aux extrémités.
    deltats_V = valeurs[2:, indice_colonne_rampe] - valeurs[:-2, indice_colonne_rampe]
    outside_zero_treshold = np.logical_or((deltats_V > zero_threshold), deltats_V < -zero_threshold)
    within_infinity_threshold = np.logical_and(deltats_V > -infinity_threshold, deltats_V < infinity_threshold)
    variation_non_nulle, = np.where(np.logical_and(outside_zero_treshold, within_infinity_threshold))
    index_debut = variation_non_nulle.min()
    index_fin = variation_non_nulle.max()

    cropped_array = valeurs[index_debut:index_fin]

    return cropped_array


from scipy.signal import boxcar
from pylab import r_


def smooth(x, smoothing_param=3):
    window_len=smoothing_param*2+1
    s=r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    w=boxcar(smoothing_param*2+1)
    y=np.convolve(w/np.sum(w),s,mode='valid')
    return y[smoothing_param:-smoothing_param] 

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import copy

def crop_ramp_better(valeurs, indice_colonne_rampe, nb_of_stds=1, if_plot=False, verbose=False):
    """
    smooths the signal and thens finds the bounds so it's good

    assumes the middle fifth of the data is the ramp
    """
    valeurs_og = copy.copy(valeurs)
    valeurs[:,indice_colonne_rampe] = smooth(valeurs[:,indice_colonne_rampe], 3)
    deltats_V = valeurs[1:, indice_colonne_rampe] - valeurs[:-1, indice_colonne_rampe]
    ramp_length = len(valeurs[:,indice_colonne_rampe])
    ramp_mid = deltats_V[int(2*ramp_length/5):int(3*ramp_length/5)]
    ramp_mean, ramp_std = np.mean(ramp_mid), np.std(ramp_mid)


    if if_plot:
        plt.plot(valeurs_og[:,indice_colonne_rampe], linewidth=3, color="orange", label="Original")
        plt.plot(valeurs[:,indice_colonne_rampe], linewidth=1.5, color="blue", label="Lissé")
        plt.xlabel("Index")
        plt.ylabel("Tension [V]")
        plt.legend(fontsize=15)
        plt.show()

        #plt.plot([0,1,2], [4,6,5], "--")
        #plt.fill_between([0,1,2], [3,5,4], [5,7,6], color="red", alpha=0.3)
        #plt.show()

        x_fit = np.linspace(0, len(valeurs[:,indice_colonne_rampe]), len(valeurs[:,indice_colonne_rampe]))
        def slope(x,a,b):
            return x*a+b

        res = curve_fit(slope, x_fit[int(2*ramp_length/5):int(3*ramp_length/5)], valeurs[int(2*ramp_length/5):int(3*ramp_length/5),indice_colonne_rampe])[0]
        y_fit = slope(x_fit,res[0],res[1])
        #plt.plot(valeurs[int(2*ramp_length/5):int(3*ramp_length/5),indice_colonne_rampe])
        plt.plot(valeurs[:,indice_colonne_rampe])
        plt.plot(x_fit, y_fit, "--")
        plt.fill_between(x_fit, y_fit-ramp_std*nb_of_stds, y_fit+ramp_std*nb_of_stds, color="blue", alpha=0.3)
        plt.xlabel("Index")
        plt.ylabel("Tension [V]")
        plt.show()



    # Find the bounds

    lo, hi = int(ramp_length/2), int(ramp_length/2)

    x = valeurs[lo,indice_colonne_rampe]
    try:
        while (x > y_fit[lo]-ramp_std*nb_of_stds and x < y_fit[lo]+ramp_std*nb_of_stds):
            lo -= 1
            x = valeurs[lo,indice_colonne_rampe]
        lo += 1
    except:
        lo = 0

    x = valeurs[hi,indice_colonne_rampe]
    try:
        while (x > y_fit[hi]-ramp_std*nb_of_stds and x < y_fit[hi]+ramp_std*nb_of_stds):
            hi += 1
            x = valeurs[hi,indice_colonne_rampe]
        hi += 1
    except:
        hi = len(valeurs)-1

    if verbose:
        print(lo, hi)

    return valeurs_og[lo:hi]