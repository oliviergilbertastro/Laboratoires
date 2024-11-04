import numpy as np
from scipy.signal import find_peaks


def get_peaks_indices(valeurs: np.array, colonne_y: int,
                                      hauteur_minimum: int = None,
                                      distance_minumum: int = None):
    """
    Cette méthode utilise la méthode find_peaks de scipy pour trouver les indices de l'array correspondant aux maximums.

    :param valeurs: array contenant toutes les valeurs
    :param colonne_y: index de la colonne contenant les valeurs dépendantes
    :param hauteur_minimum: Optionnel! Celui-ci indique la hauteur minimale entre
                            deux pics pour guider l'algorithme de scipy
                            Cette valeur devrait être inférieure à la hauteur du plus grand pic.
                            de l'array numpy (<array.max())
    :param distance_minumum:Optionnel! Celui-ci indique la distance minimale entre
                            deux pics pour guider l'algorithme de scipy
                            Cette valeur devrait être inférieure à la longueur totale
                            de l'array numpy divisée par le nombre total de pics (<(array.shape[1]/nb de pics))

    :return: liste des indices correspondant aux emplacements des maximums
    """
    peaks, _ = find_peaks(valeurs[:, colonne_y], height=hauteur_minimum, distance=distance_minumum)
    return peaks
