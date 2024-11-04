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
