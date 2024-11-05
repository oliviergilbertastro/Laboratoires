
from outils_analyse.identification_des_pics import get_peaks_indices
from outils_analyse.lecture_des_fichiers import read_csv, crop_ramp, crop_ramp_actually_good
from outils_analyse.conversion_temps_en_potentiel import \
    compute_conversion_factors
import matplotlib.pyplot as plt
import os
import matplotlib
# Grosseur du texte dans les figures
matplotlib.rcParams.update({'font.size': 18})
import parse
import sys
code_cours = "3002\lab_franck_hertz"
#print(os.path.dirname(os.path.realpath(__file__)))
parent_dir = parse.parse("{}\PHY-"+code_cours, os.path.dirname(os.path.realpath(__file__)))[0]
sys.path.append(parent_dir)
from utils import *

"""
_____________________________________________________________________________
"""

# TODO: Lire le fichier des résultats et transformer les valeurs en un array numpy
# Mettre votre code ici

import numpy as np


extension = input("Entrer l'extension du fichier à analyser:\nPHY-3002\lab_franck_hertz\courbe_excitation_electronique_simple")
data = read_csv(r"PHY-3002\lab_franck_hertz\courbe_excitation_electronique_simple"+f"{extension}.csv", 9)

#data = read_csv(r"PHY-3002\lab_franck_hertz\exemples_de_fichiers\exemple_de_donnees.csv", 9)




# Mettre vos valeurs extraites à la place du None.
valeurs_en_array = np.array(data)  # Array de trois colonnes
"""
_____________________________________________________________________________
"""
# Ne pas modifier cette section!!!

# La figure obtenue devrait correspondre à celle de figures_exemple/lecture_des_donnes_brutes
plt.figure()
plt.plot(valeurs_en_array[:, 0], valeurs_en_array[:, 2], label="Tensions du pico")
plt.plot(valeurs_en_array[:, 0], valeurs_en_array[:, 1], label="Tensions entre la G1 et le ground")
plt.xlabel("Temps [s]")
plt.ylabel("Tension [V]")
plt.legend()
plt.subplots_adjust(left=0.158, bottom=0.145, right=0.967, top=0.955, wspace=0.2, hspace=0)
plt.savefig(f"PHY-3002/Graph_FH/donnees_brutes{extension}.png")
plt.show()

"""
_____________________________________________________________________________
"""


#TODO: Retirer les valeurs qui se trouvent à l'extérieur de l'activation du générateur de rampe.
# Ne pas oublier de mettre le début de la rampe comme étant t=0.

# Mettre votre code ici







#cropped = crop_ramp(valeurs_en_array, 2, 0.0733, 10)
cropped = crop_ramp_actually_good(valeurs_en_array, 2, 10, if_plot=False)

cropped[:, 0] = cropped[:, 0]-cropped[0, 0]



# Mettre vos données croppées remises à t_0=0 dans cette variable
valeurs_cropped_debutant_par_t0 = cropped  # Array de trois colonnes

"""
_____________________________________________________________________________
"""
# Ne pas modifier cette section!!!

# La figure obtenue devrait correspondre à celle de figures_exemple/donnes_cropped
plt.figure()
plt.plot(valeurs_cropped_debutant_par_t0[:, 0], valeurs_cropped_debutant_par_t0[:, 2],
         label="Tensions du pico")
plt.plot(valeurs_cropped_debutant_par_t0[:, 0], valeurs_cropped_debutant_par_t0[:, 1],
         label="Tensions entre la G1 et le ground")
plt.xlabel("Temps [s]")
plt.ylabel("Tension [V]")
plt.legend()
plt.subplots_adjust(left=0.158, bottom=0.145, right=0.967, top=0.955, wspace=0.2, hspace=0)
plt.savefig(f"PHY-3002/Graph_FH/donnes_cropped{extension}.png")
plt.show()

"""
_____________________________________________________________________________
"""
# TODO Calculer la pente de la tension du générateur de rampe et son incertitude.
#  Afficher cette valeur et son incertitude, puis convertir les valeurs de temps
#  en valeurs de tension. La valeur devrait être près de: Pente =  -0.44003644585609436 +- 0.031069057062268257 V/s

# TODO: Ensuite, convertissez les valeurs de tensions du pico en valeur de courant, en considérant que l'échelle
#  du pico utilisé est de 3nA.

# Mettre votre code ici

from outils_analyse.fits import linear_regression

res = compute_conversion_factors(valeurs_cropped_debutant_par_t0, 0, 2)

valeurs_cropped_debutant_par_t0[:,0] = valeurs_cropped_debutant_par_t0[:,0]*np.abs(res[0])
valeurs_cropped_debutant_par_t0[:,1:] = valeurs_cropped_debutant_par_t0[:,1:]#*(3/2) #devrait être l'échelle

# Mettre vos données avec les bonnes unités à la place et vos informations par rapport à la pente ici
valeurs_avec_bonnes_unites = valeurs_cropped_debutant_par_t0  # Array de trois colonnes
facteur_valeur = res[0]  # Nombre à virgule
facteur_incertitude = res[1]  # Nombre à virgule

"""
_____________________________________________________________________________
"""

# Ne pas modifier cette section!!!

print("Pente = ", f"{facteur_valeur} +- {facteur_incertitude}")

# La figure obtenue devrait correspondre à celle de figures_exemple/donnes_avec_bonnes_unités
plt.figure()
plt.plot(valeurs_avec_bonnes_unites[:, 0], valeurs_avec_bonnes_unites[:, 1],
         label="Courant du pico")
plt.xlabel("Tension entre G1 et le ground [V]")
plt.ylabel("Courant mesuré [nA]")
plt.legend()
plt.subplots_adjust(left=0.158, bottom=0.145, right=0.967, top=0.955, wspace=0.2, hspace=0)
plt.savefig(f"PHY-3002/Graph_FH/donnes_avec_bonnes_unites{extension}.png")
plt.show()

"""
_____________________________________________________________________________
"""

# TODO :Déterminer l'emplacement approximatif des maximums.
#  Ça devrait être environ: Estimation des pics: [ 1.9844797  6.902538  11.950019  17.083782 ] V

# Mettre votre code ici


maxs = get_peaks_indices(valeurs_avec_bonnes_unites, 1, hauteur_minimum=np.max(valeurs_avec_bonnes_unites[:,1])/20, distance_minumum=(len(valeurs_avec_bonnes_unites[:,1])/7))









# Mettre vos données avec les bonnes unités à la place du None
valeurs_avec_bonnes_unites_determination_des_pics = valeurs_avec_bonnes_unites  # Array de trois colonnes
liste_des_indexes_des_pics = maxs  # Liste de nombres entiers

"""
_____________________________________________________________________________
"""
# Ne pas modifier cette section!!!

print("Estimation des pics:", valeurs_avec_bonnes_unites_determination_des_pics[liste_des_indexes_des_pics, 0])

# La figure obtenue devrait correspondre à celle de figures_exemple/estimation_des_pics
plt.figure()
plt.plot(valeurs_avec_bonnes_unites_determination_des_pics[:, 0],
         valeurs_avec_bonnes_unites_determination_des_pics[:, 1],
         label="Courant du pico")
plt.xlabel("Tension entre G1 et le ground [V]")
plt.scatter(valeurs_avec_bonnes_unites_determination_des_pics[liste_des_indexes_des_pics, 0],
            valeurs_avec_bonnes_unites_determination_des_pics[liste_des_indexes_des_pics, 1],
            label="Estimation des pics")
plt.ylabel("Courant mesuré [nA]")
plt.legend()
plt.subplots_adjust(left=0.158, bottom=0.145, right=0.967, top=0.955, wspace=0.2, hspace=0)
plt.savefig(f"PHY-3002/Graph_FH/estimation_des_pics{extension}.png")
plt.show()




# V_res
pics = valeurs_avec_bonnes_unites_determination_des_pics[liste_des_indexes_des_pics, 0]
distances_pics = pics[1:]-pics[:-1]
v_res = np.mean(distances_pics)

#print("V_res:", np.around(v_res, decimals=3), "V")
print_color(f"V_res: {np.around(v_res, decimals=3)} V")

# Potentiel de charge:
u1 = float(input("Entrer la valeur de U1 (en V):\n"))
w = valeurs_avec_bonnes_unites_determination_des_pics[liste_des_indexes_des_pics[0], 0]+u1-v_res
#print("Potentiel de charge W:", np.around(w, decimals=3), "V")
print_color(f"Potentiel de charge W: {np.around(w, decimals=3)} V")

with open(f'PHY-3002/lab_franck_hertz/analyse{extension}.txt', 'w') as output:
    message = f"Pics: {pics}\nU1: {u1} V\nV_res: {np.around(v_res, decimals=3)} V\nPotentiel de charge W: {np.around(w, decimals=3)} V"
    output.write(message)

