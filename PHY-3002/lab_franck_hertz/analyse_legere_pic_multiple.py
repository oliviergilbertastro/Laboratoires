
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


extension = "_multi"
data = read_csv(r"PHY-3002\lab_franck_hertz\courbe_excitation_electronique"+f"{extension}.csv", 9)


valeurs_en_array = np.array(data)  # Array de trois colonnes

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

# TODO: Retirer les valeurs qui se trouvent à l'extérieur de l'activation du générateur de rampe.
#  Ne pas oublier de mettre le début de la rampe comme étant t=0.

# Mettre votre code ici



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

# TODO: Calculer la pente de la tension du générateur de rampe et son incertitude.
#  Afficher cette valeur et son incertitude, puis convertir les valeurs de temps
#  en valeurs de tension.

# TODO:Ensuite, convertissez les valeurs de tensions du pico en valeurs de courant, en considérant que l'échelle
#  du pico utilisée est de 3nA.

# Mettre votre code ici


from outils_analyse.fits import linear_regression

res = compute_conversion_factors(valeurs_cropped_debutant_par_t0, 0, 2)

valeurs_cropped_debutant_par_t0[:,0] = valeurs_cropped_debutant_par_t0[:,0]*np.abs(res[0])
valeurs_cropped_debutant_par_t0[:,1:] = valeurs_cropped_debutant_par_t0[:,1:]#*(3/2) #devrait être l'échelle

# Mettre vos données avec les bonnes unités à la place et vos informations par rapport à la pente ici
valeurs_avec_bonnes_unites = valeurs_cropped_debutant_par_t0  # Array de trois colonnes
facteur_valeur = res[0]  # Nombre à virgule
facteur_incertitude = res[1]  # Nombre à virgule

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

# TODO: Déterminer l'emplacement approximatif des maximums. Ça devrait être
#  environ: Estimation des pics: [ 1.5860128  3.7164788  6.7701464  8.569206  10.486626  14.676541 16.499271 ] V

# Mettre votre code ici


maxs = get_peaks_indices(valeurs_avec_bonnes_unites, 1, hauteur_minimum=np.max(valeurs_avec_bonnes_unites[:,1])/40, distance_minumum=(len(valeurs_avec_bonnes_unites[:,1])/15))










# Mettre vos données avec les bonnes unités à la place du None
valeurs_avec_bonnes_unites_determination_des_pics = valeurs_avec_bonnes_unites  # Array de trois colonnes
liste_des_indexes_des_pics = maxs  # Liste de nombres entiers

"""
_____________________________________________________________________________
"""
# Ne pas modifier cette section!!!

print("Estimation des pics:", valeurs_avec_bonnes_unites_determination_des_pics[liste_des_indexes_des_pics, 0])

# La figure obtenue devrait correspondre à celle de figures_exemple/estimation_des_pics_multi_pics
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
u2=3.00
print(f"W experimental: {np.mean([1.789,1.764,1.817,1.791,1.809])} +/- {np.std([1.789,1.764,1.817,1.791,1.809])}")
w_exp=np.mean([1.789,1.764,1.817,1.791,1.809])
def find_combination(pic):
    """
    pic = u1 (v)
    """
    energy = pic+u2-w_exp
    # get upper bounds for each component (lower bounds are always 0):
    #        [     h_bound,           k_bound,            l_bound     ]
    bounds = [round(energy/4.89),round(energy/5.46),round(energy/6.70)]

    # Calculate all possible energy combinations
    energies = []
    delta_energy_sq = []
    linear_combinations = []
    for h in range(bounds[0]+1):
        for k in range(bounds[1]+1):
            for l in range(bounds[2]+1):
                sim_energy = h*4.89+k*5.46+l*6.70
                energies.append(sim_energy)
                delta_energy_sq.append((energy-sim_energy)**2)
                linear_combinations.append([h,k,l])
    # Get best combination
    index = delta_energy_sq.index(min(delta_energy_sq))
    return linear_combinations[index], energies[index]


u1_trouvees = []
for pic in valeurs_avec_bonnes_unites_determination_des_pics[liste_des_indexes_des_pics, 0]:
    print_color(pic)
    lin_comb, en = find_combination(pic)
    u1_trouvees.append(en-u2+w_exp)
    print(lin_comb)
    #print(f"{pic} - [h,k,l]={[h,k,l]}")





plt.figure()
plt.plot(valeurs_avec_bonnes_unites_determination_des_pics[:, 0],
         valeurs_avec_bonnes_unites_determination_des_pics[:, 1],
         label="Courant du pico")
plt.xlabel("Tension entre G1 et le ground [V]")
plt.scatter(valeurs_avec_bonnes_unites_determination_des_pics[liste_des_indexes_des_pics, 0],
            valeurs_avec_bonnes_unites_determination_des_pics[liste_des_indexes_des_pics, 1],
            label="Estimation des pics")
for u1 in u1_trouvees:
    plt.vlines(u1, 0, 1, color="red", linestyles="dashed", linewidth=2)
plt.ylabel("Courant mesuré [nA]")
plt.legend()
plt.subplots_adjust(left=0.158, bottom=0.145, right=0.967, top=0.955, wspace=0.2, hspace=0)
plt.savefig(f"PHY-3002/Graph_FH/estimation_des_pics_fit{extension}.png")
plt.show()


