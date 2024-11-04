from outils_analyse.fits import gaussian_fit, gaus, round_any

"""
_____________________________________________________________________________
"""
# TODO: Copiez le code de l'analyse légère pour un pic simple ici.
# Mettre votre code ici






"""
_____________________________________________________________________________
"""

# TODO: Faire une régression gaussienne sur chacun des pics et calculer le potentiel de contact.
#  Pour arrondir les valeurs selons leurs incertitudes, regardez la documentation de sigfig.
#  Les résultats devraient être similaires à
#    Pic 1: Moyenne: 2.04 ± 0.02 STD: -0.44 ± 0.02 Amplitude: 0.083 ± 0.004
#    Pic 2: Moyenne: 6.91 ± 0.01 STD: 0.82 ± 0.01 Amplitude: 0.459 ± 0.006
#    Pic 3: Moyenne: 12.00 ± 0.01 STD: 1.09 ± 0.01 Amplitude: 0.96 ± 0.01
#    Pic 4: Moyenne: 17.14 ± 0.01 STD: 1.26 ± 0.01 Amplitude: 1.365 ± 0.007

# Mettre votre code ici









# Mettre les paramètres des fits gaussiens pour chaque pic [Amplitude, Moyenne, STD]
peak1 = None  # Arrray de 3 éléments
peak2 = None  # Arrray de 3 éléments
peak3 = None  # Arrray de 3 éléments
peak4 = None  # Arrray de 3 éléments


"""
_____________________________________________________________________________
"""
# Ne pas modifier cette section!!!

def rounding_peaks(peaks):
    all_values = []
    for i in range(0, 3):
        all_values.append(round_any(peaks[0][i], uncertainty=peaks[1][i]))

    return all_values

print("Pic 1:", f"Moyenne: {rounding_peaks(peak1)[1]}",
      f"STD: {rounding_peaks(peak1)[2]}",
      f"Amplitude: {rounding_peaks(peak1)[0]}")
print("Pic 2:", f"Moyenne: {rounding_peaks(peak2)[1]}",
      f"STD: {rounding_peaks(peak2)[2]}",
      f"Amplitude: {rounding_peaks(peak2)[0]}")
print("Pic 3:", f"Moyenne: {rounding_peaks(peak3)[1]}",
      f"STD: {rounding_peaks(peak3)[2]}",
      f"Amplitude: {rounding_peaks(peak3)[0]}")
print("Pic 4:", f"Moyenne: {rounding_peaks(peak4)[1]}",
      f"STD: {rounding_peaks(peak4)[2]}",
      f"Amplitude: {rounding_peaks(peak4)[0]}")

"""
_____________________________________________________________________________
"""

# TODO: Faire un graphique digne d'un article qui contient l'ensemble des données de courants en fonction de la tension,
#  les emplacements approximatifs des maximums et les différents fits gaussiens effectués.
#  Ça devrait ressembler à la figure exemple_de_fichiers/exemple_graphique_f_et_h

# Mettre votre code ici
