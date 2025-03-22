import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# --- FONCTIONS DE CHARGEMENT ---
import numpy as np
import matplotlib.pyplot as plt
import os

def load_time(source):
    with open(f"PHY-3004/SPEC_Gamma/{source}.Spe", 'r') as file:
        lines = file.readlines()
    return int(lines[9].split()[0])  # Extract the first number (Live Time)


def load_text(source, max_channels=4096):
    return np.loadtxt(f"PHY-3004/SPEC_Gamma/{source}.Spe", skiprows=12, max_rows=max_channels, dtype=float)

# --- CHARGEMENT DES DONNÉES ---
sources = ['Exam/Cs_137_gain50', 'Exam/Co60_gain50', 'Exam/2e57co', 'Exam/Na22_gain50']

# Mapping pour simplifier les noms des sources pour le titre
source_titles = {
    'Exam/Cs_137_gain50': 'Cs137',
    'Exam/Co60_gain50': 'Co60',
    'Exam/2e57co': 'Co57',
    'Exam/Na22_gain50': 'Na22'
}

# Boucle pour générer et sauvegarder chaque spectre séparément
for source in sources:
    data = load_text(source)
    
    # Création du dossier spécifique pour chaque source si il n'existe pas déjà
    source_folder = f"PHY-3004/SPEC_Gamma/{source}/figure"
    if not os.path.exists(source_folder):
        os.makedirs(source_folder)
    
    # Crée une nouvelle figure pour chaque source
    plt.figure(figsize=(10, 6))
    plt.plot(data, color='black')  # Courbe noire

    # Personnalisation du graphique
    simplified_title = source_titles.get(source, source)  # Utilise le titre simplifié
    plt.xlabel("Canal", fontsize=18)
    plt.ylabel("Intensité(Count)", fontsize=18)
    plt.title(f"Spectre gamma  - {simplified_title}", fontsize=18)
    plt.legend(fontsize=18)

    # Ajustement des tailles de caractères pour les axes
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # Sauvegarde de la figure dans le dossier de la source
    file_name = f"{simplified_title}_spectre.png"  # Utilisation du titre simplifié
    plt.savefig(os.path.join(source_folder, file_name))
    
    # Fermeture de la figure après sauvegarde
    plt.close()
    
print(f"Les graphiques ont été sauvegardés dans le dossier spécifique de chaque source.")


Na22_data = load_text(sources[3])/load_time(sources[3])
Co57_data = load_text(sources[2])/load_time(sources[2])
Co60_data = load_text(sources[1])/load_time(sources[1])
Cs137_data = load_text(sources[0])/load_time(sources[0])

Na22_t = load_time(sources[3])
Co57_t = load_time(sources[2])
Co60_t = load_time(sources[1])
Cs137_t = load_time(sources[0])


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the Gaussian function with a constant C
def gaussienne(x, A, mu, sigma, C):
    return A * np.exp(-0.5 * ((x - mu) / sigma)**2) + C

# Function to perform the Gaussian fit and calculate FWHM
def fit_gaussian(data, title="Gaussian Fit", p0=None, lower_bound=0):
    x = np.arange(len(data))  # Create an x-axis corresponding to the indices of the data
    params, covariance = curve_fit(gaussienne, x, data, p0=p0)
    
    A, mu, sigma, C = params
    print(f"{title} - A: {A}, mu: {mu}, sigma: {sigma}, C: {C}")
    
    # Calculate the FWHM
    FWHM = 2 * np.sqrt(2 * np.log(2)) * sigma
    print(f"{title} - FWHM: {FWHM}")
    
    # Peak position in the original scale
    pic_position = mu + lower_bound
    print(f"{title} - Peak position in original scale: {pic_position}")
    
    # Plot the result
    plt.figure(figsize=(10, 6))
    plt.plot(x, data, label="Data", color='blue')
    plt.plot(x, gaussienne(x, *params), label="Gaussian Fit", color='red', linestyle='--')
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    
    return pic_position, FWHM # Return only the peak position

Co57_122_bounds = (100, 275)
Co57_136_bounds = (275, 350)

Co60_1173_bounds = (2600, 2932)
Co60_1333_bounds = (2932, 3371)

Cs137_661_bounds = (1400, 1800)

# Define bounds for isolating peaks
Na22_511_bounds = (1110, 1550)
Na22_1275_bounds = (2750, 3200)

Na22_data = load_text(sources[3])
Co57_data = load_text(sources[2])
Co60_data = load_text(sources[1])
Cs137_data = load_text(sources[0])










# Isolate the peaks using the defined bounds
Na22_data_511 = Na22_data[Na22_511_bounds[0]:Na22_511_bounds[1]]
Na22_data_1275 = Na22_data[Na22_1275_bounds[0]:Na22_1275_bounds[1]]

Co57_data_122 = Co57_data[Co57_122_bounds[0]:Co57_122_bounds[1]]
Co57_data_136 = Co57_data[Co57_136_bounds[0]:Co57_136_bounds[1]]

Co60_data_1173 = Co60_data[Co60_1173_bounds[0]:Co60_1173_bounds[1]]
Co60_data_1333 = Co60_data[Co60_1333_bounds[0]:Co60_1333_bounds[1]]

Cs137_data_661 = Cs137_data[Cs137_661_bounds[0]:Cs137_661_bounds[1]]

# Extract the peak positions for each isolated peak
position_Na22_511, FWHM_Na22_511 = fit_gaussian(Na22_data_511, "Gaussian Fit on Na22_data_511", p0=[max(Na22_data_511), np.argmax(Na22_data_511), 10, 0], lower_bound=Na22_511_bounds[0])
position_Na22_1275, FWHM_Na22_1275 = fit_gaussian(Na22_data_1275, "Gaussian Fit on Na22_data_1275", p0=[max(Na22_data_1275), np.argmax(Na22_data_1275), 10, 0], lower_bound=Na22_1275_bounds[0])

position_Co57_122, FWHM_Co57_122 = fit_gaussian(Co57_data_122, "Gaussian Fit on Co57_data_122", p0=[max(Co57_data_122), np.argmax(Co57_data_122), 10, 0], lower_bound=Co57_122_bounds[0])
position_Co57_136, FWHM_Co57_136 = fit_gaussian(Co57_data_136, "Gaussian Fit on Co57_data_136", p0=[max(Co57_data_136), np.argmax(Co57_data_136), 10, 0], lower_bound=Co57_136_bounds[0])

position_Co60_1173, FWHM_Co60_1173 = fit_gaussian(Co60_data_1173, "Gaussian Fit on Co60_data_1173", p0=[max(Co60_data_1173), np.argmax(Co60_data_1173), 10, 0], lower_bound=Co60_1173_bounds[0])
position_Co60_1333, FWHM_Co60_1333 = fit_gaussian(Co60_data_1333, "Gaussian Fit on Co60_data_1333", p0=[max(Co60_data_1333), np.argmax(Co60_data_1333), 10, 0], lower_bound=Co60_1333_bounds[0])

position_Cs137_661 , FWHM_Cs137_661= fit_gaussian(Cs137_data_661, "Gaussian Fit on Cs137_data_661", p0=[max(Cs137_data_661), np.argmax(Cs137_data_661), 10, 0], lower_bound=Cs137_661_bounds[0])


# Maintenant, tu as les positions du pic pour chaque source dans des variables distinctes.
print("Positions des pics :")
print("Position Na22 511:", position_Na22_511)
print("Position Na22 1275:", position_Na22_1275)
print("Position Co57 122:", position_Co57_122)
print("Position Co57 136:", position_Co57_136)
print("Position Co60 1173:", position_Co60_1173)
print("Position Co60 1333:", position_Co60_1333)
print("Position Cs137 661:", position_Cs137_661)




import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Définir une fonction linéaire pour le fit (sans intercept)
def linear(x, a):
    return a * x

# Positions des pics (canaux)
positions = np.array([position_Co57_122, position_Co57_136, position_Cs137_661, position_Co60_1173, position_Co60_1333])

# Énergies en MeV correspondantes (ordre croissant)
energies = np.array([0.122, 0.136, 0.661, 1.173, 1.333])

# Largeur à mi-hauteur (FWHM) pour chaque pic, utilisée comme incertitude (en x)
fwhm_x = np.array([FWHM_Co57_122, FWHM_Co57_136, FWHM_Cs137_661, FWHM_Co60_1173, FWHM_Co60_1333])

# Effectuer le fit linéaire avec curve_fit
params, covariance = curve_fit(linear, positions, energies, sigma=fwhm_x, absolute_sigma=True)

# Extraire la pente du fit
slope = params[0]

# Calculer l'erreur standard du fit (écart-type)
perr = np.sqrt(np.diag(covariance))

# Calculer la valeur du R^2
residuals = energies - linear(positions, *params)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((energies - np.mean(energies))**2)
r_squared = 1 - (ss_res / ss_tot)

# Tracer les données avec barres d'erreur (en x) et la régression linéaire
plt.figure(figsize=(10, 6))
plt.errorbar(positions, energies, xerr=fwhm_x, fmt='o', label="Pics expérimentaux", color='black', capsize=5)
plt.plot(positions, linear(positions, *params), label=f"Régression linéaire", color='black', linestyle='--')

# Ajouter le texte avec l'équation de la régression et la valeur de $R^2$
equation_text = f"$y = {slope:.4f} \cdot x$"
r2_text = f"$R^2 = {r_squared:.4f}$"
plt.text(0.5, 0.8, equation_text, transform=plt.gca().transAxes, fontsize=18, color='black')
plt.text(0.5, 0.75, r2_text, transform=plt.gca().transAxes, fontsize=18, color='black')

# Personnaliser le graphique
plt.xlabel('Position du pic (canal)', fontsize=18)
plt.ylabel('Énergie (MeV)', fontsize=18)
plt.title('Régression linéaire des positions des pics en fonction de l\'énergie', fontsize=18)
plt.legend(fontsize=18)
plt.grid(True)

# Ajuster la taille des graduations (ticks)
plt.tick_params(axis='both', labelsize=18)

plt.show()


# Afficher les résultats dans la console
print(f"Pente de la régression linéaire (a) : {slope:.4f}")
print(f"Erreur standard du coefficient de pente : {perr[0]:.4f}")
print(f"$R^2$ : {r_squared:.4f}")



# Pente du fit linéaire (a) que tu as trouvé précédemment
pente = slope*1000  # Remplace "slope" par la valeur de la pente de ton fit linéaire

# Appliquer le facteur de conversion de la pente à chaque position de pic

energies_MeV = np.array([0.122, 0.136, 0.511, 0.661, 1.173, 1.275, 1.333])  
position_Na22_511_converted = position_Na22_511 * pente
position_Na22_1275_converted = position_Na22_1275 * pente

position_Co57_122_converted = position_Co57_122 * pente
position_Co57_136_converted = position_Co57_136 * pente

position_Co60_1173_converted = position_Co60_1173 * pente
position_Co60_1333_converted = position_Co60_1333 * pente

position_Cs137_661_converted = position_Cs137_661 * pente

# Convertir les FWHM en appliquant la pente
FWHM_Na22_511_converted = FWHM_Na22_511 * pente
FWHM_Na22_1275_converted = FWHM_Na22_1275 * pente

FWHM_Co57_122_converted = FWHM_Co57_122 * pente
FWHM_Co57_136_converted = FWHM_Co57_136 * pente

FWHM_Co60_1173_converted = FWHM_Co60_1173 * pente
FWHM_Co60_1333_converted = FWHM_Co60_1333 * pente

FWHM_Cs137_661_converted = FWHM_Cs137_661 * pente

# Positions de référence (en keV)
positions_ref = {
    'Na22_511': 511,
    'Na22_1275': 1275,
    'Co57_122': 122,
    'Co57_136': 136,
    'Co60_1173': 1173,
    'Co60_1333': 1333,
    'Cs137_661': 661
}

# Calcul de l'écart relatif
def calcul_ecart_relatif(position_mesuree, position_ref):
    return abs(position_mesuree - position_ref) / position_ref * 100

# Positions mesurées converties
positions_mesurees = {
    'Na22_511': position_Na22_511_converted,
    'Na22_1275': position_Na22_1275_converted,
    'Co57_122': position_Co57_122_converted,
    'Co57_136': position_Co57_136_converted,
    'Co60_1173': position_Co60_1173_converted,
    'Co60_1333': position_Co60_1333_converted,
    'Cs137_661': position_Cs137_661_converted
}

# Calculer et afficher l'écart pour chaque position
for key, pos_mesuree in positions_mesurees.items():
    pos_ref = positions_ref[key]
    ecart = calcul_ecart_relatif(pos_mesuree, pos_ref)
    print(f"Écart relatif pour {key}: {ecart:.2f}%")


# Imprimer les résultats pour vérifier les valeurs converties
print(f"Position Na22 511 convertie: {position_Na22_511_converted}")
print(f"Position Na22 1275 convertie: {position_Na22_1275_converted}")

print(f"Position Co57 122 convertie: {position_Co57_122_converted}")
print(f"Position Co57 136 convertie: {position_Co57_136_converted}")

print(f"Position Co60 1173 convertie: {position_Co60_1173_converted}")
print(f"Position Co60 1333 convertie: {position_Co60_1333_converted}")

print(f"Position Cs137 661 convertie: {position_Cs137_661_converted}")

# Afficher les FWHM convertis
print(f"FWHM Na22 511 converti: {FWHM_Na22_511_converted}")
print(f"FWHM Na22 1275 converti: {FWHM_Na22_1275_converted}")

print(f"FWHM Co57 122 converti: {FWHM_Co57_122_converted}")
print(f"FWHM Co57 136 converti: {FWHM_Co57_136_converted}")

print(f"FWHM Co60 1173 converti: {FWHM_Co60_1173_converted}")
print(f"FWHM Co60 1333 converti: {FWHM_Co60_1333_converted}")

print(f"FWHM Cs137 661 converti: {FWHM_Cs137_661_converted}")




# --- DÉCROISSANCE RADIOACTIVE ---
T_Co57_days, T_Co60_years, T_Cs137_years = 271.79, 5.27, 30.17
T_Co57_s = T_Co57_days * 24 * 3600
T_Co60_s = T_Co60_years * 365.25 * 24 * 3600
T_Cs137_s = T_Cs137_years * 365.25 * 24 * 3600
lambda_Co57, lambda_Co60, lambda_Cs137 = np.log(2)/T_Co57_s, np.log(2)/T_Co60_s, np.log(2)/T_Cs137_s
date_initiale, date_finale = datetime(2021, 1, 1), datetime(2025, 2, 24)
t_seconds = (date_finale - date_initiale).total_seconds()
A0 = 1e-6 * 3.7e10
A0137 = 0.25e-6 * 3.7e10
A_Co57, A_Co60, A_Cs137 = A0 * np.exp(-lambda_Co57 * t_seconds), A0 * np.exp(-lambda_Co60 * t_seconds), A0137 * np.exp(-lambda_Cs137 * t_seconds)
T_Na22_years = 2.6018
T_Na22_s = T_Na22_years * 365.25 * 24 * 3600
lambda_Na22 = np.log(2) / T_Na22_s
A_Na22 = A0 * np.exp(-lambda_Na22 * t_seconds)
 # Vérifie si c'est bien le bon lambda pour Na-22

print('A0', A0)
print('A0137', A0137)





print('A_Co57', A_Co57)
print('A_Co60', A_Co60)
print('A_Cs137', A_Cs137)
print('A_Na22', A_Na22)
# --- BRUIT DE FOND ---
import numpy as np



# --- CÉSium-137 ---
background_Cs137 = Cs137_data[Cs137_661_bounds[1]:]
sum_noise_Cs137 = np.sum(background_Cs137)
sum_noise_Cs137_661 = sum_noise_Cs137 * (Cs137_661_bounds[1] - Cs137_661_bounds[0]) / len(background_Cs137) if len(background_Cs137) > 0 else 0

# --- Cobalt-60 ---
background_Co60 = Co60_data[Co60_1333_bounds[1]:]
sum_noise_Co60 = np.sum(background_Co60)
sum_noise_Co60_1173 = sum_noise_Co60 * (Co60_1173_bounds[1] - Co60_1173_bounds[0]) / len(background_Co60) if len(background_Co60) > 0 else 0
sum_noise_Co60_1333 = sum_noise_Co60 * (Co60_1333_bounds[1] - Co60_1333_bounds[0]) / len(background_Co60) if len(background_Co60) > 0 else 0

# --- Cobalt-57 ---
background_Co57 = Co57_data[Co57_136_bounds[1]:]
sum_noise_Co57 = np.sum(background_Co57)
sum_noise_Co57_122 = sum_noise_Co57 * (Co57_122_bounds[1] - Co57_122_bounds[0]) / len(background_Co57) if len(background_Co57) > 0 else 0
sum_noise_Co57_136 = sum_noise_Co57 * (Co57_136_bounds[1] - Co57_136_bounds[0]) / len(background_Co57) if len(background_Co57) > 0 else 0

# --- Sodium-22 ---
background_Na22 = Na22_data[Na22_1275_bounds[1]:]
sum_noise_Na22 = np.sum(background_Na22)
sum_noise_Na22_511 = sum_noise_Na22 * (Na22_511_bounds[1] - Na22_511_bounds[0]) / len(background_Na22) if len(background_Na22) > 0 else 0
sum_noise_Na22_1275 = sum_noise_Na22 * (Na22_1275_bounds[1] - Na22_1275_bounds[0]) / len(background_Na22) if len(background_Na22) > 0 else 0

# --- Affichage des résultats ---
print(f"Cs-137 : Bruit total = {sum_noise_Cs137}, Bruit normalisé 661 keV = {sum_noise_Cs137_661}")
print(f"Co-60  : Bruit total = {sum_noise_Co60}, Bruit normalisé 1173 keV = {sum_noise_Co60_1173}, Bruit normalisé 1333 keV = {sum_noise_Co60_1333}")
print(f"Co-57  : Bruit total = {sum_noise_Co57}, Bruit normalisé 122 keV = {sum_noise_Co57_122}, Bruit normalisé 136 keV = {sum_noise_Co57_136}")
print(f"Na-22  : Bruit total = {sum_noise_Na22}, Bruit normalisé 511 keV = {sum_noise_Na22_511}, Bruit normalisé 1275 keV = {sum_noise_Na22_1275}")

# --- FACTEUR GÉOMÉTRIQUE ---
radius = 2.54  # cm (pour un cristal 2" x 2")
distance = 0.1  # cm
G = (np.pi * radius**2) / (4 * np.pi * distance**2)

print('G', G)

# --- FRACTIONS DE DÉCROISSANCE PAR PIC ---
f_Co57_pic122, f_Co57_pic136 = 0.87, 0.11
f_Co60_pic1173, f_Co60_pic1333 = 0.9986, 0.9986
f_Cs137_pic661 = 0.851
f_Na22_pic1275, f_Na22_pic511 = 0.994, 1.78  

# Valeurs à définir (exemple)


# Définir les variables pour les appels de load_text



# Calcul des sommes et temps pour chaque pic
sum_Na22_511 = np.sum(Na22_data_511)
print('sum_Na22_511', sum_Na22_511)
time_Na22_511 = load_time(sources[3])  # Assuming sources[3] corresponds to Na22
print('time_Na22_511', time_Na22_511)

sum_Na22_1275 = np.sum(Na22_data_1275)
print('sum_Na22_1275', sum_Na22_1275)
time_Na22_1275 = load_time(sources[3])  # Same source, adjust as needed
print('time_Na22_1275', time_Na22_1275)


# Données : Énergies et intensités des pics
energies = np.array([1275, 511])  # en keV
intensities = np.array([np.sum(Na22_data_1275), np.sum(Na22_data_511)])  # Sommes des counts

# Modèle d'ajustement : décroissance exponentielle
def exp_decay(E, I0, lambd):
    return I0 * np.exp(-lambd * E)

# Ajustement des paramètres
params, covariance = curve_fit(exp_decay, energies, intensities, p0=[max(intensities), 0.001])

# Génération des points pour la courbe ajustée
E_fit = np.linspace(min(energies), max(energies), 100)
I_fit = exp_decay(E_fit, *params)

# Calcul du ratio
ratio_na22 = sum_Na22_511 / sum_Na22_1275

# Création du graphique
plt.figure(figsize=(8, 6))
plt.scatter(energies, intensities, color='black', label="Somme des counts dans la région d'air du pic", s=100)
plt.plot(E_fit, I_fit, linestyle='--', color='black', label="Décroissance qualitative")
plt.text(1000, 250000, f"Ratio = {ratio_na22:.2f}", fontsize=18, color='black')

# Ajouter un titre
plt.title("Counts en fonction de l'énergie", fontsize=20, color='black')

# Paramètres d'affichage
plt.xlabel("Énergie (keV)", fontsize=18, color='black')
plt.ylabel("Intensité (counts)", fontsize=18, color='black')
plt.xticks(fontsize=18, color='black')
plt.yticks(fontsize=18, color='black')
plt.legend(fontsize=18)
plt.grid(True, linestyle=':', linewidth=0.7)

# Affichage du graphique
plt.show()




sum_Co57_122 = np.sum(Co57_data_122)
print('sum_Co57_122', sum_Co57_122)
time_Co57_122 = load_time(sources[2])  # Assuming sources[2] corresponds to Co57
print('time_Co57_122', time_Co57_122)

sum_Co57_136 = np.sum(Co57_data_136)
print('sum_Co57_136', sum_Co57_136)
time_Co57_136 = load_time(sources[2])  # Same source, adjust as needed
print('time_Co57_136', time_Co57_136)

sum_Co60_1173 = np.sum(Co60_data_1173)
print('sum_Co60_1173', sum_Co60_1173)
time_Co60_1173 = load_time(sources[1])  # Assuming sources[1] corresponds to Co60
print('time_Co60_1173', time_Co60_1173)

sum_Co60_1333 = np.sum(Co60_data_1333)
print('sum_Co60_1333', sum_Co60_1333)
time_Co60_1333 = load_time(sources[1])  # Same source, adjust as needed
print('time_Co60_1333', time_Co60_1333)

sum_Cs137_661 = np.sum(Cs137_data_661)
print('sum_Cs137_661', sum_Cs137_661)
time_Cs137_661 = load_time(sources[0])  # Assuming sources[0] corresponds to Cs137
print('time_Cs137_661', time_Cs137_661)

# Calcul des epsilons

print('sum_Co57_122', sum_Co57_122)
print('sum_Co57_136', sum_Co57_136)




eps_Co57_122 = (2*sum_Co57_122 - 0*sum_noise_Co57_122) / (A_Co57 * f_Co57_pic122 * G * time_Co57_122)
eps_Co57_136 = (sum_Co57_136 - 10*sum_noise_Co57_136) / (A_Co57 * f_Co57_pic136 * G * time_Co57_136)

eps_Co60_1173 = (sum_Co60_1173 - sum_noise_Co60_1173) / (A_Co60 * f_Co60_pic1173 * G * time_Co60_1173)
eps_Co60_1333 = (sum_Co60_1333 - sum_noise_Co60_1333) / (A_Co60 * f_Co60_pic1333 * G * time_Co60_1333)

eps_Cs137_661 = (sum_Cs137_661 - sum_noise_Cs137_661) / (A_Cs137 * f_Cs137_pic661 * G * time_Cs137_661)

eps_Na22_1275 = (sum_Na22_1275 - sum_noise_Na22_1275) / (A_Na22 * f_Na22_pic1275 * G * time_Na22_1275)
eps_Na22_511 = (sum_Na22_511 - sum_noise_Na22_511) / (A_Na22 * f_Na22_pic511 * G * time_Na22_511)




# --- AFFICHAGE DES RÉSULTATS ---
print(f"Efficacité Co-57 (122 keV) : {eps_Co57_122:.6f}")
print(f"Efficacité Co-57 (136 keV) : {eps_Co57_136:.6f}")
print(f"Efficacité Co-60 (1173 keV) : {eps_Co60_1173:.6f}")
print(f"Efficacité Co-60 (1333 keV) : {eps_Co60_1333:.6f}")
print(f"Efficacité Cs-137 (661 keV) : {eps_Cs137_661:.6f}")
print(f"Efficacité Na-22 (1275 keV) : {eps_Na22_1275:.6f}")
print(f"Efficacité Na-22 (511 keV) : {eps_Na22_511:.6f}")


# --- ÉNERGIES DES PICS EN MeV ---
# --- ÉNERGIES DES PICS EN MeV (triées) ---
energies_MeV = np.array([0.122, 0.136, 0.511, 0.661, 1.173, 1.275, 1.333])  

# --- VALEURS D'EFFICACITÉ (réarrangées selon l'ordre des énergies) ---
efficacités = np.array([
    eps_Co57_122, eps_Co57_136, 
    eps_Na22_511, eps_Cs137_661, 
    eps_Co60_1173, eps_Na22_1275, 
    eps_Co60_1333
])


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- PLOT DE L'EFFICACITÉ ---
plt.figure(figsize=(8, 5))
plt.scatter(energies_MeV, efficacités, color='b', label='Efficacité expérimentale')
plt.plot(energies_MeV, efficacités, 'b--')  # Relier les points pour une meilleure visualisation

plt.xlabel("Énergie (MeV)")
plt.ylabel("Efficacité")
plt.title("Efficacité en fonction de l'énergie")
plt.grid(True)
plt.xscale('log')
plt.yscale('log')

# Modifier le format des ticks en notation décimale
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.xaxis.set_minor_formatter(ticker.NullFormatter())  # Supprime les labels des ticks mineurs
ax.yaxis.set_minor_formatter(ticker.NullFormatter())

plt.show()

# --- NORMALISATION DES EFFICACITÉS ---
efficacités_norm = efficacités / eps_Cs137_661
print('efficacités_norm', efficacités_norm)


# --- PLOT DE L'EFFICACITÉ NORMALISÉE ---
plt.figure(figsize=(10, 6))  # Augmenter la taille de la figure pour une meilleure visibilité
plt.scatter(energies_MeV, efficacités_norm, color='black', label='Efficacité normalisée')
plt.plot(energies_MeV, efficacités_norm, 'k--')  # Relier les points pour une meilleure visualisation

# Modifier les tailles de police des axes, titre et légende
plt.xlabel("Énergie (MeV)", fontsize=26)
plt.ylabel("Efficacité normalisée", fontsize=26)
plt.title("Efficacité normalisée en fonction de l'énergie", fontsize=26)
plt.grid(True)

# Changer la taille de la légende
plt.legend(fontsize=26)

# Modifier l'échelle
plt.xscale('log')
plt.yscale('log')

# Modifier le format des ticks en notation décimale et changer leur taille
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.xaxis.set_minor_formatter(ticker.NullFormatter())  # Supprime les labels des ticks mineurs
ax.yaxis.set_minor_formatter(ticker.NullFormatter())

# Changer la taille des ticks et ajuster la largeur des ticks
plt.tick_params(axis='both', which='major', labelsize=26, width=2)  # Augmenter la taille et la largeur des ticks majeurs
plt.tick_params(axis='both', which='minor', labelsize=26, width=2)  # Augmenter la taille et la largeur des ticks mineurs

plt.show()
