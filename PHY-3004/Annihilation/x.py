

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


def load_text(source, max_channels=4096):
    return np.loadtxt(f"PHY-3004/Annihilation/Data/{source}.Spe", skiprows=12, max_rows=max_channels, dtype=float)

# --- CHARGEMENT DES DONNÉES ---
sources= ['fixe_Co60Co57Cs137', 'mobile_Na22']



# Boucle pour générer et sauvegarder chaque spectre séparément
for source in sources:
    data = load_text(source)

    # Crée une nouvelle figure pour chaque source
    plt.figure(figsize=(10, 6))
    plt.plot(data, color='black')  # Courbe noire

    # Personnalisation du graphique
    
    plt.xlabel("Canal", fontsize=18)
    plt.ylabel("Intensité(Count)", fontsize=18)
    plt.title(f"Spectre gamma ", fontsize=18)
    plt.legend(fontsize=18)

    # Ajustement des tailles de caractères pour les axes
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()


data = load_text(sources[0])
Na22_data = load_text(sources[1])



    






# Définir la fonction gaussienne
def gaussienne(x, A, mu, sigma, B, C):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + B * x + C

# Function to perform the Gaussian fit and calculate FWHM
def fit_gaussian(data, title, source_pic, p0=None, lower_bound=0):
    x = np.arange(len(data))  # Create an x-axis corresponding to the indices of the data
    params, covariance = curve_fit(gaussienne, x, data, p0=p0)
    
    # Extraire les paramètres ajustés
    A, mu, sigma, B, C = params
    A_err, mu_err, sigma_err, B_err, C_err = np.sqrt(np.diag(covariance))  # Calcul des incertitudes des paramètres
    print(f"{title} - A: {A}, mu: {mu}, sigma: {sigma}, B: {B}, C: {C}")
    
    # Calculate the FWHM
    FWHM = 2 * np.sqrt(2 * np.log(2)) * sigma
    print(f"{title} - FWHM: {FWHM}")
    
    # Peak position in the original scale
    pic_position = mu + lower_bound
    print(f"{title} - Peak position in original scale: {pic_position}")
    
    # Tracer le résultat
    plt.figure(figsize=(10, 6))
    plt.plot(x, data, label="Données", color='blue')
    plt.plot(x, gaussienne(x, *params), label=f"{title}", color='red', linestyle='--')

    plt.xlabel('Canaux', fontsize=18)  
    plt.ylabel('Intensité (Counts)', fontsize=18)
    plt.legend(fontsize=16, loc='upper right')
    # Ajouter les incertitudes sur les paramètres avec fontsize=18
    plt.text(0.05, 0.9, f"$A = {A:.4f} \pm {A_err:.4f}$", transform=plt.gca().transAxes, fontsize=14, color='black')
    plt.text(0.05, 0.85, f"$\mu = {mu:.4f} \pm {mu_err:.4f}$", transform=plt.gca().transAxes, fontsize=14, color='black')
    plt.text(0.05, 0.8, f"$\sigma = {sigma:.4f} \pm {sigma_err:.4f}$", transform=plt.gca().transAxes, fontsize=14, color='black')
    plt.text(0.05, 0.75, f"$B = {B:.4f} \pm {B_err:.4f}$", transform=plt.gca().transAxes, fontsize=14, color='black')
    plt.text(0.05, 0.7, f"$C = {C:.4f} \pm {C_err:.4f}$", transform=plt.gca().transAxes, fontsize=14, color='black')
    plt.tick_params(axis='both', labelsize=18)
    save_path="PHY-3004/Annihilation/Figures"  
    save_file = os.path.join(save_path, f"{source_pic.replace(' ', '_')}.png")
    plt.savefig(save_file, dpi=300)
    plt.show()


    
    return pic_position, FWHM # Return only the peak position cette function*

Co57_122_bounds = (290, 400)
Cs137_661_bounds = (1400, 2000)
Co60_1173_bounds = (2780, 3200)
Co60_1333_bounds = (3200, 3600)


















Co57_data_122 = data[Co57_122_bounds[0]:Co57_122_bounds[1]]
Cs137_data_661 = data[Cs137_661_bounds[0]:Cs137_661_bounds[1]]
Co60_data_1173 = data[Co60_1173_bounds[0]:Co60_1173_bounds[1]]
Co60_data_1333 = data[Co60_1333_bounds[0]:Co60_1333_bounds[1]]


# Define bounds for isolating peaks
Na22_511_bounds = (1110, 1550)
Na22_1275_bounds = (2750, 3200)

# Isolate the peaks using the defined bounds
Na22_data_511 = Na22_data[Na22_511_bounds[0]:Na22_511_bounds[1]]
Na22_data_1275 = Na22_data[Na22_1275_bounds[0]:Na22_1275_bounds[1]]



position_Co57_122, FWHM_Co57_122 = fit_gaussian(Co57_data_122, "Ajustement Gaussien", 'Co57_122' ,p0=[max(Co57_data_122), np.argmax(Co57_data_122), 10, 0, 0], lower_bound=Co57_122_bounds[0])
position_Cs137_661, FWHM_Cs137_661 = fit_gaussian(Cs137_data_661, "Ajustement Gaussien ",'Cs137_661' ,p0=[max(Cs137_data_661), np.argmax(Cs137_data_661), 10, 0, 0], lower_bound=Cs137_661_bounds[0])
position_Co60_1173, FWHM_Co60_1173 = fit_gaussian(Co60_data_1173, "Ajustement Gaussien ",'Co60_1173', p0=[max(Co60_data_1173), np.argmax(Co60_data_1173), 10, 0, 0], lower_bound=Co60_1173_bounds[0])
position_Co60_1333, FWHM_Co60_1333 = fit_gaussian(Co60_data_1333, "Ajustement Gaussien ", 'Co60_1333',p0=[max(Co60_data_1333), np.argmax(Co60_data_1333), 10, 0, 0], lower_bound=Co60_1333_bounds[0])
position_Na22_511, FWHM_Na22_511 = fit_gaussian(Na22_data_511, "Ajustement Gaussien", 'Na22_511',p0=[max(Na22_data_511), np.argmax(Na22_data_511), 10, 0, 0], lower_bound=Na22_511_bounds[0])




# Définir une fonction linéaire pour le fit (sans intercept)
def linear(x, a):
    return a * x

# Positions des pics (canaux)

positions = np.array([position_Co57_122,position_Cs137_661,position_Co60_1173 , position_Co60_1333])
# Énergies en MeV correspondantes (ordre croissant)
energies = np.array([0.122, 0.661, 1.173, 1.333])

# Largeur à mi-hauteur (FWHM) pour chaque pic, utilisée comme incertitude (en x)
fwhm_x = np.array([FWHM_Co57_122,FWHM_Cs137_661,FWHM_Co60_1173,FWHM_Co60_1333])

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

x = np.linspace(0, 4000, 100)
# Tracer les données avec barres d'erreur (en x) et la régression linéaire
plt.figure(figsize=(10, 6))
plt.errorbar(positions, energies, xerr=fwhm_x, fmt='o', label="Pics expérimentaux", color='black', capsize=5)
plt.plot(x, linear(x, *params), label=f"Régression linéaire", color='black', linestyle='--')

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


energies*=1000
energies = np.append(energies, 511)
positions = np.append(positions, position_Na22_511)
fwhm_x = np.append(fwhm_x, FWHM_Na22_511)
# Affichage des résultats avec valeur absolue de l'écart et de l'écart en %
for i in range(len(energies)):
    position_ajustee = positions[i] * pente
    fwhm_ajuste = fwhm_x[i] * pente
    ecart = abs(position_ajustee - energies[i])  # Valeur absolue de l'écart
    ecart_pourcentage = abs((ecart / energies[i]) * 100)  # Valeur absolue de l'écart en %

    print(f"Énergie : {energies[i]:.3f} KeV | Position ajustée (KeV) : {position_ajustee:.3f} | "
          f"FWHM ajusté (KeV): {fwhm_ajuste:.3f} | Écart : {ecart:.3f} KeV ({ecart_pourcentage:.2f}%)")
