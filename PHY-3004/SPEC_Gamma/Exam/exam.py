import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime

# --- FONCTIONS DE CHARGEMENT ---
def load_time(source):
    with open(f"PHY-3004/SPEC_Gamma/Exam/{source}_gain50.Spe", 'r') as file:
        lines = file.readlines()
    return int("".join(lines[9].split()))

def load_text(source, max_channels=4096):
    return np.loadtxt(f"PHY-3004/SPEC_Gamma/Exam/{source}.Spe", skiprows=12, max_rows=max_channels, dtype=float)

# --- CHARGEMENT DES DONNÉES ---
sources = ['Cs_137_gain50', 'Co60_gain50', 'Co57_gain50']
colors = ['r', 'b', 'g']
plt.figure(figsize=(10, 6))
for source, color in zip(sources, colors):
    data = load_text(source)
    plt.plot(data, label=f"{source} Raw", color=color)
plt.xlabel("Canal")
plt.ylabel("Intensité")
plt.title("Spectres gamma normalisés")
plt.legend()
plt.show()

# --- CALIBRATION ÉNERGÉTIQUE ---
pic57, E57 = [224, 327], [122, 136]
pic60, E60 = [2757, 3111], [1173, 1333]
pic137, E137 = [1600], [661.7]
all_channels, all_energies = np.array(pic57 + pic60 + pic137), np.array(E57 + E60 + E137)
def linear_model(x, a):
    return a * x
popt, _ = curve_fit(linear_model, all_channels, all_energies)
slope = popt[0]
print(f"Slope = {slope:.2f}")

# --- DÉCROISSANCE RADIOACTIVE ---
T_Co57_days, T_Co60_years, T_Cs137_years = 271.79, 5.27, 30.17
T_Co57_s = T_Co57_days * 24 * 3600
T_Co60_s = T_Co60_years * 365.25 * 24 * 3600
T_Cs137_s = T_Cs137_years * 365.25 * 24 * 3600
lambda_Co57, lambda_Co60, lambda_Cs137 = np.log(2)/T_Co57_s, np.log(2)/T_Co60_s, np.log(2)/T_Cs137_s
date_initiale, date_finale = datetime(2021, 1, 1), datetime(2025, 2, 24)
t_seconds = (date_finale - date_initiale).total_seconds()
A0 = 1e-6 * 3.7e10
A_Co57, A_Co60, A_Cs137 = A0 * np.exp(-lambda_Co57 * t_seconds), A0 * np.exp(-lambda_Co60 * t_seconds), A0 * np.exp(-lambda_Cs137 * t_seconds)

# --- BRUIT DE FOND ---
background = load_text("fond")
bruit_de_fond = np.sum(background)

# --- FACTEUR GÉOMÉTRIQUE ---
radius = 2.54  # cm (pour un cristal 2" x 2")
distance = 10  # cm
G = (np.pi * radius**2) / (4 * np.pi * distance**2)

# --- FRACTIONS DE DÉCROISSANCE ---
f_Co57, f_Co60, f_Cs137 = 0.85, 0.99, 0.85  # (valeurs fictives, à vérifier dans Table 3.2)

# --- CALCUL DE L'EFFICACITÉ ---
eps_Co57 = (np.sum(load_text("Co57_gain50")) - bruit_de_fond) / (A_Co57 * f_Co57 * G * load_time("Co57_gain50"))
eps_Co60 = (np.sum(load_text("Co60_gain50")) - bruit_de_fond) / (A_Co60 * f_Co60 * G * load_time("Co60_gain50"))
eps_Cs137 = (np.sum(load_text("Cs_137_gain50")) - bruit_de_fond) / (A_Cs137 * f_Cs137 * G * load_time("Cs_137_gain50"))

print(f"Efficacité Co-57 : {eps_Co57:.6f}")
print(f"Efficacité Co-60 : {eps_Co60:.6f}")
print(f"Efficacité Cs-137 : {eps_Cs137:.6f}")
