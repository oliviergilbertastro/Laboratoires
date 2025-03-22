import numpy as np

activite = 0.331E-6 # Ci
activite = activite * 3.7E10 # Bq (désintégration/s)
print(f"Activité: {activite} désintégration/s")
# On divise par le nombre de stéradians dans une sphère complète
activite_par_sr = activite/(4*np.pi)
print(f"Activité/sr: {activite_par_sr} désintégration/s/sr")


R_DETECTOR = 5.08/2 #cm
DISTANCE_SOURCE = 13 # cm
AREA_DETECTOR = R_DETECTOR**2*np.pi
AREA_SPHERE_AT_DETECTOR = 4*np.pi*DISTANCE_SOURCE**2

SOLID_ANGLE_DETECTOR = AREA_DETECTOR/AREA_SPHERE_AT_DETECTOR*4*np.pi

print(SOLID_ANGLE_DETECTOR)
print(f"Angle solide détecteur: {SOLID_ANGLE_DETECTOR} sr")

activite_par_detector = activite_par_sr*SOLID_ANGLE_DETECTOR

print(f"Activité/détecteur: {activite_par_detector} désintégration/s/détecteur")

def read_spec(detecteur="fixe", deg=0):
    return np.loadtxt(f"PHY-3004/Annihilation/Data/angles/{detecteur}_Na22_{deg}.txt", skiprows=15, max_rows=4096, dtype=float)

