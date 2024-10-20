import numpy as np
import matplotlib.pyplot as plt


c = 299792458 # m/s
m_e = 9.109E-31 # kg
e = 1.602E-19 # C
epsilon_0 = 8.85418782E-12 # F/m
hbar = 1.05457182E-34 # m^2 kg /s
G = 6.67430E-11 # m^3 kg^-1 s^-2
M_Terre = 5.9736E24 # kg
R_Terre = 6371E3

erreur = np.abs(16*np.pi**2*epsilon_0**2*hbar**4/(m_e**2*e**4)*-2*G*M_Terre/(R_Terre**3*c**2))

print(erreur)