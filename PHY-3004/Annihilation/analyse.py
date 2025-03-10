import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

R_detector = 5.08 #cm
DISTANCE_SOURCE = 13 # cm

def read_spec(detecteur="fixe", deg=0):
    return np.loadtxt(f"PHY-3004/Annihilation/Data/angles/{detecteur}_Na22_{deg}.txt", skiprows=15, max_rows=4096, dtype=float)

data_coincidences = np.loadtxt("PHY-3004/Annihilation/Data/coincidences.txt", skiprows=2, dtype=int)
angles = data_coincidences[:,0]
N_coincidences = data_coincidences[:,1]

def A_normal_theorique(phi):
    epsilon = 2*np.arccos(DISTANCE_SOURCE*np.sin(phi/2)/R_detector)
    epsilon_0 = 2*np.arccos(0)
    return (epsilon-np.sin(epsilon))/(epsilon_0-np.sin(epsilon_0))

