import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

R_DETECTOR = 5.08/2 #cm
DISTANCE_SOURCE = 13 # cm

def read_spec(detecteur="fixe", deg=0):
    return np.loadtxt(f"PHY-3004/Annihilation/Data/angles/{detecteur}_Na22_{deg}.txt", skiprows=15, max_rows=4096, dtype=float)
