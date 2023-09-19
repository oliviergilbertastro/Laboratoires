import numpy as np
import matplotlib.pyplot as plt
from pylab import genfromtxt

def get_values_from_file(filename):
    values = genfromtxt(open("swift_xrt_lc.txt", "r"))
    return values

def create_hist(values, combinaison="Inox-Alu", bins=200):
    counts, bins = np.histogram(values, bins=bins)
    plt.stairs(counts, bins)
    plt.show()
    plt.suptitle(f'Histogramme pour la combinaison {combinaison}', size=17)
    plt.xlabel(r'Amplitude [V]', size=17)

