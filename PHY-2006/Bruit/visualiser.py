import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_values_from_file(filename):
    values = pd.read_csv(filename, delimiter="\t", decimal=",", skiprows=0)
    return values
data = get_values_from_file('PHY-2006/Bruit/labBruit1.lvm')
plt.plot(data)
plt.show()