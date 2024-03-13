import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import curve_fit
from tqdm import tqdm

h = 6.62607015E-34 #m^2.kg.s^-1
c = 299792458 #m/s
k_B = 1.380649E-23 #m^2.kg.s^-2.K^-1
b = 2.897771955E-3 #m.K

#Best-fitting coefficients are calculated in color_temperature.py from the ZWO ASI 224mc color manual and theoretical formulas
coeffs = [-36756.02512371,285792.83399003,-819480.14776133,1208529.75689052,-884389.37032566,261133.76091327]

def temperature_from_BR(br):
    """
    Returns the blackbody temperature in Kelvin from the B/R ratio
    """
    c1,c2,c3,c4,c5,c6 = coeffs
    return c1+c2*br+c3*br**2+c4*br**3+c5*br**4+c6*br**5

#Process the actual pictures
betelegeuse_im = np.array(Image.open("PHY-2006/Projet1/data/stars_pictures/betelgeuse2.tif"))
bet_red = betelegeuse_im[:, :, 0]
bet_blue = betelegeuse_im[:, :, 2]
bet_ratio = bet_blue/bet_red


ax1 = plt.subplot(131)
ax2 = plt.subplot(132, sharex=ax1, sharey=ax1)
ax3 = plt.subplot(133, sharex=ax1, sharey=ax1)
ax1.imshow(bet_red, cmap='Reds')
ax2.imshow(bet_blue, cmap='Blues')
ax3.imshow(bet_ratio)
plt.suptitle('Betelgeuse')
plt.show()

rigel_im = np.array(Image.open("PHY-2006/Projet1/data/stars_pictures/rigel.tif"))
rig_red = rigel_im[:, :, 0]
rig_blue = rigel_im[:, :, 2]
rig_ratio = rig_blue/rig_red


ax1 = plt.subplot(131)
ax2 = plt.subplot(132, sharex=ax1, sharey=ax1)
ax3 = plt.subplot(133, sharex=ax1, sharey=ax1)
ax1.imshow(rig_red, cmap='Reds')
ax2.imshow(rig_blue, cmap='Blues')
ax3.imshow(rig_ratio)
plt.suptitle('Rigel')
plt.show()

print(np.mean(bet_ratio[585:591,602:608]))
print(np.mean(rig_ratio[534:542,657:667]))

print("Betelgeuse temperature:", temperature_from_BR(0.483), 'K')
print("Rigel temperature:", temperature_from_BR(0.88), 'K')