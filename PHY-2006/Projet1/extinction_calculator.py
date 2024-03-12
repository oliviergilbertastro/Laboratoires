import numpy as np
import matplotlib.pyplot as plt


def mag_to_percentage(extinction_coeff, observation_angle=0):
    airmass = 1/np.cos(observation_angle)
    return 1-10**(-0.4*extinction_coeff*airmass)

def correction_percentage():
    """
    """

print(mag_to_percentage(0.6, observation_angle=45/180*np.pi))