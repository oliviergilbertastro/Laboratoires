import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
from photutils.datasets import make_noise_image

lambda_laser = 0.33E-6

def fonction_circulaire(x, y, lambda1, res, ouverture=0.4E-3, distance=37E-2):
    r = ouverture
    theta = np.sqrt(np.arcsin(y*res/distance)**2+np.arcsin(x*res/distance)**2)
    argument = 2*np.pi*r*np.sin(theta)/lambda1
    return ((2*j1(argument))/(argument))**2

class Diffraction():
    def __init__(self, res, taille, lambda1, noise_level, noise_deviation, amplitude):
            self.res = res
            self.taille = taille
            self.lambda1 = lambda1
            ny = nx = self.res
            y, x = np.mgrid[-ny/2:ny/2, -nx/2:nx/2]
            noise = make_noise_image((ny, nx), distribution='gaussian', mean=noise_level, stddev=noise_deviation, seed=None)
            self.data = noise
            self.data += fonction_circulaire(x,y, lambda1, self.res)*amplitude

    def show(self):
         plt.imshow(self.data)
         plt.show()


image = Diffraction(100, 1E-7, 0.33E-6, 1, 1, amplitude=1000)
image.show()

