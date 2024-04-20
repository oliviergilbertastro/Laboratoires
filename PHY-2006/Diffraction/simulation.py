import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap, ListedColormap
from scipy.special import j1
from photutils.datasets import make_noise_image


def patron_circulaire(x, y, lambda1, res, taille, ouverture=0.2E-3, distance=37E-2):
    ratio = taille/res
    r = ouverture/2
    theta = np.arctan(np.sqrt(x**2+y**2)*ratio/distance)
    argument = 2*np.pi*r*np.sin(theta)/lambda1
    return ((2*j1(argument))/(argument))**2

class Diffraction():
    def __init__(self, taille, res, lambda1, noise_level, noise_deviation, amplitude, distance_to_screen, ouverture):
            self.res = res
            self.taille = taille
            self.lambda1 = lambda1
            self.distance_to_screen = distance_to_screen
            self.ouverture = ouverture
            ny = nx = self.res
            y, x = np.mgrid[-ny/2:ny/2, -nx/2:nx/2]
            noise = np.array(make_noise_image((ny, nx), distribution='gaussian', mean=noise_level, stddev=noise_deviation, seed=None))
            self.data = np.zeros(noise.shape) + noise
            self.data = np.where(self.data > 0, self.data, 0)
            self.data += patron_circulaire(x,y, lambda1, self.res, self.taille, ouverture=self.ouverture, distance=self.distance_to_screen)*(amplitude)
            self.data = np.nan_to_num(self.data, nan=1)

    def show(self):
        print(np.nanmedian(self.data))
        x1,x2,y1,y2 = -self.taille/2*100, self.taille/2*100, -self.taille/2*100, self.taille/2*100
        fig = plt.gcf()
        fig.set_size_inches(12, 6)
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122, sharex=ax1)
        ticklabels = ax1.get_xticklabels()
        ticklabels.extend( ax1.get_yticklabels() )
        ticklabels.extend( ax2.get_xticklabels() )
        ticklabels.extend( ax2.get_yticklabels() )
        for label in ticklabels:
            label.set_fontsize(14)
        N = 256
        vals = np.ones((N, 4))
        vals[:, 0] = np.linspace(0/256, 1, N)
        vals[:, 1] = np.linspace(0/256, 0/256, N)
        vals[:, 2] = np.linspace(0/256, 0/256, N)
        custom_cmap = ListedColormap(vals)
        sim = ax1.imshow(self.data, extent=[x1,x2,y1,y2], vmin=0.003, vmax=0.04, cmap=custom_cmap)
        #fig.colorbar(sim, ax=ax1)
        ax1.set_xlabel(r'Distance (cm)', size=17)
        ax1.set_ylabel(r'Distance (cm)', size=17)
        #plt.show()

        data = np.array(self.data[:,int(self.res/2)])/np.nanmax(self.data[:,int(self.res/2)])
        angles = []
        for i in range(int(-self.res/2), int(self.res/2)):
            angles.append(i*self.taille/self.res*100)
        ax2.plot(angles, data)
        ax2.set_xlabel(r'Distance (cm)', size=17)
        ax2.set_ylabel(r'Intensité ($I/I_0$)', size=17)
        plt.suptitle(f'$\lambda$={self.lambda1*1E9}nm, DIA={self.ouverture*1000}mm, '+ r'D$_\mathrm{écran}$'+f'={self.distance_to_screen*100}cm, résolution=({self.res}x{self.res})', size=18)
        plt.show()

#res=1000-10000 est le meilleur rapport qualité/prix
image = Diffraction(
                    taille=0.01, #en mètres
                    res=2000,
                    lambda1=650E-9, #laser de 650nm
                    noise_level=0,
                    noise_deviation=0.01,
                    amplitude=1,
                    distance_to_screen=0.255, #en mètres
                    ouverture=2E-4 #en mètres
                    )
image.show()

