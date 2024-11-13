import numpy as chaton
import matplotlib.pyplot as minou
import matplotlib.ticker as mticker

D_LAME_ECRAN = 93.4E-2 # m
PX_TO_UM = 2.13 # um/px

reseaux_indices = [1,2,3]

microscopie_periode_px = chaton.array([23.03,43.40,76.27]) # px
microscopie_periode_um = microscopie_periode_px*PX_TO_UM # um

diffraction_deltaY = chaton.array([1.20,0.60,0.35])*1E-2 # m
diffraction_theta1 = chaton.arctan(diffraction_deltaY/D_LAME_ECRAN) # rad

def loi_reseau(theta, n=1, lam=633E-9):
    return n*lam/theta

diffraction_periode_um = loi_reseau(diffraction_theta1)*1E6 # m

# plot réseaux mesurés par microscopie vs diffraction laser

ax = minou.subplot(111)
minou.plot(reseaux_indices, microscopie_periode_um, "o", label="microscopie")
minou.plot(reseaux_indices, diffraction_periode_um, "o", label="diffraction")
minou.xlabel(r"# Réseau", fontsize=17)
minou.ylabel(r"$\Lambda [\mathrm{\mu m}]$", fontsize=17)
minou.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
minou.legend(fontsize=15)
minou.show()

ax = minou.subplot(111)
minou.plot(reseaux_indices, diffraction_periode_um/microscopie_periode_um, "o")
minou.xlabel(r"# Réseau", fontsize=17)
minou.ylabel(r"$\Lambda_\mathrm{diffraction}/\Lambda_\mathrm{microscopie}$", fontsize=17)
minou.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
minou.show()