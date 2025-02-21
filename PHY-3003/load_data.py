from vpython import *
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Charger les données depuis le fichier
module = 0.05
orientation = 'x'
filename = f"/Users/xavierboucher/Desktop/Laboratoires/PHY-3003/data_vec_xy_with_module_{module}_{orientation}.txt"
data = np.loadtxt(filename)

# Extraire les colonnes
t = data[:, 0]  # Temps
x = data[:, 1]  # Coordonnée x
y = data[:, 2]  # Coordonnée y

# Créer les figures
fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# Tracer x(t)
axs[0].plot(t, x, label="x(t)", color="b")
axs[0].set_ylabel("x")
axs[0].legend()
axs[0].grid()

# Tracer y(t)
axs[1].plot(t, y, label="y(t)", color="r")
axs[1].set_xlabel("Temps (s)")
axs[1].set_ylabel("y")
axs[1].legend()
axs[1].grid()

# Afficher la figure
plt.tight_layout()
plt.show()

print(path_object)
data = np.loadtxt("/Users/xavierboucher/Desktop/Laboratoires/PHY-3003/data.txt")

temps_entre_collision = data[:,0]
distance_entre_collision = data[:,1]
distance_x_entre_collision = data[:,2]
distance_y_entre_collision = data[:,3]
distance_z_entre_collision = data[:,4]

vitesse_moyenne = vector(np.sum(distance_x_entre_collision),np.sum(distance_y_entre_collision),np.sum(distance_z_entre_collision))/np.sum(temps_entre_collision)
print("MOI:")
print(f"Vitesse moyenne: {vitesse_moyenne} m/s")
print(f"Vitesse moyenne scalaire: {np.sum(distance_entre_collision)/np.sum(temps_entre_collision)} m/s")
print(f"Vitesse moyenne scalaire: {vitesse_moyenne.mag} m/s")




data = np.loadtxt("/Users/xavierboucher/Desktop/Laboratoires/PHY-3003/data1.txt")

temps_entre_collision = data[:,0]
distance_entre_collision = data[:,1]
distance_x_entre_collision = data[:,2]
distance_y_entre_collision = data[:,3]
distance_z_entre_collision = data[:,4]

vitesse_moyenne = vector(np.sum(distance_x_entre_collision),np.sum(distance_y_entre_collision),np.sum(distance_z_entre_collision))/np.sum(temps_entre_collision)
print("TOI:")
print(f"Vitesse moyenne: {vitesse_moyenne} m/s")
print(f"Vitesse moyenne scalaire: {np.sum(distance_entre_collision)/np.sum(temps_entre_collision)} m/s")
print(f"Vitesse moyenne scalaire: {vitesse_moyenne.mag} m/s")