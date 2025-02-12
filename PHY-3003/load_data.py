import numpy as np

data = np.loadtxt("data.txt")

liste_temps_entre_collision = data[:,0]
liste_distance_entre_collision = data[:,1]

print('temps moyen',np.mean(liste_temps_entre_collision))
print('distance moyenne',np.mean(liste_distance_entre_collision))