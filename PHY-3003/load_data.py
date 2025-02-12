import numpy as np

data = np.loadtxt("data.txt")

liste_temps_entre_collision = data[:,0]
liste_distance_entre_collision = data[:,1]
liste_distance_x_entre_collision, liste_distance_y_entre_collision, liste_distance_z_entre_collision = data[:,2], data[:,3],data[:,4]
print('temps moyen',np.mean(liste_temps_entre_collision))
print('distance moyenne',np.mean(liste_distance_entre_collision))


liste_vitesse_entre_collision = liste_distance_entre_collision/liste_temps_entre_collision


print('vitesse moyenne', np.mean(liste_vitesse_entre_collision))




liste_vx_entre_collision = liste_distance_x_entre_collision/liste_temps_entre_collision
liste_vy_entre_collision = liste_distance_y_entre_collision/liste_temps_entre_collision
liste_vz_entre_collision = liste_distance_z_entre_collision/liste_temps_entre_collision


print('vitesse en x moyenne', np.mean(liste_vx_entre_collision))
print('vitesse en y moyenne', np.mean(liste_vy_entre_collision))
print('vitesse en z moyenne', np.mean(liste_vz_entre_collision))

vmoy=  np.sqrt(np.mean(liste_vx_entre_collision)**2+np.mean(liste_vy_entre_collision)**2+np.mean(liste_vz_entre_collision)**2)

print('vitesse moyenne', (vmoy))