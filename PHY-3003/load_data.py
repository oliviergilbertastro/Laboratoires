from vpython import *
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
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