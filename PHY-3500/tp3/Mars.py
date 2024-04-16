from astropy.time import Time
# éphémérides
import de421
from jplephem import Ephemeris
eph = Ephemeris(de421)
# dates
lancement=Time("2020-07-30")
atterissage=Time("2021-02-18")

position, velocity = eph.position_and_velocity('mars',lancement.jd)
delta_time_seconds = (atterissage.jd-lancement.jd)*24*3600

#On transforme en m et en m/s:
position, velocity = position*1000, velocity*1000/(3600*24)
#print("Position et vitesse initiale (2020-07-30):")
#print(position)
#print(velocity)
#print(delta_time_seconds)

#Résultat attendu:
final_position = eph.position('mars',atterissage.jd)*1000
#print("\nRésultat attendu (2021-02-18):")
#print(final_position)

import numpy as np
import astropy as ap

G = ap.constants.G.value #m^3.kg^-1.s^-2
M_sun = ap.constants.M_sun.value #kg
M_mars = 0.64169E24 #kg


def module(a):
    sum = 0
    for i in a:
        sum += i**2
    return np.sqrt(sum)

def acceleration(r,t):
    '''
    r: array of position [x,y,...]
    t: time
    '''
    return -G*M_sun*np.array(r)/module(r)**3

def func(state_vec):
    r_vec = state_vec[0]
    v_vec = state_vec[1]
    fr_vec = v_vec
    fv_vec = acceleration(r_vec, 0) #Le temps ne change rien à la fonction a=-GM/r^2, donc on utilise t=0 même si techniquement l'accélération change selon le temps puisque r=r(t). La dépendance du temps est implicite.
    return np.array([fr_vec, fv_vec])

def bulirsch_stoer_mars(r0, v0, total_time=3600*24*365.25*5, H=3600*24*7, delta=1E-3):
    '''
    r0: initial position array [x0,y0,...]
    v0: initial velocity array [v0x,v0y,...]
    total_time: total time T of integration
    H: starting timestep, each n iteration of the B-S method will reduce the timestep to H/n

    Calculates an array of state vectors [r,v] for a chosen total integration time with a specific time-step
    '''
    r = np.array([r0, v0])
    r_list = []
    v_list = []
    tpoints = np.arange(0, total_time, H)
    N = len(tpoints)
    for t in tpoints:
        r_list.append(r[0])
        v_list.append(r[1])

    #On commence par faire un pas de H
    n = 1
    r1 = r+0.5*H*func(r)
    r2 = r+H*func(r1)
    R1 = np.empty([1,2,3])
    R1[0] = 0.5*(r1+r2+0.5*H*func(r2))
    error = np.array([2*H*delta, 2*H*delta, 2*H*delta])
    while error[0] > H*delta or error[1] > H*delta or error[2] > H*delta:
        n += 1
        h = H/n

        #Méthode modifiée
        r1 = r+0.5*h*func(r)
        r2 = r+h*func(r1)
        for i in range(n-1):
            r1 += h*func(r2)
            r2 += h*func(r1)

        #On extrapole
        R2 = R1.copy()
        R1 = np.empty([n,2,3])
        R1[0] = 0.5*(r1+r2+0.5*h*func(r2))
        for m in range(1,n):
            epsilon = (R1[m-1]-R2[m-1])/((n/(n-1))**(2*m)-1)
            R1[m] = R1[m-1]+epsilon
        error = np.abs(epsilon[0])
    r = R1[n-1]

    return r_list, v_list, tpoints



position, velocity = np.array(position).reshape((3,)), velocity.reshape((3,))
r_list_mars, v_list_mars, t_list_mars = bulirsch_stoer_mars(position, velocity, total_time=delta_time_seconds, H=3600*20, delta=1E-6)

#Calculate the final position deviation

final_position = np.array(final_position).reshape((3,))

deviation_vector = np.array(r_list_mars[-1])-np.array(final_position)

deviation_distance = module(deviation_vector)
#print(deviation_distance)
#5103382046.7468297393







from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

#PLOT MARS ORBIT
fig = plt.figure(figsize=(16,10))
ax = plt.axes(projection='3d')
print(np.array(r_list_mars)[:100,:])
ax.plot3D(np.array(r_list_mars)[:,0], np.array(r_list_mars)[:,1], np.array(r_list_mars)[:,2], color='blue')
#plt.plot()
ax.plot3D(0,0,0,'o', color='orange')
ax.plot3D(position[0], position[1], position[2],'o', color='green')
ax.plot3D(final_position[0], final_position[1], final_position[2],'o', color='red')
plt.axis('equal')
plt.xlabel('[m]')
plt.ylabel('[m]')
ax.set_zlabel('[m]')
plt.title('Orbite de Mars')
plt.show()