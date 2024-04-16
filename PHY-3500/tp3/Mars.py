from mpl_toolkits import mplot3d

import numpy as np
import astropy as ap

from astropy.time import Time
# éphémérides
import de421
from jplephem import Ephemeris



eph = Ephemeris(de421)
# dates
lancement=Time("2020-07-30")
times_mid = [Time("2020-08-30"), Time("2020-09-30"), Time("2020-10-30"), Time("2020-11-30"), Time("2020-12-30"), Time("2021-01-30")]
atterissage=Time("2021-02-18")

position, velocity = eph.position_and_velocity('mars',lancement.jd)
delta_time_seconds = (atterissage.jd-lancement.jd)*24*3600

#On transforme en m et en m/s:
position, velocity = np.array(position*1000).reshape((3,)), np.array(velocity*1000/(3600*24)).reshape((3,))
print("Position et vitesse initiale (2020-07-30):")
print(position)
print(velocity)

mid_positions_x = []
mid_positions_y = []
mid_positions_z = []
for i in range(len(times_mid)):
  pos = eph.position('mars', times_mid[i].jd)
  posx, posy, posz = np.array(pos*1000).reshape((3,))
  mid_positions_x.append(posx)
  mid_positions_y.append(posy)
  mid_positions_z.append(posz)

#Résultat attendu:
final_position = np.array(eph.position('mars',atterissage.jd)*1000).reshape((3,))
print("\nRésultat attendu (2021-02-18):")
print(final_position)


#Temps entre deux:
print("\nTemps d'intégration")
print(delta_time_seconds)


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

def bulirsch_stoer_mars(r0, v0, total_time=3600*24*365.25*5, H=3600*24*7, delta=1000/(3600*24*365.25)):
  '''
  r0: initial position array [x0,y0,...]
  v0: initial velocity array [v0x,v0y,...]
  total_time: total time of integration
  H: starting timestep, each n iteration of the B-S method will reduce the timestep to h=H/n
  delta: precision in length/unit time, (e.g. delta=1 is a precision of 1meter/second)

  Calculates arrays of position and velocity vectors (r_list and v_list) for a chosen total integration time with a specific time-step
  '''
  r = np.array([r0, v0])
  r_list = []
  v_list = []
  all_r_list = [r0,]
  all_t_list = [0, ]
  tpoints = np.arange(0, total_time, H)
  N = len(tpoints)
  max_n = 1
  for t in tpoints:
    r_list.append(r[0])
    v_list.append(r[1])

    #On commence par faire un pas de H
    n = 1
    r1 = r+0.5*H*func(r)
    r2 = r+H*func(r1)
    R1 = np.empty([1,2,3])
    R1[0] = 0.5*(r1+r2+0.5*H*func(r2))
    error = 2*H*delta
    while error > H*delta:
      n += 1
      h = H/n
      if n > max_n:
        max_n = n

      #Méthode modifiée
      r1 = r+0.5*h*func(r)
      r2 = r+h*func(r1)
      all_r_list.append(r1[0])
      all_t_list.append(t+0.5*h)
      all_r_list.append(r2[0])
      all_t_list.append(t+h)
      for i in range(n-1):
        r1 += h*func(r2)
        r2 += h*func(r1)


      #On extrapole
      R2 = np.copy(R1)
      R1 = np.empty([n,2,3])
      R1[0] = 0.5*(r1+r2+0.5*h*func(r2))
      for m in range(1,n):
        epsilon = (R1[m-1]-R2[m-1])/((n/(n-1))**(2*m)-1)
        R1[m] = R1[m-1]+epsilon
      error = module(epsilon[0])
    r = R1[n-1]
  print('MAX n:', max_n)
  return r_list, v_list, tpoints



N_iter = 10000
r_list_mars, v_list_mars, t_list_mars = bulirsch_stoer_mars(position, velocity, total_time=delta_time_seconds, H=3600, delta=1E-6)

#Calculate the final position deviation

deviation_vector = np.array(r_list_mars[-1])-np.array(final_position)

deviation_distance = module(deviation_vector)





import matplotlib.pyplot as plt


#PLOT MARS ORBIT
fig = plt.figure(figsize=(16,10))
ax = plt.axes(projection='3d')
ax.plot3D(np.array(r_list_mars)[:,0], np.array(r_list_mars)[:,1], np.array(r_list_mars)[:,2], '-')
#plt.plot()
ax.plot3D(0,0,0,'o', color='orange')
ax.plot3D(position[0], position[1], position[2],'o', color='green')
ax.plot3D(final_position[0], final_position[1], final_position[2],'o', color='red')
ax.plot3D(mid_positions_x, mid_positions_y, mid_positions_z,'o', color='purple')
plt.axis('equal')
plt.xlabel('[m]')
plt.ylabel('[m]')
ax.set_zlabel('[m]')
plt.title('Orbite de Mars')
plt.show()

ax1, ax2, ax3 = plt.subplot(221), plt.subplot(222), plt.subplot(223)
ax1.plot(np.array(r_list_mars)[:,0], np.array(r_list_mars)[:,1]) #XY
ax1.plot(position[0], position[1] ,'o', color='green')
ax1.plot(final_position[0], final_position[1],'o', color='red')
ax1.plot(mid_positions_x, mid_positions_y,'o', color='purple')
ax1.set_aspect('equal')
ax2.plot(np.array(r_list_mars)[:,1], np.array(r_list_mars)[:,2]) #YZ
ax2.plot(position[1], position[2] ,'o', color='green')
ax2.plot(final_position[1], final_position[2],'o', color='red')
ax2.plot(mid_positions_y, mid_positions_z,'o', color='purple')
ax2.set_aspect('equal')
ax3.plot(np.array(r_list_mars)[:,0], np.array(r_list_mars)[:,2]) #XZ
ax3.plot(position[0], position[2] ,'o', color='green')
ax3.plot(final_position[0], final_position[2],'o', color='red')
ax3.plot(mid_positions_x, mid_positions_z,'o', color='purple')
ax3.set_aspect('equal')
plt.show()