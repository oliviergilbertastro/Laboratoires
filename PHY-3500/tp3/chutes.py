import numpy as np
import matplotlib.pyplot as plt

h = 22.86 #m
g = -9.81 #m/sp^2

time = [0, ]
pos = [h, ]
speed = [0, ]

time_res = 0.1

while pos[-1] > 0:
    speed.append(speed[-1]+g*time_res)
    pos.append(pos[-1]+speed[-1]*time_res)
    time.append(time[-1]+time_res)

print(speed[-1]*-3.6)

ax1 = plt.subplot(121)
ax2 = plt.subplot(122, sharex=ax1)
ax1.plot(time, pos, label='Position')
ax2.plot(time, speed, label='Vitesse')
ax1.set_title('Position')
ax2.set_title('Vitesse')
ax1.set_xlabel('Temps [s]')
ax1.set_ylabel('Position [m]')
ax2.set_xlabel('Temps [s]')
ax2.set_ylabel('Vitesse [m/s]')
plt.show()