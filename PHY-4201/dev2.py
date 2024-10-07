from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()


def pos(t, a=0, b=10, c=1):
    return (np.exp(a*t)-1, np.sin(b*t), c*t)

positions = []
times = np.linspace(0,1,1000)
for i in range(len(times)):
    positions.append(pos(times[i]))
positions = np.array(positions)
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(positions[:,0], positions[:,1], positions[:,2], 'gray')
plt.show()