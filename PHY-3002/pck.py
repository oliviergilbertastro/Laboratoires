import numpy as np
import matplotlib.pyplot as plt





lam = 633E-9 #m
d = 3.75E-3 #m
r_41 = 23.41E-12 #m/V
n_o = 1.5220
n_e = 1.4773
l = 4*50E-3 #m

def V_func(Delta):
    return Delta*lam*d/(2*np.sqrt(2)*np.pi*l*r_41*(1/n_o**2+1/n_e**2)**(-3/2))

print(V_func(np.pi))



tensions = np.linspace(-200,200,10000)

def I_I0(V, delta_0=0):
    return 0.5*(1-np.cos(delta_0+np.pi*V/V_func(np.pi)))

ax = plt.subplot(111)
plt.plot(tensions, I_I0(tensions), linewidth=2)
plt.xlabel("Tension [V]", fontsize=17)
plt.ylabel("$I/I_0$", fontsize=17)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
plt.show()