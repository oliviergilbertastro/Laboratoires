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


tension_exp = [-172.5308693668575, -162.38724624457168, -152.2436231222858, -140.07122894172113, -128.9132589851472, -119.78398269714938, -108.62601274057543, -101.52550751085647, -90.36753755428253, -75.15210287085382, -63.99405552457708, -47.764274006860376, -31.53449248914353, -9.218552575995773, 7.011306331423867, 31.35593991314775, 52.6576103817103, 69.90173873371498, 81.05970869028891, 93.23210287085365, 103.37572599313961, 110.4762312228585, 126.70601274057535, 140.90710058971624, 44.54268092800032, 44.54268092800032, -39.64942194285328]
courant_exp = [73.84237305803101, 80.32852397605271, 83.3221262357984, 84.31999999999998, 83.3221262357984, 79.33065021185114, 75.33917418790385, 70.84875176542633, 61.867925953330214, 48.89562411728682, 38.91692454098914, 24.94674894074419, 12.473355437513094, 2.993564194027715, 0.49888929995333137, 5.488239088102134, 17.462724258520932, 31.93182722443715, 42.90840056493642, 52.8871001412341, 62.36687235186048, 68.35407687135191, 76.83597531777669, 81.82532510592554, 12.473355437513094, 12.473355437513094, 18.460578989863492]

indices_tension = []
tensions_exp = tension_exp.copy()
tension_exp.sort()
for i in range(len(tension_exp)):
    indices_tension.append(tensions_exp.index(tension_exp[i]))

tension_exp = np.array(tension_exp) # V
courant_exp = np.array(courant_exp)/84.32 # I/I_0
courant_exp = courant_exp[indices_tension]

ax = plt.subplot(111)
plt.plot(tensions, I_I0(tensions), linewidth=2, label="Théorique")
plt.plot(tension_exp, courant_exp, linewidth=2, label="Expérimentale")
plt.xlabel("Tension [V]", fontsize=17)
plt.ylabel("$I/I_0$", fontsize=17)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
plt.legend()
plt.show()


tensions_fit = np.linspace(np.min(tension_exp),np.max(tension_exp),1000)
def I_I0_fit(V, delta_0, V_pi):
    return 0.5*(1-np.cos(delta_0+np.pi*V/V_pi))

from scipy.optimize import curve_fit

res = curve_fit(I_I0_fit, tension_exp, courant_exp, p0=[9.04, 151])[0]

print(res)

ax = plt.subplot(111)
plt.plot(tensions_fit, I_I0_fit(tensions_fit, res[0], res[1]), linewidth=2, label="Ajustée")
plt.plot(tension_exp, courant_exp, linewidth=2, label="Expérimentale")
plt.xlabel("Tension [V]", fontsize=17)
plt.ylabel("$I/I_0$", fontsize=17)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
plt.legend()
plt.show()