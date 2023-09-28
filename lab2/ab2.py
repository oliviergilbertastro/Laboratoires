import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

ddp = [8.310, 8.544, 8.419, 8.522]
resistance_circuit = [550.33, 1516.3, 905.5, 1355.7]



def kirchoff(r_circuit, I_total, r_inconnue):
    return (r_circuit+r_inconnue)*I_total


res = curve_fit(kirchoff, resistance_circuit, ddp)[0]
print("Intensité du courant [A]: ", res[0])
print("Résistance interne [ohm]: ", res[1])

x = np.linspace(0, 2000, 200)


plt.plot(resistance_circuit, ddp, "o", label='data')
plt.plot(x, kirchoff(x, res[0], res[1]), label='fit')
plt.suptitle(r'R$_\mathrm{interne}$'+f' = {res[1]}$\Omega$', size=17)
plt.legend()
plt.xlabel(r'Résistance circuit [$\Omega$]', size=17)
plt.ylabel(r'Différence de potentiel aux bornes AB [V]', size=17)
plt.show()