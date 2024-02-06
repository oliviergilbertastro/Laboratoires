import numpy as np
import matplotlib.pyplot as plt

#Définir les fonctions
def calculate_n_squared(delta, F, B):
    
    return np.sqrt(1/2*((1+(F*delta)/(delta**2+B**2))+np.sqrt(1+((2*F)/(delta**2+B**2))*(delta+F/2))))

def alpha_calculer(n, F, B, k_0, delta):
    alpha = (k_0)/( n)*((F * B)/(delta**2 + B**2))

    return alpha

#Définir les variables
c = 299792458
omega_0 = 1
omega = np.linspace(0, 2 * omega_0, 1000)
k_0 = omega/(c)
# cas 1
gamma_1 = 0.1
F = 0.1  
delta_1 = omega_0**2 - omega**2
B_1 = gamma_1 * omega



n_squared_values_1 = calculate_n_squared(delta_1, F, B_1)
alpha_1 = alpha_calculer(n_squared_values_1, F, B_1, k_0, delta_1)

#cas 2
gamma_2 = 0.01
B_2 = gamma_2 * omega


n_squared_values_2 = calculate_n_squared(delta_1, F, B_2)

alpha_2 = alpha_calculer(n_squared_values_2, F, B_2, k_0, delta_1)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(omega, n_squared_values_1, label='γ = 0.1')
plt.plot(omega, n_squared_values_2, label='γ = 0.01')
plt.xlabel('ω(rad/s)')
plt.ylabel('n')
plt.title('n(ω)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(omega, alpha_1, label='γ = 0.1')
plt.plot(omega, alpha_2, label='γ = 0.01')
plt.xlabel('ω(rad/s) ')
plt.ylabel('α(1/m)')
plt.title('α(ω)')
plt.legend()

plt.tight_layout()
plt.show()

