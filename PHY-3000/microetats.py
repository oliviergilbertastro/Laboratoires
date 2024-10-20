import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

"""
Système A et B séparés par une paroie non-isolée

"""

def binomial_coeff(a,b):
    print(a,b, factorial(a)/(factorial(b)*factorial(a-b)))
    return factorial(a)/(factorial(b)*factorial(a-b))

class System_AB():
    def __init__(self, E_A, E_B, N_A, N_B):
        self.E_A = E_A
        self.E_B = E_B
        self.N_A = N_A
        self.N_B = N_B
        self.E_tot = E_A+E_B
        self.omega_tot = self.omega_total()

    def omega_total(self):
        return sum([self.omega(i, self.N_A)*self.omega(self.E_tot-i, self.N_B) for i in range(self.E_tot+1)])

    def E_(N):
        return binomial_coeff(E)

    def omega(self, E, N):
        return (N)**E


    def prob_E_A(E_A):
        return 

monSysteme = System_AB(3, 3, 2, 2)
print(monSysteme.omega_tot)