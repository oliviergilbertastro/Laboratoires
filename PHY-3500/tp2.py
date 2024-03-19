import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



#A = QR
I_i = 0
H_mi = 0

def sgn(x):
    return -1 if x < 0 else 1



def householder_qr(A):
    """
    Effectue la décomposition QR d'une matrice A à l'aide de la méthode de Householder.

    Args:
    A: Une matrice numpy.

    Returns:
    Q: La matrice orthogonale.
    R: La matrice triangulaire supérieure.
    """

    A = np.matrix(A)
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()

    for i in range(n):
        # Construire le vecteur de Householder pour la colonne i
        x = R[i:, i].T

        #On calcule la norme pour normaliser
        norm = np.sqrt(np.sum(np.square(x)))

        # Vecteur unitaire e_1
        e_1 = np.zeros_like(x)
        e_1[0, 0] = 1

        # Assemble le premier élément du vecteur v
        v_1 = norm*e_1

        # Signe du premier élément de x
        if x[0, 0] < 0:
            v_1 = -v_1

        #Assemble v
        v = (x + v_1)
        print("v_mi shape:", v.shape)
        # Calculer la matrice H_mi
        buff = np.eye(m-i) - ((2 * np.matmul(v.T, v)) / np.matmul(v, v.T))

        H = np.eye(n)
        H[i:, i:] = buff

        R = np.matmul(H, R)
        Q = np.matmul(Q, H)

    return Q, R

# Test householder_qr
A = np.matrix("10, 1 ; 2, 3")
A = [[10, 1, 5], [2, 3, 3], [7, 4, 9]]
A = np.matrix("10, 1, 5; 2, 3, 6; 6, 8, 9")


Q, R = np.linalg.qr(A)
print("--------------------------------------------")
print("np.linalg.qr")
print(f"Soit la matrice:\n{A}\n\n La matrice Q est:\n{Q}\n\n La matrice R est:\n{R}")

Q, R = householder_qr(A)
print("--------------------------------------------")
print("NOUS:")
print(f"Soit la matrice:\n{A}\n\n La matrice Q est:\n{Q}\n\n La matrice R est:\n{R}")

