#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#GlowScript 3.0 VPython

# Hard-sphere gas.

# Bruce Sherwood
# Claudine Allen
"""


from vpython import *
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle

# win = 500 # peut aider à définir la taille d'un autre objet visuel comme un histogramme proportionnellement à la taille du canevas.

# Déclaration de variables influençant le temps d'exécution de la simulation
Natoms = 200  # change this to have more or fewer atoms
dt = 1E-5  # pas d'incrémentation temporel

# Déclaration de variables physiques "Typical values"
DIM = 2 #Nombre de degrés de liberté de la simulation 
mass = 4E-3/6E23 # helium mass
Ratom = 0.01 # wildly exaggerated size of an atom
k = 1.4E-23 # Boltzmann constant
T = 300 # around room temperature

#### CANEVAS DE FOND ####
L = 1 # container is a cube L on a side
gray = color.gray(0.7) # color of edges of container and spheres below
animation = canvas( width=750, height=500) # , align='left')
animation.range = L
# animation.title = 'Théorie cinétique des gaz parfaits'
# s = """  Simulation de particules modélisées en sphères dures pour représenter leur trajectoire ballistique avec collisions. Une sphère est colorée et grossie seulement pour l’effet visuel permettant de suivre sa trajectoire plus facilement dans l'animation, sa cinétique est identique à toutes les autres particules.

# """
# animation.caption = s

#### ARÊTES DE BOÎTE 2D ####
d = L/2+Ratom
r = 0.005
cadre = curve(color=gray, radius=r)
cadre.append([vector(-d,-d,0), vector(d,-d,0), vector(d,d,0), vector(-d,d,0), vector(-d,-d,0)])

#### POSITION ET QUANTITÉ DE MOUVEMENT INITIALE DES SPHÈRES ####
Atoms = [] # Objet qui contiendra les sphères pour l'animation
p = [] # quantité de mouvement des sphères
apos = [] # position des sphères
pavg = sqrt(2*mass*(DIM/2)*k*T) # average kinetic energy in 3D p**2/(2mass) = (3/2)kT : Principe de l'équipartition de l'énergie en thermodynamique statistique classique

for i in range(Natoms):
    x = L*random()-L/2 # position aléatoire qui tient compte que l'origine est au centre de la boîte
    y = L*random()-L/2
    z = 0
    if i == 0:  # garde une sphère plus grosse et colorée parmis toutes les grises
        Atoms.append(simple_sphere(pos=vector(x,y,z), radius=0.03, color=color.magenta)) #, make_trail=True, retain=100, trail_radius=0.3*Ratom))
    else: Atoms.append(simple_sphere(pos=vector(x,y,z), radius=Ratom, color=gray))
    apos.append(vec(x,y,z)) # liste de la position initiale de toutes les sphères
#    theta = pi*random() # direction de coordonnées sphériques, superflue en 2D
    phi = 2*pi*random() # direction aléatoire pour la quantité de mouvement
    px = pavg*cos(phi)  # qte de mvt initiale selon l'équipartition
    py = pavg*sin(phi)
    pz = 0
    p.append(vector(px,py,pz)) # liste de la quantité de mouvement initiale de toutes les sphères

#initialization = pickle.load(open("PHY-3003/init.pkl", "rb"))
#apos,p = initialization[0], initialization[1]
#initialization = (apos,p)
#pickle.dump(initialization, open("PHY-3003/init.pkl", "wb"))

#### FONCTION POUR IDENTIFIER LES COLLISIONS, I.E. LORSQUE LA DISTANCE ENTRE LES CENTRES DE 2 SPHÈRES EST À LA LIMITE DE S'INTERPÉNÉTRER ####
def checkCollisions():
    hitlist = []   # initialisation
    r2 = 2*Ratom   # distance critique où les 2 sphères entre en contact à la limite de leur rayon
    r2 *= r2   # produit scalaire pour éviter une comparaison vectorielle ci-dessous
    for i in range(Natoms):
        ai = apos[i]
        for j in range(i) :
            aj = apos[j]
            dr = ai - aj   # la boucle dans une boucle itère pour calculer cette distance vectorielle dr entre chaque paire de sphère
            if mag2(dr) < r2:   # test de collision où mag2(dr) qui retourne la norme élevée au carré de la distance intersphère dr
                hitlist.append([i,j]) # liste numérotant toutes les paires de sphères en collision
    
    return hitlist



def suit(hitlist, dt, sommation, pos_précédente, apos, liste_temps_entre_collision, liste_distance_entre_collision, liste_distance_x_entre_collision, liste_distance_y_entre_collision, liste_distance_z_entre_collision):
    num = 0  # Indice fixe, pourrait être généralisé en fonction du cas d'utilisation
    collision_detectée = False  # Variable pour suivre l'état de collision

    # Vérification des collisions
    for paire in hitlist:
        if num == paire[0] or num == paire[1]:  
            collision_detectée = True
            break  # Sort de la boucle dès qu'une collision est détectée

    if collision_detectée==True:
        # Ajoute le temps écoulé entre les collisions à la liste des temps
        liste_temps_entre_collision.append(sommation*dt)
        #print('*********************')
        #print('sommation', sommation)
        

        # Initialisation des variables
        distance = 0
        vec_distance_x = 0
        vec_distance_y = 0
        vec_distance_z = 0

        pos_précédente.append(apos[num])  # Ajoute la position actuelle pour le calcul de la distance

        # Calcul des distances entre positions successives
        for i in range(1, len(pos_précédente)):
            delta_vect = pos_précédente[i] - pos_précédente[i - 1]
            delta_x, delta_y, delta_z = delta_vect.x, delta_vect.y, delta_vect.z

            vec_distance_x += np.abs(delta_x)
            vec_distance_y += np.abs(delta_y)
            vec_distance_z += np.abs(delta_z)

            distance_parcourue = np.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)
            distance += distance_parcourue

        # Sauvegarde des distances entre collisions
        liste_distance_x_entre_collision.append(vec_distance_x)
        liste_distance_y_entre_collision.append(vec_distance_y)
        liste_distance_z_entre_collision.append(vec_distance_z)
        liste_distance_entre_collision.append(distance)

        # Remise à zéro de la liste des positions, mais en gardant la dernière position
        pos_précédente.clear()
        pos_précédente.append(apos[num])
        #('pos_précédente', pos_précédente)
        sommation = 1

    if collision_detectée==False:
        # Si aucune collision n'est détectée, met à jour le temps entre collisions et ajoute la position actuelle à la liste
        sommation += 1
        #print('temps', sommation)
        
        pos_précédente.append(apos[num])
    return sommation, pos_précédente

def follow_particle(deltax, hitlist, deltapos, liste_temps_entre_collision, liste_distance_entre_collision,liste_distance_x_entre_collision,liste_distance_y_entre_collision,liste_distance_z_entre_collision, n_particle=0, iterations_since_last_col=0):
    """
    n_particle: indice de la particule à suivre
    iterations_since_last_col: nombre d'itérations depuis que la particule a subit une collision
    liste_temps_entre_collision: liste de temps entre les collisions
    liste_distance_entre_collision: liste des distances entre les collisions
    liste_distance_x_entre_collision: liste des distances x entre les collisions
    liste_distance_y_entre_collision: liste des distances y entre les collisions
    liste_distance_z_entre_collision: liste des distances z entre les collisions
    """
    iterations_since_last_col += 1
    deltapos.x += np.abs(deltax[n_particle].x)
    deltapos.y += np.abs(deltax[n_particle].y)
    deltapos.z += np.abs(deltax[n_particle].z)
    particle_hit = False
    #coliisions = []
    for ij in hitlist:
        i,j = ij[0], ij[1]
        if i == n_particle or j == n_particle:
            particle_hit = True
            #coliisions.append([i,j])
    if particle_hit:
        liste_temps_entre_collision.append(iterations_since_last_col*dt)
        liste_distance_entre_collision.append(deltapos.mag)
        liste_distance_x_entre_collision.append(np.abs(deltapos.x))
        liste_distance_y_entre_collision.append(np.abs(deltapos.y))
        liste_distance_z_entre_collision.append(np.abs(deltapos.z))
        return 0, vector(0,0,0)
    return iterations_since_last_col, deltapos

#### BOUCLE PRINCIPALE POUR L'ÉVOLUTION TEMPORELLE DE PAS dt ####
## ATTENTION : la boucle laisse aller l'animation aussi longtemps que souhaité, assurez-vous de savoir comment interrompre vous-même correctement (souvent `ctrl+c`, mais peut varier)
## ALTERNATIVE : vous pouvez bien sûr remplacer la boucle "while" par une boucle "for" avec un nombre d'itérations suffisant pour obtenir une bonne distribution statistique à l'équilibre
temps_entre_collision = dt
pos_précédente = []
liste_temps_entre_collision = []
liste_distance_entre_collision = []
liste_distance_x_entre_collision = []
liste_distance_y_entre_collision = []
liste_distance_z_entre_collision = []


sommation = 1
pos_précédente1 = []
liste_temps_entre_collision1 = []
liste_distance_entre_collision1 = []
liste_distance_x_entre_collision1 = []
liste_distance_y_entre_collision1 = []
liste_distance_z_entre_collision1 = []

iterations_since_last_col = 0
deltapos = vector(0,0,0)
from tqdm import tqdm
for k in tqdm(range(100000)):

    
    rate(300)  # limite la vitesse de calcul de la simulation pour que l'animation soit visible à l'oeil humain!

    #### DÉPLACE TOUTES LES SPHÈRES D'UN PAS SPATIAL deltax
    vitesse = []   # vitesse instantanée de chaque sphère
    deltax = []  # pas de position de chaque sphère correspondant à l'incrément de temps dt
    for i in range(Natoms):
        vitesse.append(p[i]/mass)   # par définition de la quantité de nouvement pour chaque sphère
        deltax.append(vitesse[i] * dt)   # différence avant pour calculer l'incrément de position

        Atoms[i].pos = apos[i] = apos[i] + deltax[i]  # nouvelle position de l'atome après l'incrément de temps dt

    #### CONSERVE LA QUANTITÉ DE MOUVEMENT AUX COLLISIONS AVEC LES MURS DE LA BOÎTE ####
    for i in range(Natoms):
        loc = apos[i]
        if abs(loc.x) > L/2:
            if loc.x < 0: p[i].x =  abs(p[i].x)  # renverse composante x au mur de gauche
            else: p[i].x =  -abs(p[i].x)   # renverse composante x au mur de droite
        if abs(loc.y) > L/2:
            if loc.y < 0: p[i].y = abs(p[i].y)  # renverse composante y au mur du bas
            else: p[i].y =  -abs(p[i].y)  # renverse composante y au mur du haut

    #### LET'S FIND THESE COLLISIONS!!! ####
    hitlist = checkCollisions()
    #### CONSERVE LA QUANTITÉ DE MOUVEMENT AUX COLLISIONS ENTRE SPHÈRES ####
    for ij in hitlist:

        # définition de nouvelles variables pour chaque paire de sphères en collision
        i = ij[0]  # extraction du numéro des 2 sphères impliquées à cette itération
        j = ij[1]
        ptot = p[i]+p[j]   # quantité de mouvement totale des 2 sphères
        mtot = 2*mass    # masse totale des 2 sphères
        Vcom = ptot/mtot   # vitesse du référentiel barycentrique/center-of-momentum (com) frame
        posi = apos[i]   # position de chacune des 2 sphères
        posj = apos[j]
        vi = p[i]/mass   # vitesse de chacune des 2 sphères
        vj = p[j]/mass
        rrel = posi-posj  # vecteur pour la distance entre les centres des 2 sphères
        vrel = vj-vi   # vecteur pour la différence de vitesse entre les 2 sphères

        # exclusion de cas où il n'y a pas de changements à faire
        if vrel.mag2 == 0: continue  # exactly same velocities si et seulement si le vecteur vrel devient nul, la trajectoire des 2 sphères continue alors côte à côte
        if rrel.mag > Ratom: continue  # one atom went all the way through another, la collision a été "manquée" à l'intérieur du pas deltax

        # calcule la distance et temps d'interpénétration des sphères dures qui ne doit pas se produire dans ce modèle
        dx = dot(rrel, vrel.hat)       # rrel.mag*cos(theta) où theta is the angle between vrel and rrel:
        dy = cross(rrel, vrel.hat).mag # rrel.mag*sin(theta)
        alpha = asin(dy/(2*Ratom))  # alpha is the angle of the triangle composed of rrel, path of atom j, and a line from the center of atom i to the center of atom j where atome j hits atom i
        d = (2*Ratom)*cos(alpha)-dx # distance traveled into the atom from first contact
        deltat = d/vrel.mag         # time spent moving from first contact to position inside atom

        #### CHANGE L'INTERPÉNÉTRATION DES SPHÈRES PAR LA CINÉTIQUE DE COLLISION ####
        posi = posi-vi*deltat   # back up to contact configuration
        posj = posj-vj*deltat
        pcomi = p[i]-mass*Vcom  # transform momenta to center-of-momentum (com) frame
        pcomj = p[j]-mass*Vcom
        rrel = hat(rrel)    # vecteur unitaire aligné avec rrel
        pcomi = pcomi-2*dot(pcomi,rrel)*rrel # bounce in center-of-momentum (com) frame
        pcomj = pcomj-2*dot(pcomj,rrel)*rrel
        p[i] = pcomi+mass*Vcom # transform momenta back to lab frame
        p[j] = pcomj+mass*Vcom
        apos[i] = posi+(p[i]/mass)*deltat # move forward deltat in time, ramenant au même temps où sont rendues les autres sphères dans l'itération
        apos[j] = posj+(p[j]/mass)*deltat

    # **Appel de la fonction suit() ici**
    sommation, pos_précédente = suit(hitlist, dt, sommation, pos_précédente, apos, liste_temps_entre_collision1, liste_distance_entre_collision1, liste_distance_x_entre_collision1, liste_distance_y_entre_collision1, liste_distance_z_entre_collision1)
    iterations_since_last_col, deltapos = follow_particle(deltax, hitlist, deltapos, liste_temps_entre_collision,liste_distance_entre_collision,liste_distance_x_entre_collision, liste_distance_y_entre_collision, liste_distance_z_entre_collision,n_particle=0, iterations_since_last_col=iterations_since_last_col)
np.savetxt("PHY-3003/data.txt", np.array([liste_temps_entre_collision,liste_distance_entre_collision, liste_distance_x_entre_collision, liste_distance_y_entre_collision, liste_distance_z_entre_collision]).T)

np.savetxt("PHY-3003/data1.txt", np.array([liste_temps_entre_collision1,liste_distance_entre_collision1, liste_distance_x_entre_collision1, liste_distance_y_entre_collision1, liste_distance_z_entre_collision1]).T)

