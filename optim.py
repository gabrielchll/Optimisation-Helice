import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from profiles import Profil

R = 0.06 # Rayon max hélice (m)
r0 = 0.006 # Début hélice (m)
rho = 1.2 # Masse volumique de l'air (kg/m**3)
nu = 1.5e-5 # Viscosité de l'air
m = 1.2 # Masse du drone
g_ = 9.81
omeg_max = 2000 # rad/s  
omeg_min = 500

class Helice :

    def __init__(self, R, r0, N_pale, N, profil, nom) :
        self.profil = Profil(profil)
        self.R = R
        self.r0 = r0
        self.dr = (R-r0)/N
        self.r = r0 + np.cumsum(np.r_[0.0, np.full(N, self.dr)])    # Renvoie les coordonnées délimitant les sections de pale.
        self.r1 = self.r[:-1] + self.dr/2   # Renvoie les coordonnées du milieu de chaque section de pale.
        self.N = N
        self.Cp = self.profil.Cp
        self.Ct = self.profil.Ct
        self.N_pale = N_pale
        self.nom = nom

    def __repr__(self) :
        return self.nom

def Reynolds(helice, x) :          # Renvoie le tableau du nombre de Reynolds de chaque section de la pale, permettra de faire varier Ct avec r.
    v_t = helice.r1*x[-1]*1000-x[3*helice.N:-1]
    return (x[helice.N:2*helice.N] * ca.sqrt(v_t**2 + x[2*helice.N:3*helice.N]**2))/nu

def alpha(helice, x) :                  # Renvoie l'angle alpha (pareil que dans le TD)
    v_t = helice.r1*x[-1]*1000-x[3*helice.N:-1]
    return ca.atan(x[2*helice.N:3*helice.N]/v_t)

def incidence(helice, x) :       # Renvoie l'angle d'incidence de la pale fonction de l'angle de callage et de r
    return x[:helice.N] - alpha(helice, x)

def facteur_perte_Prandt(helice, x) :
    a = alpha(helice, x)
    return 2/np.pi*ca.acos(ca.exp(-helice.N_pale/2*(helice.R-helice.r1)/(helice.r1*ca.sin(a))))

def force_vert_diff(helice, x) :          # Renvoie la liste des forces verticales produitent par chaque section de la pale
    Re = Reynolds(helice, x)
    a = alpha(helice, x)
    v_t = helice.r1*x[-1]*1000-x[3*helice.N:-1]
    return 1/2*rho*helice.N_pale*ca.sqrt(v_t**2 + x[2*helice.N:3*helice.N]**2)*(helice.Cp((x[:helice.N] - a)*180/np.pi, Re) * v_t - helice.Ct((x[:helice.N] - a)*180/np.pi, Re) * x[2*helice.N:3*helice.N]) * x[helice.N:2*helice.N] * helice.dr

def force_tang_diff(helice, x) :  # Renvoie la liste des puissances nécessaire pour faire tourner chaque section de la pale
    Re = Reynolds(helice, x)                 # Est pris en compte si on fait varier Ct avec Re
    a = alpha(helice, x)
    v_t = helice.r1*x[-1]*1000-x[3*helice.N:-1]
    return 1/2*rho*ca.sqrt(v_t**2 + x[2*helice.N:3*helice.N]**2) * (helice.Cp((x[:helice.N] - a)*180/np.pi, Re) * x[2*helice.N:3*helice.N]  + helice.Ct((x[:helice.N] - a)*180/np.pi, Re) * v_t) * x[helice.N:2*helice.N] * helice.dr * helice.N_pale

def puissance(helice, x) :           # Renvoie la puissance totale (somme des puissances élémentaires), fonction qu'on va chercher à minimiser
    dP = force_tang_diff(helice, x)*1000*x[-1]* helice.r1
    return ca.sum1(dP)

def force_vert(helice, x) :          # La somme des forces moins la valeur que l'on veut (c'est à dire on veut que la valeur renvoyé soit nulle)
    dF = force_vert_diff(helice, x)
    return ca.sum1(dF)

def get_problem(helice) :        # Définie le problème pour casadi

    x = ca.SX.sym("x", 4*helice.N+1)               # Définie la variable (les N angles de callage puis les N valeurs de la corde puis les N valeurs de vitesse induite et puis omega)

    lam = 100

    return {"x" : x,                        # Variable du problème d'optimisation
            "f" : puissance(helice, x) + lam * (ca.sumsqr(ca.diff(x[:helice.N], 2)) + 270 * ca.sumsqr(ca.diff(x[helice.N:2*helice.N], 2))),          # Fonction que l'on veut minimiser
            "g" : ca.vertcat(force_vert(helice, x) - m*g_/4, incidence(helice, x), force_vert_diff(helice, x) - 4*rho*np.pi*helice.r[:-1]*helice.dr*facteur_perte_Prandt(helice, x)*x[2*helice.N:3*helice.N]**2, force_tang_diff(helice, x) - 4*np.pi*rho*x[2*helice.N:3*helice.N]*x[3*helice.N:-1]*helice.r1*facteur_perte_Prandt(helice, x)*helice.dr, Reynolds(helice, x))}        # Les conditions (fonctions que l'on veut borner)

def hélice_optim(helice) :        # Résoud le problème défini au dessus en définissant le solver et les contraintes

    pb = get_problem(helice)     # Renvoie le dico qui définie le pb

    opts = {"ipopt": {"max_iter": 3000, "print_level": 5}}

    solver = ca.nlpsol('nlp', 'ipopt', pb, opts)          # On choisit ipopt

    solution = solver(
        x0 = np.r_[np.arctan(8/(helice.r1*1000)) + 1*np.pi/180, 0.0045, np.full(helice.N-1, 0.01), np.full(helice.N, 8), np.full(helice.N, 0.02), 1],     # On initialise l'angle de callage tel que l'angle d'incidence est de 5° partout et la corde à 1cm partout
        lbx = np.r_[np.full(helice.N, 0.0), np.full(helice.N, 0.002), np.full(helice.N, 0.01), np.full(helice.N, 0.01), omeg_min/1000],          # On minimise l'angle de callage par 0° (pour ne pas qu'il cherche des trucs absurdes) et la corde à 2 mm (pour éviter Re = 0 => 0 frottements mais toujours portance...)
        ubx = np.r_[np.full(helice.N, np.inf), 0.008, np.full(helice.N-1, 0.015), np.full(2*helice.N, np.inf), omeg_max/1000],          # On maximise l'angle de callage par l'infini et la corde par 1.5 cm (ouais c'est un peu au pif pour tester là)
        lbg = np.r_[0.0, np.full(helice.N, 0.0), np.full(2*helice.N, 0.0), np.full(helice.N, 400)],      # Condition d'égalité de force à gauche puis on ne veut pas une incidence négative (pour ne pas qu'il cherche des solutions chelous)      
        ubg = np.r_[0.0, np.full(helice.N, 8.9*np.pi/180), np.full(2*helice.N, 0.0), np.full(helice.N, 100000)])      # Condition d'égalité de la force puis on maximise l'incidence par le décollement constaté (10° dans le TD) et on maximise le reynolds par l'infini

    print(solver.stats()["return_status"])      # Statut pour voir s'il a bien convergé

    X = solution["x"]

    print(f"P = {solution["f"]} W")            # Renvoie la puissance
    print(f"omeg = {X[-1]*1000} rad/s")   # Renvoie la vitesse de rotation du moteur optimale
    print(f"r (en cm) : {helice.r1*100}") 
    print(f"Callage (en °) : {X[:helice.N]*180/np.pi}")
    print(f"Corde (en cm): {X[helice.N:2*helice.N]*100}")

    fig, ax = plt.subplots(2, 3)

    ax[0, 0].plot(helice.r1, X[:helice.N]*180/np.pi)        # Trace l'angle de calage en fonction
    ax[0, 1].plot(helice.r1, X[helice.N:2*helice.N]*100)        # Trace la corde en fonction de r
    ax[1, 0].plot(helice.r1, X[2*helice.N:3*helice.N])         # Trace la vitesse induite en fonction de r
    ax[1, 1].plot(helice.r1, force_vert_diff(helice, X))      # Trace la force verticale en fonction de r 
    ax[0, 2].plot(helice.r1, X[3*helice.N:-1])
    ax[1,2].plot(helice.r1, Reynolds(helice, X))

    ax[0, 0].set_title("Angle de calage en fonction du rayon de la section")
    ax[0, 1].set_title("Corde en fonction du rayon de la section")
    ax[1, 0].set_title("Vitesse induite verticale en fonction du rayon de la section")
    ax[1, 1].set_title("Répartition de la force verticale sur une pale")
    ax[0, 2].set_title("Vitesse induite tangentielle en fonction du rayon de la section")
    ax[1, 2].set_title("Reynolds de la section en fonction du rayon")

    ax[0, 0].set_ylabel("Angle de calage (en °)")
    ax[0, 0].set_xlabel("Rayon (en m)")
    ax[0, 1].set_ylabel("Corde (en cm)")
    ax[0, 1].set_xlabel("Rayon (en m)")
    ax[1, 0].set_ylabel("Vitesse induite verticale (en m/s)")
    ax[1, 0].set_xlabel("Rayon (en m)")
    ax[1, 1].set_ylabel("Force verticale (en N/m)")
    ax[1, 1].set_xlabel("Rayon (en m)")
    ax[0, 2].set_ylabel("Vitesse induite tangentielle (en m/s)")
    ax[0, 2].set_xlabel("Rayon (en m)")
    ax[1, 2].set_ylabel("Reynolds")
    ax[1, 2].set_xlabel("Rayon (en m)")

    fig.suptitle(f"Caractéristiques de l'hélice {helice.nom}")

    plt.show()

    return solution

def solidité(helice, x) :
    return x[helice.N:2*helice.N]*helice.N_pale/(2*np.pi*helice.r1)

h_V1 = Helice(R, r0, 4, 27, "sd7043","h_V1")

sol = hélice_optim(h_V1)

plt.plot(h_V1.r1, incidence(h_V1, sol["x"])*180/np.pi)
plt.show()

"""
def Rés_théo(helice) :
    sol = hélice_optim(helice)
    hel = sol["x"]
    omeg = np.linspace(0, 2000, 200)
    Force = []
    Puissance = []
    for i in range(0, len(omeg)) :
        Force.append(force_vert(helice, np.append(hel[:-1], omeg[i]/1000)))
        Puissance.append(puissance(helice, np.append(hel[:-1], omeg[i]/1000)))
    fig, ax = plt.subplots(2)
    ax[0].plot(omeg, np.array(Force).flatten())
    ax[1].plot(np.array(Puissance).flatten(), np.array(Force).flatten())
    plt.show()

Rés_théo(h_V1)"""



