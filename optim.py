import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import coefficients as coeff

R = 0.06 # Rayon max hélice (m)
r0 = 0.006 # Début hélice (m)
omeg = 1047 # Vitesse rotation moteur (rad/s)
Vz = 11 # Vitesse descension air (m/s)
rho = 1.2 # Masse volumique de l'air (1kg/m**3)
nu = 1.5e-5 # Viscosité de l'air
m = 0.5 # Masse du drone (choisi aléatoirement pour l'instant)
g_ = 9.81
Re_min = 50000

def Reynolds(r, x) :          # Renvoie le tableau du nombre de Reynolds de chaque section de la pale, permettra de faire varier Ct avec r.
    dr = r[1] - r[0]
    n = len(r)
    return (rho * x[n-1:] * ca.sqrt((r[:-1] + dr/2)**2*omeg**2 + Vz**2))/nu

def alpha(r) :                  # Renvoie l'angle d'incidence de l'air sur l'hélice
    dr = r[1] - r[0]
    return np.arctan(Vz/((r[:-1] + dr/2)*omeg))

def puiss_diff(r, x, Cp, Ct, N_pale) :  # Renvoie la liste des puissances nécessaire pour faire tourner chaque section de la pale
    Re = Reynolds(r, x)                 # Est pris en compte si on fait varier Ct avec Re
    a = alpha(r)
    n = len(r)
    dr = r[1] - r[0]
    return 1/2*rho*omeg*ca.sqrt((r[:-1] + dr/2)**2*omeg**2 + Vz**2) * (Cp(x[:n-1] - a) * Vz  + Ct(x[:n-1] - a) * (r[:-1] + dr/2) * omeg) * x[n-1:] * (r[:-1] + dr/2) * dr * N_pale

def puissance(r, x, Cp, Ct, N_pale) :           # Renvoie la puissance totale (somme des puissances élémentaires), fonction qu'on va chercher à minimiser
    dP = puiss_diff(r, x, Cp, Ct, N_pale)
    return ca.sum1(dP)

def force_diff(r, x, Cp, Ct) :          # Renvoie la liste des forces verticales produitent par chaque section de la pale
    dr = r[1] - r[0]
    Re = Reynolds(r, x)
    a = alpha(r)
    n = len(r)
    return 1/2*rho*ca.sqrt((r[:-1] + dr/2)**2*omeg**2 + Vz**2)*(Cp(x[:n-1] - a) * (r[:-1] + dr/2) * omeg - Ct(x[:n-1] - a) * Vz) * x[n-1:] * dr

def force_cond(r, x, Cp, Ct, N_pale) :          # La somme des forces moins la valeur que l'on veut (c'est à dire on veut que la valeur renvoyé soit nulle)
    dF = force_diff(r, x, Cp, Ct)
    return ca.sum1(dF) - (m*g_)/(4*N_pale)

def incidence(r, x) :       # Renvoie l'angle d'incidence de la pale fonction de l'angle de callage et de r
    n = len(r)
    return x[:n-1] - alpha(r)

def get_problem(N, Cp, Ct, N_pale) :        # Définie le problème pour casadi

    dr = (R - r0)/N
    r = np.r_[0.0, np.cumsum(np.full(N, dr))] + r0          # Liste des valeurs de r discrétisée
    
    x = ca.SX.sym("x", 2*(N))               # Définie la variable (les N angles de callage puis les N valeurs de la corde)

    return {"x" : x,                        # Variable du problème d'optimisation
            "f" : puissance(r, x, Cp, Ct, N_pale),          # Fonction que l'on veut minimiser
            "g" : ca.vertcat(force_cond(r, x, Cp, Ct, N_pale), incidence(r, x), Reynolds(r, x))}        # Les conditions (fonctions que l'on veut borner)

def hélice_optim(N, Cp, Ct, i_max, N_pale) :        # Résoud le problème défini au dessus en définissant le solver et les contraintes

    dr = (R - r0)/N
    r = np.r_[0.0, np.cumsum(np.full(N, dr))] + r0

    pb = get_problem(N, Cp, Ct, N_pale)     # Renvoie le dico qui définie le pb

    solver = ca.nlpsol('nlp', 'ipopt', pb)          # On choisit ipopt

    solution = solver(
        x0 = np.r_[alpha(r) + 5*np.pi/180, np.full(N, 0.01)],     # On initialise l'angle de callage tel que l'angle d'incidence est de 5° partout et la corde à 1cm partout
        lbx = np.r_[np.full(N, 0.0), np.full(N, 0.002)],          # On minimise l'angle de callage par 0° (pour ne pas qu'il cherche des trucs absurdes) et la corde à 2 mm (pour éviter Re = 0 => 0 frottements mais toujours portance...)
        ubx = np.r_[np.full(N, np.inf), np.full(N, 0.07)],          # On maximise l'angle de callage par l'infini et la corde par 7 cm (ouais c'est un peu au pif pour tester là)
        lbg = np.r_[0.0, np.full(N, 0.0), np.full(N, Re_min)],      # Condition d'égalité de force à gauche puis on ne veut pas une incidence négative (pour ne pas qu'il cherche des solutions chelous) et on minimise Re à la valeur min du TD car on se base sur ces modèles (pour l'instant)      
        ubg = np.r_[0.0, np.full(N, i_max*np.pi/180), np.full(N, np.inf)])      # Condition d'égalité de la force puis on maximise l'incidence par le décollement constaté (10° dans le TD) et on maximise le reynolds par l'infini

    print(solver.stats()["return_status"])      # Statut pour voir s'il a bien convergé

    return solution

sol_prof = hélice_optim(20, coeff.Cp, coeff.Ct, 10, 2)

r = np.r_[0.0, np.cumsum(np.full(20, (R-r0)/20))] + r0
r1 = r[:-1] + (R-r0)/40

X = sol_prof["x"]

fig, ax1 = plt.subplots()       
fig, ax2 = plt.subplots()

ax1.plot(r1, X[:20])        # Trace l'angle de callage en fonction
ax2.plot(r1, X[20:])        # Trace la corde en fonction de r

plt.show()

