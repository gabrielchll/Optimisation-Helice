import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import coefficients as coeff

R = 0.06 # Rayon max hélice (m)
r0 = 0.006 # Début hélice (m)
rho = 1.2 # Masse volumique de l'air (kg/m**3)
nu = 1.5e-5 # Viscosité de l'air
m = 1.2 # Masse du drone (choisi d'après le TD)
g_ = 9.81
Re_min = 50000 # A revoir c'est la prochaine étape
omeg_max = 2000 # rad/s
omeg_min = 500

def Reynolds(r, x) :          # Renvoie le tableau du nombre de Reynolds de chaque section de la pale, permettra de faire varier Ct avec r.
    dr = r[1] - r[0]
    n = len(r)
    return (rho * x[n-1:2*n-2] * ca.sqrt((r[:-1] + dr/2)**2*(x[-1]*1000)**2 + x[2*n-2:-1]**2))/nu

def alpha(r, x) :                  # Renvoie l'angle alpha (pareil que dans le TD)
    dr = r[1] - r[0]
    n = len(r)
    return ca.atan(x[2*n-2:-1]/((r[:-1] + dr/2)*x[-1]*1000))

def incidence(r, x) :       # Renvoie l'angle d'incidence de la pale fonction de l'angle de callage et de r
    n = len(r)
    return x[:n-1] - alpha(r, x)

def puiss_diff(r, x, Cp, Ct, N_pale) :  # Renvoie la liste des puissances nécessaire pour faire tourner chaque section de la pale
    Re = Reynolds(r, x)                 # Est pris en compte si on fait varier Ct avec Re
    a = alpha(r, x)
    n = len(r)
    dr = r[1] - r[0]
    return 1/2*rho*x[-1]*1000*ca.sqrt((r[:-1] + dr/2)**2*(x[-1]*1000)**2 + x[2*n-2:-1]**2) * (Cp(x[:n-1] - a) * x[2*n-2:-1]  + Ct(x[:n-1] - a) * (r[:-1] + dr/2) * x[-1]*1000) * x[n-1:2*n-2] * (r[:-1] + dr/2) * dr * N_pale

def puissance(r, x, Cp, Ct, N_pale) :           # Renvoie la puissance totale (somme des puissances élémentaires), fonction qu'on va chercher à minimiser
    dP = puiss_diff(r, x, Cp, Ct, N_pale)
    return ca.sum1(dP)

def force_diff(r, x, Cp, Ct) :          # Renvoie la liste des forces verticales produitent par chaque section de la pale
    dr = r[1] - r[0]
    Re = Reynolds(r, x)
    a = alpha(r, x)
    n = len(r)
    return 1/2*rho*ca.sqrt((r[:-1] + dr/2)**2*(x[-1]*1000)**2 + x[2*n-2:-1]**2)*(Cp(x[:n-1] - a) * (r[:-1] + dr/2) * x[-1] * 1000 - Ct(x[:n-1] - a) * x[2*n-2:-1]) * x[n-1:2*n-2] * dr

def force_cond(r, x, Cp, Ct, N_pale) :          # La somme des forces moins la valeur que l'on veut (c'est à dire on veut que la valeur renvoyé soit nulle)
    dF = force_diff(r, x, Cp, Ct)
    return ca.sum1(dF) - (m*g_)/(4*N_pale)

def get_problem(N, Cp, Ct, N_pale) :        # Définie le problème pour casadi

    dr = (R - r0)/N
    r = np.r_[0.0, np.cumsum(np.full(N, dr))] + r0          # Liste des valeurs de r discrétisée
    
    x = ca.SX.sym("x", 3*N+1)               # Définie la variable (les N angles de callage puis les N valeurs de la corde puis les N valeurs de vitesse induite et puis omega)

    return {"x" : x,                        # Variable du problème d'optimisation
            "f" : puissance(r, x, Cp, Ct, N_pale),          # Fonction que l'on veut minimiser
            "g" : ca.vertcat(force_cond(r, x, Cp, Ct, N_pale), incidence(r, x), N_pale*force_diff(r, x, Cp, Ct) - 4*rho*np.pi*r[:-1]*x[2*N:-1]**2*dr, Reynolds(r, x))}        # Les conditions (fonctions que l'on veut borner)

def hélice_optim(N, Cp, Ct, i_max, N_pale) :        # Résoud le problème défini au dessus en définissant le solver et les contraintes

    dr = (R - r0)/N
    r = np.r_[0.0, np.cumsum(np.full(N, dr))] + r0

    pb = get_problem(N, Cp, Ct, N_pale)     # Renvoie le dico qui définie le pb

    solver = ca.nlpsol('nlp', 'ipopt', pb)          # On choisit ipopt

    solution = solver(
        x0 = np.r_[np.arctan(8/((r[:-1] + dr/2)*1000)) + 5*np.pi/180, np.full(N, 0.01), np.full(N, 8), 1],     # On initialise l'angle de callage tel que l'angle d'incidence est de 5° partout et la corde à 1cm partout
        lbx = np.r_[np.full(N, 0.0), np.full(N, 0.002), np.full(N, 0.01), omeg_min/1000],          # On minimise l'angle de callage par 0° (pour ne pas qu'il cherche des trucs absurdes) et la corde à 2 mm (pour éviter Re = 0 => 0 frottements mais toujours portance...)
        ubx = np.r_[np.full(N, np.inf), np.full(N, 0.015), np.full(N, np.inf), omeg_max/1000],          # On maximise l'angle de callage par l'infini et la corde par 7 cm (ouais c'est un peu au pif pour tester là)
        lbg = np.r_[0.0, np.full(N, 0.0), np.full(N, 0.0), np.full(N, 10000)],      # Condition d'égalité de force à gauche puis on ne veut pas une incidence négative (pour ne pas qu'il cherche des solutions chelous) et on minimise Re à la valeur min du TD car on se base sur ces modèles (pour l'instant)      
        ubg = np.r_[0.0, np.full(N, i_max*np.pi/180), np.full(N, 0.0), np.full(N, np.inf)])      # Condition d'égalité de la force puis on maximise l'incidence par le décollement constaté (10° dans le TD) et on maximise le reynolds par l'infini

    print(solver.stats()["return_status"])      # Statut pour voir s'il a bien convergé

    return solution

sol_prof = hélice_optim(20, coeff.Cp, coeff.Ct, 10, 4)

r = np.r_[0.0, np.cumsum(np.full(20, (R-r0)/20))] + r0
r1 = r[:-1] + (R-r0)/40

X = sol_prof["x"]

print(Reynolds(r, X))
print(sol_prof["f"])
print(f"omeg = {X[-1]*1000}")

fig, ax1 = plt.subplots()       
fig, ax2 = plt.subplots()
fig, ax3 = plt.subplots()

ax1.plot(r1, X[:20])        # Trace l'angle de callage en fonction
ax2.plot(r1, X[20:40])        # Trace la corde en fonction de r
ax3.plot(r1, X[40:-1])

plt.show()

