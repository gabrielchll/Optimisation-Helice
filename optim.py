import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from profiles import Profil
import pandas as pd

R = 0.06 # Rayon max hélice (m)
r0 = 0.006 # Début hélice (m)
rho = 1.2 # Masse volumique de l'air (kg/m**3)
nu = 1.5e-5 # Viscosité de l'air
m = 1.3 # Masse du drone
g_ = 9.81
omeg_max = 1200 # rad/s  
omeg_min = 500

class Helice_squelette :

    def __init__(self, R, r0, N_pale, N, profil) :
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

class Helice(Helice_squelette) :

    def __init__(self, calage, corde, helice_squelette, nom) :
        self.__dict__.update(helice_squelette.__dict__)     # On reprend les attributs du squelette
        self.calage = calage
        self.corde = corde
        self.nom = nom

def Reynolds(helice, x) :          # Renvoie le tableau du nombre de Reynolds de chaque section de la pale, permettra de faire varier Ct avec r.
    v_t = helice.r1*x[-1]*1000-x[3*helice.N:-1]
    return (x[helice.N:2*helice.N] * ca.sqrt(v_t**2 + x[2*helice.N:3*helice.N]**2))/nu

def alpha(helice, x) :                  # Renvoie l'angle alpha (pareil que dans le TD)
    v_t = helice.r1*x[-1]*1000-x[3*helice.N:-1]
    return ca.atan(x[2*helice.N:3*helice.N] / ca.fmax(v_t, 1e-5))       # évite la division par 0 qui fait diverger

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

    incid_max = np.array([5.5, 5.5, 5.5, 5.7, 5.9, 5.9, 6, 6, 6.6, 7.5, 8.1, 8.4, 8.7, 9, 9.5, 10.2, 10.8, 11.3, 11.7, 11.7, 11.7, 12])
    Re = [2000, 5000, 8500, 12000, 15500, 19000, 22500, 26000, 29500, 33000, 36500, 40000, 43500, 47000, 50500, 54000, 57500, 61000, 64500, 68000, 71500, 80000]
        
    imax = ca.interpolant("amax", "bspline", [Re], incid_max)

    lam = 100

    return {"x" : x,                        # Variable du problème d'optimisation
            "f" : puissance(helice, x) + lam * (ca.sumsqr(ca.diff(x[:helice.N], 2)) + 270 * ca.sumsqr(ca.diff(x[helice.N:2*helice.N], 2))),          # Fonction que l'on veut minimiser
            "g" : ca.vertcat(force_vert(helice, x) - m*g_/4, incidence(helice, x) - imax(Reynolds(helice, x))*np.pi/180, force_vert_diff(helice, x) - 4*rho*np.pi*helice.r1*helice.dr*facteur_perte_Prandt(helice, x)*x[2*helice.N:3*helice.N]**2, force_tang_diff(helice, x) - 4*np.pi*rho*x[2*helice.N:3*helice.N]*x[3*helice.N:-1]*helice.r1*facteur_perte_Prandt(helice, x)*helice.dr, Reynolds(helice, x))}        # Les conditions (fonctions que l'on veut borner)

def hélice_optim(helice, courbes = True) :        # Résoud le problème défini au dessus en définissant le solver et les contraintes

    pb = get_problem(helice)     # Renvoie le dico qui définie le pb

    opts = {"ipopt": {"max_iter": 3000, "print_level": 5}}

    solver = ca.nlpsol('nlp', 'ipopt', pb, opts)          # On choisit ipopt

    solution = solver(
        x0 = np.r_[np.arctan(8/(helice.r1*1000)) + 1*np.pi/180, 0.0045, np.full(helice.N-1, 0.01), np.full(helice.N, 8), np.full(helice.N, 0.02), 1],     # On initialise l'angle de callage tel que l'angle d'incidence est de 5° partout et la corde à 1cm partout
        lbx = np.r_[np.full(helice.N, 0.0), np.full(helice.N, 0.002), np.full(helice.N, 0.01), np.full(helice.N, 0.01), omeg_min/1000],          # On minimise l'angle de callage par 0° (pour ne pas qu'il cherche des trucs absurdes) et la corde à 2 mm (pour éviter Re = 0 => 0 frottements mais toujours portance...)
        ubx = np.r_[np.full(helice.N, np.inf), 0.011, np.full(helice.N-1, 0.02), np.full(2*helice.N, np.inf), omeg_max/1000],          # On maximise l'angle de callage par l'infini et la corde par 1.5 cm (ouais c'est un peu au pif pour tester là)
        lbg = np.r_[0.0, np.full(helice.N, -np.inf), np.full(2*helice.N, 0.0), np.full(helice.N, 400)],      # Condition d'égalité de force à gauche puis on ne veut pas une incidence négative (pour ne pas qu'il cherche des solutions chelous)      
        ubg = np.r_[0.0, np.full(helice.N, 0), np.full(2*helice.N, 0.0), np.full(helice.N, 100000)])      # Condition d'égalité de la force puis on maximise l'incidence par le décollement constaté (10° dans le TD) et on maximise le reynolds par l'infini

    print(solver.stats()["return_status"])      # Statut pour voir s'il a bien convergé

    X = solution["x"]

    print(f"P = {solution["f"]} W")            # Renvoie la puissance
    print(f"omeg = {X[-1]*1000} rad/s")   # Renvoie la vitesse de rotation du moteur optimale
    print(f"r (en cm) : {helice.r1*100}") 
    print(f"Callage (en °) : {X[:helice.N]*180/np.pi}")
    print(f"Corde (en cm): {X[helice.N:2*helice.N]*100}")

    if courbes : 

        fig, ax = plt.subplots(2, 3)

        ax[0, 0].plot(helice.r1, X[:helice.N]*180/np.pi)        # Trace l'angle de calage en fonction
        ax[0, 1].plot(helice.r1, X[helice.N:2*helice.N]*100)        # Trace la corde en fonction de r
        ax[1, 0].plot(helice.r1, X[2*helice.N:3*helice.N])         # Trace la vitesse induite en fonction de r
        ax[1, 1].plot(helice.r1, force_vert_diff(helice, X))      # Trace la force verticale en fonction de r 
        ax[0, 2].plot(helice.r1, X[3*helice.N:-1])
        ax[1,2].plot(helice.r1, (X[:helice.N]-alpha(helice, X))*180/np.pi)

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

        fig.suptitle(f"Caractéristiques de l'hélice")

        plt.show()

    return solution

def performances_théo_expé(helice, omeg = None, expé=False, courbes = False):

    if omeg == None :
        Re_min = 2000
        if expé :
            df = pd.read_csv(f"Expériences/{helice.nom}.csv", on_bad_lines="skip")
            df = df[df["Motor Optical Speed (RPM)"] > 1000]
            index_max = df["Motor Optical Speed (RPM)"].idxmax()
            db = df.loc[:index_max]                     # Pour ne pas avoir la redescente
            O1 = db["Motor Optical Speed (RPM)"].to_numpy()*2*np.pi/60
            O = np.r_[np.linspace(Re_min*nu/(helice.corde[0]*helice.r0), O1[0], 40), O1, np.linspace(O1[-1], 2000, 40)]
        else :
            O = np.linspace(Re_min*nu/(helice.corde[0]*helice.r0), 2000, 100)
    else :
        O = omeg

    z = ca.SX.sym("z", 2*helice.N)
    p = ca.SX.sym("p")
    
    x = ca.vertcat(helice.calage, helice.corde, z, p)

    eqs1 = force_vert_diff(helice, x) - 4*rho*np.pi*helice.r1*helice.dr*facteur_perte_Prandt(helice, x)*x[2*helice.N:3*helice.N]**2
    eqs2 = force_tang_diff(helice, x) - 4*np.pi*rho*x[2*helice.N:3*helice.N]*x[3*helice.N:-1]*helice.r1*facteur_perte_Prandt(helice, x)*helice.dr
    
    g = ca.vertcat(eqs1, eqs2)

    nlp = {"x": z, "p": p, "f": 0, "g": g}

    solver = ca.nlpsol("solver", "ipopt", nlp)
    
    F = []
    P = []
    sols = []
    
    z0 = np.r_[helice.r1*O[0]*np.tan(helice.calage-5*np.pi/180), np.full(helice.N, 0.5)]

    for omeg in O:

        res = solver(x0=z0, p=omeg/1000, lbg=0, ubg=0, lbx=1e-5)
        
        if not solver.stats()["success"] :
            print(f"IPOPT a abandonné à omeg = {omeg:.2f} rad/s")
            break
            
        z0 = res["x"].full().ravel()
        sols.append(z0)

        F.append(float(force_vert(helice, ca.vertcat(helice.calage, helice.corde, z0, omeg/1000))))
        P.append(float(puissance(helice, ca.vertcat(helice.calage, helice.corde, z0, omeg/1000))))

    sols = np.array(sols)

    N = len(sols)
    if courbes :
        fig, ax = plt.subplots(3, 3)

        ax[0, 0].plot(O[:N]*60/(2*np.pi), F, color = "red")
        ax[0, 1].plot(P, F, color = "red")
        ax[1, 0].plot(O[:N]*60/(2*np.pi), P, color = "red")
        ax[2, 0].plot(O[:N]*60/(2*np.pi), F/(rho*O[:N]**2*(R*2)**4), color = "red")
        ax[2, 1].plot(O[:N]*60/(2*np.pi), P/(rho*O[:N]**3*(R*2)**5), color = "red")

        
        for i in range(0, 7) :
            ax[1, 1].plot(helice.r1, 180/np.pi*incidence(helice, ca.vertcat(helice.calage, helice.corde, sols[int((N-1)/7)*i], O[int((N-1)/7)*i]/1000)), label=f"{O[int((N-1)/7)*i]*60/(2*np.pi):.0f} tr/min")
            ax[0, 2].plot(helice.r1, sols[int((N-1)/7)*i][:helice.N], label=f"{O[int((N-1)/7)*i]*60/(2*np.pi):.0f} tr/min")
            ax[1, 2].plot(helice.r1, sols[int((N-1)/7)*i][helice.N:2*helice.N], label=f"{O[int((N-1)/7)*i]*60/(2*np.pi):.0f} tr/min")

        ax[0, 0].set_title("Poussée en fonction de la vitesse de rotation")
        ax[0, 1].set_title("Poussée en fonction de la puissance mécanique")
        ax[1, 0].set_title("Puissance mécanique en fonction de la vitesse de rotation")
        ax[1, 1].set_title("Incidence en fonction du rayon")
        ax[0, 2].set_title("Vitesse induite verticale en fonction du rayon")
        ax[1, 2].set_title("Vitesse induite tangentielle en fonction du rayon")
        ax[2, 0].set_title("Coefficient de portance en fonction de la vitesse de rotation")
        ax[1, 0].set_title("Coefficient de trainé en fonction de la vitesse de rotation")

        ax[0, 0].set_xlabel("Vitesse de rotation (tr/min)")
        ax[0, 0].set_ylabel("Poussée (N)")
        ax[0, 1].set_xlabel("Puissance mécanique fournie (W)")
        ax[0, 1].set_ylabel("Poussée (N)")
        ax[1, 0].set_xlabel("Vitesse de rotation (tour/min)")
        ax[1, 0].set_ylabel("Puissance mécanique fournie (W)")
        ax[1, 1].set_xlabel("Rayon (m)")
        ax[1, 1].set_ylabel("Angle d'incidence (°)")
        ax[0, 2].set_xlabel("Rayon (m)")
        ax[0, 2].set_ylabel("Vitesse induite verticale (m/s)")
        ax[1, 2].set_xlabel("Rayon (m)")
        ax[1, 2].set_ylabel("Vitesse induite tangentielle (m/s)")
        ax[2, 0].set_xlabel("Vitesse de rotation (tr/min)")
        ax[2, 0].set_ylabel("Coefficient de portance")
        ax[2, 1].set_xlabel("Vitesse de rotation (tr/min)")
        ax[2, 1].set_ylabel("Coefficient de trainé")

        if expé :
            ax[0, 0].scatter(db["Motor Optical Speed (RPM)"], -db["Thrust (N)"], color = "blue", s=1)
            ax[0, 1].scatter(-db["Torque (N·m)"]*db["Motor Optical Speed (RPM)"]*(2*np.pi)/60, -db["Thrust (N)"], color = "blue", s=1)
            ax[1, 0].scatter(db["Motor Optical Speed (RPM)"], -db["Torque (N·m)"]*db["Motor Optical Speed (RPM)"]*(2*np.pi)/60, color = "blue", s=1)
            ax[2, 0].scatter(db["Motor Optical Speed (RPM)"], -db["Thrust (N)"]/(rho*(db["Motor Optical Speed (RPM)"]*(2*np.pi/60))**2*(2*R)**4), color = "blue", s=1)
            ax[2, 1].scatter(db["Motor Optical Speed (RPM)"], -db["Torque (N·m)"]*db["Motor Optical Speed (RPM)"]*2*np.pi/(60*rho*(db["Motor Optical Speed (RPM)"]*(2*np.pi/60))**3*(2*R)**5), color = "blue", s=1)
            ax[2, 2].scatter(db["Motor Optical Speed (RPM)"], F[40:len(O1)+40]/-db["Thrust (N)"], color = "blue", s=1)
            ax[2, 2].set_title("Rapport poussée théorique, poussée expérimentale")
            ax[2, 2].set_xlabel("Vitesse de rotation (tr/min)")
            ax[2, 2].set_ylabel("F_théo/F_expé")
        plt.legend()
        plt.show()

    return F

if __name__ == "__main__" :

    incid_max_sd7043 = [6.5,6.7,7.2,7.3,7.4,7.54,7.72,8.25,8.7,8.93,9.22,9.48,9.71,9.82,9.95, 10.21,10.42,10.60,10.81,11.05,11.15,11.28]
    incid_max_clarky = np.array([5.5, 5.5, 5.5, 5.7, 5.9, 5.9, 6, 6, 6.6, 7.5, 8.1, 8.4, 8.7, 9, 9.5, 10.2, 10.8, 11.3, 11.7, 11.7, 11.7, 12])
    Re = [2000, 5000, 8500, 12000, 15500, 19000, 22500, 26000, 29500, 33000, 36500, 40000, 43500, 47000, 50500, 54000, 57500, 61000, 64500, 68000, 71500, 80000]

    amax_int = ca.interpolant("amax", "bspline", [Re], incid_max_clarky)

    gabizarre_s = Helice_squelette(R, r0, 3, 27, "clarky")
    gabriel_s = Helice_squelette(R, r0, 3, 20, "clarky")
    gabriel_4pales_s = Helice_squelette(R, r0, 4, 20, "clarky")
    ag11_4pales_s = Helice_squelette(R, r0, 4, 27, "ag11")
    clarky_VF_s = Helice_squelette(R, r0, 4, 27, "clarky")
    sd7043_s = Helice_squelette(R, r0, 4, 27, "sd7043")

    sd7043 = Helice(np.pi/180*np.array([23.3896, 24.0764, 24.7225, 25.309, 25.8278, 26.25, 26.5186, 26.5546, 26.301, 25.7552, 24.9694, 24.028, 23.0204, 22.0232, 21.0894, 20.2424, 19.4862, 18.8146, 18.2202, 17.6993, 17.2532, 16.8789, 16.5416, 16.1251, 15.3892, 14.0493, 12.1442]), 1/100*np.array([0.800001, 1.04492, 1.25659, 1.40838, 1.48614, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.49398, 1.47103, 1.42995, 1.37419, 1.30828, 1.23598, 1.15956, 1.08011, 0.997965, 0.912575, 0.820869, 0.715675, 0.59379]), sd7043_s, "sd7043_4pales_v0")
    gabizarre = Helice(np.pi/180*np.array([39.3307, 37.5131, 35.8262, 34.4722, 33.3607, 29.9771, 23.774, 18.9439, 19.3118, 19.2442, 19.3875, 19.9394, 20.8381, 21.6244, 21.3809, 20.5254, 19.5879, 18.6397, 17.6985, 16.7753, 15.8647, 15.2519, 14.7514, 14.0816, 11.0324, 9.34691, 9.29033]), np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.43596, 1.34892, 1.27696, 1.5, 1.5, 0.79401])/100, gabizarre_s, "gabizarre")
    ag11_4pales = Helice(np.pi/180*np.array([22.4222, 23.4483, 24.3515, 25.0287, 25.3581, 25.2755, 24.7976, 23.9949, 22.9605, 21.7869, 20.5544, 19.329, 18.1623, 17.0904, 16.1475, 15.3572, 14.7017, 14.1359, 13.6185, 13.1305, 12.6725, 12.2543, 11.8845, 11.5571, 11.2345, 10.8447, 10.3456]), np.array([0.600001, 0.973888, 1.30482, 1.55599, 1.71469, 1.7874, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.78829,1.75177, 1.68558, 1.58955, 1.46571, 1.31696, 1.14676, 0.960048, 0.763806, 0.565364])/100, ag11_4pales_s, "ag11_4pales")
    helicegabriel5000a15000 = Helice(np.pi/180*np.array([40.2631, 35.3814, 32.2878, 30.1048, 28.4588, 27.1606, 26.1028, 25.2193, 24.467, 23.8163, 23.2462, 22.511, 21.6298, 20.8552, 20.1694, 19.558, 19.0096, 18.5152, 18.0671, 17.6591]), np.array([0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.0144592, 0.0134184, 0.0124968, 0.0116756, 0.0109397, 0.0102769, 0.00967708, 0.00913186, 0.0086343]), gabriel_s, "helicegabriel5000a15000")
    gabriel_4pales = Helice(np.pi/180*np.array([40.2631, 35.3814, 32.2878, 30.1048, 28.4588, 27.1606, 26.1028, 25.2193, 24.467, 23.8163, 23.2462, 22.511, 21.6298, 20.8552, 20.1694, 19.558, 19.0096, 18.5152, 18.0671, 17.6591]), np.array([0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.0144592, 0.0134184, 0.0124968, 0.0116756, 0.0109397, 0.0102769, 0.00967708, 0.00913186, 0.0086343]), gabriel_4pales_s, "gabriel_4pales_s")
    clarky_4pale_V2 = Helice(np.pi/180*np.array([18.6938, 18.9671, 19.2654, 19.5213, 19.7364, 19.9694, 20.3572, 20.8546, 21.426, 21.9927, 22.4395, 22.6505, 22.5437, 22.1024, 21.3857, 20.5005, 19.5484, 18.5898, 17.645, 16.723, 15.8305, 14.9632, 14.0913, 13.1561, 12.0796, 10.8064, 9.37462]), 1/100*np.array([0.800001, 1.02699, 1.22347, 1.37314, 1.46512, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.49414, 1.4784, 1.4514, 1.41179, 1.35659, 1.2812, 1.18254, 1.06616]), clarky_VF_s, "clarky_4pale_V2")
    clarky_4pales_V3 = Helice(np.pi/180*np.array([18.6938, 18.9671, 19.2654, 19.5213, 19.7364, 19.9694, 20.3572, 20.8546, 21.426, 21.9927, 22.4395, 22.6505, 22.5437, 22.1024, 21.3857, 20.5005, 19.5484, 18.5898, 17.645, 16.723, 15.8305, 14.9632, 14.0913, 13.1561, 12.0796, 10.8064, 9.37462]), 1/100*np.array([0.800001, 1.02699, 1.22347, 1.37314, 1.46512, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.49414, 1.4784, 1.4514, 1.41179, 1.35659, 1.2812, 1.18254, 1.06616]), clarky_VF_s, "clarky_4pale_V3") 

    sol = hélice_optim(clarky_VF_s, courbes = True)

    performances_théo_expé(clarky_4pale_V2, expé = False, courbes = True)







