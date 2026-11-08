import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import neuralfoil as nf
import aerosandbox as asb
from scipy.ndimage import gaussian_filter1d

class Profil :

    def __init__(self, profil, incid_max = np.full(22, 8.9)) :

        # Données pour les interpolations
        self.name = profil
        incidence = np.linspace(-5, 24, 50)
        reynolds = np.linspace(300, 100000, 50)
        incidence_grid, Re_grid = np.meshgrid(incidence, reynolds)
        Re = np.array([2000, 5000, 8500, 12000, 15500, 19000, 22500, 26000, 29500, 33000, 36500, 40000, 43500, 47000, 50500, 54000, 57500, 61000, 64500, 68000, 71500, 80000])
        aero = nf.get_aero_from_airfoil(airfoil=asb.Airfoil(profil), alpha=incidence_grid.flatten(), Re=Re_grid.flatten(), model_size="xxxlarge")

        # Interpolation pour les courbes des coefficients de portance et trainée

        CT = aero["CD"].reshape(incidence_grid.shape)
        CP = aero["CL"].reshape(incidence_grid.shape)
        sigma_val = 2
        CT = gaussian_filter1d(CT, sigma=sigma_val, axis=0)     # évite les changements de régime trop brutale (plus lisses dans la réalité) sans non plus les effacer 
        CP = gaussian_filter1d(CP, sigma=sigma_val, axis=0)  
        CT_int = ca.interpolant("CT_int", "bspline", [incidence.tolist(), reynolds.tolist()], CT.T.ravel(order="F").tolist())
        CP_int = ca.interpolant("CP_int", "bspline", [incidence.tolist(), reynolds.tolist()], CP.T.ravel(order="F").tolist())

        self.Cp = lambda i, Re: CP_int(ca.horzcat(i, Re + 0*i).T).T
        self.Ct = lambda i, Re: CT_int(ca.horzcat(i, Re + 0*i).T).T

        # Interpolation pour l'angle d'incidence de décrochage

        self.i_max = ca.interpolant("i_max", "bspline", [Re], incid_max)    # Par défauts c'est une constante à i_max = 10° (pas idéal mais dans l'idée la liste des angles de décrochages est tout le temps renseigné)

def courbe_Cp(profil, ax, n = 22, style = "-") :

    incidence = np.linspace(0,16,50)
    colors = plt.cm.turbo(np.linspace(0, 1, n))

    # Fait en deux étapes pour avoir par défaut les mêmes Re que ceux nécessaires pour donner les angles de décrochages d'un profil (mal fait mais l'intervalle entre les Re n'était pas constant)

    ax.plot(incidence, profil.Cp(incidence, 2000), label = f"Re = 2000 {profil.name}", color = colors[0], linestyle = style)
    for i in range(0,n-2) :
        ax.plot(incidence, profil.Cp(incidence, 5000+i*int(70000/n)), label = f"Re = {5000+int(70000/n)*i} {profil.name}", color = colors[i+1], linestyle = style)
    ax.plot(incidence, profil.Cp(incidence, 80000), label = f"Re = 80000 {profil.name}", color = colors[-1], linestyle = style)

    ax.set_xlabel("incidence (en °)")
    ax.set_ylabel("Cp")
    ax.set_title("Evolution du coefficient de portance (Cp) en fonction de l'angle d'incidence pour différents reynolds")
    ax.legend()
    ax.grid()

def courbe_Ct(profil, ax, n = 22) :

    incidence = np.linspace(0,16,50)
    colors = plt.cm.turbo(np.linspace(0, 1, n))

    ax.plot(incidence, profil.Ct(incidence, 2000), label = f"Re = 2000", color = colors[0])
    for i in range(0,n) :
        ax.plot(incidence, profil.Ct(incidence, 5000+i*int(70000/n)), label = f"Re = {5000+int(70000/n)*i}", color = colors[i+1])
    ax.plot(incidence, profil.Ct(incidence, 80000), label = f"Re = 80000", color = colors[-1])

    ax.set_xlabel("incidence (en °)")
    ax.set_ylabel("Ct")
    ax.set_title("Evolution du coefficient de trainé (Ct) en fonction de l'angle d'incidence pour différents reynolds")
    ax.legend()
    ax.grid()

if __name__ == "__main__" :

    ag11 = Profil("ag11")
    clarky = Profil("clarky")
    sd7043 = Profil("sd7043")

    fig, ax = plt.subplots()

    courbe_Cp(sd7043, ax, 8)
    courbe_Cp(clarky, ax, 8, "--")
    plt.show()