import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import neuralfoil as nf
import aerosandbox as asb

incidence = np.linspace(-5, 24, 50)
reynolds = np.linspace(300, 100000, 50)

incidence_grid, Re_grid = np.meshgrid(incidence, reynolds)

class Profil :

    def __init__(self, profil) :

        aero = nf.get_aero_from_airfoil(airfoil=asb.Airfoil(profil), alpha=incidence_grid.flatten(), Re=Re_grid.flatten(), model_size="xxxlarge")
        CT = aero["CD"].reshape(incidence_grid.shape)
        CP = aero["CL"].reshape(incidence_grid.shape)      
        CT_int = ca.interpolant("CT_int", "bspline", [incidence.tolist(), reynolds.tolist()], CT.T.ravel(order="F").tolist())
        CP_int = ca.interpolant("CP_int", "bspline", [incidence.tolist(), reynolds.tolist()], CP.T.ravel(order="F").tolist())

        self.Cp = lambda i, Re: CP_int(ca.horzcat(i, Re + 0*i).T).T
        self.Ct = lambda i, Re: CT_int(ca.horzcat(i, Re + 0*i).T).T

colors = [
    "#da0b0b", "#da370b", "#da620b", "#da8e0b", "#dab90b",
    "#cfda0b", "#a4da0b", "#78da0b", "#4dda0b", "#21da0b",
    "#0bda21", "#0bda4d", "#0bda78", "#0bdaa4", "#0bdacf",
    "#0bb9da", "#0b8eda", "#0b62da", "#0b37da", "#0b0bda",
]

def courbe_Cp(profil, ax, n = 7) :
    incidence = np.linspace(0,16,50)
    for i in range(0,n) :
        ax.plot(incidence, profil.Cp(incidence, 5000+i*int(70000/n)), label = f"Re = {5000+int(70000/n)*i}", color = colors[i])
        ax.set_xlabel("incidence (en °)")
        ax.set_ylabel("Cp")
        ax.set_title("Evolution du coefficient de portance (Cp) en fonction de l'angle d'incidence pour différents reynolds")
    ax.legend()
    ax.grid()

def courbe_Ct(profil, ax, n = 7) :
    incidence = np.linspace(0,16,50)
    for i in range(0,n) :
        ax.plot(incidence, profil.Ct(incidence, 5000+i*int(70000/n)), label = f"Re = {5000+int(70000/n)*i}", color = colors[i])
        ax.set_xlabel("incidence (en °)")
        ax.set_ylabel("Ct")
        ax.set_title("Evolution du coefficient de trainé (Ct) en fonction de l'angle d'incidence pour différents reynolds")
    ax.legend()
    ax.grid()

if __name__ == "__main__" :

    ag11 = Profil("ag11")
    clarky = Profil("clarky")
    sd7043 = Profil("sd7043")

    fig, ax = plt.subplots(1, 2)

    courbe_Cp(ag11, ax[0], 10)
    courbe_Cp(clarky, ax[1], 10)
    plt.show()



    

    """fig, ax = plt.subplots()

    db = pd.read_csv("CoeffsProfils.csv", sep=";", decimal=",")
    ax.plot(db[db["profil"] == "SD7043"]["alpha_deg"], db[db["profil"] == "SD7043"]["Cz_moy"])
    ax.plot(np.linspace(0, 16, 10), sd7043.Cp(np.linspace(0, 16, 10), 49000))
    plt.show()"""

    """l = ["ag11", "ag10", "ag09", "ag08", "ag04", "ag03", "sd7003"]
    Re = 20000

    for t in range(0, len(l)) :
        p = Profil(l[t])
        plt.plot(np.linspace(0,8, 50), p.Cp(np.linspace(0,8,50), Re)/p.Ct(np.linspace(0,8,50), Re), color=colors[t])
    plt.show()"""

    """p = Profil("ag11")
    a = np.linspace(0,16,200)
    for i in range(1, 12) :
        plt.plot(a, p.Cp(a, 6000*i), color = colors[i], label= f"Re = {6000*i}")
        plt.legend()
    plt.xlabel("incidence (en °)")
    plt.show()"""
