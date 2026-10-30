import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import neuralfoil as nf
import aerosandbox as asb

# -------- Premier modèle de Cp et Ct --------------
# Modèles du TD, dépendances uniquement selon i, pas de Re

def Cp_TD(i, Re, kp = 0.1, ip = 0) :
    i_deg = i*180/np.pi
    return kp*(i_deg - ip)

def Ct_TD(i, Re, kt = 0.25, it = 20) :
    i_deg = i*180/np.pi
    return kt/(it-i_deg)

# ---------- Clarky avec librairie Neurafoil -------------

incidence = np.linspace(0, 16, 34)
reynolds = np.linspace(5000, 100000, 34)

Alpha_grid, Re_grid = np.meshgrid(incidence, reynolds)

aero = nf.get_aero_from_airfoil(airfoil=asb.Airfoil("clarky"), alpha=Alpha_grid.flatten(), Re=Re_grid.flatten(), model_size="xlarge")

Ct = aero["CD"].reshape(Alpha_grid.shape)
Cp = aero["CL"].reshape(Alpha_grid.shape)

Ct_int = ca.interpolant("Ct_int", "bspline", [incidence.tolist(), reynolds.tolist()], Ct.T.ravel(order="F").tolist())
Cp_int = ca.interpolant("Cp_int", "bspline", [incidence.tolist(), reynolds.tolist()], Cp.T.ravel(order="F").tolist())

def Ct_clarky(i, Re) :
    Re = Re + 0 * i       
    return Ct_int(ca.horzcat(i, Re).T).T

def Cp_clarky(i, Re) :
    Re = Re + 0 * i
    return Cp_int(ca.horzcat(i, Re).T).T