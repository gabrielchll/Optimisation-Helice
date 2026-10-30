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

incidence = np.linspace(-5, 24, 50)
reynolds = np.linspace(5000, 100000, 50)

Alpha_grid, Re_grid = np.meshgrid(incidence, reynolds)

aero_clarky = nf.get_aero_from_airfoil(airfoil=asb.Airfoil("clarky"), alpha=Alpha_grid.flatten(), Re=Re_grid.flatten(), model_size="xlarge")

Ct_clarky = aero_clarky["CD"].reshape(Alpha_grid.shape)
Cp_clarky = aero_clarky["CL"].reshape(Alpha_grid.shape)

Ct_int_clarky = ca.interpolant("Ct_int_clarky", "bspline", [incidence.tolist(), reynolds.tolist()], Ct_clarky.T.ravel(order="F").tolist())
Cp_int_clarky = ca.interpolant("Cp_int_clarky", "bspline", [incidence.tolist(), reynolds.tolist()], Cp_clarky.T.ravel(order="F").tolist())

def Ct_clarky(i, Re) :
    Re = Re + 0 * i       
    return Ct_int_clarky(ca.horzcat(i, Re).T).T

def Cp_clarky(i, Re) :
    Re = Re + 0 * i
    return Cp_int_clarky(ca.horzcat(i, Re).T).T

#------------------------------------- test autre profil ------------------------------------------

aero_goe79 = nf.get_aero_from_airfoil(airfoil=asb.Airfoil("goe79"), alpha=Alpha_grid.flatten(), Re=Re_grid.flatten(), model_size="xlarge")

Ct_goe79 = aero_goe79["CD"].reshape(Alpha_grid.shape)
Cp_goe79 = aero_goe79["CL"].reshape(Alpha_grid.shape)

Ct_int_goe79 = ca.interpolant("Ct_int_goe79", "bspline", [incidence.tolist(), reynolds.tolist()], Ct_goe79.T.ravel(order="F").tolist())
Cp_int_goe79 = ca.interpolant("Cp_int_goe79", "bspline", [incidence.tolist(), reynolds.tolist()], Cp_goe79.T.ravel(order="F").tolist())

def Ct_goe79(i, Re) :
    Re = Re + 0 * i       
    return Ct_int_goe79(ca.horzcat(i, Re).T).T

def Cp_goe79(i, Re) :
    Re = Re + 0 * i
    return Cp_int_goe79(ca.horzcat(i, Re).T).T

#----------------------------------------- aucun un autre -----------------------------------------

aero_ag11 = nf.get_aero_from_airfoil(airfoil=asb.Airfoil("ag11"), alpha=Alpha_grid.flatten(), Re=Re_grid.flatten(), model_size="xlarge")

Ct_ag11 = aero_ag11["CD"].reshape(Alpha_grid.shape)
Cp_ag11 = aero_ag11["CL"].reshape(Alpha_grid.shape)

Ct_int_ag11 = ca.interpolant("Ct_int_ag11", "bspline", [incidence.tolist(), reynolds.tolist()], Ct_ag11.T.ravel(order="F").tolist())
Cp_int_ag11 = ca.interpolant("Cp_int_ag11", "bspline", [incidence.tolist(), reynolds.tolist()], Cp_ag11.T.ravel(order="F").tolist())

def Ct_ag11(i, Re) :
    Re = Re + 0 * i       
    return Ct_int_ag11(ca.horzcat(i, Re).T).T

def Cp_ag11(i, Re) :
    Re = Re + 0 * i
    return Cp_int_ag11(ca.horzcat(i, Re).T).T