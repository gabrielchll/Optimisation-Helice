import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import neuralfoil as nf
import aerosandbox as asb

incidence = np.linspace(-5, 24, 50)
reynolds = np.linspace(5000, 100000, 50)

incidence_grid, Re_grid = np.meshgrid(incidence, reynolds)

class Profil :

    def __init__(self, profil) :

        aero = nf.get_aero_from_airfoil(airfoil=asb.Airfoil(profil), alpha=incidence_grid.flatten(), Re=Re_grid.flatten(), model_size="xlarge")
        CT = aero["CD"].reshape(incidence_grid.shape)
        CP = aero["CL"].reshape(incidence_grid.shape)      
        CT_int = ca.interpolant("CT_int", "bspline", [incidence.tolist(), reynolds.tolist()], CT.T.ravel(order="F").tolist())
        CP_int = ca.interpolant("CP_int", "bspline", [incidence.tolist(), reynolds.tolist()], CP.T.ravel(order="F").tolist())

        self.Cp = lambda i, Re: CP_int(ca.horzcat(i, Re + 0*i).T).T
        self.Ct = lambda i, Re: CT_int(ca.horzcat(i, Re + 0*i).T).T
