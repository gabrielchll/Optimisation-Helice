import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

# -------- Premier modèle de Cp et Ct --------------
# Modèles du TD, dépendances uniquement selon i, pas de Re

def Cp_TD(i, kp = 0.1, ip = 0) :
    i_deg = i*180/np.pi
    return kp*(i_deg - ip)

def Ct_TD(i, kt = 0.25, it = 20) :
    i_deg = i*180/np.pi
    return kt/(it-i_deg)