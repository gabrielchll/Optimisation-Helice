import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from profiles import Profil
import pandas as pd
import optim as opt
from scipy.optimize import least_squares
import copy

R = 0.06 # Rayon max hélice (m)
r0 = 0.006 # Début hélice (m)
rho = 1.2 # Masse volumique de l'air (kg/m**3)
nu = 1.5e-5 # Viscosité de l'air
m = 1.2 # Masse du drone
g_ = 9.81
omeg_max = 1300 # rad/s  
omeg_min = 500

df = pd.read_csv(f"Expériences/ag11_4pales.csv")
df = df[df["Motor Optical Speed (RPM)"] > 1000]
index_max = df["Motor Optical Speed (RPM)"].idxmax()
db = df.loc[:index_max]
thrust = db["Thrust (N)"]
O = db["Motor Optical Speed (RPM)"]

def polynome(c, x) :
    pol = 0
    for i in range(0, len(c)) :
        pol = pol + c[i]*x**(len(c) - 1 - i)
    return pol

def regression_expé_thrust(helice, omeg) :
    df = pd.read_csv(f"Expériences/{helice.nom}.csv")
    df = df[df["Motor Optical Speed (RPM)"] > 1000]
    index_max = df["Motor Optical Speed (RPM)"].idxmax()
    db = df.loc[:index_max]
    thrust = db["Thrust (N)"]
    O = db["Motor Optical Speed (RPM)"]
    coeffs = np.polyfit(O, thrust, 2)
    return polynome(coeffs, omeg)

def residus(theta, helice):
    kp, kt = theta

    df = pd.read_csv(f"Expériences/{helice.nom}.csv")
    df = df[df["Motor Optical Speed (RPM)"] > 1000]
    index_max = df["Motor Optical Speed (RPM)"].idxmax()
    db = df.loc[:index_max]
    thrust = db["Thrust (N)"]
    O = db["Motor Optical Speed (RPM)"]

    F_expé = regression_expé_thrust(helice, O)

    helice_c = copy.deepcopy(helice)
    helice_c.Cp = kp*helice.Cp
    helice_c.Ct = kt*helice.Ct

    F_théo = opt.performances_théo_expé(helice_c, O)

    res_T = (F_théo - F_expé) / np.max(F_expé)

    return res_T

if __name__ == "__main__" :

    ag11_4pales_s = opt.Helice_squelette(R, r0, 4, 27, "ag11")
    ag11_4pales = opt.Helice(np.pi/180*np.array([22.4222, 23.4483, 24.3515, 25.0287, 25.3581, 25.2755, 24.7976, 23.9949, 22.9605, 21.7869, 20.5544, 19.329, 18.1623, 17.0904, 16.1475, 15.3572, 14.7017, 14.1359, 13.6185, 13.1305, 12.6725, 12.2543, 11.8845, 11.5571, 11.2345, 10.8447, 10.3456]), np.array([0.600001, 0.973888, 1.30482, 1.55599, 1.71469, 1.7874, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.78829,1.75177, 1.68558, 1.58955, 1.46571, 1.31696, 1.14676, 0.960048, 0.763806, 0.565364])/100, ag11_4pales_s, "ag11_4pales")
       
    res = least_squares(
    residus, 
    x0=[1.0, 1.0], 
    bounds=([0.5, 0.5], [1.5, 3.0]), 
    args=(ag11_4pales,)
    )

    print(f"Facteurs correctifs trouvés : kL = {res.x[0]:.3f}, kD = {res.x[1]:.3f}")
     
    plt.plot(O, thrust)
    plt.plot(O, regression_expé_thrust(ag11_4pales, O), color = "red")
    plt.show()