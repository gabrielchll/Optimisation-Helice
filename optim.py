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
m = 0.1 # masse du drone
g_ = 9.81
Re_min = 50000

def Reynolds(r, x) :          # Renvoie le tableau array des nombres de Reynolds dépendant de la variable via la corde
    dr = r[1] - r[0]
    n = len(r)
    return (rho * x[n-1:] * ca.sqrt((r[:-1] + dr/2)**2*omeg**2 + Vz**2))/nu

def alpha(r) :
    dr = r[1] - r[0]
    return np.arctan(Vz/((r[:-1] + dr/2)*omeg))

def puissance(r, x, Cp, Ct, N_pale) :
    dP = puiss_diff(r, x, Cp, Ct, N_pale)
    return ca.sum1(dP)

def force_cond(r, x, Cp, Ct, N_pale) :
    dF = force_diff(r, x, Cp, Ct)
    return ca.sum1(dF) - (m*g_)/(4*N_pale)

def puiss_diff(r, x, Cp, Ct, N_pale) : # Renvoie la liste des puissances de chaque section de la pale
    Re = Reynolds(r, x)
    a = alpha(r)
    n = len(r)
    dr = r[1] - r[0]
    return 1/2*rho*omeg*ca.sqrt((r[:-1] + dr/2)**2*omeg**2 + Vz**2) * (Cp(x[:n-1] - a) * Vz  + Ct(x[:n-1] - a) * (r[:-1] + dr/2) * omeg) * x[n-1:] * (r[:-1] + dr/2) * dr * N_pale

def force_diff(r, x, Cp, Ct) :
    dr = r[1] - r[0]
    Re = Reynolds(r, x)
    a = alpha(r)
    n = len(r)
    return 1/2*rho*ca.sqrt((r[:-1] + dr/2)**2*omeg**2 + Vz**2)*(Cp(x[:n-1] - a) * (r[:-1] + dr/2) * omeg - Ct(x[:n-1] - a) * Vz) * x[n-1:] * dr

def incidence(r, x) :
    n = len(r)
    return x[:n-1] - alpha(r)

def get_problem(N, Cp, Ct, N_pale) :

    dr = (R - r0)/N
    r = np.r_[0.0, np.cumsum(np.full(N, dr))] + r0
    
    x = ca.SX.sym("x", 2*(N))

    return {"x" : x,
            "f" : puissance(r, x, Cp, Ct, N_pale),
            "g" : ca.vertcat(force_cond(r, x, Cp, Ct, N_pale), incidence(r, x), Reynolds(r, x))}

def hélice_optim(N, Cp, Ct, i_max, N_pale) :

    dr = (R - r0)/N
    r = np.r_[0.0, np.cumsum(np.full(N, dr))] + r0

    pb = get_problem(N, Cp, Ct, N_pale)

    solver = ca.nlpsol('nlp', 'ipopt', pb)

    solution = solver(
        x0 = np.r_[alpha(r) + 5*np.pi/180, np.full(N, 0.01)],                    # On initialise l'angle de callage à 10° partout et la corde à 1 cm partour
        lbx = np.r_[np.full(N, 0.0), np.full(N, 0.002)],                              # On minimise l'angle de callage à 0° et la corde à 2 mm (pour éviter Re = 0 => 0 frottements mais touujorus portance...)
        ubx = np.r_[np.full(N, np.inf), np.full(N, 0.07)],                           # On maximise l'angle de callage par l'infini et la corde par 1.5 cm (on ne faut pas des ailes de 3 m meme si c'est "optimal" en théorie)
        lbg = np.r_[0.0, np.full(N, 0.0), np.full(N, Re_min)],                                                                  # Condition d'égalité       
        ubg = np.r_[0.0, np.full(N, i_max*np.pi/180), np.full(N, np.inf)])

    print(solver.stats()["return_status"])

    return solution

sol_prof = hélice_optim(20, coeff.Cp, coeff.Ct, 10, 2)

r = np.r_[0.0, np.cumsum(np.full(20, (R-r0)/20))] + r0
r1 = r[:-1] + (R-r0)/40

X = sol_prof["x"]

fig, ax1 = plt.subplots()
fig, ax2 = plt.subplots()

ax1.plot(r1, X[:20])
ax2.plot(r1, X[20:])

print(sol_prof["g"][0])                     # résidu de poussée, doit être ~1e-8
print(np.array(sol_prof["g"][1:]).min())    # incidence min, doit être >= 0

plt.show()

