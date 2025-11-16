import numpy as np
import matplotlib.pyplot as plt
import scipy

# Common parameters
nu = 1.5e-5  # kinematic viscosity
L = 2 * np.pi  # domain size



def fetch_data(i):
    # Fetch velocity data from all pencil files for the i-th component
    # i: index of the velocity component to fetch (0 for u, 1 for v, 2 for w)
    # returns: dictionary with velocities data for x, y, z pencils
    velocities = {"x": [], "y": [], "z": []}
    for j in velocities.keys():
        for m in range(4):
            for n in range(4):
                filename = f"Data/pencils_{j}/{j}_{m}_{n}.txt"
                data = np.loadtxt(filename)
                velocities[j].append(data[:, i])
    return velocities

def TKE(u, v, w):
    # Compute the Turbulent Kinetic Energy (TKE) for homgeoneous isotropic turbulence
    # u, v, w: dictionnaries of list of arrays containing velocity components
    # returns: scalar value of isotropic TKE and anisotropic TKE
    k_iso = np.array([])
    k_aniso = np.array([])
    for i in u.keys():
        for j in range(len(u[i])):
            k_iso = np.append(k_iso, 1.5 * np.mean(u[i][j]*u[i][j]))
            k_aniso = np.append(k_aniso, 0.5 * (np.mean(u[i][j]**2) + np.mean(v[i][j]**2) + np.mean(w[i][j]**2)))
    return np.mean(k_iso), np.mean(k_aniso)


def diss_rate(u, nu):
    # Compute the dissipation rate of TKE
    # TKE: scalar value of Turbulent Kinetic Energy
    # nu: kinematic viscosity
    # returns: scalar value of dissipation rate
    L = 2*np.pi
    dx = L / 32768
    e_all = []
    for pencil in u["x"]:
        du_dx = (np.roll(pencil, -2) - 8*np.roll(pencil, -1) + 8*np.roll(pencil, 1) - np.roll(pencil, 2)) / (12 * dx)
        e_all.append(15*nu*np.mean(du_dx**2))
    return np.mean(e_all)


def get_pencils(n):
    u = {"x": [], "y": [], "z": []}
    directions = ["x", "y", "z"]
    for m in directions:
        for i in range(4):
            for j in range(4):
                filename = f"Data/pencils_{m}/{m}_{i}_{j}.txt"
                data = np.loadtxt(filename)
                u[m].append(data[:, n])
    return u


def get_dissation_rate(u):
    h = 2*np.pi / 32768
    res = np.array([])
    for pencils in u["x"]:
        dudx = 1/12 * (-np.roll(pencils, 2) + 8 * np.roll(pencils, 1) - 8 * np.roll(pencils, -1) + np.roll(pencils, -2)) / h
        res = np.concatenate((res, dudx))
    return 15 * nu * np.mean(res**2)


def integral_length_scale(k, e):
    # Compute the integral length scale
    # k: scalar value of Turbulent Kinetic Energy
    # e: scalar value of dissipation rate
    # returns: scalar value of integral length scale
    return k**1.5 / e


def Reynolds(k, e, nu):
    # Compute the Reynolds number
    # k: scalar value of Turbulent Kinetic Energy
    # e: scalar value of dissipation rate
    # nu: kinematic viscosity
    # returns: scalar value of Reynolds number
    L = integral_length_scale(k, e)
    return (k**2) / (nu * e)


def Kolomogorov_length_scale(nu, e):
    # Compute the Kolomogorov length scale
    # nu: kinematic viscosity
    # e: scalar value of dissipation rate
    # returns: scalar value of Kolomogorov length scale
    return (nu**3 / e)**0.25


def Taylor_microscale(k, e):
    # Compute the Taylor microscale
    # k: scalar value of Turbulent Kinetic Energy
    # e: scalar value of dissipation rate
    # returns: scalar value of Taylor microscale
    return np.sqrt(10 * nu * k / e)


def Reynolds_Taylor(k, e, nu, lambda_t):
    # Compute the Taylor Reynolds number
    # k: scalar value of Turbulent Kinetic Energy
    # e: scalar value of dissipation rate
    # nu: kinematic viscosity
    # returns: scalar value of Taylor Reynolds number
    lambda_t = Taylor_microscale(k, e)
    return lambda_t / nu * np.sqrt(2 * k / 3)


def structure_fuctions(u, v, w, r):
    indexes = np.arange(0, len(u["x"][0]), r, dtype=int)
    D11 = []
    D22 = []
    D33 = []
    for i in range(len(indexes)-1):
        D11.append(np.mean( (u["x"][0][indexes[i+1]] - u["x"][0][indexes[i]])**2 ))
        D22.append(np.mean( (v["y"][0][indexes[i+1]] - v["y"][0][indexes[i]])**2 ))
        D33.append(np.mean( (w["z"][0][indexes[i+1]] - w["z"][0][indexes[i]])**2 ))
    return np.mean(D11), ( np.mean(D22) + np.mean(D33) )/2


if __name__ == "__main__":

    u = fetch_data(0)
    v = fetch_data(1)
    w = fetch_data(2)

    k_i, k_a = TKE(u, v, w)
    print("k isotropic : "+str(k_i))
    print("k anisotropic : "+str(k_a))

    e = diss_rate(u, nu)
    print("dissipation rate e :", e)

    eta = Kolomogorov_length_scale(nu,e)
    print("Kolomogorov length scale eta :", eta)
    print("rmax :", eta*5e3)

    r = np.linspace(1, eta*5e3, 1000)

    """
    u = np.array([])
    v = np.array([])
    w = np.array([])
    k_iso_all = np.array([])
    k_anis_all = np.array([])
    e_all = np.array([])
    directions = ["x", "y", "z"]
    for m in directions:
        for i in range(4):
            for j in range(4):
                data = np.loadtxt(f"Data/pencils_{m}/{m}_{i}_{j}.txt")
                u = np.append(u, data[:, 0])
                v = np.append(v, data[:, 1])
                w = np.append(w, data[:, 2])
                k_iso_all = np.append(k_iso_all, TKE(data[:, 0], data[:, 1], data[:, 2])[0])
                k_anis_all = np.append(k_anis_all, TKE(data[:, 0], data[:, 1], data[:, 2])[1])
                e_all = np.append(e_all, diss_rate(data[:, 0], nu))

    print(np.shape(u))

    k_iso, k_anis = np.mean(k_iso_all), np.mean(k_anis_all)
    print("k isotropic : "+str(k_iso))
    print("k anisotropic : "+str(k_anis))
    e = np.mean(e_all)
    print("epsilon : "+str(e))
    
    # e = 1.4795797446290537 reference value

    L = integral_length_scale(k_iso,e)
    print("L : "+str(L))
    Re = Reynolds(k_iso,e,nu)
    print("Re : "+str(Re))
    eta = Kolomogorov_length_scale(nu,e)
    print("eta : "+str(eta))
    lamb = Taylor_microscale(k_iso,e)
    print("lambda ; "+str(lamb))
    Re_T = Reynolds_Taylor(k_iso,e,nu,lamb) 
    print("Re_T : "+str(Re_T))
    """
