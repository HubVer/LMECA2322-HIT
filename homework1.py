import numpy as np
import matplotlib.pyplot as plt
import scipy


def fetch_data(x):
    u = {"x": [], "y": [], "z": []}
    directions = ["x", "y", "z"]
    for m in directions:
        for i in range(4):
            for j in range(4):
                data = np.loadtxt(f"Data/pencils_{m}/{m}_{i}_{j}.txt")
                u[m].append(data[:, x])
    u = np.concatenate([u["x"], u["y"], u["z"]])
    return u


def TKE(u, v, w):
    # Compute the Turbulent Kinetic Energy (TKE) for homgeoneous isotropic turbulence
    # velocities: numpy array of shape (3, 4, 4, n) containing velocity components where n is the number of data points
    # returns: scalar value of TKE
    return 1.5 * np.mean(u**2), 0.5 * (np.mean(u*u) + np.mean(v*v) + np.mean(w*w))


def diss_rate(velocities, nu):
    # Compute the dissipation rate of TKE
    # TKE: scalar value of Turbulent Kinetic Energy
    # nu: kinematic viscosity
    # returns: scalar value of dissipation rate
    L = 2*np.pi
    dx = L / 32768
    du_dx = (np.roll(velocities, -2) - 8*np.roll(velocities, -1) + 8*np.roll(velocities, 1) - np.roll(velocities, 2)) / (12 * dx)
    return 15*nu*np.mean(du_dx**2)


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


def structure_fuctions(velocities):
    pass


if __name__ == "__main__":
    nu = 1.10555e-5 # kinematic viscosity used  (m^2/s)

    u = fetch_data(0) # fetch u velocity data
    v = fetch_data(1) # fetch v velocity data
    w = fetch_data(2) # fetch w velocity data

    k, k_anis = TKE(u, v, w)
    print("k : "+str(k))
    print("k anisotropic : "+str(k_anis))
    e = diss_rate(u ,nu)
    print("epsilon : "+str(e))
    L = integral_length_scale(k,e)
    print("L : "+str(L))
    Re = Reynolds(k,e,nu)
    print("Re : "+str(Re))
    eta = Kolomogorov_length_scale(nu,e)
    print("eta : "+str(eta))
    lamb = Taylor_microscale(k,e)
    print("lambda ; "+str(lamb))
    Re_T = Reynolds_Taylor(k,e,nu,lamb) 
    print("Re_T : "+str(Re_T))