import numpy as np
import matplotlib.pyplot as plt
import scipy
import os


def fetch_data():
    u_velocities = np.zeros((3, 4, 4, 32768)) # 32768 the same for all files

    directories = sorted(os.listdir('Data'))[2:] # index to remove some unwanted files
    for i, dir in enumerate(directories):
        files = sorted(os.listdir('Data/'+dir))[1:] # index to remove unwanted files

        for file in files:
            j, k = int(file[2]), int(file[4]) # get indices from file name
            u_velocities[i][j][k] = np.loadtxt('Data/'+dir+'/'+file)[:,i] # get u velocity

    return u_velocities


def TKE(velocities):
    # Compute the Turbulent Kinetic Energy (TKE) for homgeoneous isotropic turbulence
    # velocities: numpy array of shape (3, 4, 4, n) containing velocity components where n is the number of data points
    # returns: scalar value of TKE
    return 1.5 * np.mean(velocities**2)


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


if __name__ == "__main__":
    nu = 1.10555e-5 # kinematic viscosity used  (m^2/s)

    velocities = fetch_data()

    k = TKE(velocities)
    print("k : "+str(k))
    e = diss_rate(velocities,nu)
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