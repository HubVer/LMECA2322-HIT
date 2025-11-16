import numpy as np
import matplotlib.pyplot as plt
import scipy


def TKE(u, v, w):
    # Compute the Turbulent Kinetic Energy (TKE) for homgeoneous isotropic turbulence
    # u, v, w: numpy arrays containing velocity components
    # returns: scalar value of isotropic TKE and anisotropic TKE
    return 1.5 * np.mean(u**2), 0.5 * (np.mean(u*u) + np.mean(v*v) + np.mean(w*w))


def diss_rate(velocities, nu):
    # Compute the dissipation rate of TKE
    # TKE: scalar value of Turbulent Kinetic Energy
    # nu: kinematic viscosity
    # returns: scalar value of dissipation rate
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

def spectrum_1D(u, dx):
    #Compute the 1D energy spectrum of u by using FFT
    #u: numpy array containing velocity components
    #dx: spatial resolution (grid spacing)
    #returns the 1D energy spectrum with the associated wavenumber
    u = np.asarray(u, dtype=float)
    u_bis = np.fft.rfft(u)
    N = len(u)
    k = 2.0 * np.pi * np.fft.rfftfreq(N, d=dx)
    E = np.zeros_like(k, dtype=float)
    if len(k) > 1: dk = k[1] - k[0]
    else: dk = 1.0
    if N % 2 == 0:
        if len(k) > 2:
            E[1:-1] = (np.abs(u_bis[1:-1])**2) / (N**2 * dk)
        E[-1] = (np.abs(u_bis[-1])**2) / (2.0 * N**2 * dk)
    else:
        if len(k) > 1: E[1:] = (np.abs(u_bis[1:])**2) / (N**2 * dk)
    return E,k


if __name__ == "__main__":
    nu = 1.10555e-5 # kinematic viscosity used  (m^2/s)
    L = 2*np.pi # (m)
    dx = L / 32768 #32678 is the size of each pencil's data

    u = np.array([])
    v = np.array([])
    w = np.array([])
    k_iso_all = np.array([])
    k_anis_all = np.array([])
    e_all = np.array([])

    #One-dimensional energy spectra
    E11 = []      # longitudinal
    E22 = []      # transverse 1
    E33 = []      # transverse 2
    k1 = None   # k grid common for all 1D-spectrums
    # Theoritical Kolmogorov coefficients for 1D-spectrums
    C1 = 0.52      # est la constante universelle du spectre 1D longitudinal pour turbulence isotrope
    C2 = 0.30      # est la constante universelle des spectres 1D transverses.

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
                if m == "x":
                    long = data[:, 0]
                    trans22 = data[:, 1]
                    trans33 = data[:, 2]
                elif m == "y":
                    trans22 = data[:, 0]
                    long = data[:, 1]
                    trans33 = data[:, 2]
                elif m == "z":
                    trans22 = data[:, 0]
                    trans33 = data[:, 1]
                    long = data[:, 2]
                E_long,k_local = spectrum_1D(long,dx)
                E_trans22, _ = spectrum_1D(trans22, dx)
                E_trans33, _ = spectrum_1D(trans33, dx)    
                if k1 is None:
                    k1 = k_local
                E11.append(E_long)
                E22.append(E_trans22)
                E33.append(E_trans33)

    k_iso, k_anis = np.mean(k_iso_all), np.mean(k_anis_all)
    print("k isotropic : "+str(k_iso))
    print("k anisotropic : "+str(k_anis))
    e = np.mean(e_all)
    print("epsilon : "+str(e))
    L_ = integral_length_scale(k_iso,e)
    print("L : "+str(L_))
    Re = Reynolds(k_iso,e,nu)
    print("Re : "+str(Re))
    eta = Kolomogorov_length_scale(nu,e)
    print("eta : "+str(eta))
    lamb = Taylor_microscale(k_iso,e)
    print("lambda ; "+str(lamb))
    Re_T = Reynolds_Taylor(k_iso,e,nu,lamb) 
    print("Re_T : "+str(Re_T))

    E11_mean = np.mean(np.vstack(E11), axis=0)
    E22_mean = np.mean(np.vstack(E22), axis=0)
    E33_mean = np.mean(np.vstack(E33), axis=0)
    E_trans = (E22_mean + E33_mean)/2
    k_eta = k1 * eta # dimonsionless
    mask = k_eta > 0.0
    E11_dimless = E11_mean / ((e * nu**5)**0.25)
    Etrans_dimless = E_trans / ((e * nu**5)**0.25)
    k_eta_nz = k_eta[mask]
    E11_dimless_nz = E11_dimless[mask]
    Etrans_dimless_nz = Etrans_dimless[mask]
    # --- Spectres adimensionnels ---
    plt.loglog(k_eta_nz, E11_dimless_nz, color='red', label=r"$E_{11} / (\varepsilon \nu^5)^{1/4}$")
    plt.loglog(k_eta_nz, Etrans_dimless_nz, color='orange', label=r"$E_{22} / (\varepsilon \nu^5)^{1/4}$")
    # Courbes théoriques ~ C1, C2 (kη)^(-5/3)
    k_th = np.linspace(k_eta_nz.min(), k_eta_nz.max(), 200)
    plt.loglog(k_th, C1 * k_th**(-5.0/3.0), ls='--', label=r"$C_1 (k\eta)^{-5/3}$")
    plt.loglog(k_th, C2 * k_th**(-5.0/3.0), color='black', ls='--', label=r"$C_2 (k\eta)^{-5/3}$")
    # Paramètres du plot
    plt.xlabel(r"$k\eta$")
    plt.ylabel(r"$E_{ii} / (\varepsilon \nu^5)^{1/4}$")
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()

    # --- Spectres compensés ---
    E11_comp_spec = (k_eta_nz**(5.0/3.0)) * E11_dimless_nz
    Etrans_comp_spec = (k_eta_nz**(5.0/3.0)) * Etrans_dimless_nz
    plt.loglog(k_eta_nz, E11_comp_spec, color='red',
               label=r"$(k\eta)^{5/3} E_{11} / (\varepsilon \nu^5)^{1/4}$")
    plt.loglog(k_eta_nz, Etrans_comp_spec, color='orange',
               label=r"$(k\eta)^{5/3} E_{22} / (\varepsilon \nu^5)^{1/4}$")

    plt.axhline(C1,color='black', ls='--', label=r"$C_1$")
    plt.axhline(C2,color='black', ls='--', label=r"$C_2$")

    plt.xlabel(r"$k\eta$")
    plt.ylabel("Compensated spectra")
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()
