
import os, glob
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def dudx_fourth_order(u,dx):
    u = np.asarray(u, dtype=float)
    return (-np.roll(u, -2)
            + 8*np.roll(u, -1)
            - 8*np.roll(u,  1)
            +    np.roll(u,  2)) / (12.0 * dx)

def epsilon_moy(d, u, v, w, dx):
    du = dudx_fourth_order(u,dx)
    dv = dudx_fourth_order(v,dx)
    dw = dudx_fourth_order(w,dx)
    if(d == "pencils_x"):
        epsilon = 15* nu * (du**2)
    elif(d == "pencils_y"):
        epsilon = 15* nu * (dv**2)
    elif(d == "pencils_z"):
        epsilon = 15* nu * (dw**2)
    return np.mean(epsilon)
    

def structure_function(u, r_max, dx):
    u = np.asarray(u, dtype=float)
    D = []
    r_values = np.arange(1, r_max+1)
    for r in r_values:
        du = u[r:] - u[:-r]
        D.append(np.mean(du**2))
    return np.array(r_values) * dx, np.array(D)




root = Path.cwd() / "Data"
directions = ["pencils_x", "pencils_y", "pencils_z"]


L = 2*np.pi
nu = 1.10555e-5 

k_fluct_all = []
epsilon_all = []
eta_all = []
D11_all = []
D22_all = []
D33_all = []
r_list = []
E11_all = []
E22_all = []

for d in directions:
    folder = root / d
    pattern = str(folder / "*.txt")
    print("Looking for:", pattern, "| exists:", folder.is_dir())

    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} file(s) in {folder}")

    for path in files:
        try:
            u, v, w = np.loadtxt(path, dtype=float, comments="#", unpack=True)
            #print(f"Loaded {path} with {len(u)} entries.")
        except Exception as e:
            print(f"Failed to read {path}: {e}")
        
        N = len(u)
        dx = L / N
        K_f = 0.5 * (u**2 + v**2 + w**2)
        k_fluct_all.append(K_f)
        
        
        epsilon_mean = epsilon_moy(d, u, v, w, dx)
        epsilon_all.append(epsilon_mean)
        
        eta = float((nu**3 / epsilon_mean)**0.25)
        eta_all.append(eta)
        
        r_max = int(min(N//2, np.floor(5000 * eta / dx)))

        r, D11 = structure_function(u, r_max, dx)
        _, D22 = structure_function(v, r_max, dx)
        _, D33 = structure_function(w, r_max, dx)
        r_list.append(r)
        D11_all.append(D11)
        D22_all.append(D22)
        D33_all.append(D33)
        
        # FFT
        u_hat = fft(u, norm="forward")
        v_hat = fft(v, norm="forward")
        w_hat = fft(w, norm="forward")
        k = fftfreq(N, d=dx) * 2*np.pi  # en rad/m
        
        # Sélection k>0
        pos = k>0
        k_pos = k[pos]
        
        # Spectres 1D
        E11 = (np.abs(u_hat[pos])**2)
        E22 = 0.5*((np.abs(v_hat[pos])**2 + np.abs(w_hat[pos])**2))
        
        E11_all.append(E11)
        E22_all.append(E22)
        
        

#print(len(k_fluct_all))
#print(len(epsilon_all))
k_try_mean_mean = np.mean(k_fluct_all)
epsilon_mean_mean = np.mean(epsilon_all)
print("Mean kinetic energy fluctuation across all files:", k_try_mean_mean)
print("Mean epsilon across all files:", epsilon_mean_mean)

L = (k_try_mean_mean**1.5) / epsilon_mean_mean
print("Integral length scale L:", L)

Reynolds_number = (k_try_mean_mean**2) / (nu * epsilon_mean_mean)
print("Reynolds number Re:", Reynolds_number)
reynolds_bis = (L * np.sqrt(    k_try_mean_mean)) / (nu)
print("Reynolds number Re (bis):", reynolds_bis)

Kolmogorov_scale = L * (Reynolds_number)**(-0.75)
print("Kolmogorov length scale η:", Kolmogorov_scale)    

Taylor_microscale = np.sqrt(10* nu * k_try_mean_mean / epsilon_mean_mean)
print("Taylor microscale λ:", Taylor_microscale)

Reynolds_lambda = Taylor_microscale*(2*k_try_mean_mean/3)**0.5 / nu
print("Reynolds_number Reλ:", Reynolds_lambda)
Reynolds_lambda_bis = (20*Reynolds_number/3)**0.5
print("Reynolds_number Reλ (bis):", Reynolds_lambda_bis)


max_r_phys = 5000 * Kolmogorov_scale
r_common = np.linspace(Kolmogorov_scale, max_r_phys, num=5000) 
D11_interp = [np.interp(r_common, r, D) for r, D in zip(r_list, D11_all)]
D22_interp = [np.interp(r_common, r, D) for r, D in zip(r_list, D22_all)]
D33_interp = [np.interp(r_common, r, D) for r, D in zip(r_list, D33_all)]
D11_mean = np.mean(np.vstack(D11_interp), axis=0)
D22_mean = np.mean(np.vstack(D22_interp), axis=0)
D33_mean = np.mean(np.vstack(D33_interp), axis=0)
D22_mean = (D22_mean + D33_mean)/2
eta_mean = np.mean(eta_all)
print("eta mean:", eta_mean)
r_over_eta = r_common / eta_mean



plt.figure()
plt.loglog(r_over_eta, D11_mean, label=r"$D_{11}(r\hat{e}_x)$")
plt.loglog(r_over_eta, D22_mean, label=r"$D_{22}(r\hat{e}_x)$")
plt.xlabel(r"$r/\eta$")
plt.ylabel(r"$D_{ii}(r)$")
plt.legend()
plt.title("Structure Functions (longitudinal & transverse)")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.show()


D11_comp = D11_mean / (epsilon_mean_mean * r_common)**(2/3)
D22_comp = D22_mean / (epsilon_mean_mean * r_common)**(2/3)

plt.figure(figsize=(7,5))
plt.loglog(r_over_eta, D11_comp, label=r"$D_{11}/(\varepsilon r)^{2/3}$")
plt.loglog(r_over_eta, D22_comp, label=r"$D_{22}/(\varepsilon r)^{2/3}$")
plt.axhline(2.1, color='k', ls='--', label=r"$C_2 = 2.1$")
plt.axhline(2.8, color='r', ls='--', label=r"$C'_2 = 2.8$")
plt.xlabel(r"$r/\eta$")
plt.ylabel(r"Compensated $D_{ii}(r)$")
plt.title("Compensated Structure Functions")
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
#plt.xlim(1, 200)  
plt.show()





# Moyenne sur tous les fichiers
E11_mean = np.mean(np.vstack(E11_all), axis=0)
E22_mean = np.mean(np.vstack(E22_all), axis=0)

# Dimensionless
E11_dimless = E11_mean / (epsilon_mean * nu**5)**0.25
E22_dimless = E22_mean / (epsilon_mean * nu**5)**0.25
k_eta = k_pos * eta_mean

# Plot log-log
plt.figure()
plt.loglog(k_eta, E11_dimless, label='E11')
plt.loglog(k_eta, E22_dimless, label='E22')
plt.xlabel('k eta')
plt.ylabel('E / (eps nu^5)^{1/4}')
plt.legend()
plt.grid(True, which='both', ls='--', lw=0.5)
plt.title('Energy spectra (dimensionless)')
plt.show()



# Compensated
plt.figure()
plt.loglog(k_eta, (k_eta)**(5/3) * E11_dimless, label='E11 compensated')
plt.loglog(k_eta, (k_eta)**(5/3) * E22_dimless, label='E22 compensated')
plt.axhline(2.1, color='k', ls='--', label='C1')
plt.axhline(2.8, color='r', ls='--', label='C2')
plt.xlabel('k eta')
plt.ylabel('(k eta)^{5/3} E / (eps nu^5)^{1/4}')
plt.legend()
plt.grid(True, which='both', ls='--', lw=0.5)
plt.title('Compensated energy spectra')
plt.show()