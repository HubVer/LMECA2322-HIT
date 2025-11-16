import glob
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# ============================
#   Fonctions utilitaires
# ============================

def dudx_fourth_order(u, dx):
    """
    Dérivée spatiale d'ordre 4 en différences finies centrées.
    """
    u = np.asarray(u, dtype=float)
    return (-np.roll(u, -2)
            + 8.0 * np.roll(u, -1)
            - 8.0 * np.roll(u,  1)
            + 1.0 * np.roll(u,  2)) / (12.0 * dx)


def structure_function(u, r_max, dx):
    """
    Fonction de structure d'ordre 2 :
    D(r) = < [u(x+r) - u(x)]^2 >.
    On retourne r en unités physiques.
    """
    u = np.asarray(u, dtype=float)
    D = []
    r_values = np.arange(1, r_max + 1)
    for r in r_values:
        du = u[r:] - u[:-r]
        D.append(np.mean(du**2))
    return np.array(r_values) * dx, np.array(D)


def compute_1d_spectrum(u_fluct, dx):
    """
    Calcule le spectre d'énergie 1D E(k) d'un signal u'(x),
    tel que, par construction, l'intégrale du spectre soit
    cohérente avec l'énergie (idée de Parseval).

    On utilise la FFT réelle (rfft) car u_fluct est réel.
    """
    u_fluct = np.asarray(u_fluct, dtype=float)
    N = len(u_fluct)

    # FFT réelle (k >= 0)
    u_hat = np.fft.rfft(u_fluct)
    k = 2.0 * np.pi * np.fft.rfftfreq(N, d=dx)

    if len(k) > 1:
        dk = k[1] - k[0]
    else:
        dk = 1.0  # cas pathologique, pour éviter une division par zéro

    E = np.zeros_like(k, dtype=float)

    # k = 0 : mode moyenne => on le laisse à 0 dans E.
    if N % 2 == 0:
        # N pair -> dernier point = mode de Nyquist (flip-flop)
        if len(k) > 2:
            # modes "normaux" (1 ... N/2-1)
            E[1:-1] = (np.abs(u_hat[1:-1])**2) / (N**2 * dk)
        # mode de Nyquist : pas de symétrique négatif -> facteur 1/2
        E[-1] = (np.abs(u_hat[-1])**2) / (2.0 * N**2 * dk)
    else:
        # N impair -> pas de Nyquist
        if len(k) > 1:
            E[1:] = (np.abs(u_hat[1:])**2) / (N**2 * dk)

    return k, E


# ========= BONUS : AUTOCORRÉLATION =========

def autocorrelation_normalized(u, r_max, dx):
    """
    Fonction d'autocorrélation normalisée :
    R(r) = <u(x) u(x+r)> / <u^2>, avec R(0) = 1.
    """
    u = np.asarray(u, dtype=float)
    N = len(u)
    var = np.mean(u**2)

    r_values = np.arange(0, r_max + 1)  # inclut r = 0
    R = np.empty(r_max + 1, dtype=float)
    R[0] = 1.0

    for r in range(1, r_max + 1):
        prod = u[:-r] * u[r:]
        R[r] = np.mean(prod) / var

    return r_values * dx, R


# ============================
#   Paramètres globaux
# ============================

# Dossier des données

root = Path(_file_).parent / "Data"
directions = ["pencils_x", "pencils_y", "pencils_z"]


L_box = 2.0 * np.pi        # taille de la boîte (domaine périodique)
nu = 1.10555e-5            # viscosité cinématique

# Constantes théoriques pour les spectres (à adapter selon le cours)
C1 = 1.0
C2 = 1.0


# ============================
#   Accumulateurs
# ============================

# Pour les grandeurs globales
k_fluct_all = []   # valeurs locales de k(x) sur tous les pencils
epsilon_all = []   # epsilon moyen par pencil
eta_all = []       # échelle de Kolmogorov par pencil

# Pour les fonctions de structure
D11_all = []
D22_all = []
D33_all = []
r_list = []

# Pour les spectres 1D
E11_list = []      # longitudinal
E22_list = []      # transverse 1
E33_list = []      # transverse 2
k1_common = None   # grille k commune à tous les spectres

# ===== BONUS : autocorrélations =====
r_long_auto_list = []   # r pour f(r)
f_long_list = []        # f(r) longitudinal
r_trans_auto_list = []  # r pour g(r)
g_trans_list = []       # g(r) transverse (on empilera les deux composantes)


# ============================
#   Boucles sur les fichiers
# ============================

for d in directions:
    folder = root / d
    pattern = str(folder / "*.txt")
    print("Looking for:", pattern, "| exists:", folder.is_dir())

    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} file(s) in {folder}")

    for path in files:
        try:
            u, v, w = np.loadtxt(path, dtype=float, comments="#", unpack=True)
        except Exception as e:
            print(f"Failed to read {path}: {e}")
            continue

        # Vérification des NaN / infinities
        if (not np.all(np.isfinite(u))
            or not np.all(np.isfinite(v))
            or not np.all(np.isfinite(w))):
            print(f"Non finite values in {path}, skipping.")
            continue

        N = len(u)
        dx = L_box / N

        # Fluctuations (on enlève la moyenne)
        #u_fluct = u - np.mean(u)
        #v_fluct = v - np.mean(v)
        #w_fluct = w - np.mean(w)
        
        # Alternative : on n'enlève pas la moyenne car on est en HIT
        u_fluct = u
        v_fluct = v
        w_fluct = w
        # Énergie cinétique turbulente locale k(x)
        K_f = 0.5 * (u_fluct**2 + v_fluct**2 + w_fluct**2)
        k_fluct_all.append(K_f)

        # Dérivées spatiales pour epsilon
        du = dudx_fourth_order(u, dx)
        dv = dudx_fourth_order(v, dx)
        dw = dudx_fourth_order(w, dx)

        if d == "pencils_x":
            epsilon = 15.0 * nu * (du**2)
        elif d == "pencils_y":
            epsilon = 15.0 * nu * (dv**2)
        elif d == "pencils_z":
            epsilon = 15.0 * nu * (dw**2)
        else:
            continue

        epsilon_mean = np.mean(epsilon)
        epsilon_all.append(epsilon_mean)

        # Échelle de Kolmogorov locale
        eta = float((nu**3 / epsilon_mean)**0.25)
        eta_all.append(eta)

        # ===== Fonctions de structure (partie 2) =====
        r_max = int(min(N // 2, np.floor(5000.0 * eta / dx)))
        if r_max >= 1:
            r, D11 = structure_function(u, r_max, dx)
            _, D22 = structure_function(v, r_max, dx)
            _, D33 = structure_function(w, r_max, dx)

            r_list.append(r)
            D11_all.append(D11)
            D22_all.append(D22)
            D33_all.append(D33)

        # ===== Spectres 1D (partie 3) =====
        # Choix de la composante longitudinale / transverse selon la direction
        if d == "pencils_x":
            vel_long = u_fluct
            vel_trans1 = v_fluct
            vel_trans2 = w_fluct
        elif d == "pencils_y":
            vel_long = v_fluct
            vel_trans1 = u_fluct
            vel_trans2 = w_fluct
        elif d == "pencils_z":
            vel_long = w_fluct
            vel_trans1 = u_fluct
            vel_trans2 = v_fluct
        else:
            continue

        k_local, E_long = compute_1d_spectrum(vel_long, dx)
        _, E_t1 = compute_1d_spectrum(vel_trans1, dx)
        _, E_t2 = compute_1d_spectrum(vel_trans2, dx)

        # Vérification que la grille k est la même partout
        if k1_common is None:
            k1_common = k_local
        else:
            if len(k_local) != len(k1_common) or not np.allclose(k_local, k1_common):
                print("WARNING: different k-grid in", path, "- skipping spectra for this file.")
                continue

        E11_list.append(E_long)
        E22_list.append(E_t1)
        E33_list.append(E_t2)

        # ===== BONUS : autocorrélations f(r), g(r) =====
        r_max_auto = int(min(N // 2, np.floor(500.0 * eta / dx)))  # r/η <= 5×10^2
        if r_max_auto >= 1:
            # Longitudinale f(r)
            r_auto_L, f_L = autocorrelation_normalized(vel_long, r_max_auto, dx)
            r_long_auto_list.append(r_auto_L)
            f_long_list.append(f_L)

            # Deux composantes transverses -> deux réalisations de g(r)
            r_auto_T1, g_T1 = autocorrelation_normalized(vel_trans1, r_max_auto, dx)
            r_auto_T2, g_T2 = autocorrelation_normalized(vel_trans2, r_max_auto, dx)

            r_trans_auto_list.append(r_auto_T1)
            r_trans_auto_list.append(r_auto_T2)
            g_trans_list.append(g_T1)
            g_trans_list.append(g_T2)


# ============================
#   Grandeurs globales (partie 1)
# ============================

print("\n===== Grandeurs globales =====")
print("Nombre de segments utilisés pour k :", len(k_fluct_all))
print("Nombre de segments utilisés pour epsilon :", len(epsilon_all))

# On met tout k(x) bout-à-bout pour faire une moyenne globale
if len(k_fluct_all) > 0:
    k_all = np.concatenate(k_fluct_all)
    k_mean = np.mean(k_all)
else:
    k_mean = np.nan

epsilon_mean_mean = np.mean(epsilon_all) if len(epsilon_all) > 0 else np.nan
eta_mean = np.mean(eta_all) if len(eta_all) > 0 else np.nan

print("Mean kinetic energy fluctuation across all files k:", k_mean)
print("Mean epsilon across all files ε:", epsilon_mean_mean)
print("Kolmogorov length scale η (from ν^3/ε mean):", eta_mean)

# Longueur intégrale (selon la formule donnée dans le sujet)
L_int = (k_mean**1.5) / epsilon_mean_mean
print("Integral length scale L:", L_int)

# Nombre de Reynolds "global"
Reynolds_number = (k_mean**2) / (nu * epsilon_mean_mean)
print("Reynolds number Re:", Reynolds_number)

Reynolds_bis = (L_int * np.sqrt(k_mean)) / nu
print("Reynolds number Re (bis):", Reynolds_bis)

# Autre estimation de η via L et Re (comme tu faisais)
Kolmogorov_scale_Re = L_int * (Reynolds_number)**(-0.75)
print("Kolmogorov length scale η (from L, Re):", Kolmogorov_scale_Re)

# Échelle de Taylor
Taylor_microscale = np.sqrt(10.0 * nu * k_mean / epsilon_mean_mean)
print("Taylor microscale λ:", Taylor_microscale)

# Nombre de Reynolds de Taylor
Reynolds_lambda = Taylor_microscale * np.sqrt(2.0 * k_mean / 3.0) / nu
print("Reynolds_number Reλ:", Reynolds_lambda)

Reynolds_lambda_bis = np.sqrt(20.0 * Reynolds_number / 3.0)
print("Reynolds_number Reλ (bis):", Reynolds_lambda_bis)


# ============================
#   Fonctions de structure (partie 2)
# ============================

if len(r_list) > 0:
    print("\n===== Fonctions de structure =====")

    max_r_phys = 5000.0 * eta_mean
    r_common = np.linspace(eta_mean, max_r_phys, num=5000)

    D11_interp = [np.interp(r_common, r, D) for r, D in zip(r_list, D11_all)]
    D22_interp = [np.interp(r_common, r, D) for r, D in zip(r_list, D22_all)]
    D33_interp = [np.interp(r_common, r, D) for r, D in zip(r_list, D33_all)]

    D11_mean = np.mean(np.vstack(D11_interp), axis=0)
    D22_mean = np.mean(np.vstack(D22_interp), axis=0)
    D33_mean = np.mean(np.vstack(D33_interp), axis=0)

    # moyenne des composantes transverses
    D22_mean = 0.5 * (D22_mean + D33_mean)

    r_over_eta = r_common / eta_mean

    # --- D11 et D22 ---
    plt.figure()
    plt.loglog(r_over_eta, D11_mean, label=r"$D_{11}(r\hat{e}_x)$")
    plt.loglog(r_over_eta, D22_mean, label=r"$D_{22}(r\hat{e}_x)$")
    plt.xlabel(r"$r/\eta$")
    plt.ylabel(r"$D_{ii}(r)$")
    plt.legend()
    plt.title("Structure Functions (longitudinal & transverse)")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()

    # --- Fonctions de structure compensées ---
    D11_comp = D11_mean / (epsilon_mean_mean * r_common)**(2.0/3.0)
    D22_comp = D22_mean / (epsilon_mean_mean * r_common)**(2.0/3.0)

    plt.figure(figsize=(7, 5))
    plt.loglog(r_over_eta, D11_comp, label=r"$D_{11}/(\varepsilon r)^{2/3}$")
    plt.loglog(r_over_eta, D22_comp, label=r"$D_{22}/(\varepsilon r)^{2/3}$")
    plt.axhline(2.1, ls='--', label=r"$C_2 = 2.1$")
    plt.axhline(2.8, ls='--', label=r"$C'_2 = 2.8$")
    plt.xlabel(r"$r/\eta$")
    plt.ylabel(r"Compensated $D_{ii}(r)$")
    plt.title("Compensated Structure Functions")
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()
else:
    print("No structure functions were computed (no valid data found).")


# ============================
#   Spectres d'énergie 1D (partie 3)
# ============================

if k1_common is not None and len(E11_list) > 0:
    print("\n===== Spectres d'énergie 1D =====")

    E11_mean = np.mean(np.vstack(E11_list), axis=0)
    E22_mean = np.mean(np.vstack(E22_list), axis=0)
    E33_mean = np.mean(np.vstack(E33_list), axis=0)

    # Moyenne des deux composantes transverses
    Etrans_mean = 0.5 * (E22_mean + E33_mean)

    # Adimensionnalisation
    k_eta = k1_common * eta_mean
    kolmo_scale_spectrum = (epsilon_mean_mean * nu**5)**0.25

    E11_dimless = E11_mean / kolmo_scale_spectrum
    Etrans_dimless = Etrans_mean / kolmo_scale_spectrum

    # On enlève k = 0 pour les tracés log-log
    mask = k_eta > 0.0
    k_eta_nz = k_eta[mask]
    E11_dimless_nz = E11_dimless[mask]
    Etrans_dimless_nz = Etrans_dimless[mask]

    # --- Spectres adimensionnels ---
    plt.figure(figsize=(7, 5))
    plt.loglog(k_eta_nz, E11_dimless_nz, label=r"$E_{11} / (\varepsilon \nu^5)^{1/4}$")
    plt.loglog(k_eta_nz, Etrans_dimless_nz, label=r"$E_{22} / (\varepsilon \nu^5)^{1/4}$")

    # Courbes théoriques ~ C1, C2 (kη)^(-5/3) (A ADAPTER)
    k_th = np.linspace(k_eta_nz.min(), k_eta_nz.max(), 200)
    plt.loglog(k_th, C1 * k_th**(-5.0/3.0), ls='--', label=r"$C_1 (k\eta)^{-5/3}$")
    plt.loglog(k_th, C2 * k_th**(-5.0/3.0), ls='--', label=r"$C_2 (k\eta)^{-5/3}$")

    plt.xlabel(r"$k\eta$")
    plt.ylabel(r"$E_{ii} / (\varepsilon \nu^5)^{1/4}$")
    plt.title("Dimensionless 1D Energy Spectra")
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()

    # --- Spectres compensés ---
    E11_comp_spec = (k_eta_nz**(5.0/3.0)) * E11_dimless_nz
    Etrans_comp_spec = (k_eta_nz**(5.0/3.0)) * Etrans_dimless_nz

    plt.figure(figsize=(7, 5))
    plt.loglog(k_eta_nz, E11_comp_spec,
               label=r"$(k\eta)^{5/3} E_{11} / (\varepsilon \nu^5)^{1/4}$")
    plt.loglog(k_eta_nz, Etrans_comp_spec,
               label=r"$(k\eta)^{5/3} E_{22} / (\varepsilon \nu^5)^{1/4}$")

    plt.axhline(C1, ls='--', label=r"$C_1$")
    plt.axhline(C2, ls='--', label=r"$C_2$")

    plt.xlabel(r"$k\eta$")
    plt.ylabel("Compensated spectra")
    plt.title("Compensated 1D Energy Spectra")
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()
else:
    print("No 1D spectra were computed (no valid data found).")


# ============================
#   BONUS : fonctions d'autocorrélation f(r), g(r)
# ============================

if len(f_long_list) > 0 and len(g_trans_list) > 0:
    print("\n===== Autocorrelation functions (BONUS) =====")

    # On ne va pas au-delà du plus petit r_max parmi toutes les séries
    r_all_auto = r_long_auto_list + r_trans_auto_list
    max_r_each = [r_arr[-1] for r_arr in r_all_auto]
    max_r_common = min(max_r_each)

    max_r_phys_auto = min(500.0 * eta_mean, max_r_common)
    r_common_auto = np.linspace(0.0, max_r_phys_auto, num=2000)

    # Interpolation et moyenne de f(r)
    f_interp_list = []
    for r_arr, f_arr in zip(r_long_auto_list, f_long_list):
        f_interp = np.empty_like(r_common_auto)
        f_interp[0] = 1.0
        f_interp[1:] = np.interp(r_common_auto[1:], r_arr, f_arr)
        f_interp_list.append(f_interp)

    # Interpolation et moyenne de g(r) (toutes les composantes transverses)
    g_interp_list = []
    for r_arr, g_arr in zip(r_trans_auto_list, g_trans_list):
        g_interp = np.empty_like(r_common_auto)
        g_interp[0] = 1.0
        g_interp[1:] = np.interp(r_common_auto[1:], r_arr, g_arr)
        g_interp_list.append(g_interp)

    f_mean = np.mean(np.vstack(f_interp_list), axis=0)
    g_mean = np.mean(np.vstack(g_interp_list), axis=0)

    r_over_eta_auto = r_common_auto / eta_mean

    # Tracés de f(r) et g(r)
    plt.figure(figsize=(7, 5))
    plt.plot(r_over_eta_auto, f_mean, label=r"$f(r)$ (longitudinal)")
    plt.plot(r_over_eta_auto, g_mean, label=r"$g(r)$ (transverse)")
    plt.xlabel(r"$r/\eta$")
    plt.ylabel("Autocorrelation")
    plt.ylim(-0.2, 1.05)
    plt.title("Longitudinal and Transverse Autocorrelation Functions")
    plt.legend()
    plt.grid(True, ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()

    # --- Taylor microscales λ_f et λ_g par parabole osculatrice ---
    # f(r) ≈ 1 - r^2 / (2 λ_f^2) → f(r) - 1 ≈ a r^2, a = -1/(2 λ_f^2)

    r_fit_max = min(2.0 * eta_mean, max_r_phys_auto * 0.5)
    mask_fit = (r_common_auto > 0.0) & (r_common_auto <= r_fit_max)

    def fit_lambda(r_vals, F_vals):
        x = r_vals[mask_fit]**2
        y = F_vals[mask_fit] - 1.0
        if len(x) < 3:
            return np.nan
        a, _ = np.polyfit(x, y, 1)  # y ≈ a x
        if a >= 0:
            return np.nan
        return np.sqrt(-1.0 / (2.0 * a))

    lambda_f = fit_lambda(r_common_auto, f_mean)
    lambda_g = fit_lambda(r_common_auto, g_mean)

    print(f"Taylor microscale from autocorrelation (λ_f, longitudinal): {lambda_f}")
    print(f"Taylor microscale from autocorrelation (λ_g, transverse): {lambda_g}")
else:
    print("No autocorrelation functions were computed (no valid data found).")