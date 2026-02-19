import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt

from scipy import integrate, optimize
from scipy.interpolate import PchipInterpolator
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.lines as mlines

# -----------------------------
# Config
# -----------------------------
EOS_PATH = "/Users/krmt/Desktop/pyTOV-STT/eos_cgs.txt"

# Requested sets
BETAS = [-6.0, -10.0]  # (you removed -4.5)
MPHIS = [0.0, 1e-3, 5e-3, 2e-2, 5e-2]

RUN_CFG = dict(
    n_points=60,
    frac_min=0.70,
    frac_max=0.985,
    N=16001,
    r_max=250.0,
    init_params=(-1.0, -0.2),
    phi_min=1e-5,
    phi_seeds=None,     # keep your default 12 seeds
    maxiter=220
)

# Plot styling (beta color, mphi linestyle)
BETA_COLOR = {-6.0: "blue", -10.0: "green"}
MPHI_STYLE = {0.0: "-", 1e-3: "--", 5e-3: ":", 2e-2: "-."}
MPHI_CUSTOM_DASHES = {5e-2: (0, (6, 2, 1.2, 2, 1.2, 2))}  # dash-dot-dot

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# -----------------------------
# Physical constants (cgs)
# -----------------------------
c    = 2.99792458e10
G    = 6.67408e-8
Msun = 1.98847e33

Length_cm = G * Msun / c**2
Length_km = Length_cm / 1e5

# -----------------------------
# EOS load + interpolation (your conversion)
# -----------------------------
data = np.loadtxt(EOS_PATH, comments="#")

rho_b_cgs      = data[:, 0]
P_overc2_cgs    = data[:, 1]
eps_overc2_cgs  = data[:, 2]

P_mass_cgs   = P_overc2_cgs
eps_mass_cgs = eps_overc2_cgs

gcm3_to_MeVfm3 = 5.60958e-13
MeVfm3_to_km2  = 1.323e-6
conversion_factor = gcm3_to_MeVfm3 * MeVfm3_to_km2

km2_to_geom = (Length_km**2)
cgs_to_geom = conversion_factor * km2_to_geom

P_geom   = P_mass_cgs   * cgs_to_geom
eps_geom = eps_mass_cgs * cgs_to_geom
rho_b_geom = rho_b_cgs  * cgs_to_geom

idx = np.argsort(P_geom)
P_tab   = P_geom[idx]
eps_tab = eps_geom[idx]
rho_tab = rho_b_geom[idx]

P_unique, iu = np.unique(P_tab, return_index=True)
eps_unique = eps_tab[iu]
rho_unique = rho_tab[iu]

eps_of_P_interp = PchipInterpolator(P_unique, eps_unique, extrapolate=False)
rho_of_P_interp = PchipInterpolator(P_unique, rho_unique, extrapolate=False)
P_of_eps_interp = PchipInterpolator(eps_unique, P_unique, extrapolate=False)

def eps_of_P(P):
    P = np.clip(P, P_unique[0], P_unique[-1])
    return float(eps_of_P_interp(P))

def rho_of_P(P):
    P = np.clip(P, P_unique[0], P_unique[-1])
    return float(rho_of_P_interp(P))

def P_of_eps(eps):
    eps = np.clip(eps, eps_unique[0], eps_unique[-1])
    return float(P_of_eps_interp(eps))

# -----------------------------
# STT functions
# -----------------------------
def A_of_phi(phi, beta):
    return np.exp(0.5 * beta * phi**2)

def alpha_of_phi(phi, beta):
    return beta * phi

# -----------------------------
# ODEs (unchanged)
# -----------------------------
def f_int(r, y, beta, mphi):
    P, m, nu, phi, psi = y

    if r == 0.0:
        r = 1e-12

    eps = eps_of_P(P)
    V = 2.0 * (mphi**2) * (phi**2)
    A = A_of_phi(phi, beta)
    alpha = alpha_of_phi(phi, beta)

    r_minus_2m = r - 2.0*m
    if abs(r_minus_2m) < 1e-16:
        r_minus_2m = np.sign(r_minus_2m) * 1e-16 if r_minus_2m != 0 else 1e-16

    denom = r * r_minus_2m

    dPdr = -(eps + P) * (
        (m + 4.0*np.pi*A**4*(r**3)*P)/denom
        + 0.5*r*(psi**2) + alpha*psi
    )

    dmdr = 4.0*np.pi*A**4*(r**2)*eps + 0.5*r*(r-2.0*m)*(psi**2) + 0.25*(r**2)*V

    dnudr = (
        2.0*(m + 4.0*np.pi*A**4*(r**3)*P)/denom
        + r*(psi**2)
        - 0.25*(r**2)*V/(r_minus_2m)
    )

    dphidr = psi

    dpsidr_massless = (
        4.0*np.pi*A**4*r/(r_minus_2m) * (alpha*(eps-3.0*P) + r*(eps-P)*psi)
        - 2.0*(r-m)*psi/denom
    )

    mass_term = - (r/(r_minus_2m)) * (mphi**2) * phi
    dpsidr = dpsidr_massless + mass_term

    return [dPdr, dmdr, dnudr, dphidr, dpsidr]


def f_ext(r, y, beta, mphi):
    P, m, nu, phi, psi = y

    if r == 0.0:
        r = 1e-12

    r_minus_2m = r - 2.0*m
    if abs(r_minus_2m) < 1e-16:
        r_minus_2m = np.sign(r_minus_2m) * 1e-16 if r_minus_2m != 0 else 1e-16

    V = 2.0*(mphi**2)*(phi**2)

    dPdr = 0.0
    dmdr = 0.5*r*(r-2.0*m)*(psi**2) + 0.25*(r**2)*V
    dnudr = 2.0*m/(r*(r-2.0*m)) + r*(psi**2) - 0.25*(r**2)*V/(r_minus_2m)
    dphidr = psi

    dpsidr_massless = -2.0*(r-m)*psi/(r*(r-2.0*m))
    mass_term = - (r/(r_minus_2m))*(mphi**2)*phi
    dpsidr = dpsidr_massless + mass_term

    return [dPdr, dmdr, dnudr, dphidr, dpsidr]

# -----------------------------
# Shooting (unchanged)
# -----------------------------
def solve_star_shooting(P_c, beta, mphi,
                        N=16001, r_max=250.0,
                        initial_params=(-1.0, -0.1),
                        rtol=1e-10, atol=1e-35,
                        maxiter=200):

    r = np.linspace(0.0, r_max, N)
    dr = r[1] - r[0]

    def integrate_profile(nu_c, phi_c):
        y = np.zeros((N,5))
        y[0,:] = [P_c, 0.0, nu_c, phi_c, 0.0]

        ode = integrate.ode(lambda rr, yy: f_int(rr, yy, beta, mphi))
        ode.set_integrator('lsoda', rtol=rtol, atol=atol)
        ode.set_initial_value([P_c, 0.0, nu_c, phi_c, 0.0], dr)

        idx = 1
        while ode.successful() and ode.t < r[-1] and ode.y[0] > 0.0 and idx < N:
            y[idx,:] = ode.y
            ode.integrate(ode.t + dr)
            idx += 1

        idx_surface = idx-1
        if idx_surface < 10 or idx_surface >= N-10:
            return None

        y0ext = y[idx_surface,:].copy()
        y0ext[0] = 0.0

        ode = integrate.ode(lambda rr, yy: f_ext(rr, yy, beta, mphi))
        ode.set_integrator('lsoda', rtol=rtol, atol=atol)
        ode.set_initial_value(y0ext, r[idx_surface])

        idx2 = idx_surface
        while ode.successful() and ode.t < r[-1] and idx2 < N:
            y[idx2,:] = ode.y
            ode.integrate(ode.t + dr)
            idx2 += 1

        return r, y, idx_surface, idx2

    def objective(params):
        nu_c, phi_c = params
        out = integrate_profile(nu_c, phi_c)
        if out is None:
            return 1e99
        rgrid, y, idx_surface, idx2 = out
        k = min(N-2, idx2-2)
        nu_inf = y[k,2]
        phi_inf = y[k,3]
        return np.sqrt(nu_inf**2 + phi_inf**2)

    opt = optimize.minimize(
        objective,
        np.array(initial_params, dtype=float),
        method="nelder-mead",
        options={"maxiter": maxiter, "xatol": 1e-8, "fatol": 1e-10}
    )

    nu_c, phi_c = opt.x
    out = integrate_profile(nu_c, phi_c)
    if out is None:
        raise RuntimeError("Final integration failed.")

    rgrid, y, idx_surface, idx2 = out
    R_coord = rgrid[idx_surface]
    phi_surface = y[idx_surface,3]
    R_phys = A_of_phi(phi_surface, beta) * R_coord
    R_km = R_phys * Length_km

    nu_prof = y[:,2]
    dnudr = np.gradient(nu_prof, dr, edge_order=2)
    k = min(N-4, idx2-4)
    M_asymp = dnudr[k] * rgrid[k]**2 * np.exp(nu_prof[k]) / 2.0

    sol = {
        "r": rgrid, "y": y,
        "idx_surface": idx_surface,
        "R_km": R_km,
        "M": M_asymp,
        "nu_c": nu_c, "phi_c": phi_c,
        "phi_inf": y[min(N-2, idx2-2), 3],
        "opt": opt
    }
    return M_asymp, R_km, sol

# -----------------------------
# MR curve (unchanged logic/values)
# -----------------------------
def MR_curve_one_combo_shooting(beta, mphi,
                                n_points=60,
                                frac_min=0.70, frac_max=0.985,
                                N=16001, r_max=250.0,
                                init_params=(-1.0, -0.1),
                                phi_min=1e-5,
                                phi_seeds=None,
                                maxiter=220):

    if phi_seeds is None:
        phi_seeds = (-3e-1, -2e-1, -1e-1, -5e-2, -1e-2, -1e-3,
                     +1e-3, +1e-2, +5e-2, +1e-1, +2e-1, +3e-1)

    eps_c_list = np.linspace(
        eps_unique[int(frac_min*len(eps_unique))],
        eps_unique[int(frac_max*len(eps_unique))],
        n_points
    )

    M_list, R_list = [], []
    nu_guess = float(init_params[0])

    for i, eps_c in enumerate(eps_c_list):
        P_c = P_of_eps(eps_c)

        best = None
        tried = 0
        accepted = 0

        for phi0 in phi_seeds:
            tried += 1
            try:
                M, R_km, sol = solve_star_shooting(
                    P_c, beta=beta, mphi=mphi,
                    N=N, r_max=r_max,
                    initial_params=(nu_guess, float(phi0)),
                    maxiter=maxiter
                )
            except Exception:
                continue

            if abs(sol["phi_c"]) < phi_min:
                continue

            accepted += 1
            score = abs(sol["phi_inf"])

            if (best is None) or (score < best[0]):
                best = (score, M, R_km, sol)

        if best is None:
            print(f"[{i+1:03d}/{len(eps_c_list)}] beta={beta}, mphi={mphi} eps_c={eps_c:.3e} -> "
                  f"NO scalarized solution found (accepted {accepted}/{tried})")
            continue

        score, M, R_km, sol = best
        nu_guess = sol["nu_c"]

        M_list.append(M)
        R_list.append(R_km)

        print(f"[{i+1:03d}/{len(eps_c_list)}] beta={beta}, mphi={mphi} eps_c={eps_c:.3e} -> "
              f"M={M:.3f} Msun, R={R_km:.2f} km, phi_c={sol['phi_c']:.3e}, |phi_inf|={abs(sol['phi_inf']):.3e} "
              f"(accepted {accepted}/{tried})")

    return np.array(M_list), np.array(R_list)

# -----------------------------
# Multiprocess driver
# -----------------------------
def _run_one_combo_process(args):
    beta, mphi, run_cfg = args
    M, R = MR_curve_one_combo_shooting(beta=beta, mphi=mphi, **run_cfg)
    return beta, mphi, M, R

def run_all_curves_processes(betas, mphis, run_cfg, max_workers=None):
    combos = [(b, m, run_cfg) for b in betas for m in mphis]
    curves = {}

    if max_workers is None:
        cpu = os.cpu_count() or 4
        max_workers = max(1, cpu - 1)

    print(f"Total combos: {len(combos)} | process workers={max_workers}")

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_run_one_combo_process, x) for x in combos]
        for fut in as_completed(futs):
            beta, mphi, M, R = fut.result()
            curves[(beta, mphi)] = (M, R)
            print(f"DONE beta={beta}, mphi={mphi}, points={len(M)}")

    return curves

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("Starting run...")
    print("Length_km:", Length_km)
    print("EOS points:", len(P_unique))

    curves = run_all_curves_processes(BETAS, MPHIS, RUN_CFG, max_workers=None)

    # ---- Plot (main process only) ----
    plt.figure(figsize=(9, 7))

    for beta in BETAS:
        for mphi in MPHIS:
            M, R = curves.get((beta, mphi), (None, None))
            if M is None or len(M) == 0:
                continue

            color = BETA_COLOR[beta]
            if mphi in MPHI_CUSTOM_DASHES:
                ln, = plt.plot(R, M, color=color, lw=2.2)
                ln.set_dashes(MPHI_CUSTOM_DASHES[mphi][1])
            else:
                plt.plot(R, M, ls=MPHI_STYLE[mphi], color=color, lw=2.2)

    plt.xlabel("Radius R (km) [Jordan-frame]")
    plt.ylabel(r"Mass M ($M_\odot$)")
    plt.title(r"Mass–Radius (scalarized only): color = $\beta$, linestyle = $m_\varphi$")
    plt.grid(True, ls=":")

    beta_handles = [
        mlines.Line2D([], [], color=BETA_COLOR[-6.0], lw=2.5, label=r"$\beta=-6.0$"),
        mlines.Line2D([], [], color=BETA_COLOR[-10.0], lw=2.5, label=r"$\beta=-10.0$"),
    ]
    leg1 = plt.legend(handles=beta_handles, title=r"$\beta$", loc="upper right")
    plt.gca().add_artist(leg1)

    mphi_handles = [
        mlines.Line2D([], [], color="black", lw=2.5, ls="-",  label=r"$m_\varphi=0$"),
        mlines.Line2D([], [], color="black", lw=2.5, ls="--", label=r"$m_\varphi=10^{-3}$"),
        mlines.Line2D([], [], color="black", lw=2.5, ls=":",  label=r"$m_\varphi=5\times10^{-3}$"),
        mlines.Line2D([], [], color="black", lw=2.5, ls="-.", label=r"$m_\varphi=2\times10^{-2}$"),
    ]
    h = mlines.Line2D([], [], color="black", lw=2.5, label=r"$m_\varphi=5\times10^{-2}$")
    h.set_dashes(MPHI_CUSTOM_DASHES[5e-2][1])
    mphi_handles.append(h)

    plt.legend(handles=mphi_handles, title=r"$m_\varphi$", loc="lower left")
    plt.show()