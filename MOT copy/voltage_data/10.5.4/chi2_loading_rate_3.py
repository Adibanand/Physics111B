import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

DIST_MM = 14.5
DIAM_MM = 5.5
GAMMA_RAD_S = 2*np.pi*6.07e6
I_SAT = 17.0
OMEGA_MEAN = np.mean([0.008/2, 0.009/2, 0.0092/2, 0.0085/2])
P_X = 0.005426
IMPEDANCE_OHM = 1e6
RESPONSIVITY_AW = 0.45
LAMBDA = 780e-9

V_ROOM_BOUND = 0.07
SAVE = True

filename = r"MOT copy\MOTdata\bfield26A.txt"
out_root = r"MOT copy\voltage_data\10.5.4\bfield_chi2"

figures_dict = {}

# ==============================
# File Reading (unchanged)
# ==============================

voltages = []
start_reading = False

with open(filename, "r") as f:
    for line in f:
        line = line.strip()
        if line.startswith("Voltage"):
            start_reading = True
            continue
        if start_reading and line:
            voltages.append(float(line))

voltages = np.array(voltages)

fs = 10000
t = np.arange(len(voltages)) / fs

# ==============================
# Zoom + detect switch
# ==============================

V_zoomed = voltages[500::5]
t_zoomed = t[500::5]

dVdt = np.gradient(V_zoomed, t_zoomed)

i0 = np.argmax(dVdt)
i0_switch = i0 + 10

V_loading_window = V_zoomed[i0_switch+250::2]
t_loading_window = t_zoomed[i0_switch+250::2]

# ==============================
# Models
# ==============================

def loading_model_exp(t, Vbg, A, tau):
    return Vbg + A * (1 - np.exp(-t / tau))

def loading_model_betaN2(t, Vbg, Rv, tau, betaV):
    """
    Two-body loss model in voltage units:
        dVsig/dt = Rv - (1/tau)*Vsig - betaV*Vsig^2
    with Vsig(0)=0.
    Parameters:
        Vbg   : background voltage [V]
        Rv    : loading rate in voltage units [V/s]
        tau   : one-body time constant [s]
        betaV : two-body loss coefficient in voltage units [1/(V*s)]
    """
    gamma = 1.0 / tau
    betaV = np.maximum(betaV, 0.0)
    Rv = np.maximum(Rv, 0.0)

    s = np.sqrt(gamma**2 + 4.0 * betaV * Rv)

    # handle betaV -> 0 smoothly (should reduce to exponential)
    # when betaV is extremely small, s ~ gamma and expression is well behaved
    exp_term = np.exp(-s * t)
    Vsig = (2.0 * Rv * (1.0 - exp_term)) / ((s + gamma) + (s - gamma) * exp_term)

    return Vbg + Vsig

# ==============================
# Fit utilities
# ==============================

def compute_chi2(y, yfit, sigma):
    resid = y - yfit
    chi2 = np.sum((resid / sigma)**2)
    return chi2

def estimate_sigma_from_background(V_window, n=10):
    n = min(n, len(V_window))
    return np.std(V_window[:n])

def fit_exp_and_beta(t_loading_window, V_loading_window):
    t_rel = t_loading_window - t_loading_window[0]

    # noise estimate for chi^2 and optional weighting
    sigma_est = estimate_sigma_from_background(V_loading_window, n=10)
    if sigma_est == 0:
        sigma_est = 1e-6
    sigma_vec = sigma_est * np.ones_like(V_loading_window)

    # --------------------------
    # Fit 1: exponential
    # --------------------------
    Vbg0 = np.mean(V_loading_window[:5])
    Vf0  = np.mean(V_loading_window[-5:])
    A0 = max(Vf0 - Vbg0, 1e-6)
    tau0 = max((t_rel[-1] - t_rel[0]) / 3, 1e-6)

    p0_exp = [Vbg0, A0, tau0]
    bounds_exp = ([-np.inf, 0.0, 1e-9], [np.inf, np.inf, np.inf])

    popt_exp, pcov_exp = curve_fit(
        loading_model_exp,
        t_rel,
        V_loading_window,
        p0=p0_exp,
        bounds=bounds_exp,
        sigma=sigma_vec,
        absolute_sigma=True,
        maxfev=20000
    )

    Vfit_exp = loading_model_exp(t_rel, *popt_exp)
    chi2_exp = compute_chi2(V_loading_window, Vfit_exp, sigma_est)
    dof_exp = len(V_loading_window) - len(popt_exp)
    chi2red_exp = chi2_exp / dof_exp

    perr_exp = np.sqrt(np.diag(pcov_exp))
    Vbg_e, A_e, tau_e = popt_exp
    Vbg_e_err, A_e_err, tau_e_err = perr_exp

    # --------------------------
    # Fit 2: beta N^2 (voltage form)
    # --------------------------
    # Initial guesses:
    # early slope ~ Rv, tau from exponential, beta small
    Rv0 = max(A0 / tau0, 1e-6)      # V/s
    beta0 = 1e-3                    # 1/(V*s), small
    p0_beta = [Vbg0, Rv0, tau0, beta0]
    bounds_beta = ([-np.inf, 0.0, 1e-9, 0.0], [np.inf, np.inf, np.inf, np.inf])

    popt_beta, pcov_beta = curve_fit(
        loading_model_betaN2,
        t_rel,
        V_loading_window,
        p0=p0_beta,
        bounds=bounds_beta,
        sigma=sigma_vec,
        absolute_sigma=True,
        maxfev=50000
    )

    Vfit_beta = loading_model_betaN2(t_rel, *popt_beta)
    chi2_beta = compute_chi2(V_loading_window, Vfit_beta, sigma_est)
    dof_beta = len(V_loading_window) - len(popt_beta)
    chi2red_beta = chi2_beta / dof_beta

    perr_beta = np.sqrt(np.diag(pcov_beta))
    Vbg_b, Rv_b, tau_b, beta_b = popt_beta
    Vbg_b_err, Rv_b_err, tau_b_err, beta_b_err = perr_beta

    results = {
        "sigma_est": sigma_est,
        "exp": {
            "popt": popt_exp, "pcov": pcov_exp, "perr": perr_exp,
            "chi2": chi2_exp, "chi2_red": chi2red_exp, "dof": dof_exp,
            "Vfit": Vfit_exp,
            "params": {"Vbg": Vbg_e, "A": A_e, "tau": tau_e},
            "errs": {"Vbg": Vbg_e_err, "A": A_e_err, "tau": tau_e_err},
        },
        "betaN2": {
            "popt": popt_beta, "pcov": pcov_beta, "perr": perr_beta,
            "chi2": chi2_beta, "chi2_red": chi2red_beta, "dof": dof_beta,
            "Vfit": Vfit_beta,
            "params": {"Vbg": Vbg_b, "Rv": Rv_b, "tau": tau_b, "betaV": beta_b},
            "errs": {"Vbg": Vbg_b_err, "Rv": Rv_b_err, "tau": tau_b_err, "betaV": beta_b_err},
        }
    }
    return t_rel, results

# ==============================
# Use your existing window selection, then call:
# ==============================

t_rel, fit_results = fit_exp_and_beta(t_loading_window, V_loading_window)

print("\nNoise estimate used for chi^2:", fit_results["sigma_est"])

print("\nExponential fit:")
print("  params:", fit_results["exp"]["params"])
print("  errs  :", fit_results["exp"]["errs"])
print(f"  chi^2 = {fit_results['exp']['chi2']:.3f}, dof = {fit_results['exp']['dof']}, chi^2_red = {fit_results['exp']['chi2_red']:.3f}")

print("\nBeta N^2 fit:")
print("  params:", fit_results["betaN2"]["params"])
print("  errs  :", fit_results["betaN2"]["errs"])
print(f"  chi^2 = {fit_results['betaN2']['chi2']:.3f}, dof = {fit_results['betaN2']['dof']}, chi^2_red = {fit_results['betaN2']['chi2_red']:.3f}")

# ==============================
# Plot comparison
# ==============================

fig, ax = plt.subplots()
ax.plot(t_loading_window, V_loading_window, label="Data")
ax.plot(t_loading_window, fit_results["exp"]["Vfit"], linewidth=2,
        label=rf"Exp fit, $\chi^2_\nu$={fit_results['exp']['chi2_red']:.2f}")
ax.plot(t_loading_window, fit_results["betaN2"]["Vfit"], linewidth=2, linestyle="--",
        label=rf"$-\beta N^2$ fit, $\chi^2_\nu$={fit_results['betaN2']['chi2_red']:.2f}")

ax.set_xlabel("Time (s)")
ax.set_ylabel("Voltage (V)")
ax.set_title("MOT Loading Fits")
ax.legend()
plt.tight_layout()
plt.show()
