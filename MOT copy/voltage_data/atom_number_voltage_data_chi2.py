#!/usr/bin/env python
# coding: utf-8

# ## Loading Rate Voltage Data (with chi^2 test)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import json
import csv

# ==============================
# (All your constants unchanged)
# ==============================

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

filename = r"MOT copy\MOTdata\detune3686.txt"
out_root = r"MOT copy\voltage_data\10.5.3\detune"

figures_dict = {}

# ==============================
# Loading Model
# ==============================

def loading_model(t, Vbg, A, tau):
    return Vbg + A * (1 - np.exp(-t / tau))

# ==============================
# NEW: Loading fit with chi^2
# ==============================

def loading_rate(t_loading_window, V_loading_window):

    t_rel = t_loading_window - t_loading_window[0]

    # ---- initial guesses (same logic as original file) ----
    Vbg0 = np.mean(V_loading_window[:5])
    Vf0  = np.mean(V_loading_window[-5:])
    A0 = max(Vf0 - Vbg0, 1e-6)
    tau0 = (t_rel[-1] - t_rel[0]) / 3

    p0 = [Vbg0, A0, tau0]
    bounds = ([-np.inf, 0.0, 1e-9], [np.inf, np.inf, np.inf])

    popt, pcov = curve_fit(
        loading_model,
        t_rel,
        V_loading_window,
        p0=p0,
        bounds=bounds,
        maxfev=20000
    )

    Vbg, A, tau = popt
    Vf = Vbg + A
    load_rate = A / tau

    perr = np.sqrt(np.diag(pcov))
    Vbg_err, A_err, tau_err = perr

    # ============================================
    # χ² goodness-of-fit calculation
    # ============================================

    residuals = V_loading_window - loading_model(t_rel, *popt)

    # Estimate noise from early flat background region
    sigma_est = np.std(V_loading_window[:10])

    chi2 = np.sum((residuals / sigma_est)**2)
    dof = len(V_loading_window) - len(popt)
    chi2_red = chi2 / dof

    print("\nChi^2 Test Results:")
    print(f"chi^2 = {chi2:.3f}")
    print(f"dof   = {dof}")
    print(f"reduced chi^2 = {chi2_red:.3f}")

    return (
        Vbg, Vbg_err,
        A, A_err,
        tau, tau_err,
        Vf,
        load_rate,
        popt,
        chi2,
        chi2_red,
        dof
    )

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

V_loading_window = V_zoomed[i0_switch:]
t_loading_window = t_zoomed[i0_switch:]

# ==============================
# Fit
# ==============================

(
    Vbg, Vbg_err,
    A, A_err,
    tau, tau_err,
    Vf,
    load_rate_V_s,
    popt,
    chi2,
    chi2_red,
    dof
) = loading_rate(t_loading_window, V_loading_window)

print(f"\nVbg = {Vbg:.6f} ± {Vbg_err:.6f} V")
print(f"A   = {A:.6f} ± {A_err:.6f} V")
print(f"tau = {tau:.6f} ± {tau_err:.6f} s")
print(f"Loading rate A/tau = {load_rate_V_s:.6f} V/s")

# ==============================
# Plot data + fit
# ==============================

t_rel = t_loading_window - t_loading_window[0]

fig_fit, ax = plt.subplots()

ax.plot(t_loading_window, V_loading_window, label="Data")
ax.plot(
    t_loading_window,
    loading_model(t_rel, *popt),
    label="Fit",
    linewidth=2
)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Voltage (V)")
ax.set_title(f"MOT Loading Fit\nReduced chi^2 = {chi2_red:.3f}")
ax.legend()

plt.tight_layout()
plt.show()

figures_dict["loading_fit"] = fig_fit