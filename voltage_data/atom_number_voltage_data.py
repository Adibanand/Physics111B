#!/usr/bin/env python
# coding: utf-8

# ## Loading Rate Voltage Data

# In[62]:
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import json
import csv

# Constants/Measured Inputs 
DIST_MM = 14.5      # distance from MOT to focusing lens [mm]
DIAM_MM = 5.5       # diameter of focusing lens [mm]
DETUNING_MHZ = 25   # detuning (MHz)
GAMMA_RAD_S = 2*np.pi*6.07e6   # Rb D2 natural linewidth (rad/s)
I_SAT = 17.0  # saturation intensity W/m^2  (1.7 mW/cm^2)
OMEGA_MEAN = np.mean([0.008/2, 0.009/2, 0.0092/2, 0.0085/2])  # mean beam radius at MOT (m)
P_X = 0.005426 # power of input beam in X direction (W)
IMPEDANCE_OHM = 1e6 # DAQ analog I/O input impedance (Ohms)
RESPONSIVITY_AW = 0.45 # responsivity of PD3 (A/W)
LAMBDA = 780e-9


# indexes
V_ROOM_BOUND = 0.06
V_ZOOMED = 1600
SWITCH_TIME_DELAY = 25
LOADING_WINDOW_UP_BOUND = None

# SAVE?
SAVE = False

# change file name
filename = r"MOTdata\PD30_bigcirc.txt"
out_root = r"voltage_data/10.5.4"
figures_dict = {}

def save_run_outputs(
    results_dict,
    measured_dict,
    figures_dict, 
    indexes_dict,   
    data_path=filename,
    out_root=out_root,
):
    """
    Save run outputs to: out_root / f"atom_number_{file_stem}/"

    Parameters
    ----------
    data_path : str or Path
        Path to the dataset file used for this run.
    out_root : str or Path
        Parent directory to store outputs.
    results_dict : dict
        Calculated values (atom number, R_sc, photon rate, tau, etc.)
    measured_dict : dict
        Measured/assumed values (power at beam, R_load, responsivity, Omega, eta_coll, etc.)
    figures_dict : dict[str, matplotlib.figure.Figure] or None
        Mapping from filename stem to matplotlib Figure, example:
        {"loading_curve": fig1, "fit_residuals": fig2}

    Returns
    -------
    out_dir : Path
        The directory where files were saved.
    """
    data_path = Path(data_path)
    out_root = Path(out_root)
    file_stem = data_path.stem

    out_dir = out_root / f"1.5cm_{file_stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Combine everything into one record
    record = {
        "source_file": str(data_path),
        "results": results_dict,
        "measured": measured_dict,
        "indexes": indexes_dict
    }

    # 1) JSON dump (best for full fidelity)
    json_path = out_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    # 2) One-row CSV summary (nice for spreadsheets)
    # Flatten keys for CSV
    flat = {"source_file": str(data_path)}
    for k, v in (results_dict or {}).items():
        flat[f"calc__{k}"] = v
    for k, v in (measured_dict or {}).items():
        flat[f"meas__{k}"] = v

    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)

    with open(out_dir/"indexes.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "index"])  # header

        for k, v in indexes_dict.items():
            writer.writerow([k, v])
            
    # 3) Human-readable measured inputs
    txt_path = out_dir / "measured_inputs.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Measured / Assumed Inputs\n")
        f.write("------------------------\n")
        for k, v in (measured_dict or {}).items():
            f.write(f"{k}: {v}\n")

    # 4) Save plots
    if figures_dict:
        for name, fig in figures_dict.items():
            # sanitize name for filename
            safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name)
            fig_path = out_dir / f"{safe}.png"
            fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    
    print("Saved data!")
    return out_dir

# intensity at MOT
def Ix_MOT(Px):
    Ix = 2*Px/(np.pi*OMEGA_MEAN**2)
    return Ix

# scattering rate calculation
def scattering_rate(Px):
    I = Ix_MOT(Px)
    s = I / I_SAT

    frac = s / (1 + s + (2*Delta_rad_s/GAMMA_RAD_S)**2)
    R_sc = (GAMMA_RAD_S/2) * frac

    print(f"Beam radius w = {OMEGA_MEAN*1e3:.2f} mm")
    print(f"Intensity at MOT = {I:.2f} W/m^2 = {I/10:.2f} mW/cm^2")
    print(f"Saturation parameter s = {s:.2f}")
    print(f"Scattering rate R_sc = {R_sc:.2e} s^-1")
    print(f"Maximum possible (Γ/2) = {GAMMA_RAD_S/2:.2e} s^-1")

    return R_sc, s

# Model: V(t) = Vbg + A*(1 - exp(-t/tau))
def loading_model(t, Vbg, A, tau):
    return Vbg + A * (1 - np.exp(-t / tau))

def solid_angle():
    # ---- compute solid angle ----
    a_mm = DIAM_MM / 2.0
    theta = np.arctan(a_mm / DIST_MM)              # half-angle [rad]
    Omega = 2*np.pi*(1 - np.cos(theta))            # solid angle in sr
    eta = Omega / (4*np.pi)                        # fraction of isotropic emission

    print(f"Lens radius a = {a_mm:.3f} mm")
    print(f"Distance d    = {DIST_MM:.3f} mm")
    print(f"Half-angle θ  = {np.degrees(theta):.3f} deg ({theta:.6f} rad)")
    print(f"Solid angle Ω = {Omega:.6f} sr")
    print(f"Collected fraction Ω/4π = {eta:.6%}")
    return Omega

def loading_rate(t_loading_window,V_loading_window):
    Vbg0 = np.mean(V_loading_window[:max(5, int(0.001 / np.median(np.diff(t_loading_window))))])  # ~first 1 ms
    Vf0  = np.mean(V_loading_window[-max(5, int(0.002 / np.median(np.diff(t_loading_window)))):]) # ~last 2 ms
    A0 = max(Vf0 - Vbg0, 1e-6)
    tau0 = (t_rel[-1] - t_rel[0]) / 3 if t_rel[-1] > 0 else 0.002

    p0 = [Vbg0, A0, tau0]
    bounds = ([-np.inf, 0.0, 1e-9], [np.inf, np.inf, np.inf])  # enforce A>0, tau>0

    popt, pcov = curve_fit(loading_model, t_rel, V_loading_window, p0=p0, bounds=bounds, maxfev=20000)
    Vbg, A, tau = popt
    Vf = Vbg + A
    load_rate = A / tau
    # 1-sigma uncertainties
    perr = np.sqrt(np.diag(pcov))
    Vbg_err, A_err, tau_err = perr

    return Vbg, Vbg_err, A, A_err, tau, tau_err, Vf, load_rate, popt

def pd3_VtoW(voltages, Vbg, impedance_ohm, responsivity_aw):
    return (voltages-Vbg) / (impedance_ohm*responsivity_aw)

def pd3_photon_emission_rate(powers, coll_efficiency):
    h = 6.626e-34
    c = 2.99e8
    lamb = 780e-9
    E_per_photon = h*c/lamb
    return powers/(coll_efficiency*E_per_photon)

def collection_efficiency(solid_angle, T_opt=0.9):
    coll_efficiency = solid_angle*T_opt/(4*np.pi)
    print(f"Collection efficiency: {coll_efficiency}")
    return coll_efficiency

def atom_number(R_sc, photon_detection_rate):
    a_numb = photon_detection_rate/(R_sc)
    a_numb_mean = np.mean(a_numb)
    a_numb_std = np.std(a_numb)
    a_numb_std_err = np.std(a_numb)/len(photon_detection_rate)

    return a_numb_mean, a_numb_std, a_numb_std_err

# convert V/s quantity to atoms/s
# V --> P (V/G*R) --> photon Rate (P/E_photon) --> detection rate (P_rate/eta_coll) --> Atom loading rate (D_r/R_sc)
def convert_loading_rate(load_rate_V, coll_efficiency, R_sc):
    VtoW = load_rate_V/(IMPEDANCE_OHM * RESPONSIVITY_AW)
    h = 6.626e-34
    c = 2.99e8
    E_per_photon = h*c/LAMBDA
    Wtophotons = VtoW/E_per_photon
    photon_detection_rate = Wtophotons/coll_efficiency
    loading_rate_atoms_s = photon_detection_rate/R_sc
    return loading_rate_atoms_s

################### START PLOTTING ####################
# In[65]:
voltages = []
start_reading = False

with open(filename, "r") as f:
    for line in f:
        line = line.strip()

        # start reading after the Voltage header
        if line.startswith("Voltage"):
            start_reading = True
            continue

        if start_reading and line:
            voltages.append(float(line))

voltages = np.array(voltages)

# create time axis
fs = 10000  # Hz
t = np.arange(len(voltages)) / fs

fig1 = plt.figure()
plt.plot(t, voltages)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("PD3 Voltage vs Time")
plt.tight_layout()
plt.show()

figures_dict["raw data"] = fig1

# In[65]:
V_room = np.mean(voltages[voltages <= V_ROOM_BOUND])
print(f"Mean room light: {V_room}")

# In[66]:
V_zoomed = voltages[V_ZOOMED:]
t_zoomed = t[V_ZOOMED:]

fig2= plt.figure()
plt.plot(t_zoomed[:600], V_zoomed[:600])
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("Loading Window")
plt.tight_layout()
plt.show()
figures_dict["loading window"] = fig2

# In[7]:
dVdt = np.gradient(V_zoomed, t_zoomed)

fig3 = plt.figure()
plt.plot(t_zoomed[:500], dVdt[:500])
plt.xlabel("Time (s)")
plt.ylabel("dV/dt (V/s)")
plt.title("Loading Window dV/dt")
plt.tight_layout()
plt.show()
figures_dict["loading window_derivative"] = fig3

# In[67]:
# Calculate turn on time
i0 = np.argmax(dVdt)
t0 = t_zoomed[i0]

# SWITCH TIME DELAY
i0_switch = i0 + SWITCH_TIME_DELAY


# In[68]:
V_loading_window = V_zoomed[i0_switch:600]
print(f"\nLoading window length: {len(V_loading_window)}")

t_loading_window = t_zoomed[i0_switch:600]
fig4 = plt.figure()
plt.plot(t_loading_window, V_loading_window)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("Loading Window")
plt.tight_layout()
plt.show()
figures_dict["loading window_zoomed"] = fig4

####### COMPUTE LOADING RATE ######
t_rel = t_loading_window - t_loading_window[0]
print("\nFitting Function...")
Vbg, Vbg_err, A, A_err, tau, tau_err, Vf, load_rate_V_s, popt = loading_rate(t_loading_window,V_loading_window)

# In[70]:
print(f"Vbg = {Vbg:.6f} ± {Vbg_err:.6f} V")
print(f"A   = {A:.6f} ± {A_err:.6f} V")
print(f"tau = {tau:.6f} ± {tau_err:.6f} s")
print(f"Vf  = {Vf:.6f} V")
print(f"Loading rate (proportional) A/tau= {load_rate_V_s:.6f} V/s")

# 5) Plot data + fit
fig5 = plt.figure()
plt.plot(t_loading_window, V_loading_window, label="data")
plt.plot(t_loading_window, loading_model(t_rel, *popt), label="fit")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("MOT Loading Fit")
plt.legend()
plt.tight_layout()
plt.show()
figures_dict["loading_fit"] = fig4

# In[72]:
Delta_rad_s = DETUNING_MHZ *1e6*2*np.pi

####### COMPUTE SCATTERING RATE ######
print("\nCalculating Scattering Rate...")
R_sc,s = scattering_rate(P_X)

###### COMPUTE SOLID ANGLE #######
print("\nCalculating Solid Angle...")
Omega = solid_angle()

# In[75]:
clip_index = 280
V_saturation_window = V_loading_window[clip_index:]
plt.plot(t_loading_window[clip_index:], V_loading_window[clip_index:], label="data")
sat_voltage = np.mean(V_loading_window[clip_index:])
print(f"\nMean saturation voltage: {sat_voltage} V")


# In[79]:
V_to_W = pd3_VtoW(V_saturation_window, Vbg, IMPEDANCE_OHM, RESPONSIVITY_AW)
print(f"Mean Power at Saturation: {np.mean(V_to_W)*1e6} uW")

coll_efficiency = collection_efficiency(Omega)
photon_emission_rate = pd3_photon_emission_rate(V_to_W, coll_efficiency)
mean_emission_rate = np.mean(photon_emission_rate)
print(f"Mean Photon Emission Rate: {np.format_float_scientific(mean_emission_rate, precision=4)} photons/S")

# In[87]:
atom_numb_mean, atom_numb_std, atom_numb_std_err = atom_number(R_sc, photon_emission_rate)
print(f"Mean Atom Number: {np.format_float_scientific(atom_numb_mean, precision=4)}, STD Atom Number: {np.format_float_scientific(atom_numb_std, precision=4)}, STD ERR Atom Number: {np.format_float_scientific(atom_numb_std_err, precision=4)}")

#In[87]
loading_rate_atoms_s = convert_loading_rate(load_rate_V_s, coll_efficiency, R_sc)
print(f"Loading Rate (atoms/s): {np.format_float_scientific(loading_rate_atoms_s, precision=4)} atoms/s")

measured_dict = {
    "power_at_beam_W": P_X,
    "detuning_MHz": DETUNING_MHZ,
    "natural_linewidth_MHz": GAMMA_RAD_S,
    "R_load_ohm": IMPEDANCE_OHM,                 # DAQ input impedance
    "responsivity_A_per_W": RESPONSIVITY_AW,  # at 780 nm from datasheet
    "I_sat_W": I_SAT,
    "dist_mot_focuslens_mm": DIST_MM,
    "diam_pd3_focuslens_mm": DIAM_MM,
    "beam_radius_m": OMEGA_MEAN,
    "Omega_sr": Omega,
    "eta_coll": coll_efficiency,
}

results_dict = {
    "solid_angle_sr": Omega,
    "scattering_rate": R_sc,
    "saturation_parameter": s,
    "atom_number_mean": atom_numb_mean,
    "atom_number_std": atom_numb_std,
    "atom_number_std_err": atom_numb_std_err,                
    "background_voltage": Vbg, 
    "background_voltage_err": Vbg_err,
    "mean_photon_emission_rate": mean_emission_rate,
    "A": A, 
    "A_err": A_err, 
    "tau": tau, 
    "tau_err": tau_err, 
    "V_final": Vf,
    "loading_rate_atoms_s": loading_rate_atoms_s
}

indexes_dict = {
    "V_ROOM_BOUND": 0.06,
    "V_ZOOMED": 1560,
    "SWITCH_TIME_DELAY": 20,
    "LOADING_WINDOW_UP_BOUND": 2500
}

if SAVE:
    save_run_outputs(results_dict, measured_dict, figures_dict, indexes_dict)
