import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf
import matplotlib.pyplot as plt

kB = 1.380649e-23
amu = 1.66053906660e-27
m_Rb85 = 84.911789738 * amu   # kg

def P_inside_sphere(t, T, Rc, sigma0, m=m_Rb85):
    """
    Fraction of a 3D isotropic Gaussian within radius Rc after ballistic expansion.
    t in seconds, Rc and sigma0 in meters, T in kelvin.
    """
    vrms = np.sqrt(kB * T / m)
    sigma = np.sqrt(sigma0**2 + (vrms * t)**2)
    a = Rc / (np.sqrt(2) * sigma)
    return erf(a) - np.sqrt(2/np.pi) * a * np.exp(-a**2)

def Ni_model(t, T, N0, Nbg, Rc, sigma0):
    return N0 * P_inside_sphere(t, T, Rc, sigma0) + Nbg

# -------------------------
# PUT YOUR DATA HERE
# tTOF in milliseconds from your plot, Ni in raw counts (or millions, any consistent unit)
t_ms = np.array([10, 20, 30, 45], dtype=float)
Ni   = np.array([1.91794965, 1.52269223, 0.89068426, 0.52967139], dtype=float)  # from your screenshot
Ni_err   = np.array([0.1794965, 0.354, 0.1234, 0.09235], dtype=float)  # from your screenshot

t = t_ms * 1e-3  # seconds

# -------------------------
# Initial guesses (you can tweak)
T_guess = 200e-6          # 200 microkelvin
N0_guess = Ni.max()       # same units as Ni
Nbg_guess = Ni.min() * 0.05
Rc_guess = 0.5e-2         # 0.5 cm capture radius (edit if your beam size is different)
sigma0_guess = 0.2e-3     # 0.2 mm initial size guess

p0 = [T_guess, N0_guess, Nbg_guess, Rc_guess, sigma0_guess]

# Bounds to keep it physical
bounds = (
    [1e-6,    0.0,   -np.inf,  1e-4,   1e-6],   # lower: T>=1 uK, Rc>=0.1 mm, sigma0>=1 um
    [5e-2,  np.inf,   np.inf,   5e-2,   5e-3],  # upper: T<=50 mK, Rc<=5 cm, sigma0<=5 mm
)

popt, pcov = curve_fit(Ni_model, t, Ni, p0=p0, bounds=bounds, maxfev=20000)
T_fit, N0_fit, Nbg_fit, Rc_fit, sigma0_fit = popt
perr = np.sqrt(np.diag(pcov))
T_err = perr[0]

print(f"T = {T_fit*1e6:.1f} ± {T_err*1e6:.1f} µK")
print(f"Rc = {Rc_fit*1e3:.2f} mm, sigma0 = {sigma0_fit*1e3:.2f} mm")
print(f"N0 = {N0_fit:.3g}, Nbg = {Nbg_fit:.3g}")

# Plot
t_dense = np.linspace(t.min(), t.max(), 300)
plt.figure()
plt.errorbar(t_ms, Ni, yerr=Ni_err,fmt="o", capsize=4, label="data")
plt.plot(t_dense*1e3, Ni_model(t_dense, *popt), "-", label=r'fit: $T_{fit} = 213.8 \mu K$')
plt.xlabel(r"t_{TOF} (ms)")
plt.ylabel(r'$N_i$ ($10^6$ atoms)')
plt.title(r'$N_i$ (atoms) vs $t_{TOF}$ (ms)')
plt.legend()
plt.show()