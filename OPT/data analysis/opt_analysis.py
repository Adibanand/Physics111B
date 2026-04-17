import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2
import scipy.constants as sc


# -----------------------------
# Physics helpers
# -----------------------------
def get_nuclear_spin(isotope: str) -> float:
    """
    Returns the known nuclear spin used for theory/model setup.
    """
    isotope = isotope.strip().lower()
    if isotope in ["rb85", "85", "85rb", "rb-85"]:
        return 5/2
    elif isotope in ["rb87", "87", "87rb", "rb-87"]:
        return 3/2
    else:
        raise ValueError(f"Unrecognized isotope: {isotope}")


def get_isotope_label(isotope: str) -> str:
    isotope = isotope.strip().lower()
    if isotope in ["rb85", "85", "85rb", "rb-85"]:
        return "Rb85"
    elif isotope in ["rb87", "87", "87rb", "rb-87"]:
        return "Rb87"
    else:
        raise ValueError(f"Unrecognized isotope: {isotope}")


def field_sign_from_direction(field_direction: str) -> int:
    """
    '+' or 'B+' -> +1
    '-' or 'B-' -> -1
    """
    s = field_direction.strip().lower()
    if s in ["+", "b+", "pos", "positive", "plus"]:
        return +1
    elif s in ["-", "b-", "neg", "negative", "minus"]:
        return -1
    else:
        raise ValueError(f"Unrecognized field direction: {field_direction}")


def coil_field_per_current_G_per_A(N: float, R: float, h: float, z: float = 0.0) -> float:
    """
    Returns coil field conversion factor in Gauss/A.
    Assumes the same formula you have been using.
    """
    B_per_A_T = sc.mu_0 * N * R**2 / (((h - z)**2 + R**2)**1.5)
    return 1e4 * B_per_A_T  # Tesla -> Gauss


def gamma_MHz_per_G(I: float) -> float:
    """
    Zeeman slope factor in MHz/G from your lab expression.
    """
    return 2.799 / (2 * I + 1)


def linear_model(x, m, c):
    return m * x + c


# -----------------------------
# Data loading
# -----------------------------
def load_optical_pumping_csv(
    filename,
    current_col=1,
    freq_col=8,
    current_err_col=None,
    freq_err_col=None,
    skiprows=1,
    delimiter=","
):
    """
    Loads columns from CSV.
    Column indices are 0-based.
    """

    # Decide which columns to read
    cols = [current_col, freq_col]
    if current_err_col is not None:
        cols.append(current_err_col)
    if freq_err_col is not None:
        cols.append(freq_err_col)

    data = np.genfromtxt(
        filename,
        delimiter=delimiter,
        skip_header=skiprows,
        usecols=cols,
        filling_values=np.nan
    )

    # remove bad rows
    if data.ndim == 1:
        data = data[None, :]
    data = data[~np.isnan(data).any(axis=1)]

    out = {}
    out["current"] = data[:, 0]
    out["freq_MHz"] = data[:, 1]

    idx = 2
    if current_err_col is not None:
        out["current_err"] = data[:, idx]
        idx += 1
    else:
        out["current_err"] = None

    if freq_err_col is not None:
        out["freq_err"] = data[:, idx]
    else:
        out["freq_err"] = None

    return out


# -----------------------------
# Error propagation
# -----------------------------
def freq_err_from_current_err(current_err, I, N, R, h, z=0.0):
    """
    Propagate current uncertainty into frequency uncertainty.
    Returns freq uncertainty in MHz.
    """
    slope_G_per_A = coil_field_per_current_G_per_A(N, R, h, z)
    slope_MHz_per_A = gamma_MHz_per_G(I) * slope_G_per_A
    return np.abs(slope_MHz_per_A) * np.asarray(current_err)


# -----------------------------
# Fit + goodness of fit
# -----------------------------
def fit_line_with_optional_rescaling(x, y, yerr, p0=(1.0, 0.0), rescale_errors=True):
    """
    Weighted linear fit. If rescale_errors=True, rescales all y-errors
    by sqrt(reduced chi^2) so final reduced chi^2 = 1.
    """

    # initial fit
    popt, pcov = curve_fit(
        linear_model,
        x,
        y,
        p0=p0,
        sigma=yerr,
        absolute_sigma=True
    )

    m, c = popt
    m_err, c_err = np.sqrt(np.diag(pcov))

    yfit = linear_model(x, m, c)
    residuals = y - yfit

    chi2_val = np.sum((residuals / yerr) ** 2)
    ndof = len(x) - 2
    chi2_red = chi2_val / ndof
    p_val = chi2.sf(chi2_val, ndof)

    result = {
        "m": m,
        "c": c,
        "m_err": m_err,
        "c_err": c_err,
        "yfit": yfit,
        "residuals": residuals,
        "chi2": chi2_val,
        "ndof": ndof,
        "chi2_red": chi2_red,
        "p_value": p_val,
        "yerr_used": np.array(yerr, copy=True),
        "scale_factor": 1.0,
    }

    if rescale_errors:
        scale = np.sqrt(chi2_red)
        yerr_scaled = yerr * scale

        popt2, pcov2 = curve_fit(
            linear_model,
            x,
            y,
            p0=popt,
            sigma=yerr_scaled,
            absolute_sigma=True
        )

        m2, c2 = popt2
        m2_err, c2_err = np.sqrt(np.diag(pcov2))
        yfit2 = linear_model(x, m2, c2)
        residuals2 = y - yfit2

        chi2_val2 = np.sum((residuals2 / yerr_scaled) ** 2)
        chi2_red2 = chi2_val2 / ndof
        p_val2 = chi2.sf(chi2_val2, ndof)

        result = {
            "m": m2,
            "c": c2,
            "m_err": m2_err,
            "c_err": c2_err,
            "yfit": yfit2,
            "residuals": residuals2,
            "chi2": chi2_val2,
            "ndof": ndof,
            "chi2_red": chi2_red2,
            "p_value": p_val2,
            "yerr_used": yerr_scaled,
            "scale_factor": scale,
            "chi2_initial": chi2_val,
            "chi2_red_initial": chi2_red,
            "p_value_initial": p_val,
        }

    return result


# -----------------------------
# Physics extraction from fit
# -----------------------------
def infer_ambient_field_from_fit(intercept_MHz, intercept_err_MHz, I):
    """
    From c = gamma * B_amb, infer B_amb in Gauss.
    """
    gamma = gamma_MHz_per_G(I)
    B_amb = intercept_MHz / gamma
    B_amb_err = intercept_err_MHz / gamma
    return B_amb, B_amb_err


def infer_spin_from_slope(slope_MHz_per_A, slope_err_MHz_per_A, N, R, h, z=0.0):
    """
    From measured slope:
        m = [2.799/(2I+1)] * (Bcoil/I)_G_per_A
    solve for I.
    """
    alpha = coil_field_per_current_G_per_A(N, R, h, z)
    q = 2.799 * alpha / slope_MHz_per_A   # = 2I + 1
    I_fit = (q - 1) / 2

    # derivative propagation
    dI_dm = - (2.799 * alpha) / (2 * slope_MHz_per_A**2)
    I_err = np.abs(dI_dm) * slope_err_MHz_per_A

    return I_fit, I_err


# -----------------------------
# Plotting
# -----------------------------
def plot_data_and_fit(x, y, yerr, fit_result, title="Linear Fit", xlabel="Current (A)", ylabel="Frequency (MHz)"):
    plt.figure(figsize=(6, 4))

    plt.errorbar(
        x, y,
        yerr=yerr,
        fmt='.',
        linestyle='none',
        markersize=6,
        markerfacecolor='red',
        markeredgewidth=1.5,
        capsize=4,
        elinewidth=1.5,
        label='Data'
    )

    x_plot = np.linspace(np.min(x), np.max(x), 300)
    plt.plot(x_plot, linear_model(x_plot, fit_result["m"], fit_result["c"]), label='Linear fit')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_residuals(x, residuals, yerr, title="Residuals", xlabel="Current (A)"):
    plt.figure(figsize=(6, 4))

    plt.errorbar(
        x, residuals,
        yerr=yerr,
        fmt='.',
        linestyle='none',
        markersize=6,
        markerfacecolor='red',
        markeredgewidth=1.5,
        capsize=4,
        elinewidth=1.5
    )

    plt.axhline(0, linestyle='--', alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel("Residuals (MHz)")
    plt.title(title)
    
    plt.tight_layout()
    plt.show()


# -----------------------------
# Main wrapper
# -----------------------------
def analyze_optical_pumping_dataset(
    filename,
    isotope,
    field_direction,
    N,
    R,
    h,
    z=0.0,
    current_col=1,
    freq_col=8,
    current_err_col=None,
    freq_err_col=None,
    skiprows=1,
    p0=(1.0, 0.0),
    rescale_errors=True,
    make_plots=True
):
    """
    Full pipeline for one dataset.
    """

    iso_label = get_isotope_label(isotope)
    I_theory = get_nuclear_spin(isotope)
    sign = field_sign_from_direction(field_direction)

    data = load_optical_pumping_csv(
        filename,
        current_col=current_col,
        freq_col=freq_col,
        current_err_col=current_err_col,
        freq_err_col=freq_err_col,
        skiprows=skiprows
    )

    current = np.asarray(data["current"])
    freq_MHz = np.asarray(data["freq_MHz"])

    # assign sign to current, not frequency
    current_signed = sign * current

    # choose y-errors
    if data["freq_err"] is not None:
        freq_err = np.asarray(data["freq_err"])
    elif data["current_err"] is not None:
        freq_err = freq_err_from_current_err(data["current_err"], I_theory, N, R, h, z)
    else:
        raise ValueError("Need either freq_err_col or current_err_col.")

    # sort by x for prettier plots
    idx = np.argsort(current_signed)
    current_signed = current_signed[idx]
    freq_MHz = freq_MHz[idx]
    freq_err = freq_err[idx]

    fit_result = fit_line_with_optional_rescaling(
        current_signed,
        freq_MHz,
        freq_err,
        p0=p0,
        rescale_errors=rescale_errors
    )

    B_amb, B_amb_err = infer_ambient_field_from_fit(
        fit_result["c"],
        fit_result["c_err"],
        I_theory
    )

    I_fit1, I_fit_err = infer_spin_from_slope(
        fit_result["m"],
        fit_result["m_err"],
        N, R, h, z
    )

    if field_direction == "+":
        I_fit = I_fit1

    if field_direction == "-":
        I_fit = (-I_fit1) - 1


    result = {
        "filename": filename,
        "isotope": iso_label,
        "field_direction": field_direction,
        "I_theory": I_theory,
        "current_A": current_signed,
        "freq_MHz": freq_MHz,
        "freq_err_MHz": fit_result["yerr_used"],
        "fit": fit_result,
        "B_amb_G": B_amb,
        "B_amb_err_G": B_amb_err,
        "I_fit": I_fit,
        "I_fit_err": I_fit_err,
    }

    print(f"\n=== {iso_label} {field_direction} ===")
    print(f"File: {filename}")
    print(f"Slope      = {fit_result['m']:.4f} ± {fit_result['m_err']:.4f} MHz/A")
    print(f"Intercept  = {fit_result['c']:.4f} ± {fit_result['c_err']:.4f} MHz")
    print(f"chi^2      = {fit_result['chi2']:.3f}")
    print(f"dof        = {fit_result['ndof']}")
    print(f"chi^2/dof  = {fit_result['chi2_red']:.3f}")
    print(f"p-value    = {fit_result['p_value']:.4g}")
    print(f"scale fac. = {fit_result['scale_factor']:.4f}")
    print(f"B_amb      = {B_amb:.4f} ± {B_amb_err:.4f} G")
    print(f"I_fit      = {I_fit:.4f} ± {I_fit_err:.4f}")

    if make_plots:
        plot_data_and_fit(
            current_signed,
            freq_MHz,
            fit_result["yerr_used"],
            fit_result,
            title=f"{iso_label}, B{field_direction}: Frequency vs Current"
        )

        plot_residuals(
            current_signed,
            fit_result["residuals"],
            fit_result["yerr_used"],
            title=f"{iso_label}, B{field_direction}: Residuals"
        )

    return result