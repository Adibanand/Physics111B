import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2
import scipy.constants as sc


# ============================================================
# Basic helpers
# ============================================================

def get_nuclear_spin(isotope: str) -> float:
    """Known nuclear spin for each rubidium isotope."""
    s = isotope.strip().lower()
    if s in ["rb85", "85", "85rb", "rb-85"]:
        return 5 / 2
    if s in ["rb87", "87", "87rb", "rb-87"]:
        return 3 / 2
    raise ValueError(f"Unrecognized isotope: {isotope}")


def get_isotope_label(isotope: str) -> str:
    s = isotope.strip().lower()
    if s in ["rb85", "85", "85rb", "rb-85"]:
        return "Rb85"
    if s in ["rb87", "87", "87rb", "rb-87"]:
        return "Rb87"
    raise ValueError(f"Unrecognized isotope: {isotope}")


def field_sign_from_direction(field_direction: str) -> int:
    """Map B+ / B- labels onto a sign for the current."""
    s = field_direction.strip().lower()
    if s in ["+", "b+", "plus", "positive", "pos"]:
        return +1
    if s in ["-", "b-", "minus", "negative", "neg"]:
        return -1
    raise ValueError(f"Unrecognized field direction: {field_direction}")


# ============================================================
# Physics helpers
# ============================================================

def coil_field_per_current_G_per_A(N: float, R: float, h: float, z: float = 0.0) -> float:
    """
    Helmholtz-coil field conversion factor in Gauss / A.

    Convention used here matches the user's setup:
    h is the distance from the midpoint between the coils to one coil.
    """
    B_per_A_T = sc.mu_0 * N * R**2 / (((h - z) ** 2 + R**2) ** 1.5)
    return 1e4 * B_per_A_T  # Tesla -> Gauss


def gamma_MHz_per_G(I: float) -> float:
    """Zeeman slope factor from the lab relation."""
    return 2.799 / (2 * I + 1)


def infer_spin_from_freq_vs_current_slope(slope_MHz_per_A: float, slope_err_MHz_per_A: float,
                                          N: float, R: float, h: float, z: float = 0.0):
    """
    Infer nuclear spin from a fit of frequency vs current:
        nu = A |I - I0| + f0
    where A = gamma * alpha.
    """
    alpha = coil_field_per_current_G_per_A(N, R, h, z)
    q = 2.799 * alpha / slope_MHz_per_A  # = 2I + 1
    I_fit = 0.5 * (q - 1)

    dI_dA = -(2.799 * alpha) / (2 * slope_MHz_per_A**2)
    I_err = abs(dI_dA) * slope_err_MHz_per_A
    return I_fit, I_err


def infer_spin_from_current_vs_freq_slope(inv_slope_A_per_MHz: float, inv_slope_err_A_per_MHz: float,
                                          N: float, R: float, h: float, z: float = 0.0):
    """
    Infer nuclear spin from a fit of signed current vs frequency:
        I = I0 ± nu/A
    so the fitted slope magnitude is b = 1/A in A/MHz.
    Then A = 1/b and A = gamma * alpha.
    """
    A = 1.0 / inv_slope_A_per_MHz
    A_err = inv_slope_err_A_per_MHz / (inv_slope_A_per_MHz**2)
    return infer_spin_from_freq_vs_current_slope(A, A_err, N, R, h, z)


# ============================================================
# Models
# ============================================================

def linear_model(x, m, c):
    return m * x + c


def v_model_freq_vs_current(current, A, I0):
    """Frequency as a function of signed current."""
    return A * np.abs(current - I0)


def v_model_freq_vs_current_offset(current, A, I0, f0):
    """Frequency as a function of signed current with offset."""
    return A * np.abs(current - I0) + f0


def branch_model_current_vs_freq(freq, b, I0, branch_sign):
    """
    Signed current as a function of frequency on one branch:
        I = I0 + branch_sign * b * freq
    where b = 1/A in A/MHz.
    """
    return I0 + branch_sign * b * freq


def combined_current_vs_freq_model(freq_and_branch, b, I0):
    """
    Model for fitting both B+ and B- data at once with current as the dependent variable.

    freq_and_branch is a 2-row array-like:
        freq_and_branch[0] -> frequency
        freq_and_branch[1] -> branch sign (+1 or -1)
    """
    freq = np.asarray(freq_and_branch[0])
    branch = np.asarray(freq_and_branch[1])
    return I0 + branch * b * freq


# ============================================================
# Data loading
# ============================================================

def load_optical_pumping_csv(filename, current_col=7, freq_col=0,
                             current_err_col=None, freq_err_col=None,
                             skiprows=1, delimiter=","):
    """
    Load selected columns from a CSV file.
    Column indices are 0-based.
    """
    cols = [freq_col, current_col]
    if current_err_col is not None:
        cols.append(current_err_col)
    if freq_err_col is not None:
        cols.append(freq_err_col)

    data = np.genfromtxt(
        filename,
        delimiter=delimiter,
        skip_header=skiprows,
        usecols=cols,
        filling_values=np.nan,
    )

    if data.ndim == 1:
        data = data[None, :]

    data = data[~np.isnan(data).any(axis=1)]

    out = {
        "freq_MHz": data[:, 0],
        "current_A": data[:, 1],
        "current_err_A": None,
        "freq_err_MHz": None,
    }

    idx = 2
    if current_err_col is not None:
        out["current_err_A"] = data[:, idx]
        idx += 1
    if freq_err_col is not None:
        out["freq_err_MHz"] = data[:, idx]

    return out


def prepare_dataset(filename, isotope, field_direction,
                    N, R, h, z=0.0,
                    current_col=7, freq_col=0,
                    current_err_col=None, freq_err_col=None,
                    skiprows=1, delimiter=","):
    """
    Load one dataset and assign sign to the current based on B+ / B-.
    """
    sign = field_sign_from_direction(field_direction)
    I_theory = get_nuclear_spin(isotope)

    data = load_optical_pumping_csv(
        filename,
        current_col=current_col,
        freq_col=freq_col,
        current_err_col=current_err_col,
        freq_err_col=freq_err_col,
        skiprows=skiprows,
        delimiter=delimiter,
    )

    freq_MHz = np.asarray(data["freq_MHz"])
    current_A = sign * np.asarray(data["current_A"])

    # Use provided current errors directly when current is the dependent variable.
    current_err_A = None
    if data["current_err_A"] is not None:
        current_err_A = np.asarray(data["current_err_A"])

    # If frequency errors are needed for freq-vs-current fits, use provided ones or propagate.
    freq_err_MHz = None
    if data["freq_err_MHz"] is not None:
        freq_err_MHz = np.asarray(data["freq_err_MHz"])
    elif current_err_A is not None:
        alpha = coil_field_per_current_G_per_A(N, R, h, z)
        gamma = gamma_MHz_per_G(I_theory)
        freq_err_MHz = np.abs(gamma * alpha) * current_err_A

    return {
        "freq_MHz": freq_MHz,
        "current_A": current_A,
        "current_err_A": current_err_A,
        "freq_err_MHz": freq_err_MHz,
        "branch_sign": np.full_like(freq_MHz, sign, dtype=float),
    }


# ============================================================
# Generic fit utilities
# ============================================================

def compute_chi2(y_obs, y_fit, yerr, n_params):
    residuals = y_obs - y_fit
    chi2_val = np.sum((residuals / yerr) ** 2)
    ndof = len(y_obs) - n_params
    chi2_red = chi2_val / ndof
    p_val = chi2.sf(chi2_val, ndof)
    return residuals, chi2_val, ndof, chi2_red, p_val


def rescale_errors_to_unit_chi2(yerr, chi2_red):
    return yerr * np.sqrt(chi2_red)


# ============================================================
# Single-dataset linear analysis
# ============================================================

def analyze_optical_pumping_dataset(filename, isotope, field_direction,
                                    N, R, h, z=0.0,
                                    current_col=7, freq_col=0,
                                    current_err_col=None, freq_err_col=None,
                                    skiprows=1, delimiter=",",
                                    p0=(1.0, 0.0),
                                    rescale_errors=True,
                                    make_plots=True):
    """
    Analyze a single dataset by fitting frequency vs signed current to a line.
    """
    iso_label = get_isotope_label(isotope)

    data = prepare_dataset(
        filename, isotope, field_direction,
        N, R, h, z=z,
        current_col=current_col, freq_col=freq_col,
        current_err_col=current_err_col, freq_err_col=freq_err_col,
        skiprows=skiprows, delimiter=delimiter,
    )

    x = np.asarray(data["current_A"])
    y = np.asarray(data["freq_MHz"])

    if data["freq_err_MHz"] is None:
        raise ValueError("Need frequency uncertainties for the linear frequency-vs-current fit.")
    yerr = np.asarray(data["freq_err_MHz"])

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    yerr = yerr[idx]

    popt, pcov = curve_fit(
        linear_model, x, y,
        p0=p0,
        sigma=yerr,
        absolute_sigma=True,
        maxfev=20000,
    )

    m, c = popt
    m_err, c_err = np.sqrt(np.diag(pcov))
    yfit = linear_model(x, m, c)
    residuals, chi2_val, ndof, chi2_red, p_val = compute_chi2(y, yfit, yerr, 2)

    scale_factor = 1.0
    if rescale_errors:
        yerr = rescale_errors_to_unit_chi2(yerr, chi2_red)
        scale_factor = np.sqrt(chi2_red)

        popt, pcov = curve_fit(
            linear_model, x, y,
            p0=popt,
            sigma=yerr,
            absolute_sigma=True,
            maxfev=20000,
        )

        m, c = popt
        m_err, c_err = np.sqrt(np.diag(pcov))
        yfit = linear_model(x, m, c)
        residuals, chi2_val, ndof, chi2_red, p_val = compute_chi2(y, yfit, yerr, 2)

    I_fit, I_err = infer_spin_from_freq_vs_current_slope(m, m_err, N, R, h, z)
    gamma = gamma_MHz_per_G(get_nuclear_spin(isotope))
    B_amb = c / gamma
    B_amb_err = c_err / gamma

    result = {
        "filename": filename,
        "isotope": iso_label,
        "field_direction": field_direction,
        "current_A": x,
        "freq_MHz": y,
        "freq_err_MHz": yerr,
        "fit": {"m": m, "c": c, "m_err": m_err, "c_err": c_err},
        "residuals": residuals,
        "chi2": chi2_val,
        "ndof": ndof,
        "chi2_red": chi2_red,
        "p_value": p_val,
        "scale_factor": scale_factor,
        "I_fit": I_fit,
        "I_fit_err": I_err,
        "B_amb_G": B_amb,
        "B_amb_err_G": B_amb_err,
    }

    print(f"\n=== {iso_label} {field_direction} ===")
    print(f"File         : {filename}")
    print(f"Slope        = {m:.4f} ± {m_err:.4f} MHz/A")
    print(f"Intercept    = {c:.4f} ± {c_err:.4f} MHz")
    print(f"chi^2        = {chi2_val:.3f}")
    print(f"dof          = {ndof}")
    print(f"chi^2/dof    = {chi2_red:.3f}")
    print(f"p-value      = {p_val:.4g}")
    print(f"scale factor = {scale_factor:.4f}")
    print(f"I_fit        = {I_fit:.4f} ± {I_err:.4f}")
    print(f"B_amb        = {B_amb:.4f} ± {B_amb_err:.4f} G")

    if make_plots:
        plot_linear_data_and_fit(result)
        plot_linear_residuals(result)

    return result


# ============================================================
# Combined V-fit, frequency as dependent variable
# ============================================================

def analyze_isotope_combined_vfit(filename_Bplus, filename_Bminus, isotope,
                                  N, R, h, z=0.0,
                                  current_col=7, freq_col=0,
                                  current_err_col=None, freq_err_col=None,
                                  skiprows=1, delimiter=",",
                                  use_offset=True,
                                  p0=None,
                                  rescale_errors=True,
                                  make_plots=True):
    """
    Combine B+ and B- data and fit frequency vs signed current to a V-shaped model.
    """
    iso_label = get_isotope_label(isotope)

    plus = prepare_dataset(
        filename_Bplus, isotope, "+",
        N, R, h, z=z,
        current_col=current_col, freq_col=freq_col,
        current_err_col=current_err_col, freq_err_col=freq_err_col,
        skiprows=skiprows, delimiter=delimiter,
    )
    minus = prepare_dataset(
        filename_Bminus, isotope, "-",
        N, R, h, z=z,
        current_col=current_col, freq_col=freq_col,
        current_err_col=current_err_col, freq_err_col=freq_err_col,
        skiprows=skiprows, delimiter=delimiter,
    )

    current_all = np.concatenate([plus["current_A"], minus["current_A"]])
    freq_all = np.concatenate([plus["freq_MHz"], minus["freq_MHz"]])

    if plus["freq_err_MHz"] is None or minus["freq_err_MHz"] is None:
        raise ValueError("Need frequency uncertainties for the combined frequency-vs-current V fit.")
    freq_err_all = np.concatenate([plus["freq_err_MHz"], minus["freq_err_MHz"]])

    idx = np.argsort(current_all)
    current_all = current_all[idx]
    freq_all = freq_all[idx]
    freq_err_all = freq_err_all[idx]

    if p0 is None:
        I0_guess = current_all[np.argmin(freq_all)]
        f0_guess = max(0.0, np.min(freq_all))
        span = np.max(np.abs(current_all - I0_guess)) + 1e-12
        A_guess = max((np.max(freq_all) - f0_guess) / span, 1e-3)
        p0 = [A_guess, I0_guess, f0_guess] if use_offset else [A_guess, I0_guess]

    if use_offset:
        model = v_model_freq_vs_current_offset
        bounds = ([0.0, np.min(current_all), -np.inf], [np.inf, np.max(current_all), np.inf])
        n_params = 3
    else:
        model = v_model_freq_vs_current
        bounds = ([0.0, np.min(current_all)], [np.inf, np.max(current_all)])
        n_params = 2

    popt, pcov = curve_fit(
        model, current_all, freq_all,
        p0=p0,
        sigma=freq_err_all,
        absolute_sigma=True,
        bounds=bounds,
        maxfev=20000,
    )

    params = popt
    param_errs = np.sqrt(np.diag(pcov))
    freq_fit = model(current_all, *params)
    residuals, chi2_val, ndof, chi2_red, p_val = compute_chi2(freq_all, freq_fit, freq_err_all, n_params)

    scale_factor = 1.0
    if rescale_errors:
        freq_err_all = rescale_errors_to_unit_chi2(freq_err_all, chi2_red)
        scale_factor = np.sqrt(chi2_red)

        popt, pcov = curve_fit(
            model, current_all, freq_all,
            p0=popt,
            sigma=freq_err_all,
            absolute_sigma=True,
            bounds=bounds,
            maxfev=20000,
        )

        params = popt
        param_errs = np.sqrt(np.diag(pcov))
        freq_fit = model(current_all, *params)
        residuals, chi2_val, ndof, chi2_red, p_val = compute_chi2(freq_all, freq_fit, freq_err_all, n_params)

    A_fit, I0_fit = params[:2]
    A_err, I0_err = param_errs[:2]
    f0_fit = params[2] if use_offset else 0.0
    f0_err = param_errs[2] if use_offset else 0.0

    I_fit, I_err = infer_spin_from_freq_vs_current_slope(A_fit, A_err, N, R, h, z)
    alpha = coil_field_per_current_G_per_A(N, R, h, z)
    B_amb = -alpha * I0_fit
    B_amb_err = abs(alpha) * I0_err

    result = {
        "isotope": iso_label,
        "filename_Bplus": filename_Bplus,
        "filename_Bminus": filename_Bminus,
        "current_A": current_all,
        "freq_MHz": freq_all,
        "freq_err_MHz": freq_err_all,
        "A_fit": A_fit,
        "A_err": A_err,
        "I0_fit_A": I0_fit,
        "I0_err_A": I0_err,
        "f0_fit_MHz": f0_fit,
        "f0_err_MHz": f0_err,
        "residuals": residuals,
        "chi2": chi2_val,
        "ndof": ndof,
        "chi2_red": chi2_red,
        "p_value": p_val,
        "scale_factor": scale_factor,
        "use_offset": use_offset,
        "I_fit": I_fit,
        "I_fit_err": I_err,
        "B_amb_G": B_amb,
        "B_amb_err_G": B_amb_err,
    }

    print(f"\n=== {iso_label} combined B+ and B- ===")
    print(f"B+ file       : {filename_Bplus}")
    print(f"B- file       : {filename_Bminus}")
    print(f"A (V slope)   = {A_fit:.4f} ± {A_err:.4f} MHz/A")
    print(f"I0 (minimum)  = {I0_fit:.4f} ± {I0_err:.4f} A")
    if use_offset:
        print(f"f0 (offset)   = {f0_fit:.4f} ± {f0_err:.4f} MHz")
    print(f"chi^2         = {chi2_val:.3f}")
    print(f"dof           = {ndof}")
    print(f"chi^2/dof     = {chi2_red:.3f}")
    print(f"p-value       = {p_val:.4g}")
    print(f"scale factor  = {scale_factor:.4f}")
    print(f"I_fit         = {I_fit:.4f} ± {I_err:.4f}")
    print(f"B_amb         = {B_amb:.4f} ± {B_amb_err:.4f} G")

    if make_plots:
        plot_combined_vfit(result)
        plot_combined_vfit_residuals(result)

    return result


# ============================================================
# Combined branch fit, current as dependent variable
# ============================================================

def analyze_isotope_combined_current_fit(filename_Bplus, filename_Bminus, isotope,
                                         N, R, h, z=0.0,
                                         current_col=7, freq_col=0,
                                         current_err_col=None, freq_err_col=None,
                                         skiprows=1, delimiter=",",
                                         p0=None,
                                         rescale_errors=True,
                                         make_plots=True):
    """
    Combine B+ and B- data and fit signed current as a function of frequency:
        I = I0 + branch_sign * b * nu
    This matches data tables where frequency is the controlled variable and current is measured.
    """
    iso_label = get_isotope_label(isotope)

    plus = prepare_dataset(
        filename_Bplus, isotope, "+",
        N, R, h, z=z,
        current_col=current_col, freq_col=freq_col,
        current_err_col=current_err_col, freq_err_col=freq_err_col,
        skiprows=skiprows, delimiter=delimiter,
    )
    minus = prepare_dataset(
        filename_Bminus, isotope, "-",
        N, R, h, z=z,
        current_col=current_col, freq_col=freq_col,
        current_err_col=current_err_col, freq_err_col=freq_err_col,
        skiprows=skiprows, delimiter=delimiter,
    )

    freq_all = np.concatenate([plus["freq_MHz"], minus["freq_MHz"]])
    current_all = np.concatenate([plus["current_A"], minus["current_A"]])
    branch_all = np.concatenate([plus["branch_sign"], minus["branch_sign"]])

    if plus["current_err_A"] is None or minus["current_err_A"] is None:
        raise ValueError("Need current uncertainties for the combined current-vs-frequency fit.")
    current_err_all = np.concatenate([plus["current_err_A"], minus["current_err_A"]])

    idx = np.argsort(freq_all)
    freq_all = freq_all[idx]
    current_all = current_all[idx]
    branch_all = branch_all[idx]
    current_err_all = current_err_all[idx]

    if p0 is None:
        alpha = coil_field_per_current_G_per_A(N, R, h, z)
        gamma_guess = gamma_MHz_per_G(get_nuclear_spin(isotope))
        A_guess = gamma_guess * alpha
        b_guess = 1.0 / A_guess
        I0_guess = np.median(current_all)
        p0 = [b_guess, I0_guess]

    xdata = np.vstack([freq_all, branch_all])

    popt, pcov = curve_fit(
        combined_current_vs_freq_model,
        xdata,
        current_all,
        p0=p0,
        sigma=current_err_all,
        absolute_sigma=True,
        bounds=([0.0, -np.inf], [np.inf, np.inf]),
        maxfev=20000,
    )

    b_fit, I0_fit = popt
    b_err, I0_err = np.sqrt(np.diag(pcov))
    current_fit = combined_current_vs_freq_model(xdata, b_fit, I0_fit)
    residuals, chi2_val, ndof, chi2_red, p_val = compute_chi2(current_all, current_fit, current_err_all, 2)

    scale_factor = 1.0
    if rescale_errors:
        current_err_all = rescale_errors_to_unit_chi2(current_err_all, chi2_red)
        scale_factor = np.sqrt(chi2_red)

        popt, pcov = curve_fit(
            combined_current_vs_freq_model,
            xdata,
            current_all,
            p0=popt,
            sigma=current_err_all,
            absolute_sigma=True,
            bounds=([0.0, -np.inf], [np.inf, np.inf]),
            maxfev=20000,
        )

        b_fit, I0_fit = popt
        b_err, I0_err = np.sqrt(np.diag(pcov))
        current_fit = combined_current_vs_freq_model(xdata, b_fit, I0_fit)
        residuals, chi2_val, ndof, chi2_red, p_val = compute_chi2(current_all, current_fit, current_err_all, 2)

    A_fit = 1.0 / b_fit
    A_err = b_err / (b_fit**2)
    I_fit, I_err = infer_spin_from_current_vs_freq_slope(b_fit, b_err, N, R, h, z)
    alpha = coil_field_per_current_G_per_A(N, R, h, z)
    B_amb = -alpha * I0_fit
    B_amb_err = abs(alpha) * I0_err

    result = {
        "isotope": iso_label,
        "filename_Bplus": filename_Bplus,
        "filename_Bminus": filename_Bminus,
        "freq_MHz": freq_all,
        "current_A": current_all,
        "branch_sign": branch_all,
        "current_err_A": current_err_all,
        "b_fit_A_per_MHz": b_fit,
        "b_err_A_per_MHz": b_err,
        "A_fit_MHz_per_A": A_fit,
        "A_err_MHz_per_A": A_err,
        "I0_fit_A": I0_fit,
        "I0_err_A": I0_err,
        "residuals": residuals,
        "chi2": chi2_val,
        "ndof": ndof,
        "chi2_red": chi2_red,
        "p_value": p_val,
        "scale_factor": scale_factor,
        "I_fit": I_fit,
        "I_fit_err": I_err,
        "B_amb_G": B_amb,
        "B_amb_err_G": B_amb_err,
    }

    print(f"\n=== {iso_label} combined current-vs-frequency fit ===")
    print(f"B+ file       : {filename_Bplus}")
    print(f"B- file       : {filename_Bminus}")
    print(f"b             = {b_fit:.6f} ± {b_err:.6f} A/MHz")
    print(f"A = 1/b       = {A_fit:.4f} ± {A_err:.4f} MHz/A")
    print(f"I0            = {I0_fit:.4f} ± {I0_err:.4f} A")
    print(f"chi^2         = {chi2_val:.3f}")
    print(f"dof           = {ndof}")
    print(f"chi^2/dof     = {chi2_red:.3f}")
    print(f"p-value       = {p_val:.4g}")
    print(f"scale factor  = {scale_factor:.4f}")
    print(f"I_fit         = {I_fit:.4f} ± {I_err:.4f}")
    print(f"B_amb         = {B_amb:.4f} ± {B_amb_err:.4f} G")

    if make_plots:
        plot_combined_current_fit(result)
        plot_combined_current_fit_residuals(result)

    return result


# ============================================================
# Plotting
# ============================================================

def plot_linear_data_and_fit(result):
    x = result["current_A"]
    y = result["freq_MHz"]
    yerr = result["freq_err_MHz"]
    m = result["fit"]["m"]
    c = result["fit"]["c"]

    plt.figure(figsize=(6, 4))
    plt.errorbar(
        x, y, yerr=yerr,
        fmt='o', linestyle='none', markersize=6,
        markerfacecolor='white', markeredgewidth=1.5,
        capsize=4, elinewidth=1.5, label='Data'
    )
    x_plot = np.linspace(np.min(x), np.max(x), 400)
    plt.plot(x_plot, linear_model(x_plot, m, c), label='Linear fit')
    plt.xlabel("Signed current (A)")
    plt.ylabel("Frequency (MHz)")
    plt.title(f"{result['isotope']}, B{result['field_direction']}: Frequency vs Current")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_linear_residuals(result):
    x = result["current_A"]
    residuals = result["residuals"]
    yerr = result["freq_err_MHz"]

    plt.figure(figsize=(6, 4))
    plt.errorbar(
        x, residuals, yerr=yerr,
        fmt='o', linestyle='none', markersize=6,
        markerfacecolor='white', markeredgewidth=1.5,
        capsize=4, elinewidth=1.5,
    )
    plt.axhline(0, linestyle='--', alpha=0.7)
    plt.xlabel("Signed current (A)")
    plt.ylabel("Residuals (MHz)")
    plt.title(f"{result['isotope']}, B{result['field_direction']}: Residuals")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_combined_vfit(result):
    x = result["current_A"]
    y = result["freq_MHz"]
    yerr = result["freq_err_MHz"]
    A = result["A_fit"]
    I0 = result["I0_fit_A"]
    use_offset = result["use_offset"]
    f0 = result["f0_fit_MHz"]

    plt.figure(figsize=(6, 4))
    plt.errorbar(
        x, y, yerr=yerr,
        fmt='o', linestyle='none', markersize=6,
        markerfacecolor='white', markeredgewidth=1.5,
        capsize=4, elinewidth=1.5, label='Data'
    )
    x_plot = np.linspace(np.min(x), np.max(x), 500)
    if use_offset:
        y_plot = v_model_freq_vs_current_offset(x_plot, A, I0, f0)
    else:
        y_plot = v_model_freq_vs_current(x_plot, A, I0)
    plt.plot(x_plot, y_plot, label='V fit')
    plt.axvline(I0, linestyle='--', alpha=0.7, label=r'$I_0$')
    plt.xlabel("Signed current (A)")
    plt.ylabel("Frequency (MHz)")
    plt.title(f"{result['isotope']}: Combined B+ and B-")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_combined_vfit_residuals(result):
    x = result["current_A"]
    residuals = result["residuals"]
    yerr = result["freq_err_MHz"]

    plt.figure(figsize=(6, 4))
    plt.errorbar(
        x, residuals, yerr=yerr,
        fmt='o', linestyle='none', markersize=6,
        markerfacecolor='white', markeredgewidth=1.5,
        capsize=4, elinewidth=1.5,
    )
    plt.axhline(0, linestyle='--', alpha=0.7)
    plt.xlabel("Signed current (A)")
    plt.ylabel("Residuals (MHz)")
    plt.title(f"{result['isotope']}: Residuals of combined V fit")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_combined_current_fit(result):
    freq = result["freq_MHz"]
    current = result["current_A"]
    current_err = result["current_err_A"]
    branch = result["branch_sign"]
    b = result["b_fit_A_per_MHz"]
    I0 = result["I0_fit_A"]

    plt.figure(figsize=(6, 4))

    mask_plus = branch > 0
    mask_minus = branch < 0

    plt.errorbar(
        freq[mask_plus], current[mask_plus], yerr=current_err[mask_plus],
        fmt='o', linestyle='none', markersize=6,
        markerfacecolor='white', markeredgewidth=1.5,
        capsize=4, elinewidth=1.5, label='B+'
    )
    plt.errorbar(
        freq[mask_minus], current[mask_minus], yerr=current_err[mask_minus],
        fmt='s', linestyle='none', markersize=6,
        markerfacecolor='white', markeredgewidth=1.5,
        capsize=4, elinewidth=1.5, label='B-'
    )

    f_plot = np.linspace(np.min(freq), np.max(freq), 400)
    plt.plot(f_plot, branch_model_current_vs_freq(f_plot, b, I0, +1), label='Fit, B+')
    plt.plot(f_plot, branch_model_current_vs_freq(f_plot, b, I0, -1), label='Fit, B-')

    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Signed current (A)")
    plt.title(f"{result['isotope']}: Combined current-vs-frequency fit")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_combined_current_fit_residuals(result):
    freq = result["freq_MHz"]
    residuals = result["residuals"]
    current_err = result["current_err_A"]
    branch = result["branch_sign"]

    plt.figure(figsize=(6, 4))

    mask_plus = branch > 0
    mask_minus = branch < 0

    plt.errorbar(
        freq[mask_plus], residuals[mask_plus], yerr=current_err[mask_plus],
        fmt='o', linestyle='none', markersize=6,
        markerfacecolor='white', markeredgewidth=1.5,
        capsize=4, elinewidth=1.5, label='B+'
    )
    plt.errorbar(
        freq[mask_minus], residuals[mask_minus], yerr=current_err[mask_minus],
        fmt='s', linestyle='none', markersize=6,
        markerfacecolor='white', markeredgewidth=1.5,
        capsize=4, elinewidth=1.5, label='B-'
    )

    plt.axhline(0, linestyle='--', alpha=0.7)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Residuals (A)")
    plt.title(f"{result['isotope']}: Residuals of combined current-vs-frequency fit")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Summary helpers
# ============================================================

def summarize_linear_results(results):
    print("\nSingle-dataset linear-fit summary")
    print("-" * 80)
    for r in results:
        print(
            f"{r['isotope']:>4}  B{r['field_direction']:<2} | "
            f"m = {r['fit']['m']:.4f} ± {r['fit']['m_err']:.4f} MHz/A | "
            f"c = {r['fit']['c']:.4f} ± {r['fit']['c_err']:.4f} MHz | "
            f"I = {r['I_fit']:.4f} ± {r['I_fit_err']:.4f} | "
            f"B_amb = {r['B_amb_G']:.4f} ± {r['B_amb_err_G']:.4f} G"
        )


def summarize_vfit_results(results):
    print("\nCombined V-fit summary")
    print("-" * 80)
    for r in results:
        print(
            f"{r['isotope']:>4} | "
            f"A = {r['A_fit']:.4f} ± {r['A_err']:.4f} MHz/A | "
            f"I0 = {r['I0_fit_A']:.4f} ± {r['I0_err_A']:.4f} A | "
            f"I = {r['I_fit']:.4f} ± {r['I_fit_err']:.4f} | "
            f"B_amb = {r['B_amb_G']:.4f} ± {r['B_amb_err_G']:.4f} G"
        )


def summarize_current_fit_results(results):
    print("\nCombined current-vs-frequency fit summary")
    print("-" * 80)
    for r in results:
        print(
            f"{r['isotope']:>4} | "
            f"b = {r['b_fit_A_per_MHz']:.6f} ± {r['b_err_A_per_MHz']:.6f} A/MHz | "
            f"A = {r['A_fit_MHz_per_A']:.4f} ± {r['A_err_MHz_per_A']:.4f} MHz/A | "
            f"I0 = {r['I0_fit_A']:.4f} ± {r['I0_err_A']:.4f} A | "
            f"I = {r['I_fit']:.4f} ± {r['I_fit_err']:.4f} | "
            f"B_amb = {r['B_amb_G']:.4f} ± {r['B_amb_err_G']:.4f} G"
        )


__all__ = [
    "get_nuclear_spin",
    "get_isotope_label",
    "field_sign_from_direction",
    "coil_field_per_current_G_per_A",
    "gamma_MHz_per_G",
    "linear_model",
    "v_model_freq_vs_current",
    "v_model_freq_vs_current_offset",
    "branch_model_current_vs_freq",
    "combined_current_vs_freq_model",
    "load_optical_pumping_csv",
    "prepare_dataset",
    "analyze_optical_pumping_dataset",
    "analyze_isotope_combined_vfit",
    "analyze_isotope_combined_current_fit",
    "plot_linear_data_and_fit",
    "plot_linear_residuals",
    "plot_combined_vfit",
    "plot_combined_vfit_residuals",
    "plot_combined_current_fit",
    "plot_combined_current_fit_residuals",
    "summarize_linear_results",
    "summarize_vfit_results",
    "summarize_current_fit_results",
]
