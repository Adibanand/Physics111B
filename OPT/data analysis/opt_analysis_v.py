import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2
import scipy.constants as sc


# ============================================================
# Physics helpers
# ============================================================

def get_nuclear_spin(isotope: str) -> float:
    """
    Returns the known nuclear spin for the requested isotope.

    Supported inputs include:
        'Rb85', '85', '85Rb', 'Rb-85'
        'Rb87', '87', '87Rb', 'Rb-87'
    """
    s = isotope.strip().lower()
    if s in ["rb85", "85", "85rb", "rb-85"]:
        return 5 / 2
    if s in ["rb87", "87", "87rb", "rb-87"]:
        return 3 / 2
    raise ValueError(f"Unrecognized isotope: {isotope}")


def get_isotope_label(isotope: str) -> str:
    """Returns a clean isotope label."""
    s = isotope.strip().lower()
    if s in ["rb85", "85", "85rb", "rb-85"]:
        return "Rb85"
    if s in ["rb87", "87", "87rb", "rb-87"]:
        return "Rb87"
    raise ValueError(f"Unrecognized isotope: {isotope}")


def field_sign_from_direction(field_direction: str) -> int:
    """
    Converts field-direction labels to a sign.

    '+' or 'B+' -> +1
    '-' or 'B-' -> -1
    """
    s = field_direction.strip().lower()
    if s in ["+", "b+", "plus", "positive", "pos"]:
        return +1
    if s in ["-", "b-", "minus", "negative", "neg"]:
        return -1
    raise ValueError(f"Unrecognized field direction: {field_direction}")


def coil_field_per_current_G_per_A(N: float, R: float, h: float, z: float = 0.0) -> float:
    """
    Helmholtz-coil field conversion factor in Gauss/A using the convention
    from the user's lab notes/code, where h is the distance from the midpoint
    between the coils to one coil.

    Parameters
    ----------
    N : float
        Number of turns.
    R : float
        Coil radius in meters.
    h : float
        Distance from midpoint to one coil in meters.
    z : float, optional
        Axial displacement from the midpoint in meters.

    Returns
    -------
    float
        Magnetic-field conversion factor in Gauss/A.
    """
    B_per_A_T = sc.mu_0 * N * R**2 / (((h - z) ** 2 + R**2) ** 1.5)
    return 1e4 * B_per_A_T  # Tesla -> Gauss


def gamma_MHz_per_G(I: float) -> float:
    """
    Zeeman frequency coefficient in MHz/G from the optical-pumping lab model.
    """
    return 2.799 / (2 * I + 1)


# ============================================================
# Models
# ============================================================

def linear_model(x, m, c):
    """Straight-line model: y = m x + c."""
    return m * np.asarray(x) + c


def v_model(current, A, I0):
    """
    V-shaped resonance model:
        nu(I) = A * |I - I0|

    Parameters
    ----------
    current : array-like
        Signed current in A.
    A : float
        Slope of each arm of the V in MHz/A.
    I0 : float
        Current at which the total field cancels, in A.
    """
    return A * np.abs(np.asarray(current) - I0)


def v_model_offset(current, A, I0, f0):
    """
    V-shaped resonance model with vertical offset:
        nu(I) = A * |I - I0| + f0

    Useful if the minimum frequency does not go exactly to zero.
    """
    return A * np.abs(np.asarray(current) - I0) + f0


# ============================================================
# Data loading and preparation
# ============================================================

def load_optical_pumping_csv(
    filename,
    current_col=1,
    freq_col=8,
    current_err_col=None,
    freq_err_col=None,
    skiprows=1,
    delimiter=",",
):
    """
    Loads the requested columns from a CSV file.

    All column numbers are 0-indexed.

    Returns
    -------
    dict with keys:
        current, freq_MHz, current_err, freq_err
    """
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
        filling_values=np.nan,
    )

    if data.ndim == 1:
        data = data[None, :]

    data = data[~np.isnan(data).any(axis=1)]

    out = {
        "current": data[:, 0],
        "freq_MHz": data[:, 1],
        "current_err": None,
        "freq_err": None,
    }

    idx = 2
    if current_err_col is not None:
        out["current_err"] = data[:, idx]
        idx += 1

    if freq_err_col is not None:
        out["freq_err"] = data[:, idx]

    return out


def freq_err_from_current_err(current_err, I, N, R, h, z=0.0):
    """
    Propagates current uncertainty into frequency uncertainty.

    Returns frequency uncertainty in MHz.
    """
    alpha_G_per_A = coil_field_per_current_G_per_A(N, R, h, z)
    slope_MHz_per_A = gamma_MHz_per_G(I) * alpha_G_per_A
    return np.abs(slope_MHz_per_A) * np.asarray(current_err)


def prepare_dataset(
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
    delimiter=",",
):
    """
    Loads one optical-pumping dataset and returns signed current, frequency,
    and frequency uncertainty.

    The sign is assigned to current, not to frequency.
    """
    I_theory = get_nuclear_spin(isotope)
    sign = field_sign_from_direction(field_direction)

    data = load_optical_pumping_csv(
        filename,
        current_col=current_col,
        freq_col=freq_col,
        current_err_col=current_err_col,
        freq_err_col=freq_err_col,
        skiprows=skiprows,
        delimiter=delimiter,
    )

    current = np.asarray(data["current"])
    freq_MHz = np.asarray(data["freq_MHz"])
    current_signed = sign * current

    if data["freq_err"] is not None:
        freq_err = np.asarray(data["freq_err"])
    elif data["current_err"] is not None:
        freq_err = freq_err_from_current_err(data["current_err"], I_theory, N, R, h, z)
    else:
        raise ValueError("Need either freq_err_col or current_err_col.")

    idx = np.argsort(current_signed)
    current_signed = current_signed[idx]
    freq_MHz = freq_MHz[idx]
    freq_err = freq_err[idx]

    return current_signed, freq_MHz, freq_err


# ============================================================
# Generic fitting helpers
# ============================================================

def fit_model_with_optional_rescaling(
    model_func,
    x,
    y,
    yerr,
    p0,
    rescale_errors=True,
):
    """
    Weighted least-squares fit with optional global rescaling of uncertainties.

    If rescale_errors is True, all y-errors are multiplied by sqrt(chi^2_red)
    and the fit is repeated so the final reduced chi-squared is 1.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    yerr = np.asarray(yerr)

    popt, pcov = curve_fit(
        model_func,
        x,
        y,
        p0=p0,
        sigma=yerr,
        absolute_sigma=True,
    )

    yfit = model_func(x, *popt)
    residuals = y - yfit
    chi2_val = np.sum((residuals / yerr) ** 2)
    ndof = len(x) - len(popt)
    chi2_red = chi2_val / ndof
    p_val = chi2.sf(chi2_val, ndof)

    result = {
        "popt": popt,
        "pcov": pcov,
        "perr": np.sqrt(np.diag(pcov)),
        "yfit": yfit,
        "residuals": residuals,
        "chi2": chi2_val,
        "ndof": ndof,
        "chi2_red": chi2_red,
        "p_value": p_val,
        "yerr_used": np.array(yerr, copy=True),
        "scale_factor": 1.0,
        "chi2_initial": chi2_val,
        "chi2_red_initial": chi2_red,
        "p_value_initial": p_val,
    }

    if not rescale_errors:
        return result

    scale = np.sqrt(chi2_red)
    yerr_scaled = yerr * scale

    popt2, pcov2 = curve_fit(
        model_func,
        x,
        y,
        p0=popt,
        sigma=yerr_scaled,
        absolute_sigma=True,
    )

    yfit2 = model_func(x, *popt2)
    residuals2 = y - yfit2
    chi2_val2 = np.sum((residuals2 / yerr_scaled) ** 2)
    chi2_red2 = chi2_val2 / ndof
    p_val2 = chi2.sf(chi2_val2, ndof)

    return {
        "popt": popt2,
        "pcov": pcov2,
        "perr": np.sqrt(np.diag(pcov2)),
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


def fit_line_with_optional_rescaling(x, y, yerr, p0=(1.0, 0.0), rescale_errors=True):
    """Convenience wrapper for a straight-line fit."""
    result = fit_model_with_optional_rescaling(
        linear_model,
        x,
        y,
        yerr,
        p0=p0,
        rescale_errors=rescale_errors,
    )

    m, c = result["popt"]
    m_err, c_err = result["perr"]
    result.update({"m": m, "c": c, "m_err": m_err, "c_err": c_err})
    return result


def fit_v_with_optional_rescaling(
    current,
    freq_MHz,
    freq_err,
    use_offset=False,
    p0=None,
    rescale_errors=True,
):
    """
    Fits combined B+ and B- data to a V-shaped model.

    Parameters
    ----------
    use_offset : bool
        If True, fits nu(I) = A |I - I0| + f0.
        If False, fits nu(I) = A |I - I0|.
    """
    current = np.asarray(current)
    freq_MHz = np.asarray(freq_MHz)
    freq_err = np.asarray(freq_err)

    if p0 is None:
        A0 = (np.max(freq_MHz) - np.min(freq_MHz)) / (np.max(current) - np.min(current))
        I00 = current[np.argmin(freq_MHz)]
        if use_offset:
            p0 = [A0, I00, np.min(freq_MHz)]
        else:
            p0 = [A0, I00]

    model_func = v_model_offset if use_offset else v_model
    result = fit_model_with_optional_rescaling(
        model_func,
        current,
        freq_MHz,
        freq_err,
        p0=p0,
        rescale_errors=rescale_errors,
    )

    if use_offset:
        A_fit, I0_fit, f0_fit = result["popt"]
        A_err, I0_err, f0_err = result["perr"]
        result.update(
            {
                "A_fit": A_fit,
                "I0_fit_A": I0_fit,
                "f0_fit_MHz": f0_fit,
                "A_err": A_err,
                "I0_err_A": I0_err,
                "f0_err_MHz": f0_err,
                "model_func": v_model_offset,
                "use_offset": True,
            }
        )
    else:
        A_fit, I0_fit = result["popt"]
        A_err, I0_err = result["perr"]
        result.update(
            {
                "A_fit": A_fit,
                "I0_fit_A": I0_fit,
                "A_err": A_err,
                "I0_err_A": I0_err,
                "model_func": v_model,
                "use_offset": False,
            }
        )

    return result


# ============================================================
# Physics extraction from fit parameters
# ============================================================

def infer_ambient_field_from_fit(intercept_MHz, intercept_err_MHz, I):
    """
    For a straight-line fit c = gamma * B_amb, infer B_amb in Gauss.
    """
    gamma = gamma_MHz_per_G(I)
    B_amb = intercept_MHz / gamma
    B_amb_err = intercept_err_MHz / gamma
    return B_amb, B_amb_err


def infer_spin_from_slope(slope_MHz_per_A, slope_err_MHz_per_A, N, R, h, z=0.0):
    """
    From a straight-line slope or V-slope:
        slope = [2.799/(2I+1)] * alpha
    with alpha = coil_field_per_current_G_per_A.

    Solves for I and propagates uncertainty from the slope only.
    """
    alpha = coil_field_per_current_G_per_A(N, R, h, z)
    q = 2.799 * alpha / slope_MHz_per_A  # = 2I + 1
    I_fit = (q - 1) / 2

    dI_dm = -(2.799 * alpha) / (2 * slope_MHz_per_A**2)
    I_err = np.abs(dI_dm) * slope_err_MHz_per_A
    return I_fit, I_err


def infer_ambient_field_from_vminimum(I0_fit_A, I0_err_A, N, R, h, z=0.0):
    """
    From the V minimum location, infer ambient field:
        B_amb = - alpha * I0
    where alpha is in G/A.
    """
    alpha = coil_field_per_current_G_per_A(N, R, h, z)
    B_amb = -alpha * I0_fit_A
    B_amb_err = np.abs(alpha) * I0_err_A
    return B_amb, B_amb_err


# ============================================================
# Plotting
# ============================================================

def _default_errorbar_kwargs(label=None):
    return dict(
        fmt=".",
        linestyle="none",
        markersize=6,
        markerfacecolor="red",
        markeredgewidth=1.5,
        capsize=4,
        elinewidth=1.5,
        label=label,
    )


def plot_data_and_fit(
    x,
    y,
    yerr,
    fit_result,
    title="Linear Fit",
    xlabel="Current (A)",
    ylabel="Frequency (MHz)",
):
    """Plots one straight-line dataset and its fit."""
    plt.figure(figsize=(6, 4))

    plt.errorbar(x, y, yerr=yerr, **_default_errorbar_kwargs(label="Data"))

    x_plot = np.linspace(np.min(x), np.max(x), 300)
    plt.plot(x_plot, linear_model(x_plot, fit_result["m"], fit_result["c"]), label="Linear fit")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_residuals(
    x,
    residuals,
    yerr,
    title="Residuals",
    xlabel="Current (A)",
):
    """Plots residuals for one straight-line dataset."""
    plt.figure(figsize=(6, 4))

    plt.errorbar(x, residuals, yerr=yerr, **_default_errorbar_kwargs())
    plt.axhline(0, linestyle="--", alpha=0.7)

    plt.xlabel(xlabel)
    plt.ylabel("Residuals (MHz)")
    plt.title(title)
    
    plt.tight_layout()
    plt.show()


def plot_combined_vfit(result):
    """Plots combined B+ and B- data and the fitted V curve."""
    current = result["current_A"]
    freq = result["freq_MHz"]
    freq_err = result["freq_err_MHz"]
    isotope = result["isotope"]

    plt.figure(figsize=(6, 4))
    plt.errorbar(current, freq, yerr=freq_err, **_default_errorbar_kwargs(label="Data"))

    x_plot = np.linspace(np.min(current), np.max(current), 500)
    if result["use_offset"]:
        y_plot = v_model_offset(x_plot, result["A_fit"], result["I0_fit_A"], result["f0_fit_MHz"])
    else:
        y_plot = v_model(x_plot, result["A_fit"], result["I0_fit_A"])

    plt.plot(x_plot, y_plot, label="V fit")
    plt.axvline(result["I0_fit_A"], linestyle="--", alpha=0.7, label=r"$I_0$")

    plt.xlabel("Signed Current (A)")
    plt.ylabel("Frequency (MHz)")
    plt.title(f"{isotope}: Combined B+ and B-")
    
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_combined_residuals(result):
    """Plots residuals for the combined V fit."""
    current = result["current_A"]
    residuals = result["residuals"]
    freq_err = result["freq_err_MHz"]
    isotope = result["isotope"]

    plt.figure(figsize=(6, 4))
    plt.errorbar(current, residuals, yerr=freq_err, **_default_errorbar_kwargs())
    plt.axhline(0, linestyle="--", alpha=0.7)

    plt.xlabel("Signed Current (A)")
    plt.ylabel("Residuals (MHz)")
    plt.title(f"{isotope}: Residuals of Combined V Fit")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_stacked_fit_and_residuals(x, y, yerr, fit_result, title="Fit and residuals"):
    """
    Makes a 2-panel plot with data+fit on top and residuals on bottom.
    Works for straight-line fits.
    """
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(6, 6),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax1.errorbar(x, y, yerr=yerr, **_default_errorbar_kwargs(label="Data"))
    x_plot = np.linspace(np.min(x), np.max(x), 300)
    ax1.plot(x_plot, linear_model(x_plot, fit_result["m"], fit_result["c"]), label="Linear fit")
    ax1.set_ylabel("Frequency (MHz)")
    ax1.set_title(title)
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.errorbar(x, fit_result["residuals"], yerr=yerr, **_default_errorbar_kwargs())
    ax2.axhline(0, linestyle="--", alpha=0.7)
    ax2.set_xlabel("Current (A)")
    ax2.set_ylabel("Residuals")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_stacked_vfit_and_residuals(result):
    """
    Makes a 2-panel plot with combined V-fit on top and residuals on bottom.
    """
    current = result["current_A"]
    freq = result["freq_MHz"]
    freq_err = result["freq_err_MHz"]

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(6, 6),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax1.errorbar(current, freq, yerr=freq_err, **_default_errorbar_kwargs(label="Data"))
    x_plot = np.linspace(np.min(current), np.max(current), 500)
    if result["use_offset"]:
        y_plot = v_model_offset(x_plot, result["A_fit"], result["I0_fit_A"], result["f0_fit_MHz"])
    else:
        y_plot = v_model(x_plot, result["A_fit"], result["I0_fit_A"])
    ax1.plot(x_plot, y_plot, label="V fit")
    ax1.axvline(result["I0_fit_A"], linestyle="--", alpha=0.7, label=r"$I_0$")
    ax1.set_ylabel("Frequency (MHz)")
    ax1.set_title(f"{result['isotope']}: Combined B+ and B-")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.errorbar(current, result["residuals"], yerr=freq_err, **_default_errorbar_kwargs())
    ax2.axhline(0, linestyle="--", alpha=0.7)
    ax2.set_xlabel("Signed Current (A)")
    ax2.set_ylabel("Residuals")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================
# Analysis wrappers
# ============================================================

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
    delimiter=",",
    p0=(1.0, 0.0),
    rescale_errors=True,
    make_plots=True,
):
    """
    Full analysis pipeline for one dataset treated as a straight line.

    Best used for analyzing a single branch (B+ or B-) separately.
    """
    iso_label = get_isotope_label(isotope)
    I_theory = get_nuclear_spin(isotope)

    current_signed, freq_MHz, freq_err = prepare_dataset(
        filename,
        isotope,
        field_direction,
        N,
        R,
        h,
        z=z,
        current_col=current_col,
        freq_col=freq_col,
        current_err_col=current_err_col,
        freq_err_col=freq_err_col,
        skiprows=skiprows,
        delimiter=delimiter,
    )

    fit_result = fit_line_with_optional_rescaling(
        current_signed,
        freq_MHz,
        freq_err,
        p0=p0,
        rescale_errors=rescale_errors,
    )

    B_amb, B_amb_err = infer_ambient_field_from_fit(fit_result["c"], fit_result["c_err"], I_theory)
    I_fit, I_fit_err = infer_spin_from_slope(fit_result["m"], fit_result["m_err"], N, R, h, z)

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
            title=f"{iso_label}, B{field_direction}: Frequency vs Current",
            xlabel="Signed Current (A)",
        )
        plot_residuals(
            current_signed,
            fit_result["residuals"],
            fit_result["yerr_used"],
            title=f"{iso_label}, B{field_direction}: Residuals",
            xlabel="Signed Current (A)",
        )

    return result


def analyze_isotope_combined_vfit(
    filename_Bplus,
    filename_Bminus,
    isotope,
    N,
    R,
    h,
    z=0.0,
    current_col=1,
    freq_col=8,
    current_err_col=None,
    freq_err_col=None,
    skiprows=1,
    delimiter=",",
    use_offset=False,
    p0=None,
    rescale_errors=True,
    make_plots=True,
):
    """
    Combines the B+ and B- datasets for one isotope and fits them to a V-shaped model.

    This is the preferred analysis when you want to use the full dataset to infer
    nuclear spin and ambient field.
    """
    iso_label = get_isotope_label(isotope)

    current_plus, freq_plus, ferr_plus = prepare_dataset(
        filename_Bplus,
        isotope,
        "+",
        N,
        R,
        h,
        z=z,
        current_col=current_col,
        freq_col=freq_col,
        current_err_col=current_err_col,
        freq_err_col=freq_err_col,
        skiprows=skiprows,
        delimiter=delimiter,
    )

    current_minus, freq_minus, ferr_minus = prepare_dataset(
        filename_Bminus,
        isotope,
        "-",
        N,
        R,
        h,
        z=z,
        current_col=current_col,
        freq_col=freq_col,
        current_err_col=current_err_col,
        freq_err_col=freq_err_col,
        skiprows=skiprows,
        delimiter=delimiter,
    )

    current_all = np.concatenate([current_plus, current_minus])
    freq_all = np.concatenate([freq_plus, freq_minus])
    ferr_all = np.concatenate([ferr_plus, ferr_minus])

    idx = np.argsort(current_all)
    current_all = current_all[idx]
    freq_all = freq_all[idx]
    ferr_all = ferr_all[idx]

    fit_result = fit_v_with_optional_rescaling(
        current_all,
        freq_all,
        ferr_all,
        use_offset=use_offset,
        p0=p0,
        rescale_errors=rescale_errors,
    )

    I_fit, I_fit_err = infer_spin_from_slope(fit_result["A_fit"], fit_result["A_err"], N, R, h, z)
    B_amb, B_amb_err = infer_ambient_field_from_vminimum(fit_result["I0_fit_A"], fit_result["I0_err_A"], N, R, h, z)

    result = {
        "isotope": iso_label,
        "filename_Bplus": filename_Bplus,
        "filename_Bminus": filename_Bminus,
        "current_A": current_all,
        "freq_MHz": freq_all,
        "freq_err_MHz": fit_result["yerr_used"],
        **fit_result,
        "I_fit": I_fit,
        "I_fit_err": I_fit_err,
        "B_amb_G": B_amb,
        "B_amb_err_G": B_amb_err,
        "alpha_G_per_A": coil_field_per_current_G_per_A(N, R, h, z),
    }

    print(f"\n=== {iso_label} combined B+ and B- ===")
    print(f"B+ file       : {filename_Bplus}")
    print(f"B- file       : {filename_Bminus}")
    print(f"A (V slope)   = {result['A_fit']:.4f} ± {result['A_err']:.4f} MHz/A")
    print(f"I0 (minimum)  = {result['I0_fit_A']:.4f} ± {result['I0_err_A']:.4f} A")
    if result["use_offset"]:
        print(f"f0 (offset)   = {result['f0_fit_MHz']:.4f} ± {result['f0_err_MHz']:.4f} MHz")
    print(f"chi^2         = {result['chi2']:.3f}")
    print(f"dof           = {result['ndof']}")
    print(f"chi^2/dof     = {result['chi2_red']:.3f}")
    print(f"p-value       = {result['p_value']:.4g}")
    print(f"scale factor  = {result['scale_factor']:.4f}")
    print(f"I_fit         = {I_fit:.4f} ± {I_fit_err:.4f}")
    print(f"B_amb         = {B_amb:.4f} ± {B_amb_err:.4f} G")

    if make_plots:
        plot_combined_vfit(result)
        plot_combined_residuals(result)

    return result


# ============================================================
# Summary helpers
# ============================================================

def summarize_linear_results(results):
    """
    Prints a compact summary for a list of single-branch straight-line fits.
    """
    print("\nSingle-branch linear-fit summary")
    print("-" * 90)
    for r in results:
        fit = r["fit"]
        print(
            f"{r['isotope']:>4}  B{r['field_direction']:<2} | "
            f"m = {fit['m']:.4f} ± {fit['m_err']:.4f} MHz/A | "
            f"c = {fit['c']:.4f} ± {fit['c_err']:.4f} MHz | "
            f"chi2/dof = {fit['chi2_red']:.3f} | "
            f"I = {r['I_fit']:.4f} ± {r['I_fit_err']:.4f} | "
            f"B_amb = {r['B_amb_G']:.4f} ± {r['B_amb_err_G']:.4f} G"
        )


def summarize_combined_v_results(results):
    """
    Prints a compact summary for a list of combined-isotope V fits.
    """
    print("\nCombined V-fit summary")
    print("-" * 90)
    for r in results:
        line = (
            f"{r['isotope']:>4} | "
            f"A = {r['A_fit']:.4f} ± {r['A_err']:.4f} MHz/A | "
            f"I0 = {r['I0_fit_A']:.4f} ± {r['I0_err_A']:.4f} A | "
            f"chi2/dof = {r['chi2_red']:.3f} | "
            f"I = {r['I_fit']:.4f} ± {r['I_fit_err']:.4f} | "
            f"B_amb = {r['B_amb_G']:.4f} ± {r['B_amb_err_G']:.4f} G"
        )
        if r["use_offset"]:
            line += f" | f0 = {r['f0_fit_MHz']:.4f} ± {r['f0_err_MHz']:.4f} MHz"
        print(line)


# ============================================================
# Example usage
# ============================================================
# N = 135
# R = 27.5e-2
# h = 14.25e-2
# z = 0.0
#
# res_85_plus = analyze_optical_pumping_dataset(
#     filename="opt - Rb 85, +.csv",
#     isotope="Rb85",
#     field_direction="+",
#     N=N, R=R, h=h, z=z,
#     current_col=1,
#     freq_col=8,
#     current_err_col=9,
#     skiprows=1,
#     rescale_errors=True,
# )
#
# res_85_combined = analyze_isotope_combined_vfit(
#     filename_Bplus="opt - Rb 85, +.csv",
#     filename_Bminus="opt - Rb 85, -.csv",
#     isotope="Rb85",
#     N=N, R=R, h=h, z=z,
#     current_col=1,
#     freq_col=8,
#     current_err_col=9,
#     skiprows=1,
#     use_offset=False,
#     rescale_errors=True,
# )
