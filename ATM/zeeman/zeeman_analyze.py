"""
Zeeman Analysis: Load cached radial profiles and compute fractional frequency shifts + Bohr magneton.

Prerequisites: Run zeeman_preprocess.py once to create r_arr.npy, I_r_arr.npy, centers.npy.

This script is fast—it only loads the cache and runs peak detection, fitting, and linear regression.
No image loading or center-finding is performed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.constants import c, h
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# Optional matplotlib for plotting (use Agg for headless saving)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AnalyzeConfig:
    """Configuration for Zeeman analysis. Override as needed."""
    cache_dir: Path = Path(".")
    etalon_thickness_cm: float = 0.811  # 8.11 mm
    B_fields: Optional[np.ndarray] = None  # In gauss; set explicitly or via CLI
    B_scale: float = 1000.0  # If B_fields are in kG, use 1000 to convert to G

    # Peak detection
    prominence_frac: float = 0.02
    min_peak_distance: int = 10
    smooth_window: int = 7
    inner_radius: float = 40.0
    outer_radius: Optional[float] = None

    # Local fitting
    window_half_width: int = 18
    use_triplet: bool = True
    min_sigma_pi_sep_pix: float = 0.5
    max_sigma_pi_sep_pix: float = 30.0
    max_center_shift_pix: float = 25.0
    min_rings_required: int = 2

    # FSR mode: 'r2' (Fabry-Perot natural) or 'r' (linear radius)
    fsr_mode: str = "r2"
    split_mode: str = "adjacent"  # 'adjacent' = sigma-pi, 'full' = sigma+/-/2

    # FSR must come from adjacent interference orders, not split components.
    # Full triplet (sigma- to sigma+) is ~15–25 px; adjacent orders are 25+ px apart.
    min_order_spacing_r: float = 25.0

    # Output
    out_ring_csv: Optional[Path] = None
    out_summary_csv: Optional[Path] = None
    out_plot: Optional[Path] = None


# =============================================================================
# Load cache
# =============================================================================

def load_cached_profiles(cache_dir: Path) -> Tuple[List[np.ndarray], List[np.ndarray], Optional[np.ndarray]]:
    """Load r_arr, I_r_arr, centers from cache directory."""
    r_file = cache_dir / "r_arr.npy"
    i_file = cache_dir / "I_r_arr.npy"
    c_file = cache_dir / "centers.npy"

    if not r_file.exists() or not i_file.exists():
        raise FileNotFoundError(
            f"Cache not found in {cache_dir}. Run zeeman_preprocess.py first.\n"
            f"  Expected: r_arr.npy, I_r_arr.npy (and optionally centers.npy)"
        )

    r_arr = np.load(r_file, allow_pickle=True)
    I_arr = np.load(i_file, allow_pickle=True)
    centers = np.load(c_file, allow_pickle=True) if c_file.exists() else None

    r_list = [np.asarray(r, dtype=float) for r in r_arr]
    I_list = [np.asarray(i, dtype=float) for i in I_arr]
    return r_list, I_list, centers


# =============================================================================
# Helpers
# =============================================================================

def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y.copy()
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def robust_prominence(y: np.ndarray, frac: float) -> float:
    lo, hi = np.percentile(y, [5, 95])
    return max((hi - lo) * frac, 1e-9)


def gaussian(x: np.ndarray, A: float, mu: float, s: float, C: float) -> np.ndarray:
    return C + A * np.exp(-0.5 * ((x - mu) / s) ** 2)


def triplet_model(
    x: np.ndarray,
    A1: float, mu1: float, s1: float,
    A2: float, mu2: float, s2: float,
    A3: float, mu3: float, s3: float,
    C: float,
) -> np.ndarray:
    return (
        gaussian(x, A1, mu1, s1, 0)
        + gaussian(x, A2, mu2, s2, 0)
        + gaussian(x, A3, mu3, s3, 0)
        + C
    )


def doublet_model(
    x: np.ndarray,
    A1: float, mu1: float, s1: float,
    A2: float, mu2: float, s2: float,
    C: float,
) -> np.ndarray:
    return gaussian(x, A1, mu1, s1, 0) + gaussian(x, A2, mu2, s2, 0) + C


# =============================================================================
# Peak detection and ring alignment
# =============================================================================

def detect_ring_centers(
    r: np.ndarray,
    I: np.ndarray,
    prominence_frac: float = 0.02,
    min_dist: int = 10,
    smooth: int = 7,
    r_min: float = 40.0,
    r_max: Optional[float] = None,
) -> np.ndarray:
    """Return r positions of ring peaks."""
    y = moving_average(I, smooth)
    mask = r >= r_min
    if r_max is not None:
        mask &= r <= r_max
    rr = r[mask]
    yy = y[mask]
    prom = robust_prominence(yy, prominence_frac)
    peak_idx, _ = find_peaks(yy, prominence=prom, distance=min_dist)
    return rr[peak_idx]


def align_ring_guesses(
    peaks_per_image: List[np.ndarray],
    max_shift: float = 25.0,
) -> dict:
    """Map master ring index -> [(image_idx, matched_r), ...]."""
    if not peaks_per_image:
        return {}
    master = peaks_per_image[0]
    aligned = {k: [(0, float(r0))] for k, r0 in enumerate(master)}
    for img_idx, peaks in enumerate(peaks_per_image[1:], start=1):
        for k, r0 in enumerate(master):
            if len(peaks) == 0:
                continue
            j = int(np.argmin(np.abs(peaks - r0)))
            if abs(peaks[j] - r0) <= max_shift:
                aligned[k].append((img_idx, float(peaks[j])))
    return aligned


# =============================================================================
# Local fitting (triplet / doublet)
# =============================================================================

def fit_triplet_local(
    x: np.ndarray,
    y: np.ndarray,
    mu_center_guess: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    loc_idx, _ = find_peaks(y, prominence=robust_prominence(y, 0.15), distance=3)
    xpk = x[loc_idx] if len(loc_idx) else np.array([])

    if len(xpk) >= 3:
        order = np.argsort(np.abs(xpk - mu_center_guess))
        chosen = np.sort(xpk[order[:3]])
        mu1, mu2, mu3 = chosen[0], chosen[1], chosen[2]
    else:
        sep = 4.0
        mu1, mu2, mu3 = mu_center_guess - sep, mu_center_guess, mu_center_guess + sep

    C0 = np.percentile(y, 10)
    A0 = max(y.max() - C0, 1e-6)
    p0 = [0.5 * A0, mu1, 2.5, A0, mu2, 2.5, 0.5 * A0, mu3, 2.5, C0]
    bounds_lower = [0, x.min(), 0.3, 0, x.min(), 0.3, 0, x.min(), 0.3, -np.inf]
    bounds_upper = [np.inf, x.max(), 15.0, np.inf, x.max(), 15.0, np.inf, x.max(), 15.0, np.inf]

    try:
        popt, pcov = curve_fit(
            triplet_model, x, y,
            p0=p0, bounds=(bounds_lower, bounds_upper), maxfev=50000,
        )
        # Sort by center
        peaks = sorted(
            [(popt[0], popt[1], popt[2]), (popt[3], popt[4], popt[5]), (popt[6], popt[7], popt[8])],
            key=lambda t: t[1],
        )
        (_, mu1, _), (_, mu2, _), (_, mu3, _) = peaks
        ordered = np.array([popt[0], mu1, popt[2], popt[3], mu2, popt[5], popt[6], mu3, popt[8], popt[9]])
        return ordered, pcov
    except Exception:
        return np.full(10, np.nan), None


def fit_doublet_local(
    x: np.ndarray,
    y: np.ndarray,
    mu_center_guess: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    sep = 4.0
    C0 = np.percentile(y, 10)
    A0 = max(y.max() - C0, 1e-6)
    p0 = [0.7 * A0, mu_center_guess - sep / 2, 2.5, A0, mu_center_guess + sep / 2, 2.5, C0]
    bounds_lower = [0, x.min(), 0.3, 0, x.min(), 0.3, -np.inf]
    bounds_upper = [np.inf, x.max(), 15.0, np.inf, x.max(), 15.0, np.inf]
    try:
        popt, pcov = curve_fit(
            doublet_model, x, y,
            p0=p0, bounds=(bounds_lower, bounds_upper), maxfev=50000,
        )
        peaks = sorted([(popt[0], popt[1], popt[2]), (popt[3], popt[4], popt[5])], key=lambda t: t[1])
        (_, mu1, _), (_, mu2, _) = peaks
        ordered = np.array([popt[0], mu1, popt[2], popt[3], mu2, popt[5], popt[6]])
        return ordered, pcov
    except Exception:
        return np.full(7, np.nan), None


# =============================================================================
# Main analysis
# =============================================================================

def run_analysis(config: AnalyzeConfig) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Run full Zeeman analysis from cached profiles.

    Returns:
        df_ring: per-ring fit results
        df_summary: aggregated delta_wn vs B
        magneton_result: dict with slope, mu_B, intercept, etc.
    """
    r_list, I_list, centers = load_cached_profiles(config.cache_dir)
    n_profiles = len(r_list)

    if config.B_fields is None:
        raise ValueError("B_fields must be set in config")
    B_g = np.asarray(config.B_fields, dtype=float) * config.B_scale  # Convert to gauss
    if len(B_g) != n_profiles:
        raise ValueError(
            f"B_fields length ({len(B_g)}) must match number of profiles ({n_profiles})"
        )

    peaks_per_image = [
        detect_ring_centers(
            r, I,
            prominence_frac=config.prominence_frac,
            min_dist=config.min_peak_distance,
            smooth=config.smooth_window,
            r_min=config.inner_radius,
            r_max=config.outer_radius,
        )
        for r, I in zip(r_list, I_list)
    ]
    aligned = align_ring_guesses(peaks_per_image, max_shift=config.max_center_shift_pix)

    rows = []
    for master_ring_idx, matches in aligned.items():
        if len(matches) < config.min_rings_required:
            continue
        if master_ring_idx + 1 >= len(peaks_per_image[0]):
            continue

        for image_idx, ring_r_guess in matches:
            r = r_list[image_idx]
            I = I_list[image_idx]
            y = moving_average(I, config.smooth_window)

            image_peaks = peaks_per_image[image_idx]
            if len(image_peaks) < 2:
                rows.append(_fail_row(image_idx, master_ring_idx, B_g[image_idx], ring_r_guess, "not enough peaks"))
                continue
            j = int(np.argmin(np.abs(image_peaks - ring_r_guess)))
            # Find next peak that is a different INTERFERENCE ORDER (not sigma/pi of same order).
            # Split components are typically 2–10 px apart; adjacent orders are 15+ px.
            j_next = None
            for jn in range(j + 1, len(image_peaks)):
                if image_peaks[jn] - image_peaks[j] >= config.min_order_spacing_r:
                    j_next = jn
                    break
            if j_next is None:
                rows.append(_fail_row(image_idx, master_ring_idx, B_g[image_idx], ring_r_guess, "no adjacent order for FSR"))
                continue

            # Local FSR: use r^2 for Fabry-Perot or r for linear
            if config.fsr_mode == "r2":
                fsr_local = image_peaks[j_next] ** 2 - image_peaks[j] ** 2
            else:
                fsr_local = image_peaks[j_next] - image_peaks[j]

            if fsr_local <= 0:
                rows.append(_fail_row(image_idx, master_ring_idx, B_g[image_idx], ring_r_guess, "bad local FSR"))
                continue

            mask = (r >= ring_r_guess - config.window_half_width) & (r <= ring_r_guess + config.window_half_width)
            xw = r[mask]
            yw = y[mask]
            if len(xw) < 8:
                rows.append(_fail_row(image_idx, master_ring_idx, B_g[image_idx], ring_r_guess, "window too small"))
                continue

            try:
                if config.use_triplet:
                    popt, _ = fit_triplet_local(xw, yw, ring_r_guess)
                    _, mu_left, _, _, mu_center, _, _, mu_right, _, _ = popt

                    sep_left = mu_center - mu_left
                    sep_right = mu_right - mu_center
                    if not (
                        config.min_sigma_pi_sep_pix <= sep_left <= config.max_sigma_pi_sep_pix
                        and config.min_sigma_pi_sep_pix <= sep_right <= config.max_sigma_pi_sep_pix
                    ):
                        raise RuntimeError("triplet separation out of bounds")

                    if config.fsr_mode == "r2":
                        if config.split_mode == "adjacent":
                            delta_split = 0.5 * (
                                (mu_right ** 2 - mu_center ** 2)
                                + (mu_center ** 2 - mu_left ** 2)
                            )
                        else:
                            delta_split = 0.5 * (mu_right ** 2 - mu_left ** 2)
                    else:
                        if config.split_mode == "adjacent":
                            delta_split = 0.5 * (sep_left + sep_right)
                        else:
                            delta_split = 0.5 * (mu_right - mu_left)

                    frac_shift = delta_split / fsr_local
                else:
                    popt, _ = fit_doublet_local(xw, yw, ring_r_guess)
                    _, mu_left, _, _, mu_right, _, _ = popt
                    mu_center = None
                    if config.fsr_mode == "r2":
                        delta_split = 0.5 * (mu_right ** 2 - mu_left ** 2)
                    else:
                        delta_split = 0.5 * (mu_right - mu_left)
                    frac_shift = delta_split / fsr_local

                # Sanity: fractional shift should be < 0.5 (Zeeman split << FSR)
                if frac_shift > 0.5:
                    raise RuntimeError(f"frac_shift={frac_shift:.3g} > 0.5 (FSR ref likely wrong)")

                # Δν~ = frac / (2 * d_etalon) in cm^-1
                delta_wn = frac_shift / (2 * config.etalon_thickness_cm)

                rows.append({
                    "image_idx": image_idx,
                    "ring_idx": master_ring_idx,
                    "B_g": B_g[image_idx],
                    "r_guess": ring_r_guess,
                    "mu_left": float(mu_left),
                    "mu_center": float(mu_center) if config.use_triplet and mu_center is not None else None,
                    "mu_right": float(mu_right),
                    "fsr_local": float(fsr_local),
                    "frac_shift": float(frac_shift),
                    "delta_wn_cm_inv": float(delta_wn),
                    "model": "triplet" if config.use_triplet else "doublet",
                    "success": True,
                    "note": "ok",
                })
            except Exception as e:
                rows.append(_fail_row(
                    image_idx, master_ring_idx, B_g[image_idx], ring_r_guess, str(e),
                    fsr_local=fsr_local,
                ))

    df_ring = pd.DataFrame(rows)

    # Summary: median delta_wn per B
    ok = df_ring[df_ring["success"] & df_ring["delta_wn_cm_inv"].notna()]
    if ok.empty:
        raise RuntimeError("No successful ring fits. Check config or data.")

    grouped = ok.groupby("B_g")["delta_wn_cm_inv"].agg(["median", "mean", "std", "count"]).reset_index()
    grouped.rename(columns={"median": "delta_wn_cm_inv"}, inplace=True)
    q1 = ok.groupby("B_g")["delta_wn_cm_inv"].quantile(0.25)
    q3 = ok.groupby("B_g")["delta_wn_cm_inv"].quantile(0.75)
    iqr = (q3 - q1).rename("iqr")
    grouped = grouped.merge(iqr, on="B_g", how="left")
    grouped["sigma_robust"] = grouped["iqr"] / 1.349

    # Linear fit: Δν~ vs B → Bohr magneton (weighted by per-point uncertainty)
    x = grouped["B_g"].to_numpy(dtype=float)
    y = grouped["delta_wn_cm_inv"].to_numpy(dtype=float)
    sigma_y = grouped["sigma_robust"].to_numpy(dtype=float)
    # Use std as fallback when sigma_robust is NaN (e.g. single ring at that B)
    sigma_y = np.where(np.isfinite(sigma_y) & (sigma_y > 0), sigma_y, grouped["std"].to_numpy(dtype=float))
    sigma_y = np.where(np.isfinite(sigma_y) & (sigma_y > 0), sigma_y, 0.1 * np.abs(y))
    sigma_y = np.maximum(sigma_y, 1e-10)  # avoid zero for weighted fit

    def _linear(B: np.ndarray, slope: float, intercept: float) -> np.ndarray:
        return slope * B + intercept

    popt, pcov = curve_fit(_linear, x, y, p0=[(y[-1] - y[0]) / (x[-1] - x[0]), y[0]], sigma=sigma_y, absolute_sigma=True)
    slope, intercept = float(popt[0]), float(popt[1])
    slope_unc = float(np.sqrt(pcov[0, 0])) if pcov.size else None

    # μ_B from slope: d(ν~)/dB [cm^-1/G] → μ_B = h c * slope * 1e4 * 100 / component_factor
    # Error propagation: σ(μ_B) = (∂μ_B/∂slope) * σ_slope = h*c*1e4*100 * σ_slope / component
    component = 1.0 if config.split_mode == "adjacent" else 2.0
    slope_m_inv_per_T = slope * 1e4 * 100.0  # cm^-1/G -> m^-1/T
    mu_B = h * c * slope_m_inv_per_T / component
    mu_B_unc = None if slope_unc is None else h * c * (slope_unc * 1e4 * 100.0) / component

    slope0 = np.sum(x * y) / np.sum(x * x)  # Through origin
    mu_B_origin = h * c * (slope0 * 1e4 * 100.0) / component

    intercept_unc = float(np.sqrt(pcov[1, 1])) if pcov.size and pcov.shape[0] > 1 else None

    magneton_result = {
        "slope_cm_inv_per_G": float(slope),
        "slope_unc": slope_unc,
        "intercept_cm_inv": float(intercept),
        "intercept_unc": intercept_unc,
        "slope_through_origin": float(slope0),
        "mu_B_J_per_T": float(mu_B),
        "mu_B_unc_J_per_T": mu_B_unc,
        "mu_B_through_origin_J_per_T": float(mu_B_origin),
        "component_factor": component,
        "n_points": int(len(x)),
        "accepted_mu_B": 9.2740100783e-24,
    }

    return df_ring, grouped, magneton_result


def _fail_row(img_idx: int, ring_idx: int, B_g: float, r_guess: float, note: str, fsr_local: Optional[float] = None) -> dict:
    return {
        "image_idx": img_idx,
        "ring_idx": ring_idx,
        "B_g": B_g,
        "r_guess": r_guess,
        "mu_left": None,
        "mu_center": None,
        "mu_right": None,
        "fsr_local": fsr_local,
        "frac_shift": None,
        "delta_wn_cm_inv": None,
        "model": "triplet",
        "success": False,
        "note": note,
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_results(
    df_summary: pd.DataFrame,
    magneton_result: dict,
    out_path: Optional[Path] = None,
) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not available; skipping plot")
        return

    x = df_summary["B_g"].to_numpy()
    y = df_summary["delta_wn_cm_inv"].to_numpy()
    yerr = df_summary["sigma_robust"].to_numpy() if "sigma_robust" in df_summary.columns else None
    if yerr is not None:
        yerr = np.where(np.isfinite(yerr) & (yerr > 0), yerr, 0)

    slope = magneton_result["slope_cm_inv_per_G"]
    intercept = magneton_result["intercept_cm_inv"]
    mu_B = magneton_result["mu_B_J_per_T"]
    mu_B_unc = magneton_result.get("mu_B_unc_J_per_T")

    fig, ax = plt.subplots(figsize=(8, 5))
    if yerr is not None and np.any(yerr > 0):
        ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=4, capthick=1.5, label="Median ringwise shifts")
    else:
        ax.scatter(x, y, s=45, label="Median ringwise shifts")
    xx = np.linspace(x.min() * 0.95, x.max() * 1.02, 200)
    ax.plot(xx, slope * xx + intercept, label=f"Linear fit: y={slope:.4g}B+{intercept:.4g}")
    ax.set_xlabel("Magnetic field (G)")
    ax.set_ylabel(r"$\Delta \tilde{\nu}\ (\mathrm{cm}^{-1})$")
    mu_str = f"$\\mu_B = {mu_B:.3e}\\,\\mathrm{{J/T}}$"
    if mu_B_unc is not None:
        mu_str += f" $\\pm {mu_B_unc:.2e}$"
    ax.set_title(f"Zeeman splitting vs magnetic field\n{mu_str}")
    ax.legend()
    fig.tight_layout()
    if out_path:
        out_path = Path(out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f"Saved plot to {out_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze cached Zeeman profiles: fractional shifts + Bohr magneton."
    )
    parser.add_argument(
        "-c", "--cache-dir",
        type=Path,
        default=Path("."),
        help="Directory with r_arr.npy, I_r_arr.npy",
    )
    parser.add_argument(
        "-B", "--B-fields",
        type=float,
        nargs="+",
        default=[6, 6.7, 7.3, 8.05, 9.3],
        help="Magnetic field values (default: 6 6.7 7.3 8.05 9.3 in kG)",
    )
    parser.add_argument(
        "--B-unit",
        choices=["gauss", "kilogauss"],
        default="kilogauss",
        help="Unit of B-fields (default: kilogauss)",
    )
    parser.add_argument(
        "-d", "--etalon-thickness-cm",
        type=float,
        default=0.811,
        help="Etalon thickness in cm (default: 0.811)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Directory for output CSVs and plot (default: same as cache-dir)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip saving plot",
    )
    parser.add_argument(
        "--min-order-spacing",
        type=float,
        default=25.0,
        help="Min pixel spacing to treat peaks as different orders, not split components (default: 25)",
    )
    args = parser.parse_args()

    out_dir = (args.output_dir or args.cache_dir).resolve()
    B_scale = 1000.0 if args.B_unit == "kilogauss" else 1.0

    config = AnalyzeConfig(
        cache_dir=args.cache_dir,
        min_order_spacing_r=args.min_order_spacing,
        etalon_thickness_cm=args.etalon_thickness_cm,
        B_fields=np.array(args.B_fields),
        B_scale=B_scale,
        out_ring_csv=out_dir / "zeeman_ringwise_results.csv",
        out_summary_csv=out_dir / "zeeman_summary_results.csv",
        out_plot=None if args.no_plot else out_dir / "zeeman_vs_B.png",
    )

    df_ring, df_summary, magneton = run_analysis(config)

    out_dir.mkdir(parents=True, exist_ok=True)
    df_ring.to_csv(config.out_ring_csv, index=False)
    df_summary.to_csv(config.out_summary_csv, index=False)
    print(f"Saved ringwise results to {config.out_ring_csv.resolve()}")
    print(f"Saved summary to {config.out_summary_csv.resolve()}")

    if config.out_plot:
        plot_results(df_summary, magneton, config.out_plot)
    else:
        print("(Use without --no-plot to save the Zeeman vs B plot)")

    print()
    print("Results:")
    su = magneton.get("slope_unc")
    slope_str = f"{magneton['slope_cm_inv_per_G']:.6g}" + (f" ± {su:.4g}" if su is not None else "")
    print(f"  Slope (free)       = {slope_str} cm^-1/G")
    mu = magneton["mu_B_J_per_T"]
    mu_u = magneton.get("mu_B_unc_J_per_T")
    mu_str = f"{mu:.4e}" + (f" ± {mu_u:.2e}" if mu_u is not None else "")
    print(f"  Estimated μ_B     = {mu_str} J/T")
    print(f"  Accepted μ_B      = {magneton['accepted_mu_B']:.4e} J/T")


if __name__ == "__main__":
    main()
