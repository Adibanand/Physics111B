"""
He yellow line (587.6 nm) Zeeman analysis — anomalous Zeeman effect.

Expects paired sigma/pi images per VDC:
  vdc0_He_yellow_587_sigma.png, vdc0_He_yellow_587_pi.png
  vdc5_He_yellow_587_sigma.png, vdc5_He_yellow_587_pi.png
  ... (same for 10, 15, 20, 25, 30)

VDC → B (kG): 0→0.2, 5→1.5, 10→2.97, 15→4.02, 20→5.97, 25→7.35, 30→9.3

Workflow:
  1. Preprocess: python zeeman_yellow.py --preprocess -d <data_dir>
  2. Analyze: python zeeman_yellow.py -o yellow_cache
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.constants import c, h
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Import existing pipeline for preprocessing
from zeeman_analysis_pipeline import load_or_compute_profiles

# VDC → B field (kG)
VDC_TO_B_KG = {
    0: 0.2,
    5: 1.5,
    10: 2.97,
    15: 4.02,
    20: 5.97,
    25: 7.35,
    30: 9.3,
}

# Two images per VDC: sigma and pi
# vdc0_He_yellow_587_sigma.png, vdc0_He_yellow_587_pi.png (no underscore between vdc and number)
VDC_VALUES = [0, 5, 10, 15, 20, 25, 30]
B_FIELDS_KG = [VDC_TO_B_KG[v] for v in VDC_VALUES]


# =============================================================================
# Helpers (from zeeman_analyze)
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


def doublet_model(x, A1, mu1, s1, A2, mu2, s2, C):
    return gaussian(x, A1, mu1, s1, 0) + gaussian(x, A2, mu2, s2, 0) + C


def detect_ring_centers(r, I, prominence_frac=0.02, min_dist=10, smooth=7, r_min=40.0, r_max=None):
    y = moving_average(I, smooth)
    mask = r >= r_min
    if r_max is not None:
        mask &= r <= r_max
    rr, yy = r[mask], y[mask]
    prom = robust_prominence(yy, prominence_frac)
    peak_idx, _ = find_peaks(yy, prominence=prom, distance=min_dist)
    return rr[peak_idx]


def align_ring_guesses(peaks_per_image: List[np.ndarray], max_shift: float = 25.0) -> dict:
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


def fit_doublet_local(x, y, mu_center_guess):
    sep = 4.0
    C0 = np.percentile(y, 10)
    A0 = max(y.max() - C0, 1e-6)
    p0 = [0.7 * A0, mu_center_guess - sep / 2, 2.5, A0, mu_center_guess + sep / 2, 2.5, C0]
    bounds_lower = [0, x.min(), 0.3, 0, x.min(), 0.3, -np.inf]
    bounds_upper = [np.inf, x.max(), 15.0, np.inf, x.max(), 15.0, np.inf]
    try:
        popt, pcov = curve_fit(
            doublet_model, x, y, p0=p0, bounds=(bounds_lower, bounds_upper), maxfev=50000,
        )
        peaks = sorted([(popt[0], popt[1], popt[2]), (popt[3], popt[4], popt[5])], key=lambda t: t[1])
        (_, mu1, _), (_, mu2, _) = peaks
        return np.array([popt[0], mu1, popt[2], popt[3], mu2, popt[5], popt[6]]), pcov
    except Exception:
        return np.full(7, np.nan), None


# =============================================================================
# Load cache
# =============================================================================

def load_cached_profiles(cache_dir: Path) -> Tuple[List[np.ndarray], List[np.ndarray], Optional[np.ndarray]]:
    r_file = cache_dir / "r_arr.npy"
    i_file = cache_dir / "I_r_arr.npy"
    c_file = cache_dir / "centers.npy"
    if not r_file.exists() or not i_file.exists():
        raise FileNotFoundError(
            f"Cache not found in {cache_dir}. Run preprocessing first:\n"
            f"  python zeeman_yellow.py --preprocess -o yellow_cache"
        )
    r_arr = np.load(r_file, allow_pickle=True)
    I_arr = np.load(i_file, allow_pickle=True)
    centers = np.load(c_file, allow_pickle=True) if c_file.exists() else None
    return (
        [np.asarray(r, dtype=float) for r in r_arr],
        [np.asarray(i, dtype=float) for i in I_arr],
        centers,
    )


# =============================================================================
# Main analysis (sigma/pi doublet for anomalous Zeeman)
# =============================================================================

def run_yellow_analysis(
    cache_dir: Path,
    B_fields_kG: List[float],
    etalon_thickness_cm: float = 0.811,
    min_order_spacing_r: float = 25.0,
    window_half_width: int = 18,
    g_effective: float = 1.0,  # For anomalous: use theoretical g for component
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Analyze sigma/pi pairs: profiles are [sigma0, pi0, sigma5, pi5, ..., sigma30, pi30]."""
    r_list, I_list, _ = load_cached_profiles(cache_dir)
    B_g = np.array(B_fields_kG, dtype=float) * 1000.0  # kG → G
    n_vdc = len(B_fields_kG)
    n_profiles = len(r_list)
    if n_profiles != 2 * n_vdc:
        raise ValueError(
            f"Expect 2 profiles per VDC (sigma+pi). Got {n_profiles} profiles for {n_vdc} B fields."
        )

    rows = []
    for vdc_idx in range(n_vdc):
        sigma_idx, pi_idx = 2 * vdc_idx, 2 * vdc_idx + 1
        r_sigma, I_sigma = r_list[sigma_idx], I_list[sigma_idx]
        r_pi, I_pi = r_list[pi_idx], I_list[pi_idx]

        pi_peaks = detect_ring_centers(r_pi, I_pi)
        pi_smooth = moving_average(I_pi, 7)
        sigma_smooth = moving_average(I_sigma, 7)

        if len(pi_peaks) < 2:
            continue

        for ring_idx, r_pi_peak in enumerate(pi_peaks):
            # FSR from adjacent pi rings (different interference orders)
            j_next = None
            for jn in range(ring_idx + 1, len(pi_peaks)):
                if pi_peaks[jn] - pi_peaks[ring_idx] >= min_order_spacing_r:
                    j_next = jn
                    break
            if j_next is None:
                continue

            fsr_r2 = pi_peaks[j_next] ** 2 - pi_peaks[ring_idx] ** 2
            if fsr_r2 <= 0:
                continue

            # Fit sigma doublet (σ⁺, σ⁻) in window around pi peak
            mask = (r_sigma >= r_pi_peak - window_half_width) & (r_sigma <= r_pi_peak + window_half_width)
            xw, yw = r_sigma[mask], sigma_smooth[mask]
            if len(xw) < 8:
                continue

            try:
                popt, _ = fit_doublet_local(xw, yw, r_pi_peak)
                _, mu_left, _, _, mu_right, _, _ = popt
                # Sigma split in r² (σ⁺ − σ⁻) / 2 gives sigma-pi spacing
                delta_split = 0.5 * (mu_right ** 2 - mu_left ** 2)
                frac_shift = delta_split / fsr_r2
                if frac_shift > 0.5 or frac_shift < 0:
                    raise RuntimeError(f"frac_shift={frac_shift:.3g} out of range")
                delta_wn = frac_shift / (2 * etalon_thickness_cm)
                rows.append({
                    "vdc_idx": vdc_idx, "ring_idx": ring_idx, "B_g": B_g[vdc_idx],
                    "r_pi": float(r_pi_peak), "mu_sigma_left": float(mu_left), "mu_sigma_right": float(mu_right),
                    "fsr_local": float(fsr_r2), "frac_shift": float(frac_shift),
                    "delta_wn_cm_inv": float(delta_wn), "success": True, "note": "ok",
                })
            except Exception as e:
                rows.append({
                    "vdc_idx": vdc_idx, "ring_idx": ring_idx, "B_g": B_g[vdc_idx],
                    "r_pi": float(r_pi_peak), "mu_sigma_left": None, "mu_sigma_right": None,
                    "fsr_local": fsr_r2, "frac_shift": None, "delta_wn_cm_inv": None,
                    "success": False, "note": str(e),
                })

    df_ring = pd.DataFrame(rows)
    ok = df_ring[df_ring["success"] & df_ring["delta_wn_cm_inv"].notna()]
    if ok.empty:
        raise RuntimeError("No successful fits. Check config or preprocessing.")

    grouped = ok.groupby("B_g")["delta_wn_cm_inv"].agg(["median", "mean", "std", "count"]).reset_index()
    grouped.rename(columns={"median": "delta_wn_cm_inv"}, inplace=True)
    q1 = ok.groupby("B_g")["delta_wn_cm_inv"].quantile(0.25)
    q3 = ok.groupby("B_g")["delta_wn_cm_inv"].quantile(0.75)
    grouped["sigma_robust"] = (q3 - q1).values / 1.349

    x = grouped["B_g"].to_numpy(float)
    y = grouped["delta_wn_cm_inv"].to_numpy(float)
    sigma_y = np.maximum(grouped["sigma_robust"].to_numpy(float), 1e-10)
    sigma_y = np.where(np.isfinite(sigma_y), sigma_y, 0.1 * np.abs(y))

    def _linear(B, slope, intercept):
        return slope * B + intercept

    popt, pcov = curve_fit(_linear, x, y, p0=[(y[-1] - y[0]) / max(x[-1] - x[0], 1e-9), y[0]], sigma=sigma_y, absolute_sigma=True)
    slope, intercept = float(popt[0]), float(popt[1])
    slope_unc = float(np.sqrt(pcov[0, 0]))
    # μ_B = slope * h * c / g_effective (for anomalous Zeeman)
    slope_m_per_T = slope * 1e4 * 100.0
    mu_B = h * c * slope_m_per_T / g_effective
    mu_B_unc = h * c * (slope_unc * 1e4 * 100.0) / g_effective

    return df_ring, grouped, {
        "slope_cm_inv_per_G": slope, "slope_unc": slope_unc,
        "intercept_cm_inv": intercept, "mu_B_J_per_T": mu_B, "mu_B_unc_J_per_T": mu_B_unc,
        "g_effective": g_effective, "accepted_mu_B": 9.2740100783e-24,
    }




def plot_results(df_summary, magneton, out_path, line_label="He yellow line"):
    if not HAS_MATPLOTLIB:
        return
    x = df_summary["B_g"].to_numpy()
    y = df_summary["delta_wn_cm_inv"].to_numpy()
    yerr = df_summary["sigma_robust"].to_numpy() if "sigma_robust" in df_summary.columns else None
    if yerr is not None:
        yerr = np.where(np.isfinite(yerr) & (yerr > 0), yerr, 0)
    fig, ax = plt.subplots(figsize=(8, 5))
    if yerr is not None and np.any(yerr > 0):
        ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=4, capthick=1.5, label="Median shifts")
    else:
        ax.scatter(x, y, s=45, label="Median shifts")
    xx = np.linspace(x.min() * 0.95, x.max() * 1.02, 200)
    slope, intercept = magneton["slope_cm_inv_per_G"], magneton["intercept_cm_inv"]
    ax.plot(xx, slope * xx + intercept, label=f"Fit: y={slope:.4g}B+{intercept:.4g}")
    ax.set_xlabel("Magnetic field (G)")
    ax.set_ylabel(r"$\Delta \tilde{\nu}\ (\mathrm{cm}^{-1})$")
    mu_B, mu_u = magneton["mu_B_J_per_T"], magneton.get("mu_B_unc_J_per_T")
    mu_str = f"$\\mu_B = {mu_B:.3e}\\,\\mathrm{{J/T}}$"
    if mu_u is not None:
        mu_str += f" $\\pm {mu_u:.2e}$"
    ax.set_title(f"{line_label} (anomalous Zeeman) — sigma/pi\n{mu_str}")
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="He yellow line (587.6 nm) Zeeman analysis — anomalous Zeeman")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess images and save cache (run once)")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("yellow_cache"),
                        help="Cache/output directory (default: yellow_cache)")
    parser.add_argument("-d", "--data-dir", type=Path, default=Path("."),
                        help="Directory with vdc0_He_yellow_587_sigma.png, vdc0_He_yellow_587_pi.png, etc.")
    parser.add_argument("images", nargs="*", type=Path,
                        help="Explicit paths: sigma0, pi0, sigma5, pi5, ... (14 files). Overrides auto-discovery.")
    parser.add_argument("--etalon-cm", type=float, default=0.811, help="Etalon thickness (cm)")
    parser.add_argument("-g", "--g-effective", type=float, default=1.0,
                        help="Effective Landé g for μ_B extraction (anomalous Zeeman)")
    parser.add_argument("-B", "--B-fields", type=float, nargs="+", default=B_FIELDS_KG,
                        help="B fields in kG, in image order (default: 0.2,1.5,2.97,4.02,5.97,7.35,9.3)")
    parser.add_argument("--no-plot", action="store_true", help="Skip saving plot")
    parser.add_argument("--precise", action="store_true", help="Slower center search (step=1); default uses step=2 (~2-3x faster)")
    args = parser.parse_args()

    cache_dir = args.output_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Build image list: (sigma, pi) pairs per VDC
    # Expected: vdc0_He_yellow_587_sigma.png, vdc0_He_yellow_587_pi.png, ...
    if args.images:
        image_paths = [Path(p).resolve() for p in args.images]
        n_expected = 2 * len(VDC_VALUES)  # sigma and pi per VDC
        if len(image_paths) != n_expected:
            print(f"Warning: {len(image_paths)} images vs {n_expected} expected (sigma+pi per VDC)")
    else:
        image_paths = []
        for vdc in VDC_VALUES:
            base = f"vdc{vdc}_He_yellow_587"
            sigma_p = args.data_dir / f"{base}_sigma.png"
            pi_p = args.data_dir / f"{base}_pi.png"
            if not sigma_p.exists() or not pi_p.exists():
                raise FileNotFoundError(
                    f"For VDC={vdc}, expected {base}_sigma.png and {base}_pi.png in {args.data_dir}"
                )
            image_paths.extend([sigma_p, pi_p])  # sigma first, then pi

    if args.preprocess:
        print("Preprocessing He yellow images...")
        load_or_compute_profiles(
            image_paths=[str(p) for p in image_paths],
            cache_dir=cache_dir,
            force_recompute=True,
            nbins=2000,
            verbose=True,
            center_step=1 if args.precise else 2,
        )
        print(f"Cached to {cache_dir}")
        print("Run without --preprocess to analyze.")
        return

    df_ring, df_summary, magneton = run_yellow_analysis(
        cache_dir=cache_dir,
        B_fields_kG=args.B_fields,
        etalon_thickness_cm=args.etalon_cm,
        g_effective=args.g_effective,
    )

    df_ring.to_csv(cache_dir / "zeeman_yellow_ringwise.csv", index=False)
    df_summary.to_csv(cache_dir / "zeeman_yellow_summary.csv", index=False)
    print(f"Saved to {cache_dir}")

    if not args.no_plot:
        plot_results(df_summary, magneton, cache_dir / "zeeman_yellow_vs_B.png")

    mu, mu_u = magneton["mu_B_J_per_T"], magneton.get("mu_B_unc_J_per_T")
    mu_str = f"{mu:.4e}" + (f" ± {mu_u:.2e}" if mu_u else "")
    print()
    print("He yellow line (anomalous Zeeman, sigma/pi):")
    print(f"  Estimated μ_B = {mu_str} J/T (g_eff={args.g_effective})")
    print(f"  Accepted μ_B  = {magneton['accepted_mu_B']:.4e} J/T")


if __name__ == "__main__":
    main()
