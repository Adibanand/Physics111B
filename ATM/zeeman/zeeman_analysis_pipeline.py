from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from PIL import Image
from scipy.constants import c, h
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import binned_statistic, binned_statistic_2d


# -----------------------------
# Preprocessing and caching
# -----------------------------

def remove_background_poly2(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit and subtract a 2D quadratic background."""
    img = img.astype(float)
    ny, nx = img.shape
    y, x = np.indices(img.shape)

    X = np.column_stack(
        [
            np.ones(img.size),
            x.ravel(),
            y.ravel(),
            x.ravel() ** 2,
            y.ravel() ** 2,
            x.ravel() * y.ravel(),
        ]
    )

    coeffs, *_ = np.linalg.lstsq(X, img.ravel(), rcond=None)
    background = (X @ coeffs).reshape(img.shape)
    corrected = img - background
    corrected -= corrected.min()
    return corrected, background


def centroid_from_core(
    img: np.ndarray,
    blur_sigma: float = 2.0,
    top_frac: float = 0.05,
) -> tuple[float, float]:
    I = img.astype(float)
    I_s = gaussian_filter(I, blur_sigma)
    thr = np.quantile(I_s, 1.0 - top_frac)
    mask = I_s >= thr
    y, x = np.indices(I.shape)
    w = I_s * mask
    if w.sum() <= 0:
        raise ValueError("Core mask is empty; try changing blur_sigma or top_frac.")
    xc = float((x * w).sum() / w.sum())
    yc = float((y * w).sum() / w.sum())
    return xc, yc


def refine_center_by_ring_symmetry(
    img: np.ndarray,
    x0: float,
    y0: float,
    search_radius: int = 30,
    step: int = 1,
    blur_sigma: float = 1.5,
    r_min: float = 20,
    r_max: float | None = None,
    nr: int = 200,
    ntheta: int = 180,
) -> tuple[float, float]:
    """Choose the center that minimizes angular variation at fixed radius."""
    I = gaussian_filter(img.astype(float), blur_sigma)
    ny, nx = I.shape
    if r_max is None:
        r_max = min(nx, ny) * 0.45

    yy, xx = np.indices(I.shape)
    best_score = np.inf
    best_center: tuple[float, float] | None = None

    xs = np.arange(x0 - search_radius, x0 + search_radius + 1, step)
    ys = np.arange(y0 - search_radius, y0 + search_radius + 1, step)

    r_edges = np.linspace(r_min, r_max, nr + 1)
    th_edges = np.linspace(-np.pi, np.pi, ntheta + 1)

    for xc in xs:
        for yc in ys:
            r = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2)
            th = np.arctan2(yy - yc, xx - xc)
            m = (r >= r_min) & (r <= r_max)
            if m.sum() < 1000:
                continue

            stat, _, _, _ = binned_statistic_2d(
                r[m].ravel(),
                th[m].ravel(),
                I[m].ravel(),
                statistic="mean",
                bins=[r_edges, th_edges],
            )
            ang_std = np.nanstd(stat, axis=1)
            score = np.nanmean(ang_std)
            if np.isfinite(score) and score < best_score:
                best_score = score
                best_center = (float(xc), float(yc))

    if best_center is None:
        raise ValueError("Center refinement failed.")
    return best_center


def radial_profile_centered(
    img: np.ndarray,
    nbins: int = 2000,
    center: tuple[float, float] | None = None,
    search_radius: int = 40,
    r_min: float = 30,
    center_step: int = 1,  # step for center grid search; 2–3 = faster
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    img = img.astype(float)
    if center is None:
        ny, nx = img.shape
        y0, x0 = ny // 2, nx // 2
        xc, yc = refine_center_by_ring_symmetry(
            img,
            x0,
            y0,
            search_radius=search_radius,
            step=center_step,
            r_min=r_min,
            r_max=None,
        )
    else:
        xc, yc = center

    ny, nx = img.shape
    y, x = np.indices((ny, nx))
    r = np.sqrt((x - xc) ** 2 + (y - yc) ** 2).ravel()
    I = img.ravel()
    radial_mean, edges, _ = binned_statistic(r, I, statistic="mean", bins=nbins)
    r_centers = 0.5 * (edges[1:] + edges[:-1])
    return r_centers, radial_mean, (float(xc), float(yc))


def load_or_compute_profiles(
    image_paths: Iterable[str | Path],
    cache_dir: str | Path = ".",
    force_recompute: bool = False,
    nbins: int = 2000,
    verbose: bool = False,
    center_step: int = 1,  # step=2 or 3 for faster (less precise) center search
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load saved I_r_arr/r_arr/centers if present. Otherwise compute and save."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    I_path = cache_dir / "I_r_arr.npy"
    r_path = cache_dir / "r_arr.npy"
    c_path = cache_dir / "centers.npy"

    if I_path.exists() and r_path.exists() and c_path.exists() and not force_recompute:
        return (
            np.load(I_path, allow_pickle=True),
            np.load(r_path, allow_pickle=True),
            np.load(c_path, allow_pickle=True),
        )

    image_paths = list(image_paths)
    I_r_arr = []
    r_arr = []
    centers = []

    for i, image_path in enumerate(image_paths):
        if verbose:
            name = Path(image_path).name
            print(f"  [{i+1}/{len(image_paths)}] {name} ...")
        img = np.array(Image.open(image_path).convert("L")).astype(float)
        img_corr, _ = remove_background_poly2(img)
        r, I_r, center = radial_profile_centered(
            img_corr, nbins=nbins, center_step=center_step
        )
        r_arr.append(r)
        I_r_arr.append(I_r)
        centers.append(center)

    I_r_arr = np.array(I_r_arr, dtype=object)
    r_arr = np.array(r_arr, dtype=object)
    centers = np.array(centers, dtype=float)
    np.save(I_path, I_r_arr)
    np.save(r_path, r_arr)
    np.save(c_path, centers)
    return I_r_arr, r_arr, centers


# -----------------------------
# Peak detection and fitting
# -----------------------------

def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y.copy()
    kernel = np.ones(window) / float(window)
    return np.convolve(y, kernel, mode="same")


@dataclass
class PeakFindingConfig:
    smooth_window: int = 7
    prominence: float = 5.0
    distance: int = 20
    r_min: float = 20.0
    r_max: float | None = None


def find_reference_peaks(
    r: np.ndarray,
    I: np.ndarray,
    config: PeakFindingConfig,
) -> pd.DataFrame:
    mask = r >= config.r_min
    if config.r_max is not None:
        mask &= r <= config.r_max

    r_use = r[mask]
    I_use = _moving_average(np.asarray(I[mask], dtype=float), config.smooth_window)
    peak_idx, props = find_peaks(
        I_use,
        prominence=config.prominence,
        distance=config.distance,
    )
    out = pd.DataFrame(
        {
            "peak_index_local": peak_idx,
            "r_peak": r_use[peak_idx],
            "prominence": props.get("prominences", np.full_like(peak_idx, np.nan, dtype=float)),
        }
    )
    out["fsr_pix"] = np.nan
    if len(out) >= 2:
        vals = out["r_peak"].to_numpy()
        fsr = np.empty_like(vals)
        for i in range(len(vals)):
            if i == 0:
                fsr[i] = vals[i + 1] - vals[i]
            elif i == len(vals) - 1:
                fsr[i] = vals[i] - vals[i - 1]
            else:
                fsr[i] = 0.5 * ((vals[i + 1] - vals[i]) + (vals[i] - vals[i - 1]))
        out["fsr_pix"] = fsr
    return out.reset_index(drop=True)


@dataclass
class FitConfig:
    model: Literal["triplet", "doublet"] = "triplet"
    window_fraction_of_fsr: float = 0.45
    sigma_fraction_of_fsr: float = 0.10
    max_component_shift_fraction: float = 0.35
    min_amplitude: float = 0.0


def _gaussian(x: np.ndarray, amp: float, mu: float, sigma: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _triplet_model(
    x: np.ndarray,
    b0: float,
    b1: float,
    a1: float,
    mu1: float,
    a2: float,
    mu2: float,
    a3: float,
    mu3: float,
    sigma: float,
) -> np.ndarray:
    return (
        b0
        + b1 * (x - x.mean())
        + _gaussian(x, a1, mu1, sigma)
        + _gaussian(x, a2, mu2, sigma)
        + _gaussian(x, a3, mu3, sigma)
    )


def _doublet_model(
    x: np.ndarray,
    b0: float,
    b1: float,
    a1: float,
    mu1: float,
    a2: float,
    mu2: float,
    sigma: float,
) -> np.ndarray:
    return b0 + b1 * (x - x.mean()) + _gaussian(x, a1, mu1, sigma) + _gaussian(x, a2, mu2, sigma)


@dataclass
class RingFitResult:
    image_index: int
    peak_number: int
    B_field: float
    r_ref: float
    fsr_pix: float
    model: str
    success: bool
    mu_left: float | None = None
    mu_center: float | None = None
    mu_right: float | None = None
    sigma_pix: float | None = None
    sigma_minus_pi_pix: float | None = None
    sigma_sigma_halfsep_pix: float | None = None
    sigma_sigma_sep_pix: float | None = None
    fractional_shift: float | None = None
    window_left: float | None = None
    window_right: float | None = None
    message: str = ""


def fit_ring_window(
    r: np.ndarray,
    I: np.ndarray,
    r_ref: float,
    fsr_pix: float,
    fit_config: FitConfig,
) -> RingFitResult:
    half_window = fit_config.window_fraction_of_fsr * fsr_pix
    mask = (r >= r_ref - half_window) & (r <= r_ref + half_window)
    x = np.asarray(r[mask], dtype=float)
    y = np.asarray(I[mask], dtype=float)

    if x.size < 15:
        return RingFitResult(
            image_index=-1,
            peak_number=-1,
            B_field=np.nan,
            r_ref=r_ref,
            fsr_pix=fsr_pix,
            model=fit_config.model,
            success=False,
            window_left=r_ref - half_window,
            window_right=r_ref + half_window,
            message="Too few samples in fit window.",
        )

    baseline = np.percentile(y, 15)
    ymax = max(float(y.max() - baseline), 1.0)
    shift0 = 0.08 * fsr_pix
    sigma0 = max(fit_config.sigma_fraction_of_fsr * fsr_pix, 0.5 * np.median(np.diff(x)))
    max_shift = fit_config.max_component_shift_fraction * fsr_pix

    try:
        if fit_config.model == "triplet":
            p0 = [
                baseline,
                0.0,
                0.6 * ymax,
                r_ref - shift0,
                1.0 * ymax,
                r_ref,
                0.6 * ymax,
                r_ref + shift0,
                sigma0,
            ]
            lower = [
                y.min() - 2 * abs(y.min()),
                -np.inf,
                fit_config.min_amplitude,
                r_ref - max_shift,
                fit_config.min_amplitude,
                r_ref - 0.10 * fsr_pix,
                fit_config.min_amplitude,
                r_ref,
                0.2 * sigma0,
            ]
            upper = [
                y.max() + abs(y.max()),
                np.inf,
                5 * ymax,
                r_ref,
                5 * ymax,
                r_ref + 0.10 * fsr_pix,
                5 * ymax,
                r_ref + max_shift,
                4.0 * sigma0,
            ]
            popt, _ = curve_fit(_triplet_model, x, y, p0=p0, bounds=(lower, upper), maxfev=40000)
            b0, b1, a1, mu1, a2, mu2, a3, mu3, sigma = popt
            sigma_minus_pi = abs(mu2 - mu1)
            sigma_plus_pi = abs(mu3 - mu2)
            sigma_pi = 0.5 * (sigma_minus_pi + sigma_plus_pi)
            sigma_sigma = abs(mu3 - mu1)
            return RingFitResult(
                image_index=-1,
                peak_number=-1,
                B_field=np.nan,
                r_ref=r_ref,
                fsr_pix=fsr_pix,
                model=fit_config.model,
                success=True,
                mu_left=float(mu1),
                mu_center=float(mu2),
                mu_right=float(mu3),
                sigma_pix=float(sigma),
                sigma_minus_pi_pix=float(sigma_pi),
                sigma_sigma_halfsep_pix=float(0.5 * sigma_sigma),
                sigma_sigma_sep_pix=float(sigma_sigma),
                fractional_shift=float(sigma_pi / fsr_pix),
                window_left=float(r_ref - half_window),
                window_right=float(r_ref + half_window),
                message="",
            )

        p0 = [baseline, 0.0, ymax, r_ref - shift0, ymax, r_ref + shift0, sigma0]
        lower = [y.min() - 2 * abs(y.min()), -np.inf, fit_config.min_amplitude, r_ref - max_shift, fit_config.min_amplitude, r_ref, 0.2 * sigma0]
        upper = [y.max() + abs(y.max()), np.inf, 5 * ymax, r_ref, 5 * ymax, r_ref + max_shift, 4.0 * sigma0]
        popt, _ = curve_fit(_doublet_model, x, y, p0=p0, bounds=(lower, upper), maxfev=40000)
        b0, b1, a1, mu1, a2, mu2, sigma = popt
        sigma_sigma = abs(mu2 - mu1)
        return RingFitResult(
            image_index=-1,
            peak_number=-1,
            B_field=np.nan,
            r_ref=r_ref,
            fsr_pix=fsr_pix,
            model=fit_config.model,
            success=True,
            mu_left=float(mu1),
            mu_center=None,
            mu_right=float(mu2),
            sigma_pix=float(sigma),
            sigma_minus_pi_pix=None,
            sigma_sigma_halfsep_pix=float(0.5 * sigma_sigma),
            sigma_sigma_sep_pix=float(sigma_sigma),
            fractional_shift=float(0.5 * sigma_sigma / fsr_pix),
            window_left=float(r_ref - half_window),
            window_right=float(r_ref + half_window),
            message="",
        )
    except Exception as exc:  # noqa: BLE001
        return RingFitResult(
            image_index=-1,
            peak_number=-1,
            B_field=np.nan,
            r_ref=r_ref,
            fsr_pix=fsr_pix,
            model=fit_config.model,
            success=False,
            window_left=float(r_ref - half_window),
            window_right=float(r_ref + half_window),
            message=str(exc),
        )


def analyze_zeeman_from_profiles(
    r_arr: np.ndarray,
    I_r_arr: np.ndarray,
    B_fields: Iterable[float],
    reference_index: int = 0,
    peak_config: PeakFindingConfig | None = None,
    fit_config: FitConfig | None = None,
    peak_numbers: Iterable[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-ring fit results and reference-peak metadata.

    Assumes each profile corresponds to one magnetic field value and the reference
    profile sets the unsplit ring positions / local FSR in pixels.
    """
    peak_config = peak_config or PeakFindingConfig()
    fit_config = fit_config or FitConfig()
    B_fields = np.asarray(list(B_fields), dtype=float)
    if len(B_fields) != len(r_arr):
        raise ValueError("B_fields must match the number of radial profiles.")

    ref_peaks = find_reference_peaks(r_arr[reference_index], I_r_arr[reference_index], peak_config)
    if peak_numbers is None:
        peak_numbers = ref_peaks.index.to_list()
    peak_numbers = list(peak_numbers)

    rows: list[dict] = []
    for image_index, (r, I_r, B) in enumerate(zip(r_arr, I_r_arr, B_fields)):
        for peak_number in peak_numbers:
            peak = ref_peaks.iloc[peak_number]
            fit = fit_ring_window(np.asarray(r, dtype=float), np.asarray(I_r, dtype=float), float(peak.r_peak), float(peak.fsr_pix), fit_config)
            row = fit.__dict__.copy()
            row["image_index"] = image_index
            row["peak_number"] = int(peak_number)
            row["B_field"] = float(B)
            rows.append(row)

    return pd.DataFrame(rows), ref_peaks


# -----------------------------
# Physics conversion helpers
# -----------------------------

def fractional_shift_to_wavenumber_cm(
    fractional_shift: np.ndarray | pd.Series,
    etalon_thickness_cm: float,
) -> np.ndarray:
    fsr_cm_inv = 1.0 / (2.0 * etalon_thickness_cm)
    return np.asarray(fractional_shift, dtype=float) * fsr_cm_inv


def estimate_bohr_magneton_from_slope(
    slope_cm_inv_per_gauss: float,
    component_factor: float = 1.0,
) -> float:
    """Convert slope d(wn_cm^-1)/dB_G to mu_B in J/T.

    component_factor = 1 for sigma-pi spacing in the normal Zeeman effect.
    component_factor = 2 for sigma+ minus sigma- full separation.
    """
    slope_cm_inv_per_T = slope_cm_inv_per_gauss * 1.0e4
    slope_m_inv_per_T = slope_cm_inv_per_T * 100.0
    return h * c * slope_m_inv_per_T / component_factor


@dataclass
class MagnetonFitResult:
    slope_cm_inv_per_gauss: float
    intercept_cm_inv: float
    mu_B_J_per_T: float
    mu_B_unc_J_per_T: float | None
    used_component_factor: float
    n_points: int


def fit_bohr_magneton(
    fit_df: pd.DataFrame,
    etalon_thickness_cm: float,
    component_factor: float = 1.0,
    aggregate: Literal["median", "mean"] = "median",
) -> tuple[pd.DataFrame, MagnetonFitResult]:
    good = fit_df.loc[fit_df["success"] & fit_df["fractional_shift"].notna()].copy()
    if good.empty:
        raise ValueError("No successful ring fits found.")

    good["delta_wn_cm_inv"] = fractional_shift_to_wavenumber_cm(
        good["fractional_shift"], etalon_thickness_cm
    )

    grouped = getattr(good.groupby("B_field")["delta_wn_cm_inv"], aggregate)().reset_index()
    x = grouped["B_field"].to_numpy(dtype=float)*1e3 #convert to gauss
    y = grouped["delta_wn_cm_inv"].to_numpy(dtype=float)

    if len(x) < 2:
        raise ValueError("Need at least two magnetic-field points for a fit.")

    coeffs, cov = np.polyfit(x, y, 1, cov=True)
    slope, intercept = coeffs
    slope_unc = float(np.sqrt(cov[0, 0])) if cov.size else None

    mu_B = estimate_bohr_magneton_from_slope(slope, component_factor=component_factor)
    mu_B_unc = None if slope_unc is None else estimate_bohr_magneton_from_slope(slope_unc, component_factor=component_factor)

    result = MagnetonFitResult(
        slope_cm_inv_per_gauss=float(slope),
        intercept_cm_inv=float(intercept),
        mu_B_J_per_T=float(mu_B),
        mu_B_unc_J_per_T=None if mu_B_unc is None else float(mu_B_unc),
        used_component_factor=float(component_factor),
        n_points=int(len(x)),
    )
    return grouped, result


def add_expected_bohr_magneton_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mu_B_reference_J_per_T"] = 9.2740100783e-24
    return df
