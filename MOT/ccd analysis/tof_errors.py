
"""
tof_errors.py

Fast, mostly-analytic uncertainty propagation utilities for the Physics111B
MOT TOF + atom-number analysis notebooks.

Designed to "drop in" with minimal changes:
- Your notebook already builds a per-file dict `r` (from analyze_file)
  and then a DataFrame `df = pd.DataFrame(rows)`.

This module helps you:
- Extract 1σ uncertainties from the Gaussian fit dictionaries (x0_Sigma, y0_Sigma, w0x_Sigma, w0y_Sigma, etc.)
- Propagate to:
  * x, y (meters)
  * vx, vy (two-point or weighted linear fit)
  * gravity g (weighted quadratic fit)
  * temperature (Poisson-limited shot-noise model, optional)
  * atom number N (Route B) with shot + systematic fractional errors

Notes / assumptions:
- "Statistical" errors: position/width uncertainties from your fit OR a fallback estimate.
- "Systematic" errors (QE, transmission, collection efficiency, exposure time, scattering rate) are handled as
  fractional uncertainties you can tune.

All functions are vectorized and avoid Monte Carlo by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.constants as c

ArrayLike = Union[np.ndarray, float]


# -------------------------
# Basic weighted regressions
# -------------------------

def weighted_line_fit(t: np.ndarray, y: np.ndarray, sy: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Weighted least squares fit: y = a + b t

    Returns: (a, b, sa, sb)
    where sa/sb are 1σ uncertainties from covariance (assuming sy are 1σ).
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    sy = np.asarray(sy, float)

    w = 1.0 / np.maximum(sy, 1e-30) ** 2

    S   = np.sum(w)
    St  = np.sum(w * t)
    Stt = np.sum(w * t * t)
    Sy  = np.sum(w * y)
    Sty = np.sum(w * t * y)

    Delta = S * Stt - St**2
    if Delta <= 0:
        raise ValueError("Singular design matrix in weighted_line_fit; check t values and uncertainties.")

    a = (Stt * Sy - St * Sty) / Delta
    b = (S * Sty - St * Sy) / Delta

    sa = np.sqrt(Stt / Delta)
    sb = np.sqrt(S / Delta)
    return float(a), float(b), float(sa), float(sb)


def weighted_quad_fit_for_g(t: np.ndarray, y: np.ndarray, sy: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """
    Weighted least squares fit: y = y0 + vy t + c2 t^2
    Interpreting c2 = -g/2  =>  g = -2 c2

    Returns: (y0, vy, g, sy0, svy, sg)
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    sy = np.asarray(sy, float)
    w = 1.0 / np.maximum(sy, 1e-30) ** 2

    X = np.vstack([np.ones_like(t), t, t**2]).T  # (N,3)
    XT_W = X.T * w  # (3,N)

    M = XT_W @ X     # (3,3)
    rhs = XT_W @ y   # (3,)

    beta = np.linalg.solve(M, rhs)
    Cov = np.linalg.inv(M)

    y0, vy, c2 = beta
    sy0, svy, sc2 = np.sqrt(np.maximum(np.diag(Cov), 0))

    g = -2.0 * c2
    sg =  2.0 * sc2
    return float(y0), float(vy), float(g), float(sy0), float(svy), float(sg)


# -------------------------
# Primitive propagation utils
# -------------------------

def pos_errors_m(x0_px: ArrayLike, y0_px: ArrayLike,
                 sx0_px: ArrayLike, sy0_px: ArrayLike,
                 px_to_mx: ArrayLike, px_to_my: ArrayLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert positions and 1σ uncertainties from pixels to meters using per-axis scale factors."""
    x0_px = np.asarray(x0_px, float)
    y0_px = np.asarray(y0_px, float)
    sx0_px = np.asarray(sx0_px, float)
    sy0_px = np.asarray(sy0_px, float)
    px_to_mx = np.asarray(px_to_mx, float)
    px_to_my = np.asarray(px_to_my, float)

    x_m  = x0_px * px_to_mx
    y_m  = y0_px * px_to_my
    sx_m = sx0_px * px_to_mx
    sy_m = sy0_px * px_to_my
    return x_m, y_m, sx_m, sy_m


def velocity_two_point(xi: ArrayLike, xf: ArrayLike, sxi: ArrayLike, sxf: ArrayLike, t: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """v = (xf - xi)/t, with σ_v = sqrt(σ_xf^2 + σ_xi^2)/t"""
    xi = np.asarray(xi, float); xf = np.asarray(xf, float)
    sxi = np.asarray(sxi, float); sxf = np.asarray(sxf, float)
    t = np.asarray(t, float)

    v = (xf - xi) / t
    sv = np.sqrt(sxf**2 + sxi**2) / np.maximum(t, 1e-30)
    return v, sv


# -------------------------
# Temperature (optional): Poisson-limited width error model
# -------------------------

def temperature_with_error(wf_m: ArrayLike, wi_m: ArrayLike,
                           S_f: ArrayLike, S_i: ArrayLike,
                           t_s: ArrayLike,
                           mass: float = 85 * c.atomic_mass) -> Tuple[np.ndarray, np.ndarray]:
    """
    Consistent with your earlier per-file estimator:
        T = (m/(3 k t^2)) * (wf^2 - wi^2)

    Shot-noise model:
      sigma_w ≈ w / sqrt(2 N_gamma)
    We take N_gamma ∝ S (background-subtracted ROI sum). This gives correct scaling and is fast.

    Returns: (T [K], sigma_T [K])
    """
    wf_m = np.asarray(wf_m, float)
    wi_m = np.asarray(wi_m, float)
    S_f  = np.asarray(S_f, float)
    S_i  = np.asarray(S_i, float)
    t_s  = np.asarray(t_s, float)

    A = mass / (3 * c.k * np.maximum(t_s, 1e-30)**2)

    S_f = np.maximum(S_f, 1.0)
    S_i = np.maximum(S_i, 1.0)

    sigma_wf = wf_m / np.sqrt(2 * S_f)
    sigma_wi = wi_m / np.sqrt(2 * S_i)

    T = A * (wf_m**2 - wi_m**2)
    sigma_T = A * np.sqrt((2*wf_m*sigma_wf)**2 + (2*wi_m*sigma_wi)**2)
    return T, sigma_T


# -------------------------
# Atom number: Route B uncertainties
# -------------------------

@dataclass
class AtomNumberUncertainty:
    """Fractional systematic uncertainties for Route B atom number (tune these)."""
    rel_eta: float = 0.20     # collection efficiency geometry
    rel_R: float = 0.15       # scattering rate uncertainty
    rel_texp: float = 0.10    # exposure time uncertainty
    rel_QE: float = 0.20      # camera QE uncertainty
    rel_Topt: float = 0.20    # optics transmission uncertainty


def atom_number_error(N: ArrayLike, S_ADU: ArrayLike,
                      syst: AtomNumberUncertainty = AtomNumberUncertainty(),
                      include_shot: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given atom number estimate N and a signal proxy S_ADU, compute:
      σ_N  and  fractional σ_N/N.

    Shot noise: rel_shot ≈ 1/sqrt(S_ADU) (proxy for 1/sqrt(N_gamma))
    Total fractional: sqrt(rel_shot^2 + rel_eta^2 + rel_R^2 + rel_texp^2 + rel_QE^2 + rel_Topt^2)
    """
    N = np.asarray(N, float)
    S_ADU = np.asarray(S_ADU, float)

    rel_shot = (1.0 / np.sqrt(np.maximum(S_ADU, 1.0))) if include_shot else 0.0

    rel_tot = np.sqrt(rel_shot**2 +
                      syst.rel_eta**2 +
                      syst.rel_R**2 +
                      syst.rel_texp**2 +
                      syst.rel_QE**2 +
                      syst.rel_Topt**2)
    sN = N * rel_tot
    return sN, rel_tot


# -------------------------
# Helpers: extract sigmas from your fit_dicts
# -------------------------

def _safe_get_fit_sigma(fit_dict: Optional[Dict[str, Any]], key: str) -> float:
    """Return fit_dict[key] as float if present and finite, else np.nan."""
    if not isinstance(fit_dict, dict):
        return np.nan
    v = fit_dict.get(key, np.nan)
    try:
        v = float(v)
    except Exception:
        return np.nan
    return v if np.isfinite(v) else np.nan


def add_fit_sigmas_to_row(row: Dict[str, Any],
                          fit_before_key: str = "fit_before",
                          fit_after_key: str = "fit_after") -> Dict[str, Any]:
    """
    Add common 1σ fit uncertainties (in pixels) onto a per-file dict row:

      sx0_before_px, sy0_before_px, sx0_after_px, sy0_after_px,
      sw0x_before_px, sw0y_before_px, sw0x_after_px, sw0y_after_px
    """
    fb = row.get(fit_before_key, None)
    fa = row.get(fit_after_key, None)

    row["sx0_before_px"] = _safe_get_fit_sigma(fb, "x0_Sigma")
    row["sy0_before_px"] = _safe_get_fit_sigma(fb, "y0_Sigma")
    row["sx0_after_px"]  = _safe_get_fit_sigma(fa, "x0_Sigma")
    row["sy0_after_px"]  = _safe_get_fit_sigma(fa, "y0_Sigma")

    row["sw0x_before_px"] = _safe_get_fit_sigma(fb, "w0x_Sigma")
    row["sw0y_before_px"] = _safe_get_fit_sigma(fb, "w0y_Sigma")
    row["sw0x_after_px"]  = _safe_get_fit_sigma(fa, "w0x_Sigma")
    row["sw0y_after_px"]  = _safe_get_fit_sigma(fa, "w0y_Sigma")
    return row


def add_default_position_sigmas_if_missing(df: pd.DataFrame, fallback_px: float = 0.5) -> pd.DataFrame:
    """
    If x0_Sigma/y0_Sigma weren't available, fill missing values with a conservative
    pixel-level uncertainty (default 0.5 px). Returns a copy.
    """
    out = df.copy()
    for col in ["sx0_after_px", "sy0_after_px", "sx0_before_px", "sy0_before_px"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
        out[col] = out[col].fillna(fallback_px)
        out.loc[~np.isfinite(out[col]), col] = fallback_px
    return out


# -------------------------
# High-level dataset fits: vx, vy, g with uncertainties
# -------------------------

@dataclass
class TOFPhysicsResults:
    vx: float
    svx: float
    vy: float
    svy: float
    g: float
    sg: float


def tof_physics_with_errors(df: pd.DataFrame,
                            *,
                            use_after: bool = True,
                            px_to_mx: Optional[ArrayLike] = None,
                            px_to_my: Optional[ArrayLike] = None,
                            pos_sigma_floor_m: float = 0.0) -> TOFPhysicsResults:
    """
    Compute vx, vy, g using weighted fits with per-point position uncertainties.

    Requires df contains:
      - t_s
      - x0_after_px/y0_after_px and sx0_after_px/sy0_after_px   (or *_before_*)
      - either per-row px_to_mx/px_to_my columns OR pass px_to_mx/px_to_my scalars.

    pos_sigma_floor_m: adds in quadrature to each σ (helps if fit sigmas are unrealistically tiny).
    """
    if len(df) < 3:
        raise ValueError("Need at least 3 TOF points to fit vx/vy/g with uncertainties.")

    d = df.copy()
    t = d["t_s"].to_numpy(dtype=float)

    tag = "after" if use_after else "before"
    x0 = d[f"x0_{tag}_px"].to_numpy(dtype=float)
    y0 = d[f"y0_{tag}_px"].to_numpy(dtype=float)
    sx0 = pd.to_numeric(d.get(f"sx0_{tag}_px", np.nan), errors="coerce").to_numpy(dtype=float)
    sy0 = pd.to_numeric(d.get(f"sy0_{tag}_px", np.nan), errors="coerce").to_numpy(dtype=float)

    # Determine pixel->meter scaling
    if px_to_mx is None:
        if "px_to_mx" in d.columns:
            px_to_mx_arr = d["px_to_mx"].to_numpy(dtype=float)
        else:
            raise ValueError("px_to_mx not provided and not present in df.")
    else:
        px_to_mx_arr = np.asarray(px_to_mx, float) + np.zeros_like(x0)

    if px_to_my is None:
        if "px_to_my" in d.columns:
            px_to_my_arr = d["px_to_my"].to_numpy(dtype=float)
        else:
            raise ValueError("px_to_my not provided and not present in df.")
    else:
        px_to_my_arr = np.asarray(px_to_my, float) + np.zeros_like(y0)

    x_m, y_m, sx_m, sy_m = pos_errors_m(x0, y0, sx0, sy0, px_to_mx_arr, px_to_my_arr)

    if pos_sigma_floor_m > 0:
        sx_m = np.sqrt(sx_m**2 + pos_sigma_floor_m**2)
        sy_m = np.sqrt(sy_m**2 + pos_sigma_floor_m**2)

    _, vx, _, svx = weighted_line_fit(t, x_m, sx_m)
    _, vy, g, _, svy, sg = weighted_quad_fit_for_g(t, y_m, sy_m)

    return TOFPhysicsResults(vx=vx, svx=svx, vy=vy, svy=svy, g=g, sg=sg)


def attach_position_velocity_errors(df: pd.DataFrame,
                                    *,
                                    px_to_mx: Optional[ArrayLike] = None,
                                    px_to_my: Optional[ArrayLike] = None) -> pd.DataFrame:
    """
    Adds common per-row columns:
      x_after_m, y_after_m, sx_after_m, sy_after_m
      x_before_m, y_before_m, sx_before_m, sy_before_m
      vx_2pt, svx_2pt, vy_2pt, svy_2pt   (two-point between before/after, per row)

    Requires df:
      x0_before_px, x0_after_px, y0_before_px, y0_after_px
      sx0_before_px, sx0_after_px, sy0_before_px, sy0_after_px
      t_s
    """
    d = df.copy()

    if px_to_mx is None:
        if "px_to_mx" in d.columns:
            px_to_mx_arr = d["px_to_mx"].to_numpy(dtype=float)
        else:
            raise ValueError("px_to_mx not provided and not present in df.")
    else:
        px_to_mx_arr = np.asarray(px_to_mx, float) + np.zeros(len(d))

    if px_to_my is None:
        if "px_to_my" in d.columns:
            px_to_my_arr = d["px_to_my"].to_numpy(dtype=float)
        else:
            raise ValueError("px_to_my not provided and not present in df.")
    else:
        px_to_my_arr = np.asarray(px_to_my, float) + np.zeros(len(d))

    # after
    x_m, y_m, sx_m, sy_m = pos_errors_m(
        d["x0_after_px"].to_numpy(float),
        d["y0_after_px"].to_numpy(float),
        d["sx0_after_px"].to_numpy(float),
        d["sy0_after_px"].to_numpy(float),
        px_to_mx_arr, px_to_my_arr
    )
    d["x_after_m"], d["y_after_m"], d["sx_after_m"], d["sy_after_m"] = x_m, y_m, sx_m, sy_m

    # before
    x_m, y_m, sx_m, sy_m = pos_errors_m(
        d["x0_before_px"].to_numpy(float),
        d["y0_before_px"].to_numpy(float),
        d["sx0_before_px"].to_numpy(float),
        d["sy0_before_px"].to_numpy(float),
        px_to_mx_arr, px_to_my_arr
    )
    d["x_before_m"], d["y_before_m"], d["sx_before_m"], d["sy_before_m"] = x_m, y_m, sx_m, sy_m

    # two-point velocities per row
    t = d["t_s"].to_numpy(float)
    vx, svx = velocity_two_point(d["x_before_m"].to_numpy(float), d["x_after_m"].to_numpy(float),
                                 d["sx_before_m"].to_numpy(float), d["sx_after_m"].to_numpy(float), t)
    vy, svy = velocity_two_point(d["y_before_m"].to_numpy(float), d["y_after_m"].to_numpy(float),
                                 d["sy_before_m"].to_numpy(float), d["sy_after_m"].to_numpy(float), t)
    d["vx_2pt"], d["svx_2pt"], d["vy_2pt"], d["svy_2pt"] = vx, svx, vy, svy
    return d
