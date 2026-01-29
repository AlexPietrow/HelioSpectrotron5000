#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectrum scroller backend (FastAPI)

Loads precomputed TAPAS tellurics from:
  telat_alt0.npy  (alt=0 m)
  telat_alt1.npy  (alt=2500 m)

Each telluric file must be a NumPy array of shape (2, N):
  row 0: wavelength [Å]
  row 1: transmittance (dimensionless)

Invariants:
- PHYSICALLY CORRECT ORDER:
      y_final = (solar * telluric) ⊗ LSF
- 2D strip is ALWAYS tiled from y_final (post-processing).
- Tellurics displayed are convolved for UI consistency:
      t_disp = telluric ⊗ LSF

Controls (via query params):
- labels=0/1     : line-name overlay
- legend=0/1     : matplotlib legend
- tellurics=0/1  : multiply by telluric transmission + plot tellurics overlay
- refinf=0/1     : overlay the un-convolved reference spectrum (R=∞) on top of the convolved spectrum
- unit=A/nm      : plotting unit only (selection in Å)
- R500           : resolving power at 500 nm; uses FWHM_A = 5000 / R500
- flux=norm/cgs/flam : normalized or absolute (per Hz / per Å)
- theme=light/dark/auto : plot theme (auto currently treated as light unless frontend passes)
- transparent=0/1 : if 1, figure background is transparent

Run:
  uvicorn backend:app --reload --port 8000
"""

import io
import os
import gc
import sys
import traceback
from typing import Optional, Tuple, Dict

import numpy as np
import astropy.units as u

try:
    from specutils.utils.wcs_utils import air_to_vac
except Exception as _e:
    air_to_vac = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles


# ----------------------------
# CONFIG
# ----------------------------
GLOBAL_WMIN_A = 3290.0
GLOBAL_WMAX_A = 12500.0

DEFAULT_WIDTH_A = 25.0
OVERLAP = 0.10
DEFAULT_STEP_A = DEFAULT_WIDTH_A * (1.0 - OVERLAP)

# Default "no smoothing" (treated as "∞" by apply_resolution_R500)
DEFAULT_R500 = 1e12

DPI = 160
REPEAT_2D = 120

# Theme colors (match your frontend dark theme intent)
THEME_LIGHT = {
    "bg": "#ffffff",
    "panel": "#ffffff",
    "text": "#111827",
    "muted": "#6b7280",
    "border": "#d1d5db",
    "spec": "#000000",
    "ref": "#000000",
    "tell": "#ff0000",
    "grid": "#e5e7eb",
}
THEME_DARK = {
    "bg": "#0b1220",
    "panel": "#111827",
    "text": "#e5e7eb",
    "muted": "#9ca3af",
    "border": "#334155",
    "spec": "#ffffff",
    "ref": "#ffffff",
    "tell": "#ff3b3b",   # red but pops on dark
    "grid": "#334155",
}

# Resolve paths relative to this file
HERE = os.path.dirname(os.path.abspath(__file__))

# Ensure bundled ISPy submodule is importable (repo has ISPy/ISPy/...)
ISPY_BUNDLE = os.path.join(HERE, "ISPy")
if os.path.isdir(ISPY_BUNDLE) and ISPY_BUNDLE not in sys.path:
    sys.path.insert(0, ISPY_BUNDLE)

from ISPy.spec import atlas as ispy_atlas  # noqa: E402

# Telluric files (TAPAS precomputed)
TELL_FILE_ALT0 = os.environ.get("TELL_FILE_ALT0", os.path.join(HERE, "telat_alt0.npy"))  # 0 m
TELL_FILE_ALT1 = os.environ.get("TELL_FILE_ALT1", os.path.join(HERE, "telat_alt1.npy"))  # 2500 m

# Frontend
INDEX_HTML = os.environ.get("INDEX_HTML", os.path.join(HERE, "index.html"))

# Line lists (FILES IN REPO ROOT)
OLD_LINE_CSV = os.path.join(HERE, "moore_clean02012026.csv")
NEW_LINE_CSV = os.path.join(HERE, "babcock_clean_02012026.csv")


# ----------------------------
# ISPy atlas (lazy)
# ----------------------------
def get_fts():
    """Return a FRESH ISPy atlas instance.

    ISPy's atlas.to() mutates internal arrays in-place. Reusing a single instance
    across requests will corrupt units when switching perHz. Therefore, we create
    a new instance for every fetch.
    """
    return ispy_atlas.atlas()


def get_atlas_range() -> Tuple[float, float]:
    fts = get_fts()
    w = np.asarray(getattr(fts, "wave"), dtype=float)
    w = w[np.isfinite(w)]
    if w.size == 0:
        raise RuntimeError("Could not determine ISPy atlas wavelength range (fts.wave missing/empty).")
    return float(np.nanmin(w)), float(np.nanmax(w))


def fetch_ispy_air_norm(w0: float, w1: float):
    """Return wavelength [Å] and normalized intensity for [w0, w1].
    Robust to empty atlas coverage: returns empty arrays instead of raising.
    """
    pad = 0.5
    fts = get_fts()
    try:
        wav, I, cont = fts.get(w0 - pad, w1 + pad, cgs=True, nograv=True, perHz=False)
    except Exception:
        return np.array([], dtype=float), np.array([], dtype=float)

    wav = np.asarray(wav, dtype=float)
    I = np.asarray(I, dtype=float)
    cont = np.asarray(cont, dtype=float)

    if wav.size == 0 or I.size == 0 or cont.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    cont_safe = np.where(cont > 0, cont, np.nan)
    I_norm = I / cont_safe
    I_norm = np.clip(I_norm, 0, np.nanmax(I_norm))

    sel = (wav >= w0) & (wav <= w1)
    wav = wav[sel]
    I_norm = I_norm[sel]

    if wav.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    return wav, I_norm


def fetch_ispy_air_cgs_fnu(w0: float, w1: float):
    """Return wavelength [Å] and absolute intensity in cgs per Hz (I_nu) for [w0, w1]."""
    pad = 0.5
    fts = get_fts()
    try:
        wav, I, cont = fts.get(w0 - pad, w1 + pad, cgs=True, nograv=True, perHz=True)
    except Exception:
        return np.array([], dtype=float), np.array([], dtype=float)

    wav = np.asarray(wav, dtype=float)
    I = np.asarray(I, dtype=float)

    if wav.size == 0 or I.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    sel = (wav >= w0) & (wav <= w1)
    return wav[sel], I[sel]


def fetch_ispy_air_cgs_flam(w0: float, w1: float):
    """Return wavelength [Å] and absolute intensity in cgs per Å (I_lambda) for [w0, w1]."""
    pad = 0.5
    fts = get_fts()
    try:
        wav, I, cont = fts.get(w0 - pad, w1 + pad, cgs=True, nograv=True, perHz=False)
    except Exception:
        return np.array([], dtype=float), np.array([], dtype=float)

    wav = np.asarray(wav, dtype=float)
    I = np.asarray(I, dtype=float)

    if wav.size == 0 or I.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    sel = (wav >= w0) & (wav <= w1)
    return wav[sel], I[sel]


# ----------------------------
# Gaussian convolution (no SciPy)
# ----------------------------
def gaussian_kernel_1d(sigma: float) -> np.ndarray:
    """Normalized 1D Gaussian kernel."""
    if sigma <= 0:
        return np.array([1.0], dtype=float)
    radius = int(np.ceil(4.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= np.sum(k)
    return k


def convolve_reflect(y: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Convolve 1D array with kernel using reflect padding."""
    if k.size == 1:
        return y
    pad = k.size // 2
    ypad = np.pad(y, pad_width=pad, mode="reflect")
    return np.convolve(ypad, k, mode="valid")


def apply_resolution_R500(w: np.ndarray, y: np.ndarray, R500: float) -> np.ndarray:
    """Degrade y(w) by Gaussian with FWHM defined from resolving power at 500 nm."""
    R500 = float(R500)
    if not np.isfinite(R500) or R500 <= 0:
        return y
    if R500 >= 1e8:
        return y

    fwhm_A = 5000.0 / R500
    sigma_A = fwhm_A / 2.355

    dw = np.diff(w)
    dw_med = float(np.nanmedian(dw)) if dw.size else np.nan
    if not np.isfinite(dw_med) or dw_med <= 0:
        return y

    sigma_samples = sigma_A / dw_med
    if sigma_samples <= 0:
        return y

    k = gaussian_kernel_1d(sigma_samples)
    return convolve_reflect(y, k)


# ----------------------------
# Tellurics (TAPAS lookup)
# ----------------------------

# ----------------------------
# Air/Vacuum wavelength conversion helpers
# ----------------------------
def air_to_vac_A(w_air_A: np.ndarray) -> np.ndarray:
    """Convert air wavelengths [Å] -> vacuum wavelengths [Å] using specutils.air_to_vac.

    Returns a NumPy float array in Å.
    """
    if air_to_vac is None:
        raise RuntimeError(
            "specutils is required for air/vacuum conversion, but specutils.utils.wcs_utils.air_to_vac "
            "could not be imported."
        )
    w_air_A = np.asarray(w_air_A, dtype=float)
    return air_to_vac(w_air_A * u.AA).to_value(u.AA)


def vac_to_air_A(w_vac_A: np.ndarray, n_iter: int = 6) -> np.ndarray:
    """Convert vacuum wavelengths [Å] -> air wavelengths [Å] by numerically inverting air_to_vac_A.

    This avoids relying on a separate vac_to_air implementation while remaining consistent with
    the chosen air_to_vac conversion.
    """
    w_vac_A = np.asarray(w_vac_A, dtype=float)
    # Initial guess: air ~= vacuum to first order (good starting point in optical/NIR)
    w_air = w_vac_A.copy()

    # Newton iterations
    # Use a small step in Å for derivative estimation
    eps = 1e-3
    for _ in range(int(n_iter)):
        f = air_to_vac_A(w_air) - w_vac_A
        # derivative d(air_to_vac)/d(air)
        fp = (air_to_vac_A(w_air + eps) - air_to_vac_A(w_air - eps)) / (2 * eps)
        # guard against pathological zeros
        fp = np.where(np.abs(fp) > 0, fp, np.nan)
        step = np.where(np.isfinite(fp), f / fp, 0.0)
        w_air = w_air - step
    return w_air


def to_air_bounds(start_A: float, end_A: float, medium: str) -> tuple[float, float]:
    """Interpret (start,end) in the requested medium and return (start,end) in air Å for ISPy/TAPAS."""
    medium = (medium or "air").strip().lower()
    if medium in ("vac", "vacuum"):
        s_air = float(vac_to_air_A(np.array([start_A], dtype=float))[0])
        e_air = float(vac_to_air_A(np.array([end_A], dtype=float))[0])
        return s_air, e_air
    return float(start_A), float(end_A)


def air_to_medium_A(w_air_A: np.ndarray, medium: str) -> np.ndarray:
    """Convert air Å to requested medium Å (air or vacuum)."""
    medium = (medium or "air").strip().lower()
    if medium in ("vac", "vacuum"):
        return air_to_vac_A(w_air_A)
    return np.asarray(w_air_A, dtype=float)

def _load_telat(path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Telluric file not found: {path}\n"
            f"Put it next to backend.py, or set env vars:\n"
            f"  TELL_FILE_ALT0=/path/to/telat_alt0.npy\n"
            f"  TELL_FILE_ALT1=/path/to/telat_alt1.npy"
        )
    arr = np.load(path, allow_pickle=False)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != 2:
        raise ValueError(f"{path} has shape {arr.shape}, expected (2, N).")
    w = arr[0, :].ravel()
    t = arr[1, :].ravel()
    if w.size < 2:
        raise ValueError(f"{path} too small.")
    if not np.all(np.diff(w) > 0):
        idx = np.argsort(w)
        w, t = w[idx], t[idx]
    return w, t


# alt in meters
TELLURICS: Dict[int, Tuple[np.ndarray, np.ndarray]] = {
    0:    _load_telat(TELL_FILE_ALT0),
    2500: _load_telat(TELL_FILE_ALT1),
}

# Telluric coverage
_tmins = [float(np.nanmin(TELLURICS[k][0])) for k in TELLURICS]
_tmaxs = [float(np.nanmax(TELLURICS[k][0])) for k in TELLURICS]
TELL_WMIN = max(_tmins)
TELL_WMAX = min(_tmaxs)

# Atlas coverage
ATLAS_WMIN, ATLAS_WMAX = get_atlas_range()

# Final intersection
WMIN = max(GLOBAL_WMIN_A, TELL_WMIN, ATLAS_WMIN)
WMAX = min(GLOBAL_WMAX_A, TELL_WMAX, ATLAS_WMAX)

if not (np.isfinite(WMIN) and np.isfinite(WMAX) and WMIN < WMAX):
    raise RuntimeError(
        "Invalid wavelength intersection:\n"
        f"  GLOBAL: {GLOBAL_WMIN_A}–{GLOBAL_WMAX_A}\n"
        f"  TELL:   {TELL_WMIN}–{TELL_WMAX}\n"
        f"  ATLAS:  {ATLAS_WMIN}–{ATLAS_WMAX}\n"
        f"  FINAL:  {WMIN}–{WMAX}\n"
    )

# ----------------------------
# Line overlays (Moore + IA/strength) — robust loaders
# ----------------------------
old_wav = old_ids = old_forced = None
new_wav = new_strength = new_ids = new_forced = None



def clean_wavelength(val):
    """Extract first numeric token from wavelength string."""
    import re
    import pandas as pd
    if isinstance(val, (int, float)) and not pd.isna(val):
        return float(val)
    if not isinstance(val, str):
        return np.nan
    m = re.search(r"\d+(?:\.\d+)?", val)
    return float(m.group(0)) if m else np.nan


def clean_ew(val):
    """Extract numeric part of EW (e.g. '14N' → 14)."""
    import re
    import pandas as pd
    if isinstance(val, (int, float)) and not pd.isna(val):
        return float(val)
    if not isinstance(val, str):
        return np.nan
    m = re.search(r"[-+]?\d+(?:\.\d+)?", val)
    return float(m.group(0)) if m else np.nan


def clean_strength(val):
    """Extract numeric strength from messy strings."""
    import re
    import pandas as pd
    if isinstance(val, (int, float)) and not pd.isna(val):
        return float(val)
    if not isinstance(val, str):
        return np.nan
    m = re.search(r"[-+]?\d+(?:\.\d+)?", val)
    return float(m.group(0)) if m else np.nan


def _read_csv_auto(path: str):
    """Try comma, then semicolon."""
    import pandas as pd
    df = pd.read_csv(path)
    if len(df.columns) == 1 and ";" in str(df.columns[0]):
        df = pd.read_csv(path, sep=";")
    return df


def bin_lines(wav: np.ndarray, ids: np.ndarray, bin_A: float = 0.2):
    """
    Keep at most one line per wavelength bin.
    Chooses the first occurrence in each bin.
    Returns: wav_binned, ids_binned, idx (indices into the ORIGINAL arrays)
    """
    if wav is None or ids is None or len(wav) == 0:
        return wav, ids, np.array([], dtype=int)

    wav = np.asarray(wav, dtype=float)
    ids = np.asarray(ids)

    bins = np.round(wav / float(bin_A)).astype(int)
    _, idx = np.unique(bins, return_index=True)
    idx = np.sort(idx)

    return wav[idx], ids[idx], idx





def select_labels_windowed_binned(
    wav_A: np.ndarray,
    strength: np.ndarray | None,
    labels: np.ndarray,
    forced: np.ndarray | None,
    start_A: float,
    end_A: float,
    *,
    bin_A: float = 0.8,
    max_labels: int = 60,
):
    """Select line labels for a wavelength window in an atlas-faithful way.

    Rules:
      - Only consider lines within [start_A, end_A].
      - Lines flagged as *forced* are always included (if within the window).
      - Remaining (normal) lines are grouped into wavelength bins (bin_A) and the
        strongest line per bin is kept (if strength is available; otherwise first per bin).
      - Final list is sorted by wavelength.
      - max_labels limits ONLY the non-forced selections; forced lines are never dropped.

    Parameters
    ----------
    wav_A : array
        Line wavelengths (in the SAME medium as start_A/end_A).
    strength : array or None
        Strength metric (larger = stronger). If None, selection per bin falls back to first.
    labels : array
        Line label strings.
    forced : array or None
        Boolean array; True indicates forced inclusion. If None, all False.
    """
    wav_A = np.asarray(wav_A, dtype=float)
    labels = np.asarray(labels)

    if strength is None:
        strength = np.zeros_like(wav_A, dtype=float)
        have_strength = False
    else:
        strength = np.asarray(strength, dtype=float)
        have_strength = True

    if forced is None:
        forced = np.zeros_like(wav_A, dtype=bool)
    else:
        forced = np.asarray(forced, dtype=bool)

    m = np.isfinite(wav_A) & (wav_A >= start_A) & (wav_A <= end_A)
    if not np.any(m):
        return wav_A[:0], strength[:0], labels[:0]

    w = wav_A[m]
    s = strength[m]
    lab = labels[m]
    f = forced[m]

    # Forced lines: always keep (within the window)
    w_for = w[f]
    s_for = s[f]
    lab_for = lab[f]

    # Normal lines: apply bin/strength logic
    w_n = w[~f]
    s_n = s[~f]
    lab_n = lab[~f]

    if w_n.size == 0:
        # Only forced lines
        o = np.argsort(w_for, kind="mergesort")
        return w_for[o], s_for[o], lab_for[o]

    # Treat non-finite strength as very weak
    s_n = np.where(np.isfinite(s_n), s_n, -np.inf)

    # Bin relative to start_A for stability under panning
    b = np.floor((w_n - start_A) / float(bin_A)).astype(np.int64)

    # Sort by bin, stable
    order = np.argsort(b, kind="mergesort")
    b_s = b[order]
    w_s = w_n[order]
    s_s = s_n[order]
    lab_s = lab_n[order]

    # Bin boundaries
    edges = np.r_[0, 1 + np.flatnonzero(b_s[1:] != b_s[:-1]), len(b_s)]

    keep = []
    for i0, i1 in zip(edges[:-1], edges[1:]):
        if have_strength:
            j = i0 + int(np.nanargmax(s_s[i0:i1]))
        else:
            j = i0
        keep.append(j)
    keep = np.asarray(keep, dtype=np.int64)

    w_keep = w_s[keep]
    s_keep = s_s[keep]
    lab_keep = lab_s[keep]

    # Sort by wavelength
    o_keep = np.argsort(w_keep, kind="mergesort")
    w_keep = w_keep[o_keep]
    s_keep = s_keep[o_keep]
    lab_keep = lab_keep[o_keep]

    # Apply max_labels to NON-forced only
    if max_labels is not None and w_keep.size > max_labels:
        w_keep = w_keep[:max_labels]
        s_keep = s_keep[:max_labels]
        lab_keep = lab_keep[:max_labels]

    # Merge forced + normal and sort by wavelength
    w_all = np.concatenate([w_for, w_keep])
    s_all = np.concatenate([s_for, s_keep])
    lab_all = np.concatenate([lab_for, lab_keep])

    o_all = np.argsort(w_all, kind="mergesort")
    return w_all[o_all], s_all[o_all], lab_all[o_all]


def load_moore_lines(path: str):
    """Moore list CSV expected columns: (wavelength or wav), ew, id.

    Markers:
      - '*' => forced (always shown if within window)
      - '-' => excluded (never shown)

    Markers can appear either:
      - in an extra flag column (recommended), or
      - embedded in the id string (e.g. "Hα*" or "Hα -").

    Returns: wav, id, forced
    """
    if not path or not os.path.exists(path):
        print(f"[LINES] Moore missing: {path}", flush=True)
        return None, None, None

    df = _read_csv_auto(path)

    # accept either naming convention
    if "wavelength" not in df.columns and "wav" in df.columns:
        df = df.rename(columns={"wav": "wavelength"})

    if "id" not in df.columns:
        raise ValueError(f"Moore CSV columns missing. Found: {list(df.columns)}")

    # Optional flag column: first column not in the expected set
    expected = {"wavelength", "wav", "ew", "id"}
    extra_cols = [c for c in df.columns if c not in expected]
    flag_col = extra_cols[-1] if len(extra_cols) > 0 else None

    df["wavelength"] = df["wavelength"].apply(clean_wavelength)
    if "ew" in df.columns:
        df["ew"] = df["ew"].apply(clean_ew)
    df["id"] = df["id"].fillna("").astype(str)

    # --- marker parsing helpers (conservative) ---
    def _id_is_forced(s: str) -> bool:
        s = (s or "").strip()
        return s.endswith("*") or s.startswith("*") or (" *" in s) or ("* " in s)

    def _id_is_excluded(s: str) -> bool:
        s = (s or "").strip()
        return s.endswith("-") or s.startswith("-") or (" -" in s) or ("- " in s) or (" - " in s)

    forced_id = df["id"].map(_id_is_forced)

    if flag_col is not None:
        flag_s = df[flag_col].fillna("").astype(str).str.strip()
        forced_flag = flag_s.str.contains("*", regex=False)
        exclude_flag = flag_s.str.contains("-", regex=False)
    else:
        forced_flag = False
        exclude_flag = False

    exclude_id = df["id"].map(_id_is_excluded)

    forced = (forced_id | forced_flag).to_numpy(bool)
    excluded = (exclude_id | exclude_flag).to_numpy(bool)

    # Drop excluded lines entirely (exclusion wins over forcing)
    if np.any(excluded):
        df = df.loc[~excluded].copy()
        forced = forced[~excluded]

    # Strip markers from label text
    df["id"] = (
        df["id"]
        .str.replace("*", "", regex=False)
        .str.replace("-", "", regex=False)
        .str.strip()
    )

    df = df.dropna(subset=["wavelength"])
    df = df[df["id"].str.strip() != ""]

    if len(df) == 0:
        return np.array([], dtype=float), np.array([], dtype=str), np.array([], dtype=bool)

    return df["wavelength"].to_numpy(float), df["id"].to_numpy(str), np.asarray(forced, dtype=bool)



def load_ia_lines(path: str):
    """IA/strength CSV expected columns: wav, strength, id, and optional flag column.

    Markers:
      - '*' => forced (always shown if within window)
      - '-' => excluded (never shown)

    Markers can appear either:
      - in an extra flag column (recommended), or
      - embedded in the id string (e.g. "Ca II 8542*" or "Ca II 8542 -").

    Exclusion wins over forcing.

    Returns: wav, strength, id, forced
    """
    if not path or not os.path.exists(path):
        print(f"[LINES] IA missing: {path}", flush=True)
        return None, None, None, None

    df = _read_csv_auto(path)
    if not all(c in df.columns for c in ("wav", "strength", "id")):
        raise ValueError(f"IA CSV columns missing. Found: {list(df.columns)}")

    # Optional flag column: any extra column beyond required set
    extra_cols = [c for c in df.columns if c not in ("wav", "strength", "id")]
    flag_col = extra_cols[-1] if len(extra_cols) > 0 else None

    # Clean core columns
    df["wav"]      = df["wav"].apply(clean_wavelength)
    df["strength"] = df["strength"].apply(clean_strength)
    df["id"]       = df["id"].fillna("").astype(str)

    # Conservative marker parsing: only treat '*' or '-' as markers when separated or at ends
    def _id_is_forced(s: str) -> bool:
        s = (s or "").strip()
        return s.endswith("*") or s.startswith("*") or (" *" in s) or ("* " in s)

    def _id_is_excluded(s: str) -> bool:
        s = (s or "").strip()
        return s.endswith("-") or s.startswith("-") or (" -" in s) or ("- " in s) or (" - " in s)

    forced_id  = df["id"].map(_id_is_forced)
    exclude_id = df["id"].map(_id_is_excluded)

    if flag_col is not None:
        flag_s = df[flag_col].fillna("").astype(str).str.strip()
        forced_flag  = flag_s.str.contains("*", regex=False)
        exclude_flag = flag_s.str.contains("-", regex=False)
    else:
        forced_flag  = False
        exclude_flag = False

    forced = (forced_id | forced_flag).to_numpy(bool)
    excluded = (exclude_id | exclude_flag).to_numpy(bool)

    # Strip markers from label text
    df["id"] = (
        df["id"]
        .str.replace("*", "", regex=False)
        .str.replace("-", "", regex=False)
        .str.strip()
    )

    # Build a single boolean mask and apply it to BOTH df and forced to keep alignment
    m = np.ones(len(df), dtype=bool)

    # Exclusion (wins)
    if np.any(excluded):
        m &= ~excluded

    # Required finite data
    m &= np.isfinite(df["wav"].to_numpy(float))
    m &= np.isfinite(df["strength"].to_numpy(float))

    # Strength threshold
    m &= (df["strength"].to_numpy(float) >= -5)

    # Remove telluric IDs
    m &= ~df["id"].str.contains("atm", case=False, na=False).to_numpy(bool)

    df = df.loc[m].copy()
    forced = forced[m]

    if len(df) == 0:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=str), np.array([], dtype=bool)

    return (
        df["wav"].to_numpy(float),
        df["strength"].to_numpy(float),
        df["id"].to_numpy(str),
        np.asarray(forced, dtype=bool),
    )


try:
    old_wav, old_ids, old_forced = load_moore_lines(OLD_LINE_CSV)

    # DO NOT pre-bin Moore here. Pre-binning can delete forced lines before selection.
    if old_wav is None or old_ids is None:
        old_wav = old_ids = old_forced = None
    else:
        old_wav = np.asarray(old_wav, dtype=float)
        old_ids = np.asarray(old_ids, dtype=str)
        old_forced = np.asarray(old_forced, dtype=bool) if old_forced is not None else np.zeros_like(old_wav, dtype=bool)

    print(f"[INFO] Moore CSV={OLD_LINE_CSV} n={0 if old_wav is None else len(old_wav)}", flush=True)
    print(f"[INFO] Moore forced n={0 if old_forced is None else int(np.sum(old_forced))}", flush=True)

except Exception as e:
    print(f"[WARN] Moore line CSV not loaded: {e}", flush=True)
    old_wav = old_ids = old_forced = None



try:
    new_wav, new_strength, new_ids, new_forced = load_ia_lines(NEW_LINE_CSV)
    print(f"[INFO] IA CSV={NEW_LINE_CSV}  n={0 if new_wav is None else len(new_wav)}", flush=True)
except Exception as e:
    print(f"[WARN] IA line CSV not loaded: {e}", flush=True)
    new_wav = new_strength = new_ids = new_forced = None


# ----------------------------
# Theme helpers
# ----------------------------
def _pick_theme(theme: str) -> Dict[str, str]:
    t = (theme or "light").strip().lower()
    if t == "dark":
        return THEME_DARK
    if t == "auto":
        # backend cannot know OS preference; frontend should pass explicit dark/light.
        # Treat auto as light for now.
        return THEME_LIGHT
    return THEME_LIGHT


def _style_axes(ax, theme_dict: Dict[str, str]):
    # Face/background
    ax.set_facecolor(theme_dict["panel"])

    # Spines
    for s in ax.spines.values():
        s.set_color(theme_dict["border"])

    # Ticks + labels
    ax.tick_params(colors=theme_dict["text"], which="both")
    ax.xaxis.label.set_color(theme_dict["text"])
    ax.yaxis.label.set_color(theme_dict["text"])
    ax.title.set_color(theme_dict["text"])

    # Grid off by default; if you ever enable, use theme_dict["grid"]
    ax.grid(False)


def _style_legend(leg, theme_dict: Dict[str, str]):
    if leg is None:
        return
    frame = leg.get_frame()
    frame.set_facecolor(theme_dict["panel"])
    frame.set_edgecolor(theme_dict["border"])
    frame.set_alpha(1.0)
    for txt in leg.get_texts():
        txt.set_color(theme_dict["text"])


# ----------------------------
# Rendering
# ----------------------------
def render_segment_png(
    start: float,
    end: float,
    R500: float,
    alt_m: int,
    labels_on: bool,
    legend_on: bool,
    tellurics_on: bool,
    refinf_on: bool,
    medium: str,
    unit: str,
    flux: str = "norm",
    theme: str = "light",
    transparent: bool = False,
    show1d: bool = True,
    show2d: bool = True,
) -> bytes:
    """Render [start,end] slice.

    Notes
    -----
    - `start`/`end` are interpreted in the requested *medium* (air/vac) in Å.
    - All internal computation is done in AIR Å to match ISPy + telluric tables.
    - `unit` only affects plotting.
    """
    unit = (unit or "A").strip().lower()
    plot_in_nm = unit in ("nm", "nanometer", "nanometers")

    medium = (medium or "air").strip().lower()

    # ISPy atlas wavelengths are in AIR Å; convert request bounds to AIR for internal slicing
    start_air, end_air = to_air_bounds(start, end, medium)

    flux = (flux or "norm").strip().lower()
    theme_dict = _pick_theme(theme)

    # Fetch solar spectrum (AIR Å)
    if flux == "norm":
        w_air, y_base = fetch_ispy_air_norm(start_air, end_air)
    elif flux == "cgs":
        w_air, y_base = fetch_ispy_air_cgs_fnu(start_air, end_air)
    elif flux == "flam":
        w_air, y_base = fetch_ispy_air_cgs_flam(start_air, end_air)
    else:
        w_air, y_base = fetch_ispy_air_norm(start_air, end_air)

    show1d = bool(show1d)
    show2d = bool(show2d)
    if (not show1d) and (not show2d):
        show1d = True  # safety

    # Figure layout
    if show1d and show2d:
        fig, (ax1, ax2) = plt.subplots(
            nrows=2,
            figsize=(12, 4.8),
            gridspec_kw={"height_ratios": [2, 1]},
            constrained_layout=True,
        )
    elif show1d and (not show2d):
        fig, ax1 = plt.subplots(nrows=1, figsize=(12, 3.2), constrained_layout=True)
        ax2 = None
    else:
        fig, ax2 = plt.subplots(nrows=1, figsize=(12, 2.2), constrained_layout=True)
        ax1 = None

    # Background + axes styling
    fig.patch.set_facecolor(theme_dict["bg"])
    if ax1 is not None:
        _style_axes(ax1, theme_dict)
    if ax2 is not None:
        _style_axes(ax2, theme_dict)

    # Empty slice safety
    if w_air.size == 0:
        if ax1 is not None:
            ax1.text(0.5, 0.5, "Empty slice", ha="center", va="center",
                     transform=ax1.transAxes, color=theme_dict["text"])
            ax1.axis("off")
        if ax2 is not None:
            ax2.text(0.5, 0.5, "Empty slice", ha="center", va="center",
                     transform=ax2.transAxes, color=theme_dict["text"])
            ax2.axis("off")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=DPI, transparent=transparent)
        plt.close(fig)
        gc.collect()
        return buf.getvalue()

    # Tellurics (AIR Å)
    twav, tint = TELLURICS.get(int(alt_m), TELLURICS[2500])
    t_seg = np.interp(w_air, twav, tint)

    if not tellurics_on:
        t_seg = np.ones_like(t_seg)
        tint_for_display = np.ones_like(w_air)
    else:
        tint_for_display = t_seg

    # ---- PHYSICALLY CORRECT ORDER ----
    y_highres = np.asarray(y_base * t_seg, dtype=float)
    y_final = np.asarray(apply_resolution_R500(w_air, y_highres, R500), dtype=float)
    y_ref = y_highres  # "∞" reference (only meaningful if R500 is finite)
    t_disp = np.asarray(apply_resolution_R500(w_air, tint_for_display, R500), dtype=float)

    # Convert wavelengths for display medium (AIR -> AIR/VAC)
    w_disp_A = air_to_medium_A(w_air, medium)
    w_plot = (w_disp_A / 10.0) if plot_in_nm else w_disp_A
    start_plot = (start / 10.0) if plot_in_nm else start
    end_plot = (end / 10.0) if plot_in_nm else end
    unit_label = "nm" if plot_in_nm else "Å"

    # Colors
    spec_col = theme_dict["spec"]
    ref_col = theme_dict["ref"]
    tell_col = theme_dict["tell"]

    # --- 1D panel ---
    if ax1 is not None:
        ax1.set_xlim(start_plot, end_plot)

        if flux in ("cgs", "flam"):
            # Optional reference overlay
            if refinf_on and (np.isfinite(R500) and R500 < 1e8):
                ax1.plot(w_plot, y_ref, color=ref_col, lw=1.0, alpha=0.45, zorder=1,
                         label="REF ∞" if legend_on else None)

            ax1.plot(w_plot, y_final, color=spec_col, lw=2.0, zorder=2,
                     label="Spectrum" if legend_on else None)

            # y-limits based on percentile
            y_cont = float(np.nanpercentile(y_final, 99.0)) if y_final.size else 1.0
            if not np.isfinite(y_cont) or y_cont <= 0:
                y_cont = float(np.nanmax(y_final)) if np.isfinite(np.nanmax(y_final)) else 1.0
            if y_cont <= 0:
                y_cont = 1.0
            headroom = 1.20
            ax1.set_ylim(0, headroom * y_cont)

            axT = None
            if tellurics_on:
                axT = ax1.twinx()
                axT.set_zorder(ax1.get_zorder() - 1)
                ax1.patch.set_visible(False)
                axT.patch.set_visible(False)
                axT.set_facecolor("none")
                for s in axT.spines.values():
                    s.set_color(theme_dict["border"])
                axT.tick_params(colors=theme_dict["text"], which="both")
                axT.yaxis.label.set_color(theme_dict["text"])
                axT.set_ylim(0, headroom)
                axT.plot(w_plot, t_disp, color=tell_col, lw=1.2, zorder=3,
                         label="Tellurics" if legend_on else None)
                axT.set_ylabel("Telluric transmission")
                axT.grid(False)

            ax1.set_ylabel("Intensity (cgs per Å)" if flux == "flam" else "Intensity (cgs per Hz)")

            if legend_on:
                handles, labs = ax1.get_legend_handles_labels()
                if tellurics_on and axT is not None:
                    h2, l2 = axT.get_legend_handles_labels()
                    handles += h2
                    labs += l2
                leg = ax1.legend(handles, labs, loc="upper right", frameon=True, fontsize=10)
                _style_legend(leg, theme_dict)

        else:
            # Normalized: plot tellurics in same axis for simplicity
            if tellurics_on:
                ax1.plot(w_plot, t_disp, color=tell_col, lw=1.2, zorder=3,
                         label="Tellurics" if legend_on else None)

            if refinf_on and (np.isfinite(R500) and R500 < 1e8):
                ax1.plot(w_plot, y_ref, color=ref_col, lw=1.0, alpha=0.45, zorder=1,
                         label="REF ∞" if legend_on else None)

            ax1.plot(w_plot, y_final, color=spec_col, lw=2.0, zorder=2,
                     label="Spectrum" if legend_on else None)

            ax1.set_ylim(0, 1.20)
            ax1.set_ylabel("Normalized intensity")

            if legend_on:
                leg = ax1.legend(loc="upper right", frameon=True, fontsize=10)
                _style_legend(leg, theme_dict)

        r_txt = "∞" if (np.isfinite(R500) and R500 >= 1e8) else f"{R500:g}"
        ax1.set_title(f"{start_plot:.3f}–{end_plot:.3f} {unit_label}   (R@500nm={r_txt}, alt={alt_m} m, flux={flux})")

        if ax2 is None:
            ax1.set_xlabel(f"Wavelength [{unit_label}]")

        # ---- Line overlays (only on ax1) ----
        if labels_on:
            MAX_LABELS = 60

            # Moore list
            if old_wav is not None and old_ids is not None:
                moore_wav_med = air_to_vac_A(old_wav) if medium in ("vac", "vacuum") else old_wav
                moore_forced = old_forced if (old_forced is not None and len(old_forced) == len(old_wav)) else None

                pw_moore, _, pi_moore = select_labels_windowed_binned(
                    moore_wav_med,
                    strength=None,
                    labels=old_ids,
                    forced=moore_forced,
                    start_A=start,
                    end_A=end,
                    bin_A=0.8,
                    max_labels=MAX_LABELS,
                )
                for x_med, lab in zip(pw_moore, pi_moore):
                    x_plot = (x_med / 10.0) if plot_in_nm else x_med
                    ax1.axvline(x_plot, ymin=0.0, ymax=0.82, lw=0.4, alpha=0.5, zorder=0, color=spec_col)
                    ax1.text(
                        x_plot, 0.84, lab,
                        transform=ax1.get_xaxis_transform(),
                        rotation=45, fontsize=8,
                        ha="center", va="bottom",
                        color=spec_col,
                    )

            # IA / strength list
            if new_wav is not None and new_ids is not None and new_strength is not None and new_forced is not None:
                ia_wav_med = air_to_vac_A(new_wav) if medium in ("vac", "vacuum") else new_wav
                ia_forced = new_forced if (new_forced is not None and len(new_forced) == len(new_wav)) else None

                pw_ia, _, pi_ia = select_labels_windowed_binned(
                    ia_wav_med,
                    new_strength,
                    new_ids,
                    ia_forced,
                    start_A=start,
                    end_A=end,
                    bin_A=0.8,
                    max_labels=MAX_LABELS,
                )
                for x_med, lab in zip(pw_ia, pi_ia):
                    x_plot = (x_med / 10.0) if plot_in_nm else x_med
                    ax1.axvline(x_plot, ymin=0.0, ymax=0.82, lw=0.4, alpha=0.5, zorder=0, color=spec_col)
                    ax1.text(
                        x_plot, 0.84, lab,
                        transform=ax1.get_xaxis_transform(),
                        rotation=45, fontsize=8,
                        ha="center", va="bottom",
                        color=spec_col,
                    )

    # --- 2D panel ---
    if ax2 is not None:
        y_strip = np.asarray(y_final, dtype=float)

        p1 = float(np.nanpercentile(y_strip, 1))
        p99 = float(np.nanpercentile(y_strip, 99))
        if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
            p1, p99 = float(np.nanmin(y_strip)), float(np.nanmax(y_strip))
        if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
            p1, p99 = 0.0, 1.0

        y_strip_n = (y_strip - p1) / (p99 - p1) if (p99 > p1) else y_strip * 0.0
        y_strip_n = np.clip(y_strip_n, 0.0, 1.0)

        img2d = np.tile(y_strip_n[np.newaxis, :], (REPEAT_2D, 1))
        ax2.imshow(
            img2d,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="gray",
            extent=[w_plot[0], w_plot[-1], 0, 1.0],
        )
        ax2.set_xlim(start_plot, end_plot)
        ax2.set_xlabel(f"Wavelength [{unit_label}]")
        ax2.set_yticks([])
        ax2.tick_params(colors=theme_dict["text"], which="both")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, transparent=transparent)
    plt.close(fig)
    gc.collect()
    return buf.getvalue()

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI()

app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")

@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return PlainTextResponse("ok")


@app.get("/meta", response_class=PlainTextResponse)
def meta():
    return PlainTextResponse(
        f"WMIN={WMIN}\nWMAX={WMAX}\n"
        f"ATLAS_WMIN={ATLAS_WMIN}\nATLAS_WMAX={ATLAS_WMAX}\n"
        f"TELL_WMIN={TELL_WMIN}\nTELL_WMAX={TELL_WMAX}\n"
        f"DEFAULT_WIDTH_A={DEFAULT_WIDTH_A}\nDEFAULT_STEP_A={DEFAULT_STEP_A}\n"
        f"DEFAULT_R500={DEFAULT_R500}\n"
        f"TELL0={TELL_FILE_ALT0}\nTELL1={TELL_FILE_ALT1}\nINDEX={INDEX_HTML}\n"
        f"OLD_LINE_CSV={OLD_LINE_CSV}\nNEW_LINE_CSV={NEW_LINE_CSV}\n"
        f"N_MOORE={(len(old_wav) if old_wav is not None else 0)}\n"
        f"N_IA={(len(new_wav) if new_wav is not None else 0)}\n"
    )


@app.get("/", response_class=HTMLResponse)
def index():
    if not os.path.exists(INDEX_HTML):
        return PlainTextResponse(f"index.html not found at: {INDEX_HTML}", status_code=500)

    html = open(INDEX_HTML, "r", encoding="utf-8").read()
    html = html.replace("__WMIN__",  f"{WMIN:.6f}")
    html = html.replace("__WMAX__",  f"{WMAX:.6f}")
    html = html.replace("__WIDTH__", f"{DEFAULT_WIDTH_A:.6f}")
    html = html.replace("__STEP__",  f"{DEFAULT_STEP_A:.6f}")
    html = html.replace("__R500__",  f"{DEFAULT_R500:.6f}")

    return HTMLResponse(
        html,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/segment.png")
def segment_png(
    start: float,
    width: Optional[float] = None,
    R500: Optional[float] = None,
    alt: Optional[int] = 2500,      # 0 or 2500 (meters)
    labels: Optional[int] = 1,      # 0/1
    legend: Optional[int] = 0,      # 0/1
    tellurics: Optional[int] = 1,   # 0/1
    refinf: Optional[int] = 0,      # 0/1 overlay REF ∞ (unconvolved)
    medium: Optional[str] = "air",  # 'air' or 'vac'
    unit: Optional[str] = "A",      # 'A' or 'nm' (plotting only)
    flux: Optional[str] = "norm",   # 'norm' or 'cgs' or 'flam'
    theme: Optional[str] = "light", # 'light'|'dark'|'auto'
    transparent: Optional[int] = 0, # 0/1
    show1d: Optional[int] = 1,      # 0/1
    show2d: Optional[int] = 1,      # 0/1
):
    try:
        start = float(start)
        width = float(width) if width is not None else DEFAULT_WIDTH_A
        R500 = DEFAULT_R500 if (R500 is None) else float(R500)

        # clamp
        start = max(WMIN, min(start, WMAX - 0.1))
        width = max(0.1, min(width, WMAX - start))
        end = start + width

        alt_m = int(alt) if int(alt) in (0, 2500) else 2500
        labels_on = bool(int(labels)) if labels is not None else True
        legend_on = bool(int(legend)) if legend is not None else False
        tellurics_on = bool(int(tellurics)) if tellurics is not None else True
        refinf_on = bool(int(refinf)) if refinf is not None else False

        theme = (theme or "light")
        transparent_on = bool(int(transparent)) if transparent is not None else False

        png = render_segment_png(
            start, end,
            R500=R500,
            alt_m=alt_m,
            labels_on=labels_on,
            legend_on=legend_on,
            tellurics_on=tellurics_on,
            refinf_on=refinf_on,
            medium=medium,
            unit=unit,
            flux=flux,
            theme=theme,
            transparent=transparent_on,
            show1d=bool(int(show1d)) if show1d is not None else True,
            show2d=bool(int(show2d)) if show2d is not None else True,
        )
        return Response(
            content=png,
            media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )

    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        return Response(
            content=tb.encode("utf-8", errors="replace"),
            media_type="text/plain; charset=utf-8",
            status_code=500,
            headers={"Cache-Control": "no-store"},
        )


@app.get("/segment.txt", response_class=PlainTextResponse)
def segment_txt(
    start: float,
    width: Optional[float] = None,
    R500: Optional[float] = None,
    alt: Optional[int] = 2500,      # 0 or 2500 (meters)
    labels: Optional[int] = 1,      # unused; kept for symmetry
    legend: Optional[int] = 0,      # unused; kept for symmetry
    tellurics: Optional[int] = 1,   # 0/1
    medium: Optional[str] = "air",  # 'air' or 'vac'
    unit: Optional[str] = "A",      # output unit: 'A' or 'nm'
    flux: Optional[str] = "norm",   # 'norm' or 'cgs' or 'flam'
):
    try:
        start = float(start)
        width = float(width) if width is not None else DEFAULT_WIDTH_A
        R500 = DEFAULT_R500 if (R500 is None) else float(R500)

        # clamp
        start = max(WMIN, min(start, WMAX - 0.1))
        width = max(0.1, min(width, WMAX - start))
        end = start + width
        medium = (medium or "air").strip().lower()
        start_air, end_air = to_air_bounds(start, end, medium)

        alt_m = int(alt) if int(alt) in (0, 2500) else 2500

        flux = (flux or "norm").strip().lower()
        plot_cgs = (flux != "norm")

        if flux == "norm":
            w, y_base = fetch_ispy_air_norm(start_air, end_air)
        elif flux == "cgs":
            w, y_base = fetch_ispy_air_cgs_fnu(start_air, end_air)
        elif flux == "flam":
            w, y_base = fetch_ispy_air_cgs_flam(start_air, end_air)
        else:
            w, y_base = fetch_ispy_air_norm(start_air, end_air)

        if w.size == 0:
            return PlainTextResponse("# empty slice\n", status_code=200)

        twav, tint = TELLURICS.get(int(alt_m), TELLURICS[2500])
        t_seg = np.interp(w, twav, tint)

        tellurics_on = bool(int(tellurics)) if tellurics is not None else True
        if not tellurics_on:
            t_seg = np.ones_like(t_seg)

        y_highres = np.asarray(y_base * t_seg, dtype=float)
        y_final   = np.asarray(apply_resolution_R500(w, y_highres, R500), dtype=float)

        w_disp = air_to_medium_A(w, medium)

        u = (unit or "A").strip().lower()
        if u in ("nm", "nanometer", "nanometers"):
            w_out = w_disp / 10.0
            unit_label = "nm"
        else:
            w_out = w_disp
            unit_label = "A"

        col = ("y_final_flam" if flux == "flam" else "y_final_cgs") if plot_cgs else "y_final_norm"
        lines = [f"# wavelength[{unit_label}]   {col}"]
        lines += [f"{ww:.6f}\t{yy:.8f}" for ww, yy in zip(w_out, y_final)]
        return PlainTextResponse("\n".join(lines) + "\n", status_code=200)

    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        return PlainTextResponse(tb, status_code=500)


@app.get("/hover.json", response_class=JSONResponse)
def hover_json(
    start: float,
    x: float,
    width: Optional[float] = None,
    R500: Optional[float] = None,
    alt: Optional[int] = 2500,      # 0 or 2500 (meters)
    tellurics: Optional[int] = 1,   # 0/1
    medium: Optional[str] = "air",  # 'air' or 'vac'
    unit: Optional[str] = "A",      # x unit: 'A' or 'nm' (same as plotting unit)
    flux: Optional[str] = "norm",   # 'norm' or 'cgs' or 'flam'
):
    try:
        start = float(start)
        x = float(x)
        width = float(width) if width is not None else DEFAULT_WIDTH_A
        R500 = DEFAULT_R500 if (R500 is None) else float(R500)

        # clamp segment
        start = max(WMIN, min(start, WMAX - 0.1))
        width = max(0.1, min(width, WMAX - start))
        end = start + width

        alt_m = int(alt) if int(alt) in (0, 2500) else 2500

        # interpret x in requested unit and convert to Å for interpolation
        u = (unit or "A").strip().lower()
        if u in ("nm", "nanometer", "nanometers"):
            x_A = x * 10.0
            unit_label = "nm"
        else:
            x_A = x
            unit_label = "A"
        medium = (medium or "air").strip().lower()

        # Convert request window and hover position to AIR Å for internal interpolation
        start_air, end_air = to_air_bounds(start, end, medium)
        if medium in ("vac", "vacuum"):
            x_air = float(vac_to_air_A(np.array([x_A], dtype=float))[0])
        else:
            x_air = x_A

        # data in Å
        flux = (flux or "norm").strip().lower()
        if flux == "norm":
            w, y_base = fetch_ispy_air_norm(start_air, end_air)
        elif flux == "cgs":
            w, y_base = fetch_ispy_air_cgs_fnu(start_air, end_air)
        elif flux == "flam":
            w, y_base = fetch_ispy_air_cgs_flam(start_air, end_air)
        else:
            w, y_base = fetch_ispy_air_norm(start_air, end_air)

        if w.size == 0:
            return JSONResponse(
                {"ok": False, "reason": "empty slice", "unit": unit_label, "x": x, "y": None},
                status_code=200,
            )

        twav, tint = TELLURICS.get(int(alt_m), TELLURICS[2500])
        t_seg = np.interp(w, twav, tint)

        tellurics_on = bool(int(tellurics)) if tellurics is not None else True
        if not tellurics_on:
            t_seg = np.ones_like(t_seg)

        y_highres = np.asarray(y_base * t_seg, dtype=float)
        y_final   = np.asarray(apply_resolution_R500(w, y_highres, R500), dtype=float)

        if not (w[0] <= x_air <= w[-1]):
            return JSONResponse(
                {"ok": True, "unit": unit_label, "x": x, "y": None, "note": "x out of slice"},
                status_code=200,
            )

        y = float(np.interp(x_air, w, y_final))

        return JSONResponse({"ok": True, "unit": unit_label, "x": x, "y": y}, status_code=200)

    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        return JSONResponse({"ok": False, "error": tb}, status_code=500)

@app.get("/favicon.svg", include_in_schema=False)
def favicon():
    return FileResponse(os.path.join(HERE, "favicon.svg"))

