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
- labels=0/1  : line-name overlay
- legend=0/1  : matplotlib legend
- unit=A/nm   : plotting unit only (selection in Å)
- R500        : resolving power at 500 nm; uses FWHM_A = 5000 / R500

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, PlainTextResponse

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
OLD_LINE_CSV = os.path.join(HERE, "moore_binned_0p5A.csv")
NEW_LINE_CSV = os.path.join(HERE, "babcock_binned_0p5A.csv")


# ----------------------------
# ISPy atlas (lazy)
# ----------------------------
_fts = None
_ATLAS_WMIN = None
_ATLAS_WMAX = None

def get_fts():
    global _fts, _ATLAS_WMIN, _ATLAS_WMAX
    if _fts is None:
        _fts = ispy_atlas.atlas()
        # Determine atlas wavelength coverage once
        try:
            w = np.asarray(getattr(_fts, "wave"), dtype=float)
            w = w[np.isfinite(w)]
            if w.size > 0:
                _ATLAS_WMIN = float(np.nanmin(w))
                _ATLAS_WMAX = float(np.nanmax(w))
        except Exception:
            _ATLAS_WMIN = None
            _ATLAS_WMAX = None
    return _fts

def get_atlas_range() -> Tuple[float, float]:
    _ = get_fts()
    if _ATLAS_WMIN is None or _ATLAS_WMAX is None:
        raise RuntimeError("Could not determine ISPy atlas wavelength range (fts.wave missing/empty).")
    return _ATLAS_WMIN, _ATLAS_WMAX

def fetch_ispy_air_norm(w0: float, w1: float):
    """Return wavelength [Å] and normalized intensity for [w0, w1].
    Robust to empty atlas coverage: returns empty arrays instead of raising.
    """
    pad = 0.5
    fts = get_fts()
    try:
        wav, I, cont = fts.get(w0 - pad, w1 + pad, cgs=True, nograv=True, perHz=True)
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

# ----------------------------
# Gaussian convolution (no SciPy)
# ----------------------------

def fetch_ispy_air_cgs(w0: float, w1: float):
    """Return wavelength [Å] and absolute intensity in cgs for [w0, w1].

    Notes
    -----
    This uses the ISPy FTS interface with cgs=True and perHz=True, matching the
    normalized pathway. The returned intensity is *not* continuum-normalized.
    """
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
old_wav = old_ids = None
new_wav = new_ids = None

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
    """
    if wav is None or ids is None or len(wav) == 0:
        return wav, ids

    bins = np.round(wav / bin_A).astype(int)

    _, idx = np.unique(bins, return_index=True)
    idx = np.sort(idx)

    return wav[idx], ids[idx]

def fnu_to_flam(lam_A, Fnu):
    lam_cm = np.asarray(lam_A) * 1e-8
    Fnu = np.asarray(Fnu)
    return Fnu * 2.99792458e10 / lam_cm**2 * 1e-8

def load_moore_lines(path: str):
    """Moore list CSV expected columns: (wavelength or wav), ew, id."""
    if not path or not os.path.exists(path):
        print(f"[LINES] Moore missing: {path}", flush=True)
        return None, None

    df = _read_csv_auto(path)

    # accept either naming convention
    if "wavelength" not in df.columns and "wav" in df.columns:
        df = df.rename(columns={"wav": "wavelength"})

    if not all(c in df.columns for c in ("wavelength", "ew", "id")):
        raise ValueError(f"Moore CSV columns missing. Found: {list(df.columns)}")

    df["wavelength"] = df["wavelength"].apply(clean_wavelength)
    df["ew"]         = df["ew"].apply(clean_ew)
    df["id"]         = df["id"].fillna("").astype(str)

    # basic filtering like you do for IA
    df = df.dropna(subset=["wavelength"])
    df = df[df["id"].str.strip() != ""]

    if len(df) == 0:
        return np.array([], dtype=float), np.array([], dtype=str)

    return df["wavelength"].to_numpy(float), df["id"].to_numpy(str)

def load_ia_lines(path: str):
    """IA/strength CSV expected columns: wav, strength, id."""
    if not path or not os.path.exists(path):
        print(f"[LINES] IA missing: {path}", flush=True)
        return None, None

    df = _read_csv_auto(path)
    if not all(c in df.columns for c in ("wav", "strength", "id")):
        raise ValueError(f"IA CSV columns missing. Found: {list(df.columns)}")

    df["wav"]      = df["wav"].apply(clean_wavelength)
    df["strength"] = df["strength"].apply(clean_strength)
    df["id"]       = df["id"].fillna("").astype(str)

    df = df.dropna(subset=["wav", "strength"])
    df = df[df["strength"] >= -5]
    df = df[~df["id"].str.contains("atm", case=False, na=False)]

    if len(df) == 0:
        return np.array([], dtype=float), np.array([], dtype=str)

    return df["wav"].to_numpy(float), df["id"].to_numpy(str)

try:
    old_wav, old_ids = load_moore_lines(OLD_LINE_CSV)
    old_wav, old_ids = bin_lines(old_wav, old_ids, bin_A=0.9)
    print(f"[INFO] Moore CSV={OLD_LINE_CSV}  n={0 if old_wav is None else len(old_wav)}", flush=True)
except Exception as e:
    print(f"[WARN] Moore line CSV not loaded: {e}", flush=True)
    old_wav = old_ids = None

try:
    new_wav, new_ids = load_ia_lines(NEW_LINE_CSV)
    print(f"[INFO] IA CSV={NEW_LINE_CSV}  n={0 if new_wav is None else len(new_wav)}", flush=True)
except Exception as e:
    print(f"[WARN] IA line CSV not loaded: {e}", flush=True)
    new_wav = new_ids = None

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
    unit: str,
    flux: str = "norm",
) -> bytes:
    """Render [start,end] slice (selection in Å). unit affects plotting only."""
    unit = (unit or "A").strip().lower()
    plot_in_nm = unit in ("nm", "nanometer", "nanometers")

    flux = (flux or "norm").strip().lower()
    plot_cgs = (flux != "norm")
    if flux == "norm":
        w, y_base = fetch_ispy_air_norm(start, end)
    else:
        w, y_base = fetch_ispy_air_cgs(start, end)

    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        figsize=(12, 4.8),
        gridspec_kw={"height_ratios": [2, 1]},
        constrained_layout=True,
    )

    if w.size == 0:
        ax1.text(0.5, 0.5, "Empty slice", ha="center", va="center", transform=ax1.transAxes)
        ax1.axis("off")
        ax2.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=DPI)
        plt.close(fig)
        gc.collect()
        return buf.getvalue()

    twav, tint = TELLURICS.get(int(alt_m), TELLURICS[2500])

    # Interpolate tellurics onto atlas wavelength cut
    t_seg = np.interp(w, twav, tint)

    # ---- PHYSICALLY CORRECT ORDER ----
    y_highres = np.asarray(y_base * t_seg, dtype=float)
    y_final   = np.asarray(apply_resolution_R500(w, y_highres, R500), dtype=float)

    if flux == "flam":
        y_final = fnu_to_flam(w, y_final)

    # tellurics for DISPLAY: convolve telluric transmission too
    t_disp = np.asarray(apply_resolution_R500(w, t_seg, R500), dtype=float)

    # Convert plotting units (data selection remains in Å)
    w_plot     = (w / 10.0) if plot_in_nm else w
    start_plot = (start / 10.0) if plot_in_nm else start
    end_plot   = (end / 10.0) if plot_in_nm else end
    unit_label = "nm" if plot_in_nm else "Å"

    ax1.set_xlim(start_plot, end_plot)

    if plot_cgs:
        # Spectrum on left axis.
        ax1.plot(w_plot, y_final, color="black", lw=2.0, zorder=2, label="Spectrum" if legend_on else None)

        # Dynamic y padding to move the spectrum down (headroom for labels)
        ymin = float(np.nanmin(y_final))
        ymax = float(np.nanmax(y_final))

        if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
            span = ymax - ymin
            top_pad = 0.20 * span
            bot_pad = 0.02 * span
            ax1.set_ylim(ymin - bot_pad, ymax + top_pad)
        else:
            ax1.set_ylim(0, 1.0)

        # Y label depends on flux mode
        if flux == "flam":
            ax1.set_ylabel("Intensity (cgs; per Å)")
        else:
            ax1.set_ylabel("Intensity (cgs; per Hz)")

        # Tellurics axis: create it whenever tellurics OR label-lines need it.
        axT = None
        if tellurics_on or labels_on:
            axT = ax1.twinx()
            axT.set_ylim(0.0, 1.18)  # headroom for the red curve
            axT.set_ylabel("Telluric transmission")

        # Tellurics curve (red) lives on the telluric axis in cgs/flam modes
        if tellurics_on and axT is not None:
            axT.plot(
                w_plot, t_disp,
                color="red", lw=1.0, zorder=3,
                label="Tellurics" if legend_on else None
            )

        # Legend: only merge handles if tellurics were actually plotted
        if legend_on:
            h1, l1 = ax1.get_legend_handles_labels()
            if tellurics_on and axT is not None:
                h2, l2 = axT.get_legend_handles_labels()
                ax1.legend(
                    h1 + h2, l1 + l2,
                    loc="upper right",
                    frameon=True,
                    facecolor="white",
                    edgecolor="black",
                    framealpha=1.0,
                    fontsize=10,
                )
            else:
                ax1.legend(
                    loc="upper right",
                    frameon=True,
                    facecolor="white",
                    edgecolor="black",
                    framealpha=1.0,
                    fontsize=10,
                )

    else:
        if legend_on:
            # Keep your original behavior (legend implies tellurics drawn)
            ax1.plot(w_plot, t_disp,  color="red",   lw=1.0, zorder=3, label="Tellurics")
            ax1.plot(w_plot, y_final, color="black", lw=2.0, zorder=2, label="Spectrum")
        else:
            if tellurics_on:
                ax1.plot(w_plot, t_disp, color="red", lw=1.0, zorder=3)
            ax1.plot(w_plot, y_final, color="black", lw=2.0, zorder=2)

        ax1.set_ylim(0, 1.20)
        ax1.set_ylabel("Normalized intensity")

        if legend_on:
            ax1.legend(loc="upper right", frameon=True, facecolor="white",
                       edgecolor="black", framealpha=1.0, fontsize=10)

        axT = None  # not used in norm mode

    r_txt = "∞" if (np.isfinite(R500) and R500 >= 1e8) else f"{R500:g}"
    flux_txt = "cgs" if plot_cgs else "norm"
    ax1.set_title(f"{start_plot:.3f}–{end_plot:.3f} {unit_label}   (R={r_txt}, alt={alt_m} m, flux={flux_txt})")

    # ----------------------------
    # Line overlays (gated by labels_on)
    # ----------------------------
    if labels_on:
        MAX_LABELS = 60

        if old_wav is not None:
            mask = (old_wav >= start) & (old_wav <= end)
            pw = old_wav[mask]
            pi = old_ids[mask]
            if pw.size > MAX_LABELS:
                pw = pw[:MAX_LABELS]
                pi = pi[:MAX_LABELS]
            for x, lab in zip(pw, pi):
                x_plot = (x / 10.0) if plot_in_nm else x

                # Vertical label lines:
                # - norm: keep exactly as before on ax1 (0..1 in data; your norm ylim is 0..1.2)
                # - cgs/flam: draw on telluric axis and stop at 1.05 transmission
                if plot_cgs and (axT is not None):
                    axT.plot([x_plot, x_plot], [0.0, 1.0], lw=0.4, alpha=0.5, zorder=0, color="k")
                else:
                    ax1.plot([x_plot, x_plot], [0.0, 1.0], lw=0.4, alpha=0.5, zorder=0, color="k")

                ax1.text(
                    x_plot, 0.84, lab,
                    transform=ax1.get_xaxis_transform(),
                    rotation=45,
                    fontsize=8,
                    ha="center",
                    va="bottom",
                    color="k",
                )

        if new_wav is not None:
            mask = (new_wav >= start) & (new_wav <= end)
            pw = new_wav[mask]
            pi = new_ids[mask]
            if pw.size > MAX_LABELS:
                pw = pw[:MAX_LABELS]
                pi = pi[:MAX_LABELS]
            for x, lab in zip(pw, pi):
                x_plot = (x / 10.0) if plot_in_nm else x

                if plot_cgs and (axT is not None):
                    axT.plot([x_plot, x_plot], [0.0, 1.05], lw=0.4, alpha=0.7, zorder=0, color="k")
                else:
                    ax1.plot([x_plot, x_plot], [0.0, 1.0], lw=0.4, alpha=0.7, zorder=0, color="k")

                ax1.text(
                    x_plot, 0.84, lab,
                    transform=ax1.get_xaxis_transform(),
                    rotation=45,
                    fontsize=8,
                    ha="center",
                    va="bottom",
                )

    # 2D strip: STRICTLY from y_final (after all processing)
    # 2D strip is always shown normalized for contrast.
    y_strip = np.asarray(y_final, dtype=float)
    p1 = float(np.nanpercentile(y_strip, 1))
    p99 = float(np.nanpercentile(y_strip, 99))
    if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
        p1, p99 = float(np.nanmin(y_strip)), float(np.nanmax(y_strip))
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

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    plt.close(fig)
    gc.collect()
    return buf.getvalue()

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI()

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
    alt: Optional[int] = 2500,   # 0 or 2500 (meters)
    labels: Optional[int] = 1,   # 0/1
    legend: Optional[int] = 0,   # 0/1
    tellurics: Optional[int] = 1,   # 0/1
    unit: Optional[str] = "A",   # 'A' or 'nm' (plotting only)
    flux: Optional[str] = "norm",  # 'norm' or 'cgs' or 'flam'
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

        png = render_segment_png(
            start, end,
            R500=R500,
            alt_m=alt_m,
            labels_on=labels_on,
            legend_on=legend_on,
            tellurics_on=tellurics_on,
            unit=unit,
            flux=flux,
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
    alt: Optional[int] = 2500,   # 0 or 2500 (meters)
    labels: Optional[int] = 1,   # unused; kept for symmetry
    legend: Optional[int] = 0,   # unused; kept for symmetry
    unit: Optional[str] = "A",   # output unit: 'A' or 'nm'
    flux: Optional[str] = "norm",  # 'norm' or 'cgs' or 'flam'
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

        # data in Å
        flux = (flux or "norm").strip().lower()
        plot_cgs = (flux != "norm")
        if flux == "norm":
            w, y_base = fetch_ispy_air_norm(start, end)
        else:
            w, y_base = fetch_ispy_air_cgs(start, end)

        if w.size == 0:
            return PlainTextResponse("# empty slice\n", status_code=200)

        twav, tint = TELLURICS.get(int(alt_m), TELLURICS[2500])
        t_seg = np.interp(w, twav, tint)

        # Physically correct final spectrum
        y_highres = np.asarray(y_base * t_seg, dtype=float)
        y_final   = np.asarray(apply_resolution_R500(w, y_highres, R500), dtype=float)

        if flux == "flam":
            y_final = fnu_to_flam(w, y_final)

        # Output units
        u = (unit or "A").strip().lower()
        if u in ("nm", "nanometer", "nanometers"):
            w_out = w / 10.0
            unit_label = "nm"
        else:
            w_out = w
            unit_label = "A"

        col = "y_final_cgs" if plot_cgs else "y_final_norm"
        lines = [f"# wavelength[{unit_label}]\t{col}"]
        lines += [f"{ww:.6f}\t{yy:.8f}" for ww, yy in zip(w_out, y_final)]
        return PlainTextResponse("\n".join(lines) + "\n", status_code=200)

    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        return PlainTextResponse(tb, status_code=500)
