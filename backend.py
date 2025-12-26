#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectrum scroller backend (FastAPI)

SkyCalc removed.
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
- Plot styling:
  1) spectrum black lw=2
  2) tellurics red lw=1 behind spectrum
  3) no legend
  4) 2D grayscale

Extras:
- labels toggle (labels=0/1) controls line-name overlay
- hard-coded line lists (same as your working atlas.py):
    csv/all_pages_clean_filtered.csv   (Moore-like: wavelength, ew, id)
    csv2/all_pages_lines_whole.csv     (IA/strength: wav, strength, id)

Run:
  uvicorn backend:app --reload --port 8000
"""

import io
import os
import gc
import traceback
from typing import Optional, Tuple, Dict

import numpy as np

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, PlainTextResponse

from ISPy.spec import atlas as ispy_atlas


# ----------------------------
# CONFIG
# ----------------------------
GLOBAL_WMIN_A = 3000.0
GLOBAL_WMAX_A = 12000.0

DEFAULT_WIDTH_A = 25.0
OVERLAP = 0.10
DEFAULT_STEP_A = DEFAULT_WIDTH_A * (1.0 - OVERLAP)

# Default "no smoothing" (treated as "∞" by apply_resolution_R500)
DEFAULT_R500 = 1e12

DPI = 160
REPEAT_2D = 120

# Resolve paths relative to this file
HERE = os.path.dirname(os.path.abspath(__file__))

# Telluric files (TAPAS precomputed)
TELL_FILE_ALT0 = os.environ.get("TELL_FILE_ALT0", os.path.join(HERE, "telat_alt0.npy"))  # 0 m
TELL_FILE_ALT1 = os.environ.get("TELL_FILE_ALT1", os.path.join(HERE, "telat_alt1.npy"))  # 2500 m

# Frontend
INDEX_HTML = os.environ.get("INDEX_HTML", os.path.join(HERE, "index.html"))

# Line lists (HARD-CODED like your working atlas.py)
OLD_LINE_CSV = os.path.join(HERE, "csv",  "all_pages_clean_filtered_clean.csv")
NEW_LINE_CSV = os.path.join(HERE, "csv2", "all_pages_lines_wav_strength_id.csv")


# ----------------------------
# ISPy atlas (lazy to avoid reload thrash)
# ----------------------------
_fts = None

def get_fts():
    global _fts
    if _fts is None:
        _fts = ispy_atlas.atlas()
    return _fts

def fetch_ispy_air_norm(w0: float, w1: float):
    """Return wavelength [Å] and normalized intensity for [w0, w1]."""
    pad = 0.5
    fts = get_fts()
    wav, I, cont = fts.get(w0 - pad, w1 + pad, cgs=True, nograv=True, perHz=True)
    wav = np.asarray(wav, dtype=float)
    I = np.asarray(I, dtype=float)
    cont = np.asarray(cont, dtype=float)

    cont_safe = np.where(cont > 0, cont, np.nan)
    I_norm = I / cont_safe
    I_norm = np.clip(I_norm, 0, np.nanmax(I_norm))

    sel = (wav >= w0) & (wav <= w1)
    return wav[sel], I_norm[sel]


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
    """
    Degrade y(w) by Gaussian with FWHM defined from resolving power at 500 nm:
      FWHM_A = 5000 / R500  (Å)
    """
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

_wmins = [float(np.nanmin(TELLURICS[k][0])) for k in TELLURICS]
_wmaxs = [float(np.nanmax(TELLURICS[k][0])) for k in TELLURICS]
WMIN = max(GLOBAL_WMIN_A, max(_wmins))
WMAX = min(GLOBAL_WMAX_A, min(_wmaxs))

if not (np.isfinite(WMIN) and np.isfinite(WMAX) and WMIN < WMAX):
    raise RuntimeError(f"Invalid telluric wavelength intersection: WMIN={WMIN}, WMAX={WMAX}")


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
    """Try comma, then semicolon. Returns DataFrame."""
    import pandas as pd
    df = pd.read_csv(path)
    if len(df.columns) == 1 and ";" in str(df.columns[0]):
        df = pd.read_csv(path, sep=";")
    return df

def load_moore_lines(path: str):
    """
    Moore list CSV expected columns: wavelength, ew, id
    Filters: ew >= 0, ew > 5, remove 'atm'
    Binning: 1 Å integer bin, keep strongest EW per bin
    """
    import pandas as pd
    if not path or not os.path.exists(path):
        print(f"[LINES] Moore missing: {path}", flush=True)
        return None, None

    df = _read_csv_auto(path)
    if not all(c in df.columns for c in ("wavelength", "ew", "id")):
        raise ValueError(f"Moore CSV columns missing. Found: {list(df.columns)}")

    df["wavelength"] = df["wavelength"].apply(clean_wavelength)
    df["ew"]         = df["ew"].apply(clean_ew)
    df["id"]         = df["id"].fillna("").astype(str)

    df = df.dropna(subset=["wavelength", "ew"])
    df = df[df["ew"] >= 0]
    df = df[~df["id"].str.contains("atm", case=False, na=False)]
    df = df[df["ew"] > 5.0]

    if len(df) == 0:
        return np.array([], dtype=float), np.array([], dtype=str)

    df["bin"] = np.floor(df["wavelength"]).astype(int)
    idx_max = df.groupby("bin")["ew"].idxmax()
    df = df.loc[idx_max].copy().sort_values("wavelength")

    return df["wavelength"].to_numpy(float), df["id"].to_numpy(str)

def load_ia_lines(path: str):
    """
    IA/strength CSV expected columns: wav, strength, id
    Filters: strength >= -5, remove 'atm'
    """
    import pandas as pd
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
def render_segment_png(start: float, end: float, R500: float, alt_m: int, labels_on: bool) -> bytes:
    """
    alt_m: 0 or 2500 (meters)

    NOTE: Uses explicit Figure/Canvas (no pyplot) and closes deterministically to avoid
    "More than 20 figures have been opened" warnings in a long-running server.
    """
    fig = None
    buf = io.BytesIO()
    try:
        w, f_norm = fetch_ispy_air_norm(start, end)

        # Create figure explicitly (no pyplot state)
        fig = Figure(figsize=(12, 4.8), dpi=DPI)
        FigureCanvas(fig)  # attaches Agg renderer

        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        # Compact layout similar to constrained_layout without relying on pyplot
        fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.10, hspace=0.25)

        if w.size == 0:
            ax1.text(0.5, 0.5, "Empty slice", ha="center", va="center", transform=ax1.transAxes)
            ax1.axis("off")
            ax2.axis("off")
            fig.savefig(buf, format="png", dpi=DPI)
            return buf.getvalue()

        twav, tint = TELLURICS.get(int(alt_m), TELLURICS[2500])

        # Interpolate tellurics onto atlas wavelength cut
        t_seg = np.interp(w, twav, tint)

        # ---- PHYSICALLY CORRECT ORDER ----
        y_highres = np.asarray(f_norm * t_seg, dtype=float)
        y_final   = np.asarray(apply_resolution_R500(w, y_highres, R500), dtype=float)

        # Plot tellurics (red behind) and final spectrum (black)
        ax1.plot(w, t_seg,   color="red",   lw=1.0, zorder=3)
        ax1.plot(w, y_final, color="black", lw=2.0, zorder=2)

        ax1.set_xlim(start, end)
        ax1.set_ylim(0, 1.20)
        ax1.set_ylabel("Normalized intensity")
        r_txt = "∞" if (np.isfinite(R500) and R500 >= 1e8) else f"{R500:g}"
        ax1.set_title(f"{start:.2f}–{end:.2f} Å   (R@500nm={r_txt}, alt={alt_m} m)")

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
                    ax1.plot([x, x], [0.0, 1.0], lw=0.4, alpha=0.5, zorder=0, color="k")
                    ax1.text(
                        x, 0.84, lab,
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
                    ax1.plot([x, x], [0.0, 1.0], lw=0.4, alpha=0.7, zorder=0, color="k")
                    ax1.text(
                        x, 0.84, lab,
                        transform=ax1.get_xaxis_transform(),
                        rotation=45,
                        fontsize=8,
                        ha="center",
                        va="bottom",
                    )

        # 2D strip: STRICTLY from y_final (after all processing)
        img2d = np.tile(y_final[np.newaxis, :], (REPEAT_2D, 1))
        ax2.imshow(
            img2d,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="gray",
            extent=[w[0], w[-1], 0, 1.0],
        )
        ax2.set_xlim(start, end)
        ax2.set_xlabel("Wavelength [Å]")
        ax2.set_yticks([])

        fig.savefig(buf, format="png", dpi=DPI)
        return buf.getvalue()

    finally:
        # Deterministic cleanup for long-running server
        if fig is not None:
            fig.clear()
        gc.collect()


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
        f"WMIN={WMIN}\nWMAX={WMAX}\nDEFAULT_WIDTH_A={DEFAULT_WIDTH_A}\nDEFAULT_STEP_A={DEFAULT_STEP_A}\n"
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
    labels: Optional[int] = 1,   # 0/1 toggle from HTML
):
    try:
        start = float(start)
        width = float(width) if width is not None else DEFAULT_WIDTH_A
        R500 = DEFAULT_R500 if (R500 is None) else float(R500)

        # clamp
        width = max(0.1, min(width, WMAX - WMIN))
        end = min(start + width, WMAX)

        # only allow 0/2500
        alt_m = int(alt) if int(alt) in (0, 2500) else 2500

        if start < WMIN or start > WMAX:
            return Response(content=b"", media_type="image/png", status_code=416)

        labels_on = bool(int(labels)) if labels is not None else True

        png = render_segment_png(start, end, R500=R500, alt_m=alt_m, labels_on=labels_on)
        return Response(content=png, media_type="image/png")

    except Exception:
        tb = traceback.format_exc()
        return PlainTextResponse(tb, status_code=500)
