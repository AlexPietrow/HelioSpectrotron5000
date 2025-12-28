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
      y_final = (solar * telluric) convolved by LSF
- 2D strip is ALWAYS from y_final (post-processing), no exceptions.

Frontend contract:
- /segment.png accepts start/width in Å by default.
- If unit=nm, frontend may send nm; backend converts to Å internally.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Use a non-interactive backend
import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from fastapi import FastAPI, Response, Query
from fastapi.responses import HTMLResponse

# Optional: ISPy atlas (you already vendor it as ./ISPy)
from ISPy.spec import atlas as ispy_atlas

app = FastAPI()

HERE = Path(__file__).resolve().parent

# -----------------------
# CONFIG
# -----------------------
DEFAULT_WIDTH_A = 20.0
DEFAULT_R500 = 200_000.0
REPEAT_2D = 40

# Telluric file env vars (optional)
TELL_FILE_ALT0 = os.environ.get("TELL_FILE_ALT0", str(HERE / "telat_alt0.npy"))
TELL_FILE_ALT1 = os.environ.get("TELL_FILE_ALT1", str(HERE / "telat_alt1.npy"))

# Line list CSVs (optional; keep your existing names)
OLD_LINE_CSV = HERE / "all_pages_clean_filtered_clean.csv"
NEW_LINE_CSV = HERE / "all_pages_lines_wav_strength_id.csv"

# -----------------------
# Helpers
# -----------------------
def _load_telat(path: str) -> Tuple[np.ndarray, np.ndarray]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Telluric file not found: {p}\n"
            f"Put it next to backend.py, or set env vars:\n"
            f"  TELL_FILE_ALT0=/path/to/telat_alt0.npy\n"
            f"  TELL_FILE_ALT1=/path/to/telat_alt1.npy\n"
        )

    arr = np.load(p, allow_pickle=False)
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[0] != 2:
        raise ValueError(f"Telluric file must have shape (2, N). Got {arr.shape} from {p}")

    wav_A = np.asarray(arr[0], dtype=float)
    trans = np.asarray(arr[1], dtype=float)

    # sort by wavelength if needed
    if np.any(np.diff(wav_A) < 0):
        idx = np.argsort(wav_A)
        wav_A = wav_A[idx]
        trans = trans[idx]

    return wav_A, trans


def _load_lines_csv(path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not path.exists():
        return None, None
    try:
        import pandas as pd
        df = pd.read_csv(path)
    except Exception:
        return None, None

    # Heuristic columns
    wav_col = None
    id_col = None
    for c in df.columns:
        lc = c.lower()
        if wav_col is None and ("wav" in lc or "lambda" in lc or lc in ("wavelength", "wave", "wl")):
            wav_col = c
        if id_col is None and (lc in ("id", "ident", "identifier", "species") or "id" in lc):
            id_col = c

    if wav_col is None:
        return None, None

    wav = np.asarray(df[wav_col], dtype=float)
    ids = np.asarray(df[id_col], dtype=str) if id_col is not None else np.asarray([""] * len(wav), dtype=str)

    # cleanup obvious nans
    m = np.isfinite(wav)
    return wav[m], ids[m]


def _gaussian_kernel(sigma_pix: float, half_width: int = 50) -> np.ndarray:
    if sigma_pix <= 0:
        return np.array([1.0], dtype=float)
    x = np.arange(-half_width, half_width + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma_pix) ** 2)
    k /= np.sum(k)
    return k


def _convolve(y: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if kernel.size == 1:
        return y
    return np.convolve(y, kernel, mode="same")


def _interp(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    # safe linear interp with edge clamp
    return np.interp(x, xp, fp, left=fp[0], right=fp[-1])


# -----------------------
# Load static resources
# -----------------------
TEL_WAV_A_0, TEL_TRANS_0 = _load_telat(TELL_FILE_ALT0)
TEL_WAV_A_1, TEL_TRANS_1 = _load_telat(TELL_FILE_ALT1)

old_wav, old_ids = _load_lines_csv(OLD_LINE_CSV)
new_wav, new_ids = _load_lines_csv(NEW_LINE_CSV)

# Preload ISPy atlas once
atlas = ispy_atlas.satlas()

# -----------------------
# Core render
# -----------------------
def render_segment_png(start: float, end: float, R500: float, alt_m: int, labels_on: bool, legend_on: bool, unit: str) -> bytes:
    # Wavelength grid in Å for rendering
    n = 2200  # fixed resolution in the image; frontend "zoom" changes start/width
    w = np.linspace(start, end, n, dtype=float)

    # Solar atlas intensity
    # ISPy atlas expects nm in some contexts, but satlas().getatlas returns wavelengths in Å (ISPy does).
    # We'll use satlas.getatlas which returns (wav_A, intensity).
    wa, Ia = atlas.getatlas()  # full atlas arrays
    y_solar = _interp(w, wa, Ia)

    # Telluric transmission (choose altitude)
    if int(alt_m) == 0:
        t_seg = _interp(w, TEL_WAV_A_0, TEL_TRANS_0)
    else:
        t_seg = _interp(w, TEL_WAV_A_1, TEL_TRANS_1)

    # Multiply (physical)
    y_mul = y_solar * t_seg

    # Convolution: translate R@500nm into sigma in pixels approximately
    # sigma_lambda ≈ (lambda / R) / (2*sqrt(2 ln 2)) for gaussian (FWHM -> sigma)
    lam_ref = 5000.0  # Å
    if np.isfinite(R500) and R500 > 0:
        fwhm_A = lam_ref / R500
        sigma_A = fwhm_A / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    else:
        sigma_A = 0.0
    dw = (end - start) / (n - 1)
    sigma_pix = sigma_A / dw if dw > 0 else 0.0
    k = _gaussian_kernel(sigma_pix, half_width=80)
    y_final = _convolve(y_mul, k)

    # Figure
    fig = plt.figure(figsize=(10, 4.8), dpi=120)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 1], hspace=0.10)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Unit handling: internal wavelength is Å. Optionally display in nm.
    u = (unit or "A").strip().lower()
    use_nm = u in ("nm", "nanometer", "nanometers")
    w_plot = w / 10.0 if use_nm else w
    start_plot = start / 10.0 if use_nm else start
    end_plot = end / 10.0 if use_nm else end

    # Plot tellurics (red behind) and final spectrum (black)
    ln_tell, = ax1.plot(w_plot, t_seg,   color="red",   lw=1.0, zorder=3, label="Tellurics")
    ln_spec, = ax1.plot(w_plot, y_final, color="black", lw=2.0, zorder=2, label="Solar × telluric ⊗ LSF")

    ax1.set_xlim(start_plot, end_plot)
    ax1.set_ylim(0, 1.20)
    ax1.set_ylabel("Normalized intensity")

    r_txt = "∞" if (np.isfinite(R500) and R500 >= 1e8) else f"{R500:g}"
    xunit = "nm" if use_nm else "Å"
    ax1.set_title(f"{start_plot:.2f}–{end_plot:.2f} {xunit}   (R@500nm={r_txt}, alt={alt_m} m)")

    if legend_on:
        ax1.legend(loc="upper right", frameon=False, fontsize=9)

    # ----------------------------
    # Line overlays (gated by labels_on)
    # ----------------------------
    if labels_on:
        MAX_LABELS = 60

        def _x(val_A: float) -> float:
            return val_A / 10.0 if use_nm else val_A

        if old_wav is not None:
            mask = (old_wav >= start) & (old_wav <= end)
            pw = old_wav[mask]
            pi = old_ids[mask]
            if pw.size > MAX_LABELS:
                pw = pw[:MAX_LABELS]
                pi = pi[:MAX_LABELS]
            for xA, lab in zip(pw, pi):
                x = _x(float(xA))
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
            for xA, lab in zip(pw, pi):
                x = _x(float(xA))
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
        extent=[w_plot[0], w_plot[-1], 0, 1.0],
    )
    ax2.set_xlim(start_plot, end_plot)
    ax2.set_xlabel(f"Wavelength [{xunit}]")
    ax2.set_yticks([])

    # Render
    bio = io.BytesIO()
    fig.savefig(bio, format="png", bbox_inches="tight")
    plt.close(fig)
    return bio.getvalue()


# -----------------------
# Routes
# -----------------------
@app.get("/", response_class=HTMLResponse)
def root():
    html = (HERE / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/segment.png")
def segment_png(
    start: float = Query(..., description="Start wavelength (Å by default; nm if unit=nm)"),
    width: Optional[float] = Query(None, description="Width (Å by default; nm if unit=nm)"),
    R500: Optional[float] = Query(None, description="Resolving power at 500 nm"),
    alt: Optional[int] = Query(2500, description="Altitude in m (0 or 2500)"),
    labels: Optional[int] = Query(1, description="0/1 show line labels"),
    legend: Optional[int] = Query(0, description="0/1 show legend"),
    unit: Optional[str] = Query("A", description='"A" (Å) or "nm"'),
):
    # inputs
    start = float(start)
    width = float(width) if width is not None else DEFAULT_WIDTH_A
    R500 = DEFAULT_R500 if (R500 is None) else float(R500)

    u = (unit or "A").strip().lower()
    use_nm = u in ("nm", "nanometer", "nanometers")
    if use_nm:
        # frontend provides nm; internal is Å
        start *= 10.0
        width *= 10.0

    end = start + width

    alt_m = int(alt) if alt is not None else 2500
    labels_on = bool(int(labels)) if labels is not None else True
    legend_on = bool(int(legend)) if legend is not None else False

    png = render_segment_png(start, end, R500=R500, alt_m=alt_m, labels_on=labels_on, legend_on=legend_on, unit=unit)
    return Response(content=png, media_type="image/png")
