"""
One-time preprocessing: remove the smooth Rayleigh/ozone/continuum extinction from
the HS5000 TAPAS transmission tables (telat_alt0.npy, telat_alt1.npy), in place.

Only line and band absorption is kept, so solar x TAPAS stays continuum-normalized.
The smooth part of tau = -ln(T) is estimated as the minimum per 1 A bin (i.e. the
maximum transmission per bin), then the 1st percentile over 400 A, smoothed with a
100 A Gaussian. The window is wider than any O2 or H2O band, so band pseudo-continua
(overlapping line wings) are kept. Same recipe as tools/telfree_atlas.py.

The tables keep their format, shape (2, N) float32 (wavelength [A], transmission),
so backend.py loads them unchanged. A table that was already stripped is detected
by the STRIPPED marker file next to it and skipped.

Usage: ipython3 tools/strip_tapas.py
"""
import os
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
TABLES = ["telat_alt0.npy", "telat_alt1.npy"]
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f} s] {msg}", flush=True)


def gaussian_kernel_1d(sigma):
    radius = int(np.ceil(4.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def convolve_reflect(y, k):
    r = (k.size - 1) // 2
    ypad = np.pad(y, r, mode="reflect")
    return np.convolve(ypad, k, mode="valid")


def strip_smooth_extinction(w, t):
    edges = np.arange(w[0], w[-1] + 1.0, 1.0)
    starts = np.searchsorted(w, edges[:-1])
    keep = np.r_[np.diff(starts) > 0, True] & (starts < w.size)
    starts = starts[keep]
    centers = edges[:-1][keep] + 0.5
    tmax = np.maximum.reduceat(t, starts)
    bmin = -np.log(np.clip(tmax, 1e-9, 1.0))
    half = 200
    padded = np.pad(bmin, half, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    base = np.percentile(win, 1, axis=1)
    base = convolve_reflect(base, gaussian_kernel_1d(100.0))
    out = np.array(t, dtype=float, copy=True)
    step = 500_000
    for i in range(0, out.size, step):
        j = min(i + step, out.size)
        out[i:j] *= np.exp(np.interp(w[i:j], centers, base))
    np.minimum(out, 1.0, out=out)
    return out


def run():
    for name in TABLES:
        path = os.path.join(ROOT, name)
        marker = path + ".STRIPPED"
        if not os.path.exists(path):
            log(f"{name}: not found, skipping")
            continue
        if os.path.exists(marker):
            log(f"{name}: already stripped (marker present), skipping")
            continue
        log(f"{name}: loading")
        arr = np.load(path, allow_pickle=False)
        if arr.ndim != 2 or arr.shape[0] != 2:
            log(f"{name}: unexpected shape {arr.shape}, skipping")
            continue
        w = np.asarray(arr[0], dtype=float)
        t = np.asarray(arr[1], dtype=float)
        order = np.argsort(w) if not np.all(np.diff(w) > 0) else None
        if order is not None:
            w, t = w[order], t[order]
        log(f"{name}: {w.size} points, {w[0]:.1f}-{w[-1]:.1f} A, stripping")
        ts = strip_smooth_extinction(w, t)
        for lo, hi, lab in [(5000, 5010, "5000 A continuum"), (5880, 5900, "Na D / H2O"),
                            (6860, 6880, "O2 B band"), (7600, 7620, "O2 A band")]:
            m = (w > lo) & (w < hi)
            if m.any():
                log(f"  {lab:18s} max T {t[m].max():.4f} -> {ts[m].max():.4f}   "
                    f"min T {t[m].min():.4f} -> {ts[m].min():.4f}")
        out = np.vstack([w, ts]).astype(np.float32)
        np.save(path, out)
        with open(marker, "w") as f:
            f.write("smooth Rayleigh/ozone extinction removed by tools/strip_tapas.py\n")
        log(f"{name}: written ({out.nbytes / 1e6:.1f} MB)")
    log("done")


if __name__ == "__main__":
    run()
