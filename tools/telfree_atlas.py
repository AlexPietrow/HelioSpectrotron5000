"""
Telluric-free version of the Hamburg FTS disk-center atlas (ISPy) using the
HS5000 TAPAS 2500 m combined transmission (telat_alt1.npy).

Per 20 A chunk: one tau scale a, one velocity shift v and one Gaussian
broadening s (km/s), fitted on the isolated TAPAS lines in that chunk with a
local linear continuum per line and a robust loss. The atlas is divided by the
model, smoothed only under the tellurics, and flagged.

Outputs (OUT_DIR):
  telfree_atlas.npz          wav, atlas, corrected, smoothed, T_model, flag
  telfree_atlas.npy          compact float32 table for HS5000 (wav, I/I_c, flag)
  telfree_params.txt         per-chunk fit parameters
Usage: ipython3 telfree_atlas.py
"""
import os
import time
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter1d, percentile_filter
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.interpolate import make_smoothing_spline

ATLAS_FILE = None  # None = from installed ISPy
TAPAS_NPY = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'telat_alt1.npy')
OUT_DIR = 'telfree_out'
CHUNK = 20.0           # A
MIN_DEPTH = 0.01       # TAPAS line depth to count as a fit line
ISOLATION = 0.005      # no other TAPAS line deeper than this inside the fit window
MAX_LINES = 8
C_KMS = 299792.458
SEG_EDGES = [3909, 4558, 5798, 7487, 8968]   # inferred from the wavenumber sampling
O2_BANDS = [(6270, 6330, 'O2 gamma'), (6860, 6960, 'O2 B'), (7590, 7710, 'O2 A')]

T0 = time.time()


def log(msg):
    print(f'[{time.time() - T0:7.1f} s] {msg}', flush=True)


def load_atlas(path):
    if path is None:
        import ISPy
        path = os.path.join(os.path.dirname(ISPy.__file__), 'data', 'fts_disk_center_SI.idlsave')
    d = sio.readsav(path)
    return np.asarray(d['ftswav'], float), np.asarray(d['ftsint_SI'] / d['ftscnt_SI'], float)


def tapas_lines(path):
    t = np.load(path).astype(float)
    w = t[0]
    tau = -np.log(np.clip(t[1], 1e-9, 1.0))
    # Smooth Rayleigh/O3/continuum part of tau: minimum per 1 A bin, 1st percentile over 400 A,
    # then smoothed over 100 A. The window is wider than any band, so O2/H2O band
    # pseudo-continua (overlapping line wings) stay in the line opacity.
    edges = np.arange(w[0], w[-1] + 1.0, 1.0)
    idx = np.digitize(w, edges) - 1
    bmin = np.full(edges.size, np.nan)
    np.fmin.at(bmin, idx, tau)
    bc = edges + 0.5
    ok = np.isfinite(bmin)
    bmin = np.interp(bc, bc[ok], bmin[ok])
    base = gaussian_filter1d(percentile_filter(bmin, 1, size=401), 100, mode='nearest')
    tau_s = np.interp(w, bc, base)
    return w, np.clip(tau - tau_s, 0.0, None)


def tmodel(x, tw, tt, a, v, s):
    lam = x / (1 + v / C_KMS)
    T = np.exp(-a * np.interp(lam, tw, tt, left=0.0, right=0.0))
    sig_pix = s / C_KMS * np.median(x) / np.median(np.diff(x))
    return gaussian_filter1d(T, sig_pix, mode='nearest') if sig_pix > 0.05 else T


def pick_lines(tw, tt, lo, hi, hw):
    m = (tw > lo) & (tw < hi)
    w, d = tw[m], 1 - np.exp(-0.5 * tt[m])   # depth at a typical scale of 0.5
    strong = find_peaks(d, height=MIN_DEPTH)[0]
    weak = find_peaks(d, height=ISOLATION)[0]
    good = []
    for p in strong:
        lc = w[p]
        if lc - hw < lo or lc + hw > hi:
            continue
        others = w[weak][(np.abs(w[weak] - lc) < hw) & (weak != p)]
        if others.size == 0:
            good.append((d[p], lc))
    good.sort(reverse=True)
    return sorted(lc for _, lc in good[:MAX_LINES])


def fit_chunk(aw, af, tw, tt, centers, hw):
    # skip candidates sitting on deep solar lines
    centers = [lc for lc in centers if np.median(af[np.abs(aw - lc) < hw]) > 0.6]
    if len(centers) < 2:
        return None, None, np.nan, 0
    wins = [np.abs(aw - lc) < hw for lc in centers]

    def resid(p, use):
        out = []
        for i, (lc, m) in enumerate(zip(centers, wins)):
            if not use[i]:
                continue
            c0, c1 = p[3 + 2 * i: 5 + 2 * i]
            x = aw[m]
            out.append(af[m] - (c0 + c1 * (x - lc)) * tmodel(x, tw, tt, *p[:3]))
        return np.concatenate(out)

    p0 = [0.5, 0.0, 0.3]
    lb = [0.0, -3.0, 0.0]
    ub = [5.0, 3.0, 3.0]
    for m in wins:
        p0 += [float(np.clip(np.median(af[m]), 0.31, 1.49)), 0.0]
        lb += [0.3, -2.0]
        ub += [1.5, 2.0]
    use = np.ones(len(centers), bool)
    sol = least_squares(resid, p0, bounds=(lb, ub), x_scale='jac', loss='soft_l1', f_scale=0.005, args=(use,))
    # drop lines that are badly fitted (likely solar blends) and refit once
    rms = np.array([np.std(af[m] - (sol.x[3 + 2 * i] + sol.x[4 + 2 * i] * (aw[m] - lc)) *
                           tmodel(aw[m], tw, tt, *sol.x[:3])) for i, (lc, m) in enumerate(zip(centers, wins))])
    use = rms < 3 * np.median(rms) + 1e-4
    if use.sum() >= 2 and use.sum() < len(centers):
        sol = least_squares(resid, sol.x, bounds=(lb, ub), x_scale='jac', loss='soft_l1', f_scale=0.005, args=(use,))
    f = resid(sol.x, use)
    s2 = np.sum(f ** 2) / max(f.size - len(p0), 1)
    err = np.sqrt(np.abs(np.diag(np.linalg.pinv(sol.jac.T @ sol.jac) * s2)))
    return sol.x[:3], err[:3], np.sqrt(s2), int(use.sum())


def smooth_under(x, corr, T):
    out = corr.copy()
    peaks = find_peaks(1 - T, height=0.01)[0]
    for n, pk in enumerate(peaks):
        if n % 2000 == 0:
            log(f'  smoothing line {n + 1}/{peaks.size} at {x[pk]:.0f} A')
        hw = max(0.08, x[pk] * 1.2e-5)
        s = np.abs(x - x[pk]) < hw
        good = s & np.isfinite(corr) & (T > 0.1)
        if good.sum() < 8:
            continue
        try:
            spl = make_smoothing_spline(x[good], corr[good], w=T[good] ** 2)
        except Exception:
            continue
        # blend weight: high under the telluric, tapered to zero at the window edges (no steps)
        taper = np.cos(0.5 * np.pi * np.abs(x[s] - x[pk]) / hw) ** 2
        b = np.clip((1 - T[s]) / 0.02, 0, 1) * taper
        out[s] = (1 - b) * corr[s] + b * spl(x[s])
    return out


def band_label(lc):
    for lo, hi, name in O2_BANDS:
        if lo <= lc <= hi:
            return name
    return '-'


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    log('reading atlas and TAPAS')
    aw, af = load_atlas(ATLAS_FILE)
    tw, tt = tapas_lines(TAPAS_NPY)
    edges = np.arange(aw[0], min(aw[-1], tw[-1]) + CHUNK, CHUNK)
    nch = len(edges) - 1
    rows = []
    T_all = np.ones_like(af)
    have = np.zeros(af.size, bool)
    for k in range(nch):
        lo, hi = edges[k], min(edges[k + 1], tw[-1])
        m = (aw >= lo) & (aw < hi)
        if not m.any():
            continue
        hw = max(0.08, 0.5 * (lo + hi) * 1.2e-5)
        cen = pick_lines(tw, tt, lo, hi, hw)
        maxdepth = 1 - np.exp(-0.5 * np.max(tt[(tw > lo) & (tw < hi)], initial=0))
        if len(cen) >= 2:
            ts = (tw > lo - 2) & (tw < hi + 2)
            par, err, rms, nused = fit_chunk(aw, af, tw[ts], tt[ts], cen, hw)
            status = 'fit' if par is not None else 'nofit'
        else:
            par, err, rms, nused = None, None, np.nan, 0
            status = 'nofit'
        rows.append([0.5 * (lo + hi), lo, hi, len(cen), nused, maxdepth, par, err, rms, status])
        if k % 10 == 0 or k == nch - 1:
            log(f'chunk {k + 1}/{nch} {lo:.0f}-{hi:.0f} A: {status}, {nused} lines'
                + (f', a={par[0]:.3f} v={par[1]:+.2f} s={par[2]:.2f} km/s' if par is not None else ''))
    # quality cut; rejected or unfitted chunks take the nearest good chunk in the same FTS segment
    seg_of = lambda lam: int(np.searchsorted(SEG_EDGES, lam))
    for r in rows:
        par, err = r[6], r[7]
        good = (par is not None and r[4] >= 3 and err[0] < 0.05 and abs(par[1]) < 1.5
                and par[2] < 1.5 and err[1] < 0.5)
        r.append('good' if good else ('rejected' if par is not None else 'nofit'))
    goods = [r for r in rows if r[10] == 'good']
    log(f'{len(goods)}/{len(rows)} chunks pass the quality cut, applying model')
    for r in rows:
        if r[10] != 'good':
            same = [g for g in goods if seg_of(g[0]) == seg_of(r[0])] or goods
            j = int(np.argmin([abs(g[0] - r[0]) for g in same]))
            r[11:] = [same[j][6], same[j][0]]
        else:
            r[11:] = [r[6], r[0]]
        m = (aw >= r[1]) & (aw < r[2])
        T_all[m] = tmodel(aw[m], tw, tt, *r[11])
        have[m] = True
    corr = af / T_all
    corr[T_all < 0.1] = np.nan
    log('smoothing under tellurics')
    smooth = smooth_under(aw, corr, T_all)
    flag = np.zeros(af.size, int)
    flag[T_all < 0.98] = 1
    flag[T_all < 0.90] = 2
    flag[T_all < 0.10] = 3
    flag[~have] = -1        # beyond TAPAS coverage, not corrected
    corr[~have] = af[~have]
    smooth[~have] = af[~have]

    np.savez_compressed(os.path.join(OUT_DIR, 'telfree_atlas.npz'), wav=aw, atlas=af, corrected=corr,
                        smoothed=smooth, T_model=T_all, flag=flag)
    with open(os.path.join(OUT_DIR, 'telfree_params.txt'), 'w') as f:
        f.write('# per-chunk telluric fit, HS5000 TAPAS 2500 m combined transmission\n'
                '# fitted values (a, v, s with errors) are NaN where no fit was possible\n'
                '# quality: good | rejected (failed quality cut) | nofit (fewer than 2 isolated lines)\n'
                '# a_used, v_used, s_used: parameters applied to this chunk; from_lam: chunk they were taken from\n'
                '# segment: FTS segment index from the inferred edges ' + str(SEG_EDGES) + '\n'
                '# lam_c lo hi n_cand n_used max_tapas_depth a a_err v_kms v_err s_kms s_err rms quality '
                'a_used v_used s_used from_lam segment band\n')
        for r in rows:
            par = r[6] if r[6] is not None else [np.nan] * 3
            err = r[7] if r[7] is not None else [np.nan] * 3
            u = r[11]
            f.write(f'{r[0]:.1f} {r[1]:.1f} {r[2]:.1f} {r[3]} {r[4]} {r[5]:.4f} {par[0]:.4f} {err[0]:.4f} '
                    f'{par[1]:.4f} {err[1]:.4f} {par[2]:.4f} {err[2]:.4f} {r[8]:.5f} {r[10]} '
                    f'{u[0]:.4f} {u[1]:.4f} {u[2]:.4f} {r[12]:.1f} {seg_of(r[0])} {band_label(r[0]).replace(" ", "_")}\n')
    # compact table for HS5000 (backend.py, TELFREE_FILE): wavelength, I/I_c, flag as float32,
    # with saturated (NaN) pixels linearly interpolated so solar * TAPAS stays finite
    ok = np.isfinite(smooth)
    hs = smooth.copy()
    hs[~ok] = np.interp(aw[~ok], aw[ok], smooth[ok])
    np.save(os.path.join(OUT_DIR, 'telfree_atlas.npy'),
            np.vstack([aw, hs, flag]).astype(np.float32))
    log(f'wrote {OUT_DIR}/telfree_atlas.npz, telfree_atlas.npy (for HS5000) and telfree_params.txt')
    log('done')


if __name__ == '__main__':
    run()
