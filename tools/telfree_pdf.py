"""
PDF comparison of the Hamburg disk-center atlas and its telluric-free version.
Page 1: per-chunk telluric parameters vs wavelength with inferred FTS segment edges.
Then: 10 A panels of atlas (gray), telluric-free (black) and the telluric model (blue).
Usage: ipython3 telfree_pdf.py   (after telfree_atlas.py)
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfWriter
import gc

OUT_DIR = 'telfree_out'
PANEL = 10.0            # A per panel
PER_PAGE = 6
BATCH = 20              # pages per temporary PDF (keeps memory low)
DPI = 150
SEG_EDGES = [3909, 4558, 5798, 7487, 8968]   # inferred from jumps in the wavenumber sampling
T0 = time.time()


def log(msg):
    print(f'[{time.time() - T0:7.1f} s] {msg}', flush=True)


def letter(ax, l):
    ax.text(0.02, 0.06, f'{l})', transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='none', pad=2))


def param_page(pdf):
    p = np.genfromtxt(os.path.join(OUT_DIR, 'telfree_params.txt'), dtype=None, encoding=None, comments='#')
    col = lambda i: np.array([r[i] for r in p])
    lam = col(0).astype(float)
    a, ae, v, ve, s, se = (col(i).astype(float) for i in (6, 7, 8, 9, 10, 11))
    q, band = col(13), col(19)
    good, rej = q == 'good', q == 'rejected'
    o2 = band != '-'
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    for ax, y, e, lab, yl in zip(axes, [a, v, s], [ae, ve, se],
                                 ['tau scale a', 'shift v (km/s)', 'broadening s (km/s)'],
                                 [(0, 1.5), (-2, 1), (0, 1.5)]):
        ax.errorbar(lam[good & ~o2], y[good & ~o2], e[good & ~o2], fmt='o', ms=3, color='C0', lw=0.6,
                    label='H2O-dominated')
        ax.errorbar(lam[good & o2], y[good & o2], e[good & o2], fmt='s', ms=4, color='C3', lw=0.6,
                    label='O2 band')
        ax.plot(lam[rej], np.clip(y[rej], *yl), 'x', ms=4, color='0.6', label='rejected by quality cut')
        for x in SEG_EDGES:
            ax.axvline(x, color='0.5', lw=0.8, ls='--')
        ax.set_ylim(*yl)
        ax.set_ylabel(lab)
    axes[0].legend(loc='upper left', ncol=3, frameon=False)
    axes[2].set_xlabel('Wavelength (A, air)')
    axes[2].set_xticks(np.arange(3500, 12501, 500))
    axes[2].tick_params(axis='x', rotation=45)
    for ax, l in zip(axes, 'abc'):
        letter(ax, l)
    fig.text(0.5, 0.005, 'Dashed lines: FTS segment edges inferred from the wavenumber sampling. '
             'Rejected chunks are clipped to the plot range.', ha='center', fontsize='small')
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    pdf.savefig(fig)
    plt.close(fig)


def make_page(pg, starts, w, at, sm, T, fl):
    fig, axes = plt.subplots(PER_PAGE, 1, figsize=(11, 14))
    for i, ax in enumerate(axes):
        j = pg * PER_PAGE + i
        if j >= len(starts):
            ax.axis('off')
            continue
        lo, hi = starts[j], starts[j] + PANEL
        i0, i1 = np.searchsorted(w, [lo, hi])
        x = w[i0:i1]
        ax.plot(x, at[i0:i1], color='0.6', lw=0.8, label='atlas', rasterized=True)
        ax.plot(x, sm[i0:i1], 'k', lw=0.6, label='telluric-free', rasterized=True)
        ax.plot(x, T[i0:i1], 'C0', lw=0.5, label='telluric model', rasterized=True)
        f = fl[i0:i1]
        if (f == 3).any():
            ax.fill_between(x, 0, 1.1, where=f == 3, color='C3', alpha=0.15, lw=0, label='saturated, no data')
        if (f == -1).any():
            ax.fill_between(x, 0, 1.1, where=f == -1, color='0.85', alpha=0.5, lw=0, label='no TAPAS model')
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1.1)
        ax.set_xticks(np.arange(lo, hi + 0.01, 1.0))
        ax.ticklabel_format(axis='x', useOffset=False, style='plain')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylabel('I/I_c')
        if i == 0:
            ax.legend(loc='lower right', ncol=3, frameon=False, fontsize='small')
    axes[-1].set_xlabel('Wavelength (A, air)')
    fig.tight_layout()
    return fig


def run():
    d = np.load(os.path.join(OUT_DIR, 'telfree_atlas.npz'))
    w, at, sm, T, fl = d['wav'], d['atlas'], d['smoothed'], d['T_model'], d['flag']
    starts = np.arange(np.floor(w[0]), w[-1], PANEL)
    npages = int(np.ceil(len(starts) / PER_PAGE))
    parts = [os.path.join(OUT_DIR, '_part000.pdf')]
    with PdfPages(parts[0]) as pdf:
        param_page(pdf)
    log('parameter page done')
    for b0 in range(0, npages, BATCH):
        part = os.path.join(OUT_DIR, f'_part{b0 + 1:03d}.pdf')
        parts.append(part)
        with PdfPages(part) as pdf:
            for pg in range(b0, min(b0 + BATCH, npages)):
                fig = make_page(pg, starts, w, at, sm, T, fl)
                pdf.savefig(fig, dpi=DPI)
                plt.close(fig)
        gc.collect()
        log(f'pages {b0 + 1}-{min(b0 + BATCH, npages)}/{npages} written')
    writer = PdfWriter()
    for part in parts:
        writer.append(part)
    with open(os.path.join(OUT_DIR, 'telfree_atlas_comparison.pdf'), 'wb') as f:
        writer.write(f)
    for part in parts:
        os.remove(part)
    log(f'wrote {OUT_DIR}/telfree_atlas_comparison.pdf')


if __name__ == '__main__':
    run()
