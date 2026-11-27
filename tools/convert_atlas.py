"""
One-time conversion of the ISPy FTS disk-center atlas (IDL save file) into a plain
NumPy archive, so that the HS5000 server needs neither ISPy, scipy nor astropy.

Source : ISPy/ISPy/data/fts_disk_center_SI.idlsave  (git submodule)
Output : fts_atlas.npz next to backend.py, with
           wave     wavelength [A, air, no gravitational redshift]  (float64)
           inu_si   disk-center intensity I_nu  [W m^-2 sr^-1 Hz^-1] (float32, as stored)
           cont_si  continuum I_nu             [W m^-2 sr^-1 Hz^-1] (float32, as stored)
The unit conversions (SI -> cgs, I_nu -> I_lambda per Angstrom) are done in backend.py
with the same expressions as ISPy.spec.atlas (commit 0790aa5 or later).

Usage: ipython3 tools/convert_atlas.py
"""
import os
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SRC = os.path.join(ROOT, "ISPy", "ISPy", "data", "fts_disk_center_SI.idlsave")
OUT = os.path.join(ROOT, "fts_atlas.npz")
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:5.1f} s] {msg}", flush=True)


def run():
    try:
        from scipy.io import readsav
    except ImportError:
        print("scipy is needed for this one-time conversion: "
              "/usr/local/opt/python@3.13/bin/python3.13 -m pip install scipy")
        return
    if not os.path.exists(SRC):
        print(f"source not found: {SRC}\n(run: git submodule update --init ISPy)")
        return
    log(f"reading {SRC}")
    d = readsav(SRC)
    wave = np.asarray(d["ftswav"], dtype=np.float64)
    inu = np.asarray(d["ftsint_si"], dtype=np.float32)
    cont = np.asarray(d["ftscnt_si"], dtype=np.float32)
    if not np.all(np.diff(wave) > 0):
        print("wavelength axis is not strictly increasing, refusing to write")
        return
    log(f"{wave.size} points, {wave[0]:.3f}-{wave[-1]:.3f} A")
    np.savez_compressed(OUT, wave=wave, inu_si=inu, cont_si=cont)
    log(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")
    # read-back check
    z = np.load(OUT)
    assert np.array_equal(z["wave"], wave) and np.array_equal(z["inu_si"], inu) and np.array_equal(z["cont_si"], cont)
    log("read-back check passed")


if __name__ == "__main__":
    run()
