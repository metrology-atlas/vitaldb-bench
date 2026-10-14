# SPDX-License-Identifier: MIT
"""PPG foot (intersecting tangent) for every exported beat, and the SNUADC port. This machine only.

    python tools/foot_export.py --root R --beats DIR --out DIR2 [--part 0 --of 3]

For each case in DIR (beats_export: r_t, p_t, up_index, ppg_first_index, ppg_count), re-reads SNUADC/PLETH
over the stretch, band-passes it 0.5-10 Hz without phase shift (as for the upstroke), and finds each beat's
foot by the intersecting tangent: the tangent at the steepest upstroke meets the horizontal line through the
minimum in the 0.4 s before it. Writes DIR2/<case>.npz with f_t (seconds, the same clock and convention as
p_t; NaN where not found) and the SNUADC device's port.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).parent))
from pat import PPG, sample_times
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital

FS = 500.0


def feet(x: np.ndarray, up: np.ndarray, fs: float = FS, back_s: float = 0.4) -> np.ndarray:
    """Sub-sample foot positions (samples) of the beats whose steepest upstroke is at sample ``up``."""
    sos = signal.butter(2, (0.5, 10), btype="bandpass", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, x.astype(np.float64))
    d = np.gradient(y)
    w = int(back_s * fs)
    out = np.full(up.size, np.nan)
    for k, u in enumerate(up):
        if u - w < 0 or u + 2 >= y.size or d[u] <= 0:
            continue
        # sub-sample steepest point and its slope (parabola through the derivative)
        den = d[u - 1] - 2 * d[u] + d[u + 1]
        off = float(np.clip(0.5 * (d[u - 1] - d[u + 1]) / den, -0.5, 0.5)) if den != 0 else 0.0
        yu = y[u] + off * d[u]
        ymin = float(np.min(y[u - w : u]))
        f = u + off + (ymin - yu) / d[u]
        if u - w <= f <= u:
            out[k] = f
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--beats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--time-budget", type=float, default=160)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = [f for i, f in enumerate(sorted(Path(a.beats).glob("*.npz"))) if i % a.of == a.part]
    t0, n = time.time(), 0
    for f in files:
        if (out / f.name).exists():
            continue
        if time.time() - t0 > a.time_budget:
            break
        c = int(f.stem)
        z = np.load(f)
        v = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={PPG})
        tr = v.tracks[PPG]
        tp = sample_times(tr)
        pa, cnt = int(z["ppg_first_index"]), int(z["ppg_count"])
        x = np.asarray(tr.values[pa : pa + cnt])
        fpos = feet(x, np.asarray(z["up_index"]) - pa)
        f_t = tp[pa] + fpos / FS
        port = ""
        for name, dev in v.devices.items():
            if "SNUADC" in str(name):
                port = str(getattr(dev, "port", "") or "")
        np.savez_compressed(out / f.name, f_t=f_t, port=np.array(port))
        n += 1
    print(json.dumps({"part": a.part, "new": n, "done": len(list(out.glob("*.npz")))}))


if __name__ == "__main__":
    main()
