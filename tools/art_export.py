# SPDX-License-Identifier: MIT
"""Arterial pressure upstroke time for every exported beat (kept on this machine only).

    python tools/art_export.py --root R --beats DIR --out DIR2 [--part 0 --of 3]

For each case in DIR (``beats_export``), reads SNUADC/ART over the stretch and writes DIR2/<case>.npz with
a_t: the filtered steepest ART upstroke between 0.05 and 0.45 s after each R-peak (sub-sample), NaN where
the arterial signal is not physiological (10 s windows: median 30-200 mmHg, pulse pressure 15-150 mmHg)
or no upstroke is found. Cases without a usable arterial line get a .skip file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).parent))
from pat import sample_times
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import (
    read_vital,
)

warnings.filterwarnings("ignore")
ART = "SNUADC/ART"
FS = 500.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--beats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--time-budget", type=float, default=150)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = [f for i, f in enumerate(sorted(Path(a.beats).glob("*.npz"))) if i % a.of == a.part]
    t_start, n = time.time(), 0
    for f in files:
        c = int(f.stem)
        if (out / f.name).exists() or (out / f"{c:04d}.skip").exists():
            continue
        if time.time() - t_start > a.time_budget:
            break
        z = np.load(f)
        r_t = z["r_t"]
        case = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={ART})
        tr = case.tracks.get(ART)
        reason = None
        a_t = np.full(r_t.size, np.nan)
        if tr is None or tr.srate != 500 or not tr.record_times.size:
            reason = "no ART"
        else:
            ta = sample_times(tr)
            i0, i1 = int(np.searchsorted(ta, r_t[0] - 2)), int(np.searchsorted(ta, r_t[-1] + 2))
            seg_t = ta[i0:i1]
            if seg_t.size < 600 * FS:
                reason = "ART too short"
            else:
                x = tr.physical()[i0:i1].astype(np.float64)
                gaps = np.flatnonzero(np.abs(np.diff(seg_t) - 1 / FS) > 0.01 / FS)
                w = int(10 * FS)
                nw = x.size // w
                X = x[: nw * w].reshape(nw, w)
                med = np.median(X, axis=1)
                pp = np.quantile(X, 0.95, axis=1) - np.quantile(X, 0.05, axis=1)
                ok_w = (med > 30) & (med < 200) & (pp > 15) & (pp < 150)
                ok_s = np.r_[np.repeat(ok_w, w), np.zeros(x.size - nw * w, bool)]
                for g in gaps:  # never across a break in the samples
                    ok_s[max(g - 250, 0) : g + 250] = False
                if ok_w.mean() < 0.2:
                    reason = "no arterial line (ART not pulsatile)"
                else:
                    sos = signal.butter(2, (0.5, 10), btype="bandpass", fs=FS, output="sos")
                    d = np.gradient(signal.sosfiltfilt(sos, np.nan_to_num(x)))
                    for k, rt in enumerate(r_t):
                        i_a = int(np.searchsorted(seg_t, rt + 0.05))
                        i_b = int(np.searchsorted(seg_t, rt + 0.45))
                        if i_a < 2 or i_b + 2 >= x.size or not ok_s[i_a:i_b].all():
                            continue
                        m = i_a + int(np.argmax(d[i_a:i_b]))
                        if d[m] <= 0 or m in (i_a, i_b - 1):
                            continue
                        den = d[m - 1] - 2 * d[m] + d[m + 1]
                        off = 0.5 * (d[m - 1] - d[m + 1]) / den if den != 0 else 0.0
                        a_t[k] = seg_t[m] + float(np.clip(off, -0.5, 0.5)) / FS
        if reason:
            (out / f"{c:04d}.skip").write_text(reason)
        else:
            np.savez_compressed(out / f.name, a_t=a_t)
        n += 1
    print(json.dumps({"part": a.part, "new": n, "seconds": round(time.time() - t_start, 1)}))


if __name__ == "__main__":
    main()
