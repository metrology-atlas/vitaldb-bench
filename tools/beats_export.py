# SPDX-License-Identifier: MIT
"""Beat-level R-peak and PPG upstroke times for later timing analyses (kept on this machine only).

    python tools/beats_export.py --root R --cases cases.json --out DIR [--part 0 --of 3] [--minutes 90]

Longest continuous ECG_II + PLETH stretch (up to ``--minutes``); sub-sample R-peaks and filtered steepest
PPG upstrokes (as ``pat_res``), beat-matched. Writes DIR/<case>.npz with r_t, p_t (absolute seconds),
the stretch start/end and the sample index of every PPG upstroke in the raw stream.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pat import ECG, PPG, r_peaks, sample_times
from pat_res import ecg_subsample, ppg_onsets
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import (
    read_vital,
)

from physiosync import match_beats

warnings.filterwarnings("ignore")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=90.0)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--time-budget", type=float, default=150)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = [c for i, c in enumerate(json.loads(Path(a.cases).read_text())["cases"]) if i % a.of == a.part]
    t_start, slowest, n = time.time(), 0.0, 0
    for c in cases:
        f = out / f"{c:04d}.npz"
        if f.exists() or (out / f"{c:04d}.skip").exists():
            continue
        if time.time() - t_start + max(1.5 * slowest, 5) > a.time_budget:
            break
        t1 = time.time()
        case = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={ECG, PPG})
        e, p = case.tracks.get(ECG), case.tracks.get(PPG)
        reason = None
        if e is None or p is None or not e.record_times.size or not p.record_times.size or e.srate != 500:
            reason = "no 500 Hz ECG_II and PLETH"
        else:
            fs = 500.0
            te, tp = sample_times(e), sample_times(p)
            gap = np.flatnonzero(np.abs(np.diff(te) - 1 / fs) > 0.01 / fs)
            be = np.r_[0, gap + 1, te.size]
            k = int(np.argmax(np.diff(be)))
            ea, eb = be[k], min(be[k + 1], be[k] + int(a.minutes * 60 * fs))
            pa, pb = np.searchsorted(tp, te[ea]), np.searchsorted(tp, te[eb - 1])
            if eb - ea < 1200 * fs or pb - pa < 1200 * fs or np.max(np.abs(np.diff(tp[pa:pb]) - 1 / fs)) > 0.01 / fs:
                reason = "no 20 min continuous stretch"
            else:
                xe = e.physical()[ea:eb].astype(np.float64)
                rp = r_peaks(xe, fs)
                up_i, up_f = ppg_onsets(np.asarray(p.values[pa:pb]), fs)
                rt = te[ea] + ecg_subsample(xe, rp, fs) / fs
                pt = tp[pa] + up_f / fs
                i, j, _ = match_beats(rt, pt)
                np.savez_compressed(f, r_t=rt[i], p_t=pt[j], ppg_first_index=np.int64(pa), ppg_count=np.int64(pb - pa),
                                    ppg_t0=tp[pa], up_index=up_i[j] + pa, ecg_t0=te[ea])
        if reason:
            (out / f"{c:04d}.skip").write_text(reason)
        slowest = max(slowest, time.time() - t1)
        n += 1
    print(json.dumps({"part": a.part, "new": n, "seconds": round(time.time() - t_start, 1)}))


if __name__ == "__main__":
    main()
