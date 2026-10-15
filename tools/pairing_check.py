# SPDX-License-Identifier: MIT
"""Beat pairing by inter-beat-interval matching (physiosync.pair_beats, on hrvmatch) against the old
most-common-delay rule (physiosync.match_beats), on the benchmark stretches. Per-case numbers, this machine.

    python tools/pairing_check.py --root R --beats DIR --cases 1,2,3 --out out.jsonl [--expected 0.45 0.95]

Per case, over the beats_export stretch: all ECG R peaks (sub-sample) and all PPG feet (intersecting tangent
of the 0.5-10 Hz band-passed PLETH, at every steepest upstroke), unpaired; then both pairings, their median
delay, how many beats each pairs, and how often they agree on which R peak a pulse belongs to.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from foot_export import feet
from pat import ECG, PPG, r_peaks, sample_times
from pat_res import ecg_subsample, ppg_onsets
from physiosync import match_beats, pair_beats
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital

FS = 500.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--beats", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expected", type=float, nargs=2, default=(0.45, 0.95))
    a = ap.parse_args()
    with open(a.out, "a") as fh:
        for c in [int(x) for x in a.cases.split(",") if x]:
            res: dict = {"case": c}
            t1 = time.time()
            try:
                z = np.load(Path(a.beats) / f"{c:04d}.npz")
                v = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={ECG, PPG})
                e, p = v.tracks[ECG], v.tracks[PPG]
                te, tp = sample_times(e), sample_times(p)
                pa, cnt = int(z["ppg_first_index"]), int(z["ppg_count"])
                ea = int(np.searchsorted(te, float(z["ecg_t0"])))
                eb = int(np.searchsorted(te, tp[pa + cnt - 1]))
                xe = e.physical()[ea:eb].astype(np.float64)
                r_all = te[ea] + ecg_subsample(xe, r_peaks(xe, FS), FS) / FS
                xp = np.asarray(p.values[pa : pa + cnt])
                up_i, _ = ppg_onsets(xp, FS)
                f = feet(xp, up_i)
                p_all = np.sort(tp[pa] + f[np.isfinite(f)] / FS)
                i, j, _ = match_beats(r_all, p_all)
                d_old = p_all[j] - r_all[i]
                new = pair_beats(r_all, p_all, expected_delay_s=tuple(a.expected))
                old_map = dict(zip(j.tolist(), i.tolist()))
                both = [k for k in new["p_index"].tolist() if k in old_map]
                agree = np.mean([old_map[k] == rk for k, rk in zip(new["p_index"].tolist(), new["r_index"].tolist())
                                 if k in old_map]) if both else None
                res.update({"r_peaks": int(r_all.size), "ppg_feet": int(p_all.size),
                            "old": {"paired": int(i.size), "delay_median_ms": round(float(np.median(d_old)) * 1000, 1)},
                            "new": {"paired": int(new["p_index"].size),
                                    "delay_median_ms": None if not new["delay_s"].size else round(float(np.median(new["delay_s"])) * 1000, 1),
                                    **{k: v_ for k, v_ in new["qc"].items() if k in ("windows", "good", "ambiguous", "paired_by_rhythm", "paired_by_window")}},
                            "agree_share": None if agree is None else round(float(agree), 4),
                            "median_rr_ms": round(float(np.median(np.diff(r_all))) * 1000, 1)})
            except Exception as ex:  # noqa: BLE001
                res["error"] = f"{type(ex).__name__}: {ex}"[:300]
            res["seconds"] = round(time.time() - t1, 1)
            fh.write(json.dumps(res) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
