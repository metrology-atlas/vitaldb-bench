# SPDX-License-Identifier: MIT
"""Per-case rate of the ECG-to-PPG sawtooth, to compare with the SNUADC clock drift of the same case.

    python tools/saw_rate.py --root R --cases cases.json --out saw.jsonl [--part 0 --of 4] [--minutes 90]

On the longest stretch where ECG_II and PLETH are both continuous (up to ``--minutes``), the beat-matched
PAT series of ``pat_res`` (filtered steepest upstroke, sub-sample R-peak) is tracked for resets. Resets
whose local step is 24-42 ms are frame slips (one 33.3 ms frame). The sawtooth rate is estimated twice:

``slope``     one slope shared by all teeth (each tooth its own offset), least squares on 15-beat medians;
              ppm, positive = PAT shrinking between slips.
``interval``  33.33 ms / median time between consecutive slips (only from slips with no gap between).
Output: per-case numbers only.
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
from pat_res import ecg_subsample, ppg_onsets, resets
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import (
    read_vital,
)

warnings.filterwarnings("ignore")
FRAME_MS = 1000 / 30


def shared_slope(t: np.ndarray, p: np.ndarray, starts: list[int], min_beats: int = 60) -> tuple[float, int, float]:
    """Least-squares slope common to all teeth (tooth = beats between consecutive starts)."""
    edges = list(starts) + [t.size]
    num = den = 0.0
    used = 0
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < min_beats:
            continue
        # 15-beat medians inside the tooth, away from the reset edges
        a2, b2 = a + 15, b - 15
        n = (b2 - a2) // 15
        if n < 4:
            continue
        tt = t[a2 : a2 + 15 * n].reshape(n, 15).mean(1)
        pp = np.median(p[a2 : a2 + 15 * n].reshape(n, 15), 1)
        tc, pc = tt - tt.mean(), pp - pp.mean()
        num += float((tc * pc).sum())
        den += float((tc * tc).sum())
        used += 1
    if den == 0:
        return float("nan"), 0, float("nan")
    s = num / den
    return s, used, den


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
    done = {json.loads(x)["case"] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    cases = [c for i, c in enumerate(json.loads(Path(a.cases).read_text())["cases"]) if i % a.of == a.part]
    t_start, slowest = time.time(), 0.0
    with out.open("a") as fh:
        for c in cases:
            if c in done:
                continue
            if time.time() - t_start + max(1.5 * slowest, 5) > a.time_budget:
                break
            t1 = time.time()
            res: dict = {"case": c}
            try:
                case = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={ECG, PPG})
                e, p = case.tracks.get(ECG), case.tracks.get(PPG)
                if e is None or p is None or not e.record_times.size or not p.record_times.size or e.srate != 500:
                    res["skip"] = "no 500 Hz ECG_II and PLETH"
                else:
                    fs = 500.0
                    te, tp = sample_times(e), sample_times(p)
                    gap = np.flatnonzero(np.abs(np.diff(te) - 1 / fs) > 0.01 / fs)
                    be = np.r_[0, gap + 1, te.size]
                    k = int(np.argmax(np.diff(be)))
                    ea, eb = be[k], min(be[k + 1], be[k] + int(a.minutes * 60 * fs))
                    pa, pb = np.searchsorted(tp, te[ea]), np.searchsorted(tp, te[eb - 1])
                    if eb - ea < 1200 * fs or pb - pa < 1200 * fs or np.max(np.abs(np.diff(tp[pa:pb]) - 1 / fs)) > 0.01 / fs:
                        res["skip"] = "no 20 min continuous stretch"
                    else:
                        xe = e.physical()[ea:eb].astype(np.float64)
                        rp = r_peaks(xe, fs)
                        up_i, up_f = ppg_onsets(np.asarray(p.values[pa:pb]), fs)
                        rf = ecg_subsample(xe, rp, fs)
                        tr, ts = te[ea] + rf / fs, tp[pa] + up_f / fs
                        j0 = np.searchsorted(ts, tr + 0.2)
                        diffs = np.concatenate([ts[np.clip(j0 + m, 0, ts.size - 1)] - tr for m in range(4)])
                        diffs = diffs[(diffs > 0.2) & (diffs < 1.6)]
                        h, edges = np.histogram(diffs, np.arange(0.2, 1.6, 0.004))
                        delay = float(edges[np.argmax(h)] + 0.002)
                        j = np.clip(np.searchsorted(ts, tr + delay), 1, ts.size - 1)
                        j = np.where(np.abs(ts[j - 1] - tr - delay) < np.abs(ts[j] - tr - delay), j - 1, j)
                        pat = ts[j] - tr
                        rr = float(np.median(np.diff(tr)))
                        good = np.abs(pat - delay) < 0.25 * rr
                        pg, tg = pat[good], tr[good] - tr[0]
                        dp = np.diff(pg)
                        sd = float(1.4826 * np.median(np.abs(dp - np.median(dp))) / np.sqrt(2) * 1000)
                        s = resets(tg, pg, 0.016)
                        # starts as beat indices, with their local steps
                        st_t = np.array(s["reset_t_s"])
                        st_ms = np.array(s["steps_ms"])
                        n = min(st_t.size, st_ms.size)
                        st_t, st_ms = st_t[-n:], st_ms[-n:]
                        slip = (st_ms >= 24) & (st_ms <= 42)
                        idx_all = [0] + [int(np.searchsorted(tg, x)) for x in st_t]
                        sl, used, _ = shared_slope(tg, pg, idx_all)
                        ts_slip = st_t[slip]
                        iv = np.diff(ts_slip)
                        res.update({
                            "minutes": round((eb - ea) / fs / 60, 1), "beats": int(good.sum()), "sd_beat_ms": round(sd, 2),
                            "pat_median_ms": round(float(np.median(pg)) * 1000, 1), "n_resets": int(n), "n_slips": int(slip.sum()),
                            "slip_steps_ms": st_ms[slip].round(2).tolist(), "slip_t_s": ts_slip.round(1).tolist(),
                            "slope_ppm": None if not np.isfinite(sl) else round(-sl * 1e6, 2), "teeth_used": used,
                            "interval_s_median": None if not iv.size else round(float(np.median(iv)), 1),
                            "interval_ppm": None if not iv.size else round(FRAME_MS / 1000 / float(np.median(iv)) * 1e6, 2),
                            "abs_start": float(te[ea]),
                        })
            except Exception as ex:  # noqa: BLE001
                res["error"] = f"{type(ex).__name__}: {ex}"
            res["seconds"] = round(time.time() - t1, 2)
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            slowest = max(slowest, res["seconds"])
    print(json.dumps({"part": a.part, "done": sum(1 for _ in out.open()), "seconds": round(time.time() - t_start, 1)}))


if __name__ == "__main__":
    main()
