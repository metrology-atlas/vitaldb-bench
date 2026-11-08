# SPDX-License-Identifier: MIT
"""Synthetic check of tools/numeric_recompute_lag.py: with a known monitor, does fitting the delay as the atlas does
and then recomputing give a lag of zero? Or does the method itself give the -1 s seen on the release?

    python examples/sim_recompute_lag.py --out $W/numeric_recompute_lag_sim [--recordings 100] [--hours 1] [--seed 3]

Heartbeats as examples/sim_pr_window_routes.py. HR = floor(60 / mean of the last 8 R-R intervals within 15% of their
median). Two monitors:
  beat   a new value after every beat, ready a processing delay d0 after it;
  grid   the monitor recomputes on its own 2 s grid (random phase), from the beats so far, and the value is ready
         d0 after that update.
Either way the recorder polls every 2 s (gaps 2.000/1.999/2.001 s, a random phase of its own) and stamps the latest
ready value. Then, as the atlas does: the delay d is fitted in the uniform-phase convention
(tools/pleth_hr_derivation.phase_delay, truncated), and, as tools/numeric_recompute_lag.py does, HR is recomputed at
every beat, placed at the beat + d + WAIT (1 s), and the lag of best agreement read at the plateau midpoints. The
true mean delay from the window's close to the stamp is also given, against d + WAIT. Writes OUT/summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from sim_pr_window_routes import beats  # noqa: E402
from numeric_recompute_lag import WAIT, lag_stats, peak, stamp_at  # noqa: E402
from pleth_hr_derivation import phase_delay  # noqa: E402

D0 = 3.0


def mean_iv(rr, n=8, band=0.15):
    """Mean of the last n intervals within band of their median, at each interval (NaN before n)."""
    out = np.full(rr.size, np.nan)
    w = np.lib.stride_tricks.sliding_window_view(rr, n)
    med = np.median(w, axis=1)[:, None]
    keep = np.abs(w - med) <= band * med
    out[n - 1:] = np.where(keep, w, 0).sum(axis=1) / np.maximum(keep.sum(axis=1), 1)
    return out


def polls(rng, end):
    gaps = rng.choice([2.0, 1.999, 2.001], size=int(end / 1.9) + 10, p=[0.52, 0.31, 0.17])
    p = rng.uniform(0, 2) + np.r_[0, np.cumsum(gaps)]
    return p[p < end]


def run(rng, kind, n, hours):
    rows, true_mean, fitted = [], [], []
    for _ in range(n):
        r = beats(rng, hours)
        rr, te = np.diff(r), r[1:]
        m = mean_iv(rr)
        ok = np.isfinite(m)
        val = np.floor(60.0 / m)
        if kind == "beat":
            ready, v, close = te[ok] + D0, val[ok], te[ok]
        else:
            u = np.arange(rng.uniform(0, 2), te[-1], 2.0)
            j = np.searchsorted(te, u, side="right") - 1
            k = (j >= 0) & ok[np.clip(j, 0, None)]
            ready, v, close = u[k] + D0, val[j[k]], te[j[k]]
        t = polls(rng, r[-1])
        i = np.searchsorted(ready, t, side="right") - 1
        s = i >= 0
        t, x, cl = t[s], v[i[s]], close[i[s]]
        true_mean.append(float(np.mean(t - cl)))
        gap = np.r_[2.0, np.diff(t)]
        ph = phase_delay("hr", ("n", np.where(ok, m, np.nan), 8), t, x.astype(int), gap, te)
        d = ph["hr|trunc"][0]
        fitted.append(d)
        rec = stamp_at(te[ok] + d + WAIT, val[ok], t)
        rows.append({"hr": lag_stats(t, x, rec)})
    return {"recordings": n, "fitted_d_mean_s": round(float(np.mean(fitted)), 3),
            "true_mean_delay_s": round(float(np.mean(true_mean)), 3),
            "d_plus_wait_s": round(float(np.mean(fitted)) + WAIT, 3),
            "lag_s": round(float(peak(rows, "hr")), 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--recordings", type=int, default=100)
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    res = {"d0_s": D0, "wait_s": WAIT, "monitors": {k: run(rng, k, a.recordings, a.hours) for k in ("beat", "grid")}}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
