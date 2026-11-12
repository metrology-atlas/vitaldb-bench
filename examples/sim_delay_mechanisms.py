# SPDX-License-Identifier: MIT
"""Which monitor behaviour makes the HR delay fit and the recompute test disagree by about 1 s?

    python examples/sim_delay_mechanisms.py --out DIR --monitor onpoll [--recordings 20] [--hours 1] [--seed 5]

Follows up metrology-atlas study numeric-delay-check: on the release the monitor's HR changes about 1 s later than its
fitted delay plus half the poll gap predicts (early delay group 1.43 s, late 0.46 s), while examples/sim_recompute_lag.py
reads a lag of about zero for a monitor updating every beat or on a free-running 2 s grid. Here the HR (60 / mean of
the last 8 R-R intervals within 15% of their median, truncated) reaches the recorder by one of these routes:

  beat     a new value after every beat, ready D0 after it (as sim_recompute_lag);
  grid2    recomputed on the monitor's own 2 s grid (random phase), ready D0 after the update (as sim_recompute_lag);
  grid1    the same on a 1 s grid;
  onpoll   computed when the recorder asks: the value at each poll is the rule applied to the beats up to poll - D0,
           so its age has no spread from a phase;
  prevpoll computed at each poll but sent at the next one (a one-poll buffer): the value at poll k is the rule
           applied to the beats up to poll k-1 - D0;
  debounce2, debounce3
           a new value after every beat, as beat, but the displayed value changes only once the newly computed value
           has stayed the same for 2 (3) consecutive beats; otherwise the previous displayed value is held;
  slew1, slew2
           a new value after every beat, but the displayed value moves towards the newly computed one by at most 1 (2)
           bpm per beat (a rate limit);
  stale25, stale50, stale75
           as beat, but at a share of polls (25, 50, 75%) the recorder receives the value it would have received one
           poll earlier (a value held over by a buffer or a missed update), so the age is a mixture one poll apart.

The recorder polls every 2 s (gaps 2.000/1.999/2.001 s) and stamps the latest value it has. Then, as the atlas does,
the delay d is fitted by the labels' likelihood (tools/numeric_delay_variability.profile: uniform phase over the gap
since the previous record, d from the window's close, the argmax of the profile), and, as
tools/numeric_recompute_lag.py does, HR is recomputed at every beat, placed at beat + d + WAIT (1 s), stamped at the
monitor's stamps, and the lag of best agreement read at the plateau midpoints. Also the true mean delay from the
window's close to the stamp, the width of the profile's 95% interval (a flat profile means d is not identified), and,
as examples/hr_change_age.py reads the release, the age of the transition beat at each value change minus d (its
mean, median and share in [0, 2) s), so the release's change ages can be compared with each monitor's.
Options make the heartbeats less ideal: beat-to-beat R-R variability (--hrv), ectopic pairs (--ectopic), and an
analyst whose R-peak detector jitters (--det-sd) and misses beats (--miss); the monitor sees the true beats, the fit,
the recomputation and the tests see the detected ones. Writes OUT/<monitor><tag>.json.
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
from sim_recompute_lag import mean_iv, polls  # noqa: E402
from numeric_recompute_lag import WAIT, lag_stats, peak, stamp_at  # noqa: E402
from numeric_delay_variability import D, profile  # noqa: E402

D0 = 3.0
SPAN = 225.0
OFFS = np.round(np.arange(-3.0, 6.0001, 0.1), 2)


def route(kind, rng, te, val, ok, t):
    """(value ready times, values, window-close times) of the monitor's outputs for one recording."""
    tv, vv = te[ok], val[ok]
    if kind == "beat":
        return tv + D0, vv, tv
    if kind.startswith("debounce"):
        k = int(kind[-1])
        shown = vv.copy()
        cur, run = vv[0], 0
        for i in range(1, vv.size):
            run = run + 1 if vv[i] == vv[i - 1] else 1
            if vv[i] != cur and run >= k:
                cur = vv[i]
            shown[i] = cur
        return tv + D0, shown, tv
    if kind.startswith("stale"):
        return tv + D0, vv, tv
    if kind.startswith("slew"):
        k = float(kind[-1])
        shown = vv.copy()
        for i in range(1, vv.size):
            shown[i] = shown[i - 1] + np.clip(vv[i] - shown[i - 1], -k, k)
        return tv + D0, shown, tv
    if kind in ("grid2", "grid1"):
        step = 2.0 if kind == "grid2" else 1.0
        u = np.arange(rng.uniform(0, step), te[-1], step)
        j = np.searchsorted(tv, u, side="right") - 1
        k = j >= 0
        return u[k] + D0, vv[j[k]], tv[j[k]]
    if kind in ("onpoll", "prevpoll"):
        ask = t - D0 if kind == "onpoll" else np.r_[np.nan, t[:-1]] - D0
        j = np.searchsorted(tv, ask, side="right") - 1
        k = (j >= 0) & np.isfinite(ask)
        # ready exactly at the poll that carries it (a tiny epsilon so the recorder stamps it at that poll)
        return t[k] - 1e-6, vv[j[k]], tv[j[k]]
    raise ValueError(kind)


def change_ages(ht, hv, tc, y):
    """At each stamp whose value differs from the previous one, stamp minus the latest transition beat to that value
    (as examples/hr_change_age.py)."""
    trans = np.r_[False, y[1:] != y[:-1]]
    ages = []
    for i in np.flatnonzero(np.r_[False, hv[1:] != hv[:-1]]):
        t, x = ht[i], hv[i]
        j1 = np.searchsorted(tc, t, side="right") - 1
        j0 = np.searchsorted(tc, t - 12.0, side="left")
        if j1 <= j0:
            continue
        js = np.arange(j1, j0 - 1, -1)
        hit = js[(y[js] == x) & trans[js]]
        if hit.size:
            ages.append(t - tc[hit[0]])
    return np.array(ages)


def realistic(rng, r, hrv, ectopic):
    """Beat-to-beat variability (each R-R times 1 + N(0, hrv)) and ectopic pairs (an interval 30% short, the next 30%
    long) at the given rate per beat."""
    rr = np.diff(r) * (1 + rng.normal(0, hrv, r.size - 1))
    e = np.flatnonzero(rng.random(rr.size - 1) < ectopic)
    rr[e] *= 0.7
    rr[e + 1] *= 1.3
    return r[0] + np.r_[0, np.cumsum(rr)]


def detected(rng, r, det_sd, miss):
    """The analyst's R peaks: each true beat jittered by N(0, det_sd) s, a share `miss` of beats not found."""
    keep = rng.random(r.size) >= miss
    return np.sort(r[keep] + rng.normal(0, det_sd, keep.sum()))


def run(rng, kind, n, hours, hrv=0.0, ectopic=0.0, det_sd=0.0, miss=0.0):
    rows, true_mean, fitted, widths, cages, profs, exact = [], [], [], [], [], [], []
    for _ in range(n):
        r = realistic(rng, beats(rng, hours), hrv, ectopic)
        rr, te = np.diff(r), r[1:]
        m = mean_iv(rr)
        ok = np.isfinite(m)
        val = np.floor(60.0 / m)
        t = polls(rng, r[-1])
        ready, v, close = route(kind, rng, te, val, ok, t)
        # from here on, the analyst's view: R peaks as detected, the rule recomputed from them
        ra = detected(rng, r, det_sd, miss)
        rra = np.diff(ra)
        rra = np.where((rra > 0.25) & (rra < 2.5), rra, np.nan)
        good = np.isfinite(rra)
        te_a, rra = ra[1:][good], rra[good]
        m_a = mean_iv(rra)
        ok_a = np.isfinite(m_a)
        te_a, m_a = te_a[ok_a], m_a[ok_a]
        val_a = np.floor(60.0 / m_a)
        o = np.argsort(ready)
        ready, v, close = ready[o], v[o], close[o]
        i = np.searchsorted(ready, t, side="right") - 1
        if kind.startswith("stale"):
            i_old = np.searchsorted(ready, t - 2.0, side="right") - 1
            pick = rng.random(t.size) < int(kind[5:]) / 100
            i = np.where(pick, i_old, i)
        s = i >= 0
        t2, x, cl = t[s], v[i[s]], close[i[s]]
        true_mean.append(float(np.mean(t2 - cl)))
        gap = np.r_[2.0, np.diff(t2)]
        d, ci, *_ = profile(t2, x.astype(float), gap, te_a, 60.0 / m_a, 1.0, "floor", False, SPAN, D, 0.05)
        fitted.append(d)
        widths.append(ci[1] - ci[0])
        cages.append(change_ages(t2, x, te_a, val_a) - d)
        prof = []
        for o in OFFS:
            j = np.searchsorted(te_a, t2 - d - o, side="right") - 1
            okj = j >= 0
            prof.append(float(np.mean(val_a[j[okj]] == x[okj])))
        profs.append(prof)
        rec = stamp_at(te_a + d + WAIT, val_a, t2)
        rows.append({"hr": lag_stats(t2, x, rec)})
    fm, tm = float(np.mean(fitted)), float(np.mean(true_mean))
    return {"monitor": kind, "recordings": n, "hours": hours, "d0_s": D0, "wait_s": WAIT,
            "hrv": hrv, "ectopic": ectopic, "det_sd_s": det_sd, "miss": miss,
            "fitted_d_mean_s": round(fm, 3), "fitted_d_sd_s": round(float(np.std(fitted)), 3),
            "profile_ci_width_mean_s": round(float(np.mean(widths)), 3),
            "true_mean_delay_s": round(tm, 3), "d_plus_wait_s": round(fm + WAIT, 3),
            "true_minus_d_plus_wait_s": round(tm - fm - WAIT, 3),
            "lag_s": round(float(peak(rows, "hr")), 3),
            "change_age_minus_d": {"mean": round(float(np.mean(np.concatenate(cages))), 3),
                                   "median": round(float(np.median(np.concatenate(cages))), 3),
                                   "share_0_2s": round(float(np.mean((np.concatenate(cages) >= 0) & (np.concatenate(cages) < 2))), 3)},
            "match_profile_peak_offset_s": float(OFFS[int(np.argmax(np.mean(profs, axis=0)))]),
            "match_profile_peak_share": round(float(np.max(np.mean(profs, axis=0))), 4),
            "match_profile_half_max_s": half_max(np.mean(profs, axis=0))}


def half_max(m):
    """The span of offsets where the mean match profile is at least half way from its minimum to its peak."""
    h = m.min() + (m.max() - m.min()) / 2
    a = np.flatnonzero(m >= h)
    return [float(OFFS[a[0]]), float(OFFS[a[-1]])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--monitor", required=True, choices=("beat", "grid2", "grid1", "onpoll", "prevpoll", "debounce2", "debounce3", "slew1", "slew2", "stale25", "stale50", "stale75"))
    ap.add_argument("--recordings", type=int, default=20)
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--hrv", type=float, default=0.0, help="beat-to-beat R-R variability (relative SD)")
    ap.add_argument("--ectopic", type=float, default=0.0, help="ectopic pairs per beat")
    ap.add_argument("--det-sd", type=float, default=0.0, help="the analyst's R-peak jitter (s)")
    ap.add_argument("--miss", type=float, default=0.0, help="share of beats the analyst's detector misses")
    ap.add_argument("--tag", default="", help="suffix for the output file name")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    res = run(rng, a.monitor, a.recordings, a.hours, a.hrv, a.ectopic, a.det_sd, a.miss)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.monitor}{a.tag}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
