# SPDX-License-Identifier: MIT
"""How the Solar8000 computes its pulse rate (PLETH_HR): recomputed from the plethysmogram's pulses, per recording.

    python tools/pleth_hr_derivation.py scan --root R --cases 1,2,3 --out part.jsonl [--time-budget 170]
    python tools/pleth_hr_derivation.py agg part*.jsonl --out DIR

Why: the comparison of pulse rate with heart rate (examples/hr_vs_pulse_rate.py) puts PLETH_HR's window centre about
8.5 s before its stamp, and its strongest line below the diagonal sits at a constant pulse-to-heart-rate ratio of
about 0.88, which a window of a fixed number of pulses averaged by a mean (N/(N+1) after one missed pulse, N about
7-8) would give, and a median would not. The numeric derivation pilot found a median of about 12 beats. This settles
it directly.

Pulses: SNUADC/PLETH on its record stamps (the file's own sample times), band-passed 0.5-10 Hz; one pulse per
steepest upstroke (at least 0.3 s apart, prominence from the recording's own upstroke sizes). Each pulse is moved
earlier by the PLETH's hardware delay (0.385 s, the atlas's pleth-chain-delay), so pulse times are the monitor's own.

Candidates at each PLETH_HR record stamp t, using the pulses up to t - d, for d from 0 to 10 s in 0.25 s steps:
  mean|n|N     60 / mean of the last N pulse intervals          N = 4..16
  median|n|N   60 / median of the last N pulse intervals        N = 4..16
  mean|t|T     60 / mean of the intervals ending in the last T s  T = 4..16 s
  count|t|T    pulses in the last T s, per minute                T = 4..16 s
  tmean|n|N    60 / mean of the last N intervals without the longest and shortest   N = 6..14
  rmean|n|N    60 / mean of the last N intervals within 25% of their median         N = 6..14
each truncated to a whole number (as HR is) and also rounded. Score: the share of records reproduced exactly. Up to
3,000 records per recording, spread over its first 4 hours of plethysmogram.

scan writes, per recording and candidate, the best delay and its exact-match share (and the share at a few delays
around it); agg ranks the candidates (wins, pooled exact matches), gives the winner's delay and implied window
centre (delay + half the window, at the recording's median pulse interval), and how many recordings a mean beats a
median of the same N. Aggregates only; per-recording output stays on this machine.
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
from pat import ppg_points, sample_times  # noqa: E402

warnings.filterwarnings("ignore")
CHAIN_S = 0.385
DELAYS = np.round(np.arange(0, 10.001, 0.25), 2)
NS = list(range(4, 17))
TS = list(range(4, 17))
MAX_RECORDS = 3000
MAX_HOURS = 4.0


def pulses(t, x, fs):
    ok = np.isfinite(x)
    if ok.mean() < 0.5:
        return np.empty(0)
    x = np.where(ok, x, np.nanmedian(x))
    _, d = ppg_points(x, fs)
    prom = 0.3 * (np.quantile(d, 0.995) - np.quantile(d, 0.5))
    pk, _ = signal.find_peaks(d, distance=int(0.3 * fs), prominence=prom)
    pk = pk[ok[pk]]
    return t[pk] - CHAIN_S


def roll_n(iv, N, how):
    """Statistic of the N intervals ending at each interval index (NaN before N are available)."""
    out = np.full(iv.size, np.nan)
    if iv.size < N:
        return out
    w = np.lib.stride_tricks.sliding_window_view(iv, N)
    if how == "mean":
        out[N - 1:] = w.mean(axis=1)
    elif how == "median":
        out[N - 1:] = np.median(w, axis=1)
    elif how == "tmean":           # mean without the longest and the shortest interval
        out[N - 1:] = (w.sum(axis=1) - w.max(axis=1) - w.min(axis=1)) / (N - 2)
    elif how == "rmean":           # mean of the intervals within 25% of the window's median
        med = np.median(w, axis=1)[:, None]
        keep = np.abs(w - med) <= 0.25 * med
        out[N - 1:] = np.where(keep, w, 0).sum(axis=1) / np.maximum(keep.sum(axis=1), 1)
    return out


def scan_case(c, root):
    from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
    v = read_vital(Path(root) / "vital_files" / f"{c:04d}.vital", track_names={"SNUADC/PLETH", "Solar8000/PLETH_HR"})
    pl, pr = v.tracks.get("SNUADC/PLETH"), v.tracks.get("Solar8000/PLETH_HR")
    if pl is None or pr is None or pr.record_times.size < 200:
        return {"case": c, "skip": "no PLETH with PLETH_HR"}
    tp, xp = sample_times(pl), pl.physical().astype(float)
    keep = tp < tp[0] + MAX_HOURS * 3600
    tp, xp = tp[keep], xp[keep]
    pt = pulses(tp, xp, float(pl.srate))
    if pt.size < 200:
        return {"case": c, "skip": "too few pulses"}
    iv = np.diff(pt)                       # interval i ends at pulse i+1
    te = pt[1:]
    good_iv = (iv > 0.25) & (iv < 2.5)
    ts, ys = pr.record_times.astype(float), pr.physical().astype(float)
    k = np.isfinite(ys) & (ys >= 30) & (ys <= 200) & (ts > te[0] + 30) & (ts < te[-1])
    ts, ys = ts[k], ys[k]
    if ts.size < 100:
        return {"case": c, "skip": "too few PLETH_HR values over the pulses"}
    if ts.size > MAX_RECORDS:
        ts, ys = ts[np.linspace(0, ts.size - 1, MAX_RECORDS).astype(int)], ys[np.linspace(0, ts.size - 1, MAX_RECORDS).astype(int)]
    yi = np.round(ys).astype(int)
    cands = {}
    for N in NS:
        cands[f"mean|n|{N}"] = ("n", roll_n(iv, N, "mean"), N)
        cands[f"median|n|{N}"] = ("n", roll_n(iv, N, "median"), N)
        if 6 <= N <= 14:
            cands[f"tmean|n|{N}"] = ("n", roll_n(iv, N, "tmean"), N)
            cands[f"rmean|n|{N}"] = ("n", roll_n(iv, N, "rmean"), N)
    cs = np.r_[0.0, np.cumsum(iv)]
    out = {"case": c, "records": int(ts.size), "pulses": int(pt.size), "median_interval_s": round(float(np.median(iv[good_iv])), 3),
           "cands": {}}
    for name, how in [(f"mean|t|{T}", ("t", "mean", T)) for T in TS] + [(f"count|t|{T}", ("t", "count", T)) for T in TS]:
        cands[name] = how
    for name, spec in cands.items():
        best = None
        rows = []
        for d in DELAYS:
            q = ts - d
            j = np.searchsorted(te, q, side="right") - 1          # last interval ended by q
            ok = j >= 0
            val = np.full(q.size, np.nan)
            if spec[0] == "n":
                stat = spec[1]
                val[ok] = 60.0 / stat[j[ok]]
            else:
                _, kind, T = spec
                j0 = np.searchsorted(te, q - T, side="right")       # intervals ending in (q - T, q]
                n_ = j - j0 + 1
                m = ok & (n_ >= 1)
                if kind == "mean":
                    val[m] = 60.0 / ((cs[j[m] + 1] - cs[j0[m]]) / n_[m])
                else:
                    val[m] = 60.0 * n_[m] / T
            fin = np.isfinite(val)
            trunc = np.mean(np.floor(val[fin]).astype(int) == yi[fin]) if fin.any() else 0.0
            rnd = np.mean(np.round(val[fin]).astype(int) == yi[fin]) if fin.any() else 0.0
            rows.append((float(d), float(trunc), float(rnd)))
        r = np.array(rows)
        for col, lab in ((1, "trunc"), (2, "round")):
            i = int(np.argmax(r[:, col]))
            out["cands"][f"{name}|{lab}"] = [r[i, 0], round(r[i, col], 4)]
    return out


def scan(a) -> None:
    out = Path(a.out)
    done = {json.loads(l)["case"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    t0 = time.time()
    with open(out, "a") as fh:
        for c in [int(x) for x in a.cases.split(",") if x]:
            if c in done:
                continue
            if a.time_budget and time.time() - t0 > a.time_budget:
                break
            t1 = time.time()
            try:
                r = scan_case(c, a.root)
            except Exception as err:  # noqa: BLE001
                r = {"case": c, "error": f"{type(err).__name__}: {err}"[:300]}
            r["secs"] = round(time.time() - t1, 1)
            fh.write(json.dumps(r) + "\n")
            fh.flush()


def agg(a) -> None:
    rows = [json.loads(l) for p in a.parts for l in Path(p).read_text().splitlines() if l.strip()]
    rows = list({r["case"]: r for r in rows}.values())
    ok = [r for r in rows if r.get("cands")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = sorted(ok[0]["cands"]) if ok else []
    wins = {n: 0 for n in names}
    pooled = {n: [0.0, 0] for n in names}
    for r in ok:
        sc = {n: r["cands"][n][1] for n in names}
        wins[max(sc, key=sc.get)] += 1
        for n in names:
            pooled[n][0] += sc[n] * r["records"]; pooled[n][1] += r["records"]
    rank = sorted(names, key=lambda n: -pooled[n][0] / pooled[n][1])
    top = rank[0]
    delays = np.array([r["cands"][top][0] for r in ok])
    kind, _, w = top.split("|")[0], top.split("|")[1], top.split("|")[2]
    unit = top.split("|")[1]
    half = np.array([(int(w) * r["median_interval_s"] if unit == "n" else int(w)) / 2 for r in ok])
    mean_vs_median = {}
    for N in NS:
        for lab in ("trunc", "round"):
            mn, md = f"mean|n|{N}|{lab}", f"median|n|{N}|{lab}"
            if mn in names and md in names:
                mean_vs_median[f"{N}|{lab}"] = int(sum(r["cands"][mn][1] > r["cands"][md][1] for r in ok))
    fam_best = {}
    good = [r for r in ok if max(v[1] for v in r["cands"].values()) >= 0.8]
    for fam in ("mean", "median", "tmean", "rmean"):
        keys_ = [n for n in names if n.startswith(fam + "|n|") and n.endswith("|round")]
        if keys_:
            fam_best[fam] = {"pooled_exact_all": round(max(pooled[n][0] / pooled[n][1] for n in keys_), 4),
                             "best": max(keys_, key=lambda n: pooled[n][0] / pooled[n][1])}
    fam_wins = {}
    for r in good:
        sc = {}
        for fam in fam_best:
            sc[fam] = max(r["cands"][n][1] for n in names if n.startswith(fam + "|n|") and n.endswith("|round"))
        w_ = max(sc, key=sc.get)
        fam_wins[w_] = fam_wins.get(w_, 0) + 1
    summ = {"recordings_read": len(rows), "analysed": len(ok), "families": fam_best,
            "family_wins_well_reproduced": fam_wins, "well_reproduced": len(good),
            "ranking": [{"candidate": n, "pooled_exact": round(pooled[n][0] / pooled[n][1], 4), "wins": wins[n]} for n in rank[:15]],
            "best": {"candidate": top, "delay_median_s": float(np.median(delays)),
                     "delay_p10_p90_s": [float(np.percentile(delays, 10)), float(np.percentile(delays, 90))],
                     "window_centre_median_s": round(float(np.median(delays + half)), 2)},
            "recordings_where_mean_beats_median_of_same_N": mean_vs_median}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    for k in ("--root", "--cases", "--out"):
        s.add_argument(k, required=True)
    s.add_argument("--time-budget", type=float, default=0)
    g = sub.add_parser("agg")
    g.add_argument("parts", nargs="+")
    g.add_argument("--out", required=True)
    a = ap.parse_args()
    scan(a) if a.cmd == "scan" else agg(a)


if __name__ == "__main__":
    main()
