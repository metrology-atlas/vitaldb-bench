# SPDX-License-Identifier: MIT
"""Does a device round or truncate its numerics? Compare each numeric with the same quantity computed from the
device's own waveform. This machine only.

    python tools/numeric_vs_wave.py scan --root R --cases 1,2,3 --out DIR [--time-budget S]   # per case JSON
    python tools/numeric_vs_wave.py agg --out DIR                                              # summary.csv

Pairs (numeric <- waveform statistic over a window ending `lag` s before the numeric's stamp):
  Primus/ETCO2      <- max  of Primus/CO2       windows 5, 7, 10 s; lags 0, 0.5, 1 s
  Primus/PIP_MBAR   <- max  of Primus/AWP       same
  Primus/MAWP_MBAR  <- mean of Primus/AWP       same
  Solar8000/ART_SBP <- max  of SNUADC/ART       windows 2, 4, 8 s; lags 0, 1, 2, 3 s (another device's output)
  Solar8000/ART_MBP <- mean of SNUADC/ART       same
  Solar8000/ART_DBP <- min  of SNUADC/ART       same
For each pair, window and lag, per case: how many numerics equal round(W) and how many equal floor(W) (W the
waveform value, rounded to 1e-6), over all numerics and over those where round and floor differ (frac(W) >= 0.5).
A device that truncates matches floor; one that rounds matches round. Both match rates also show how well the
window reproduces the device's own computation; only the better window and lag per pair are informative.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# Waveform scale overrides, tried alongside the stored gain and offset: the pilot (19 Oct) found the Primus CO2
# waveform's stored gain 0.1 reads 4/3 of ETCO2, while recordings stored with 0.075 agree with it.
OVERRIDE = {"Primus/CO2@0.075": ("Primus/CO2", 0.075, 0.0), "Primus/AWP@0.1": ("Primus/AWP", 0.1, -20.0)}

PAIRS = [
    ("Primus/ETCO2", "Primus/CO2", "max", (5, 7, 10), (0, 0.5, 1)),
    ("Primus/ETCO2", "Primus/CO2@0.075", "max", (5, 7, 10), (0, 0.5, 1)),
    ("Primus/PIP_MBAR", "Primus/AWP", "max", (5, 7, 10), (0, 0.5, 1)),
    ("Primus/PIP_MBAR", "Primus/AWP@0.1", "max", (5, 7, 10), (0, 0.5, 1)),
    ("Primus/MAWP_MBAR", "Primus/AWP", "mean", (5, 7, 10), (0, 0.5, 1)),
    ("Primus/MAWP_MBAR", "Primus/AWP@0.1", "mean", (5, 7, 10), (0, 0.5, 1)),
    ("Solar8000/ART_SBP", "SNUADC/ART", "max", (2, 4, 8), (0, 1, 2, 3)),
    ("Solar8000/ART_MBP", "SNUADC/ART", "mean", (2, 4, 8), (0, 1, 2, 3)),
    ("Solar8000/ART_DBP", "SNUADC/ART", "min", (2, 4, 8), (0, 1, 2, 3)),
]


def wave(tr, gain=None, offset=None):
    fs = float(tr.srate)
    rt = np.asarray(tr.record_times, float); sz = np.asarray(tr.record_sizes, int)
    x = np.asarray(tr.values, float)
    st = np.r_[0, np.cumsum(sz)[:-1]]
    o = np.argsort(rt, kind="stable")
    t = np.concatenate([rt[k] + np.arange(sz[k]) / fs for k in o])
    v = np.concatenate([x[st[k]:st[k] + sz[k]] for k in o])
    g = float(tr.gain or 1) if gain is None else gain
    o0 = float(tr.offset or 0) if offset is None else offset
    v = v * g + o0
    ok = np.isfinite(v)
    return t[ok], v[ok]


def stat(t, v, tn, win, lag, how):
    e = np.searchsorted(t, tn - lag, side="right"); s = np.searchsorted(t, tn - lag - win, side="right")
    good = (e - s) >= 2
    s, e = s[good], e[good]
    if s.size == 0:
        return good, np.zeros(0)
    if how == "mean":
        c = np.r_[0, np.cumsum(v)]
        w = (c[e] - c[s]) / (e - s)
    else:
        idx = np.empty(2 * s.size, np.int64); idx[0::2] = s; idx[1::2] = np.minimum(e, v.size - 1)
        f = np.maximum if how == "max" else np.minimum
        w = f.reduceat(v, idx)[0::2]
        # reduceat over [s, e) when e < len(v); the last window is clipped to e = len(v) - 1, which is fine here
    return good, w


def scan_case(root, case):
    from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital  # noqa: PLC0415
    v = read_vital(Path(root) / "vital_files" / f"{case:04d}.vital")
    out = {}
    cache = {}
    for num, wav, how, wins, lags in PAIRS:
        src, g_, o_ = OVERRIDE.get(wav, (wav, None, None))
        if num not in v.tracks or src not in v.tracks:
            continue
        if wav not in cache:
            cache[wav] = wave(v.tracks[src], g_, o_)
        t, x = cache[wav]
        if t.size < 100:
            continue
        nt = v.tracks[num]
        tn = np.asarray(nt.record_times, float); vn = np.asarray(nt.values, float)
        ok = np.isfinite(vn) & (tn >= t[0] + 12) & (tn <= t[-1])
        tn, vn = tn[ok], vn[ok]
        if tn.size < 20:
            continue
        res = {"_gain": [float(v.tracks[src].gain), float(v.tracks[src].offset)]}
        for w in wins:
            for lg in lags:
                good, W = stat(t, x, tn, w, lg, how)
                if W.size == 0:
                    continue
                N = vn[good]; W = np.round(W, 6)
                r = np.floor(W + 0.5); f = np.floor(W)
                dif = r != f
                res[f"{w}|{lg}"] = [int(N.size), int((N == r).sum()), int((N == f).sum()), int(dif.sum()),
                                    int((N[dif] == r[dif]).sum()), int((N[dif] == f[dif]).sum()),
                                    float(np.median(N - W)),
                                    float(np.median(N[W > 0] / W[W > 0])) if (W > 0).any() else None]
        out[num + ("" if wav == src else " vs " + wav)] = res
    return out


def scan(a):
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for c in [int(x) for x in a.cases.split(",") if x]:
        f = od / f"{c}.json"
        if f.exists():
            continue
        if time.time() - t0 > a.time_budget:
            break
        try:
            r = scan_case(a.root, c)
        except Exception as e:  # noqa: BLE001
            r = {"error": repr(e)[:200]}
        tmp = od / f"{c}.tmp"
        tmp.write_text(json.dumps(r)); tmp.rename(f)


def agg(a):
    import pandas as pd
    rows = []
    for f in Path(a.out).glob("*.json"):
        r = json.loads(f.read_text())
        for num, d in r.items():
            if num == "error":
                continue
            gain = d.get("_gain", [None, None])
            for k, v in d.items():
                if k.startswith("_"):
                    continue
                w, lg = k.split("|")
                rows.append(dict(case=int(f.stem), track=num, window=float(w), lag=float(lg), n=v[0], m_round=v[1],
                                 m_floor=v[2], n_dif=v[3], d_round=v[4], d_floor=v[5], med_resid=v[6],
                                 med_ratio=v[7] if len(v) > 7 else None, wave_gain=gain[0], wave_offset=gain[1]))
    d = pd.DataFrame(rows)
    g = d.groupby(["track", "window", "lag"]).agg(cases=("case", "nunique"), n=("n", "sum"), m_round=("m_round", "sum"),
                                                   m_floor=("m_floor", "sum"), n_dif=("n_dif", "sum"),
                                                   d_round=("d_round", "sum"), d_floor=("d_floor", "sum"),
                                                   med_resid=("med_resid", "median")).reset_index()
    for k in ("m_round", "m_floor"):
        g[k + "_share"] = (g[k] / g.n).round(3)
    for k in ("d_round", "d_floor"):
        g[k + "_share"] = (g[k] / g.n_dif).round(3)
    g.to_csv(Path(a.out) / "summary.csv", index=False)
    d.to_csv(Path(a.out) / "by_case.csv", index=False)
    # the waveform's stored gain against the numeric: per configuration, the median numeric / waveform ratio
    cfg = d[d.window.isin([7.0, 4.0]) & (d.lag == 0)].groupby(["track", "wave_gain", "wave_offset"]).agg(
        cases=("case", "nunique"), ratio_med=("med_ratio", "median"), ratio_p10=("med_ratio", lambda x: x.quantile(.1)),
        ratio_p90=("med_ratio", lambda x: x.quantile(.9)), resid_med=("med_resid", "median")).reset_index()
    cfg.to_csv(Path(a.out) / "gain_check.csv", index=False)
    print(cfg.to_string(index=False)); print()
    g["best_match"] = g[["m_round_share", "m_floor_share"]].max(axis=1)
    best = g.loc[g.groupby("track").best_match.idxmax()]
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(g.sort_values(["track", "window", "lag"])[["track", "window", "lag", "cases", "n", "m_round_share", "m_floor_share",
                                                      "n_dif", "d_round_share", "d_floor_share", "med_resid"]].to_string(index=False))
    print("\nbest window per track:\n", best[["track", "window", "lag", "cases", "m_round_share", "m_floor_share",
                                              "d_round_share", "d_floor_share", "med_resid"]].to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scan"); s.add_argument("--root", required=True); s.add_argument("--cases", required=True)
    s.add_argument("--out", required=True); s.add_argument("--time-budget", type=float, default=1e9)
    g = sp.add_parser("agg"); g.add_argument("--out", required=True)
    a = p.parse_args()
    {"scan": scan, "agg": agg}[a.cmd](a)


if __name__ == "__main__":
    main()
