# SPDX-License-Identifier: MIT
"""Is AWP data lost in the Primus pauses, or did only its stamps jump? This machine only.

    python tools/primus_pause_test.py scan --root R --pauses primus_awp_pauses.csv --cases 1,2,3 --out DIR
    python tools/primus_pause_test.py agg --out DIR

The Primus AWP stops for 19-23.5 s every 7,230 s while CO2, from the same device and breaths, carries on
(observation primus-awp-pauses; pauses from tools/correction_schedules.py). For each pause: the lag of AWP behind
CO2 by cross-correlation (both resampled to 10 Hz on their stamps, band-passed 0.05-1 Hz, lags -40..+40 s) in the
120 s before the pause and in the 120 s after it. If AWP data are lost, the lag is the same after the pause; if
only the stamps jumped forward by the pause length, AWP after the pause sits that much later. Breaths repeat, so
each pause is ambiguous on its own: the agg step pools pauses at different breathing rates. Per pause it also
records the lag change at the best-correlated peak and the correlation at 0 and at the pause length.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ORIGIN = 4102444800.0
FS = 10.0


def series(tr):
    rt = np.asarray(tr.record_times, float) - ORIGIN; sz = np.asarray(tr.record_sizes, int); fs = float(tr.srate)
    v = np.asarray(tr.values, float) * float(tr.gain) + float(tr.offset)
    st = np.r_[0, np.cumsum(sz)[:-1]]
    t = np.concatenate([rt[k] + np.arange(sz[k]) / fs for k in range(rt.size)])
    o = np.argsort(t, kind="stable")
    return t[o], v[o]


def grid(t, v, a, b):
    g = np.arange(a, b, 1 / FS)
    m = (t >= a - 1) & (t <= b + 1)
    if m.sum() < 50:
        return None
    y = np.interp(g, t[m], v[m], left=np.nan, right=np.nan)
    # mark grid points far from any sample (inside a gap) as missing
    idx = np.searchsorted(t[m], g)
    idx = np.clip(idx, 1, m.sum() - 1)
    tt = t[m]
    dist = np.minimum(np.abs(tt[idx] - g), np.abs(tt[idx - 1] - g))
    y[dist > 0.2] = np.nan
    return y


def bandpass(y):
    from scipy.signal import butter, sosfiltfilt
    if y is None or np.isnan(y).mean() > 0.1:
        return None
    y = pd.Series(y).interpolate(limit_direction="both").to_numpy()
    sos = butter(2, [0.05, 1.0], btype="band", fs=FS, output="sos")
    z = sosfiltfilt(sos, y - y.mean())
    s = z.std()
    return z / s if s > 0 else None


def xcorr(a, b, maxlag):
    """corr of a(t) with b(t + lag) for lag in -maxlag..maxlag samples (positive: b later)."""
    n = a.size
    lags = np.arange(-maxlag, maxlag + 1)
    r = np.empty(lags.size)
    for i, L in enumerate(lags):
        if L >= 0:
            x, y = a[: n - L], b[L:]
        else:
            x, y = a[-L:], b[: n + L]
        r[i] = np.dot(x, y) / x.size
    return lags / FS, r


def scan_case(root, case, pauses):
    from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital  # noqa: PLC0415
    v = read_vital(Path(root) / "vital_files" / f"{case:04d}.vital", track_names={"Primus/AWP", "Primus/CO2"})
    if "Primus/AWP" not in v.tracks or "Primus/CO2" not in v.tracks:
        return []
    ta, va = series(v.tracks["Primus/AWP"]); tc, vc = series(v.tracks["Primus/CO2"])
    out = []
    for _, p in pauses.iterrows():
        t_end = float(p.t)            # end of the AWP record before the pause (s from origin)
        gap = float(p.step)
        rec = {"case": case, "t": t_end, "pause_s": gap}
        for side, (a, b) in {"before": (t_end - 125, t_end - 5), "after": (t_end + gap + 5, t_end + gap + 125)}.items():
            A = bandpass(grid(ta, va, a, b)); C = bandpass(grid(tc, vc, a, b))
            if A is None or C is None:
                rec[side] = None
                continue
            lags, r = xcorr(C, A, int(40 * FS))
            k = int(np.argmax(r))
            # breath period from CO2 autocorrelation
            l2, ac = xcorr(C, C, int(15 * FS)); m = (l2 > 1.5)
            per = float(l2[m][np.argmax(ac[m])]) if m.any() else None
            rec[side] = {"lag_s": float(lags[k]), "r": float(r[k]), "period_s": per,
                         "r_at_0": float(r[np.argmin(np.abs(lags))]),
                         "r_at_gap": float(r[np.argmin(np.abs(lags - gap))]) if gap <= 40 else None,
                         "curve": [round(float(x), 3) for x in r[::5]]}  # 0.5 s steps, -40..40 s
        out.append(rec)
    return out


def scan(a):
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    P = pd.read_csv(a.pauses)
    t0 = time.time()
    for c in [int(x) for x in a.cases.split(",") if x]:
        f = od / f"{c}.json"
        if f.exists():
            continue
        if time.time() - t0 > a.time_budget:
            break
        pc = P[P.case == c]
        try:
            res = scan_case(a.root, c, pc) if len(pc) else []
        except Exception as e:  # noqa: BLE001
            res = [{"case": c, "error": repr(e)[:200]}]
        tmp = od / f"{c}.tmp"; tmp.write_text(json.dumps(res)); tmp.rename(f)


def agg(a):
    """Per pause: the AWP-behind-CO2 lag before the pause (searched within +-5 s, where the physiological lag lies);
    after the pause, the correlation at that same lag (r_same: AWP lost) against the correlation at that lag plus
    the pause length (r_jump: only the stamps jumped). The before-window at lag + pause (r_alias) measures how
    ambiguous the breathing rhythm makes each pause; only pauses with r_alias well below r_same are decisive."""
    lags = np.round(np.arange(-40, 40.01, 0.5), 1)
    rows = []
    for f in Path(a.out).glob("*.json"):
        for r in json.loads(f.read_text()):
            if "error" in r or not r.get("before") or not r.get("after"):
                continue
            cb, ca = np.array(r["before"]["curve"]), np.array(r["after"]["curve"])
            near = np.abs(lags) <= 5
            k = np.flatnonzero(near)[np.argmax(cb[near])]
            lag0 = lags[k]
            def at(curve, x):
                if abs(x) > 40:
                    return np.nan
                return float(curve[np.argmin(np.abs(lags - x))])
            P = r["pause_s"]
            rows.append(dict(case=r["case"], pause=P, lag_before=lag0, r_before=float(cb[k]),
                             r_same=at(ca, lag0), r_jump=at(ca, lag0 + P), r_back=at(ca, lag0 - P),
                             r_alias=at(cb, lag0 + P), period=r["before"]["period_s"]))
    d = pd.DataFrame(rows)
    d.to_csv(Path(a.out) / "pauses.csv", index=False)
    ok = d[d.r_before > 0.5]
    dec = ok[(ok.r_alias < 0.5 * ok.r_before)]
    print(f"pauses: {len(d)}; AWP and CO2 correlated before the pause (r > 0.5): {len(ok)}; decisive "
          f"(rhythm not aliasing the pause length): {len(dec)} in {dec.case.nunique()} recordings")
    print("lag of AWP behind CO2 before the pause (s) quantiles", ok.lag_before.quantile([.1, .5, .9]).round(2).tolist())
    for nm, x in (("all correlated", ok), ("decisive", dec)):
        if len(x):
            print(f"  {nm}: median r_same {x.r_same.median():.2f}, r_jump {x.r_jump.median():.2f}, r_back {x.r_back.median():.2f}; "
                  f"share r_same > r_jump {(x.r_same > x.r_jump).mean():.2f}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for x, c, lab in ((ok[~ok.index.isin(dec.index)], "#bbbbbb", "rhythm aliases the pause"), (dec, "#1f5fa6", "decisive")):
        ax.scatter(x.r_jump, x.r_same, s=8, color=c, label=f"{lab} ({len(x):,})")
    ax.plot([-1, 1], [-1, 1], color="#7f7f7f", ls=":", lw=1)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_xlabel("after the pause: correlation if only the stamps jumped")
    ax.set_ylabel("after the pause: correlation if AWP data were lost")
    ax.set_title("Above the diagonal: AWP lost; below: stamps jumped", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(Path(a.out) / "primus_pause_test.png", dpi=120)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scan"); s.add_argument("--root", required=True); s.add_argument("--pauses", required=True)
    s.add_argument("--cases", required=True); s.add_argument("--out", required=True); s.add_argument("--time-budget", type=float, default=1e9)
    g = sp.add_parser("agg"); g.add_argument("--out", required=True)
    a = p.parse_args()
    {"scan": scan, "agg": agg}[a.cmd](a)


if __name__ == "__main__":
    main()
