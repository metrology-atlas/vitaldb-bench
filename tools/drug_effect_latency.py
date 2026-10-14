# SPDX-License-Identifier: MIT
"""Sanity check of the clocks of devices with no shared signal (Orchestra pumps, FMS rapid infuser, Invos):
is the time from an infusion event to its physiological effect plausible? Per case, this machine only.

    python tools/drug_effect_latency.py --root R --cases 1,2 --out out.jsonl
    python tools/drug_effect_latency.py --agg out*.jsonl --summary summary.json

Times are the files' stamps (all numerics carry recorder stamps). Events and responses:
  propofol start   Orchestra/PPF20_RATE from 0 (>= 5 min) to > 0, with BIS >= 80 over the minute before:
                   latency to BIS < 60 (held 30 s). Expected: tens of seconds to ~2 min (arm-brain circulation
                   plus effect-site equilibration).
  vasopressor step Orchestra/PHEN_RATE or NEPI_RATE rising by >= 50% (or from 0), none in the 5 min before:
                   MAP (Solar8000/ART_MBP, 2 s) over the next 5 min; latency to half the peak rise (rise >= 5
                   mmHg). Expected: ~30-120 s (line dead space and circulation).
  remifentanil up  Orchestra/RFTN20_RATE rising by >= 50% (or from 0): HR (Solar8000/HR) latency to half the peak
                   fall (fall >= 3 bpm) within 10 min. Expected: 1-5 min.
  infuser start    FMS/FLOW_RATE crossing 100 mL/min within 30 s of < 10: latency to half the peak rise of CVP and MAP
                   within 3 min (rise >= 2 mmHg). Expected: seconds to tens of seconds.
  Invos            SCO2_L/R against Solar8000/PLETH_SPO2: lag maximising the correlation of 10 s means after a
                   10 min median detrend (+-300 s), per case.
Every event-type also gets the same measurement at random times with the same exclusions (control).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital

TRACKS = {"Orchestra/PPF20_RATE", "Orchestra/PHEN_RATE", "Orchestra/NEPI_RATE", "Orchestra/RFTN20_RATE", "BIS/BIS",
          "Solar8000/ART_MBP", "Solar8000/HR", "Solar8000/CVP", "FMS/FLOW_RATE", "Invos/SCO2_L", "Invos/SCO2_R",
          "Solar8000/PLETH_SPO2"}


def tv(v, n, lo=None, hi=None):
    tr = v.tracks.get(n)
    if tr is None or not tr.record_times.size:
        return None
    t = np.asarray(tr.record_times, float)
    x = np.asarray(tr.physical(), float)[: t.size]
    o = np.argsort(t, kind="stable")
    t, x = t[o], x[o]
    ok = np.isfinite(x)
    if lo is not None:
        ok &= x >= lo
    if hi is not None:
        ok &= x <= hi
    return t[ok], x[ok]


def at(s, times):
    t, x = s
    i = np.searchsorted(t, times, side="right") - 1
    ic = np.clip(i, 0, t.size - 1)
    return np.where((i >= 0) & (np.abs(times - t[ic]) <= 10), x[ic], np.nan)


def rate_events(s, min_quiet=300.0, rel=0.5, from_zero_only=False, min_zero=300.0):
    """Times where a pump rate starts (from 0 held >= min_zero) or rises by >= rel with no other rise before."""
    t, x = s
    ev = []
    last = -np.inf
    for i in range(1, t.size):
        a, b = x[i - 1], x[i]
        if b <= a:
            continue
        start = a <= 0 < b
        if start:
            j = np.searchsorted(t, t[i] - min_zero)
            if np.any(x[j:i] > 0):
                start = False
        rise = (not from_zero_only) and a > 0 and b >= a * (1 + rel)
        if (start or rise) and t[i] - last >= min_quiet:
            ev.append(float(t[i]))
            last = t[i]
        elif b > a:
            last = t[i]
    return ev


def half_latency(resp, t0, horizon, sign, min_change, base_win=60.0, step=2.0):
    """Latency to half the peak change of resp (sign +1 rise, -1 fall) within horizon after t0."""
    tt = t0 + np.arange(-base_win, horizon + step, step)
    y = at(resp, tt)
    pre = y[tt < t0]
    post = y[tt >= t0]
    tp = tt[tt >= t0]
    if np.isfinite(pre).sum() < 0.5 * pre.size or np.isfinite(post).sum() < 0.5 * post.size:
        return None
    base = np.nanmedian(pre)
    ch = sign * (post - base)
    pk = np.nanmax(ch)
    if not np.isfinite(pk) or pk < min_change:
        return {"responded": False, "peak": round(float(pk), 2) if np.isfinite(pk) else None}
    k = np.flatnonzero(ch >= pk / 2)
    return {"responded": True, "latency": float(tp[k[0]] - t0), "peak": round(float(pk), 2),
            "t_peak": float(tp[int(np.nanargmax(ch))] - t0)}


def controls(t_lo, t_hi, avoid, n, rng, gap=600.0):
    out = []
    for c in rng.uniform(t_lo, t_hi, 20 * n):
        if len(out) >= n:
            break
        if all(abs(c - a) > gap for a in avoid):
            out.append(float(c))
    return out


def run_case(root, c, seed):
    v = read_vital(Path(root) / "vital_files" / f"{c:04d}.vital", track_names=TRACKS)
    rng = np.random.default_rng(seed + c)
    res = {"case": c}
    bis = tv(v, "BIS/BIS", 1, 100)
    mbp = tv(v, "Solar8000/ART_MBP", 20, 200)
    hr = tv(v, "Solar8000/HR", 25, 220)
    cvp = tv(v, "Solar8000/CVP", -5, 40)

    def block(name, events, resp, horizon, sign, min_change, need=None):
        out = {"events": [], "controls": []}
        if resp is None or not events:
            return out
        for e in events:
            if need is not None and not need(e):
                continue
            r = half_latency(resp, e, horizon, sign, min_change)
            if r is not None:
                out["events"].append({"t": round(e, 1), **r})
        if out["events"]:
            t_lo, t_hi = resp[0][0] + 120, resp[0][-1] - horizon
            for cc in controls(t_lo, t_hi, events, 5 * len(out["events"]), rng):
                if need is not None and not need(cc):
                    continue
                r = half_latency(resp, cc, horizon, sign, min_change)
                if r is not None:
                    out["controls"].append(r)
        res[name] = out

    ppf = tv(v, "Orchestra/PPF20_RATE", 0)
    if ppf is not None and bis is not None:
        ev = rate_events(ppf, from_zero_only=True)

        def awake(e):
            y = at(bis, e - np.arange(0, 60, 2.0))
            return np.isfinite(y).sum() >= 20 and np.nanmin(y) >= 80

        out = {"events": [], "controls": []}
        for e in ev:
            if not awake(e):
                continue
            tt = e + np.arange(0, 600, 2.0)
            y = at(bis, tt)
            below = np.r_[(y < 60).astype(int), 0]
            run = np.convolve(below, np.ones(15, int), "valid")
            k = np.flatnonzero(run >= 15)
            out["events"].append({"t": round(e, 1), "responded": bool(k.size), "latency": float(tt[k[0]] - e) if k.size else None})
        res["propofol_bis"] = out
    for n, key in (("Orchestra/PHEN_RATE", "phenylephrine_map"), ("Orchestra/NEPI_RATE", "norepinephrine_map")):
        s = tv(v, n, 0)
        if s is not None and mbp is not None:
            block(key, rate_events(s), mbp, 300, +1, 5)
    rf = tv(v, "Orchestra/RFTN20_RATE", 0)
    if rf is not None and hr is not None:
        block("remifentanil_hr", rate_events(rf, min_quiet=600), hr, 600, -1, 3)
    fms = tv(v, "FMS/FLOW_RATE", 0)
    if fms is not None:
        t, x = fms
        ev = []
        for i in range(1, t.size):
            if x[i - 1] < 100 <= x[i]:
                j = np.searchsorted(t, t[i] - 30)
                if np.any(x[j:i] < 10):
                    ev.append(float(t[i]))
        ev = [e for k, e in enumerate(ev) if k == 0 or e - ev[k - 1] > 300]
        if cvp is not None:
            block("infuser_cvp", ev, cvp, 180, +1, 2)
        if mbp is not None:
            block("infuser_map", ev, mbp, 180, +1, 2)
    spo2 = tv(v, "Solar8000/PLETH_SPO2", 50, 100)
    for side in ("L", "R"):
        so = tv(v, f"Invos/SCO2_{side}", 15, 100)
        if so is None or spo2 is None:
            continue
        t0 = max(so[0][0], spo2[0][0])
        t1 = min(so[0][-1], spo2[0][-1])
        if t1 - t0 < 1800:
            continue
        g = np.arange(t0, t1, 10.0)
        a, b = at(spo2, g), at(so, g)

        def detr(y):
            from scipy.ndimage import median_filter  # noqa: PLC0415
            f = np.where(np.isfinite(y), y, np.nanmedian(y))
            return y - median_filter(f, 61)

        a, b = detr(a), detr(b)
        best = None
        for L in range(-30, 31):
            if L >= 0:
                x1, x2 = a[: a.size - L], b[L:]
            else:
                x1, x2 = a[-L:], b[: b.size + L]
            ok = np.isfinite(x1) & np.isfinite(x2)
            if ok.sum() < 100 or np.std(x1[ok]) == 0 or np.std(x2[ok]) == 0:
                continue
            r = float(np.corrcoef(x1[ok], x2[ok])[0, 1])
            if best is None or r > best[1]:
                best = (L * 10, r)
        if best:
            res[f"invos_{side}_vs_spo2"] = {"lag_s": best[0], "r": round(best[1], 3)}
    return res


def agg(paths, out):
    R = []
    for p in paths:
        R += [json.loads(l) for l in open(p)]
    S = {"cases": len(R)}
    for key in ("propofol_bis", "phenylephrine_map", "norepinephrine_map", "remifentanil_hr", "infuser_cvp", "infuser_map"):
        E = [e for r in R if key in r for e in r[key]["events"]]
        C = [e for r in R if key in r for e in r[key].get("controls", [])]
        if not E:
            continue
        lat = [e["latency"] for e in E if e.get("responded") and e.get("latency") is not None]
        clat = [e["latency"] for e in C if e.get("responded") and e.get("latency") is not None]
        S[key] = {"cases": sum(1 for r in R if key in r and r[key]["events"]), "events": len(E),
                  "responded": len(lat), "responded_share": round(len(lat) / len(E), 3),
                  "control_responded_share": round(len(clat) / len(C), 3) if C else None,
                  "latency_s_p10_25_50_75_90": np.percentile(lat, [10, 25, 50, 75, 90]).round(0).tolist() if lat else None,
                  "control_latency_s_p50": float(np.median(clat)) if clat else None,
                  "negative_or_zero_latency": int(sum(x <= 0 for x in lat))}
    for side in ("L", "R"):
        k = f"invos_{side}_vs_spo2"
        X = [r[k] for r in R if k in r]
        if X:
            good = [x for x in X if x["r"] >= 0.3]
            S[k] = {"cases": len(X), "r_ge_0.3": len(good),
                    "lag_s_p10_50_90": np.percentile([x["lag_s"] for x in good], [10, 50, 90]).round(0).tolist() if good else None,
                    "r_p50": float(np.median([x["r"] for x in X]))}
    json.dump(S, open(out, "w"), indent=1)
    print(json.dumps(S, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root")
    ap.add_argument("--cases")
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--agg", nargs="+")
    ap.add_argument("--summary")
    a = ap.parse_args()
    if a.agg:
        agg(a.agg, a.summary)
        return
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)["case"] for l in open(a.out)}
    for c in [int(x) for x in a.cases.split(",") if x]:
        if c in done:
            continue
        try:
            r = run_case(a.root, c, a.seed)
        except Exception as e:  # noqa: BLE001
            r = {"case": c, "error": repr(e)[:300]}
        with open(a.out, "a") as fh:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
