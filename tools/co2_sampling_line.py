# SPDX-License-Identifier: MIT
"""How far the stored Primus CO2 waveform trails the airway pressure, breath by breath (the sidestream sampling
line). This machine only for the scan; the aggregate is published.

    python tools/co2_sampling_line.py scan --root R --cases 1,2,3 --out part.jsonl [--minutes 60]
    python tools/co2_sampling_line.py agg PARTS... --summary summary.json

Both are Primus waveforms (62.5 Hz); each track's sample times come from its own record stamps. Marker of each
breath's inspiration start: AWP rising through 30% of its local range (20 s window), and the CO2 falling through 50%
of its local range (fresh gas reaching the analyser). Lag = CO2 fall - AWP rise, the first fall 0.3-10 s after the
rise. A second marker, expiration start (AWP falling through 50%, CO2 rising through 50%), checks the first. Per
recording: lag percentiles over up to --minutes, starting 5 min in. Resumable (recordings already in --out are
skipped). Whole release 29 Oct 2026: median lag 3.60 s (IQR 3.45-3.79), within-recording IQR 0.08 s.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital

FS = 62.5


def times(t):
    rt, rs = np.asarray(t.record_times, float), np.asarray(t.record_sizes)
    return np.concatenate([r + np.arange(s) / FS for r, s in zip(rt, rs)])


def crossings(x, tt, frac, up):
    w = int(FS * 20)
    lo = ndimage.percentile_filter(x, 5, size=w, mode="nearest")
    hi = ndimage.percentile_filter(x, 95, size=w, mode="nearest")
    ok = (hi - lo) > 0.2 * np.nanmedian(hi - lo) + 1e-9
    th = lo + frac * (hi - lo)
    a, b = x[:-1] - th[:-1], x[1:] - th[1:]
    idx = np.nonzero((a < 0) & (b >= 0) & ok[1:])[0] if up else np.nonzero((a > 0) & (b <= 0) & ok[1:])[0]
    f = a[idx] / (a[idx] - b[idx] + 1e-12)
    tc = tt[idx] + f * (tt[idx + 1] - tt[idx])
    keep, last = [], -1e9
    for t in tc:
        if t - last >= 1.5:
            keep.append(t)
            last = t
    return np.array(keep)


def lags(t_awp, t_co2):
    j = np.searchsorted(t_co2, t_awp + 0.3)
    ok = j < t_co2.size
    lag = np.full(t_awp.size, np.nan)
    lag[ok] = t_co2[j[ok]] - t_awp[ok]
    return lag[(lag > 0.3) & (lag < 10)]


def run_case(root, case, minutes):
    v = read_vital(Path(root) / "vital_files" / f"{case:04d}.vital", track_names={"Primus/AWP", "Primus/CO2"})
    ta, tc = v.tracks.get("Primus/AWP"), v.tracks.get("Primus/CO2")
    if ta is None or tc is None or not ta.record_times.size or not tc.record_times.size:
        return {"case": case, "skip": "no AWP or CO2"}
    xa, xc = np.asarray(ta.values, float), np.asarray(tc.values, float)
    sa, sc = times(ta), times(tc)
    t0 = max(sa[0], sc[0]) + 300
    t1 = min(sa[-1], sc[-1], t0 + 60 * minutes)
    ma, mc = (sa >= t0) & (sa < t1), (sc >= t0) & (sc < t1)
    if ma.sum() < FS * 120 or mc.sum() < FS * 120:
        return {"case": case, "skip": "too short"}
    xa, sa, xc, sc = xa[ma], sa[ma], xc[mc], sc[mc]
    xa = np.where(np.isfinite(xa), xa, np.nanmedian(xa))
    xc = np.where(np.isfinite(xc), xc, np.nanmedian(xc))
    insp_a, insp_c = crossings(xa, sa, 0.3, True), crossings(xc, sc, 0.5, False)
    exp_a, exp_c = crossings(xa, sa, 0.5, False), crossings(xc, sc, 0.5, True)
    out = {"case": case, "minutes": round((t1 - t0) / 60, 1), "breaths_awp": int(insp_a.size)}
    for name, lg in (("insp", lags(insp_a, insp_c)), ("exp", lags(exp_a, exp_c))):
        if lg.size >= 20:
            out[name] = {"n": int(lg.size), "p10_25_50_75_90": np.round(np.percentile(lg, [10, 25, 50, 75, 90]), 3).tolist(),
                         "share_of_breaths": round(lg.size / max(insp_a.size, 1), 3)}
    return out


def scan(a):
    out = Path(a.out)
    done = {json.loads(line)["case"] for line in out.read_text().splitlines() if line.strip()} if out.exists() else set()
    for c in [int(x) for x in a.cases.split(",") if x]:
        if c in done:
            continue
        t1 = time.time()
        try:
            res = run_case(a.root, c, a.minutes)
        except Exception as e:  # noqa: BLE001
            res = {"case": c, "error": repr(e)[:300]}
        res["secs"] = round(time.time() - t1, 1)
        with out.open("a") as fh:
            fh.write(json.dumps(res) + "\n")


def agg(a):
    R = [json.loads(line) for p in a.parts for line in open(p) if line.strip()]
    ok = [r for r in R if "insp" in r]
    med = np.array([r["insp"]["p10_25_50_75_90"][2] for r in ok])
    iqr = np.array([r["insp"]["p10_25_50_75_90"][3] - r["insp"]["p10_25_50_75_90"][1] for r in ok])
    ex = np.array([r["exp"]["p10_25_50_75_90"][2] for r in ok if "exp" in r])
    cs = np.array([r["case"] for r in ok])
    blocks = {f"{lo}-{lo + 799}": round(float(np.median(med[(cs >= lo) & (cs < lo + 800)])), 3)
              for lo in range(0, int(cs.max()) + 1, 800) if ((cs >= lo) & (cs < lo + 800)).any()}
    q = lambda x, p: [round(float(v), 3) for v in np.percentile(x, p)]
    s = {"recordings": len(R), "errors": sum("error" in r for r in R), "skipped": sum("skip" in r for r in R),
         "with_lag": len(ok), "lag_s_p5_10_25_50_75_90_95": q(med, [5, 10, 25, 50, 75, 90, 95]),
         "lag_robust_sd_s": round(float(1.4826 * np.median(np.abs(med - np.median(med)))), 3),
         "within_recording_iqr_s_median_p90": q(iqr, [50, 90]), "expiration_marker_lag_s_p10_50_90": q(ex, [10, 50, 90]),
         "share_below_2_5_s": round(float(np.mean(med < 2.5)), 3), "median_by_case_block": blocks}
    Path(a.summary).write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scan")
    s.add_argument("--root", required=True); s.add_argument("--cases", required=True); s.add_argument("--out", required=True)
    s.add_argument("--minutes", type=float, default=60)
    g = sp.add_parser("agg")
    g.add_argument("parts", nargs="+"); g.add_argument("--summary", required=True)
    a = p.parse_args()
    scan(a) if a.cmd == "scan" else agg(a)


if __name__ == "__main__":
    main()
