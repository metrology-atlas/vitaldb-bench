# SPDX-License-Identifier: MIT
"""BIS against SNUADC on their own sample counts, interval by interval, from the shared ECG-EEG bursts.

    python tools/direct_rate_intervals.py scan --root R --cases 1,2,3 --out part.jsonl [--time-budget 0]
    python tools/direct_rate_intervals.py agg part*.jsonl [--pc-rates DIR] [--keys ports.jsonl] [--out DIR]

Each device's samples are placed by their own sample count (physiosync.stamp_clock with the rate set to zero:
one offset per stretch between real gaps; the 17.7 ms the SNUADC loses at each step-mode reset added back). On that
timeline neither the recorder's corrections nor the PC enter, so the EEG-minus-ECG burst lag ramps at the rate of
the BIS's crystal against the SNUADC's (ppm; negative when the lag falls, as in the 191-case pilot's -72 ppm).

scan writes, per recording: the good-window burst lags (10-min windows stepped by 5 min, ECG II ~ EEG1 and EEG2),
each device's correction times and stretch breaks, and its stamp-fitted rate against the PC.
agg fits the lag's slope in every interval between consecutive events of either device (a correction or a stretch
break) with at least 4 good windows over 40 minutes, and reports per recording the median interval rate, the
spread of interval rates against what the slope errors alone predict, and, with --pc-rates, the closure against
the two devices' rates from their correction steps (direct = -(BIS - SNUADC), both against the PC).
It also tests the anticorrelation of the two devices' interval drifts against the PC (device-correction-schedules,
r = -0.54): if the oscillators really move in opposite directions, the direct rate must vary between intervals by
about the sum of their swings; if it does not, the anticorrelation lies in how the PC rates are measured.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from physiosync import corrected_times, stamp_clock  # noqa: E402
from shared_artifacts import BAND, DT, burst_probe, envelope  # noqa: E402
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital  # noqa: E402

TR = {"ecg": ("SNUADC/ECG_II", -1, 0.0177), "eeg1": ("BIS/EEG1_WAV", 1, 0.0), "eeg2": ("BIS/EEG2_WAV", 1, 0.0)}
GOOD_PEAK, GOOD_MARGIN, LAG_RANGE_MS = 0.3, 0.15, (-2500.0, 500.0)
MIN_WIN, MIN_SPAN_S = 4, 2400.0


def scan_case(root, c):
    v = read_vital(Path(root) / "vital_files" / f"{c:04d}.vital", track_names={n for n, _, _ in TR.values()})
    times, xs, clocks = {}, {}, {}
    for k, (n, sign, loss) in TR.items():
        tr = v.tracks.get(n)
        if tr is None or not tr.record_times.size:
            continue
        x = tr.physical().astype(np.float32)
        x = np.nan_to_num(x, nan=float(np.nanmedian(x)))
        st = tr.record_offsets[: tr.record_times.size]
        fs = float(tr.srate)
        clk = stamp_clock(tr.record_times, st, fs, correction_sign=sign, event_loss_s=loss)
        counts = dict(clk, rate_ppm=0.0)          # same stretches and losses, no rate: the device's own count
        times[k] = corrected_times(counts, st, tr.record_sizes[: st.size])[: x.size]
        xs[k] = (x, fs)
        brk = [float(clk["t0"] + st[i] / fs) for i in clk["breaks"][1:] if i < st.size]
        clocks[k] = {"rate_pc_ppm": round(clk["rate_ppm"], 2), "source": clk["rate_source"], "teeth": clk["teeth"],
                     "events_s": [round(t, 1) for t in clk["corrections"]["event_times_s"]],
                     "breaks_s": [round(t, 1) for t in brk], "median_event_s": clk["corrections"]["median_event_s"]}
    if not {"ecg", "eeg1"} <= set(times):
        return {"case": c, "skip": "no ECG or EEG"}
    t0 = float(np.floor(min(t[0] for t in times.values())))
    n = int((max(t[-1] for t in times.values()) - t0) / DT) + 1
    env = {k: envelope(times[k], xs[k][0][: times[k].size], xs[k][1], BAND[k], t0, n) for k in times}
    b = burst_probe(env, 600, 300, 20)
    keep = {}
    for pair in ("ecg~eeg1", "ecg~eeg2"):
        keep[pair] = [w[:4] for w in b.get(pair, [])
                      if w[2] >= GOOD_PEAK and w[2] - w[3] >= GOOD_MARGIN and LAG_RANGE_MS[0] < w[1] < LAG_RANGE_MS[1]]
    # event times on the stamps are close enough to place interval boundaries (windows are 10 min)
    return {"case": c, "t0": t0, "clocks": clocks, "windows": keep}


def scan(a):
    cases = [int(x) for x in a.cases.split(",") if x]
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)["case"] for l in open(a.out, encoding="utf-8") if l.strip()}
    t_start = time.time()
    with open(a.out, "a", encoding="utf-8") as f:
        for c in cases:
            if c in done:
                continue
            if a.time_budget and time.time() - t_start > a.time_budget:
                break
            t1 = time.time()
            try:
                r = scan_case(a.root, c)
            except Exception as e:  # noqa: BLE001
                r = {"case": c, "error": f"{type(e).__name__}: {e}"[:300]}
            r["seconds"] = round(time.time() - t1, 1)
            f.write(json.dumps(r) + "\n")
            f.flush()


def intervals(r, pair="ecg~eeg1"):
    """Slope (ppm) of the lag in each interval between events of either device; with its standard error."""
    W = np.array(r["windows"].get(pair, []), float)
    if len(W) < MIN_WIN:
        return []
    t = W[:, 0] + r["t0"]
    lag = W[:, 1] / 1000.0
    ev = sorted(set(sum((r["clocks"][k]["events_s"] + r["clocks"][k]["breaks_s"] for k in ("ecg", "eeg1")
                         if k in r["clocks"]), [])))
    edges = np.r_[-np.inf, ev, np.inf]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (t > lo + 300) & (t < hi - 300)          # windows wholly inside the interval
        if s.sum() < MIN_WIN or np.ptp(t[s]) < MIN_SPAN_S:
            continue
        x, y = t[s] - t[s].mean(), lag[s]
        b = float(np.sum(x * y) / np.sum(x * x))
        res = y - y.mean() - b * x
        se = float(np.sqrt(np.sum(res ** 2) / max(len(x) - 2, 1) / np.sum(x * x)))
        out.append({"mid": float(t[s].mean()), "n": int(s.sum()), "span_h": float(np.ptp(t[s]) / 3600),
                    "ppm": b * 1e6, "se_ppm": se * 1e6})
    return out


def agg(a):
    import pandas as pd
    R = [json.loads(l) for p in a.parts for l in open(p, encoding="utf-8") if l.strip()]
    rows, ints = [], []
    for r in R:
        if "windows" not in r:
            continue
        I = intervals(r)
        for i in I:
            ints.append(dict(case=r["case"], **i))
        if not I:
            continue
        p = np.array([i["ppm"] for i in I]); se = np.array([i["se_ppm"] for i in I])
        w = 1 / np.maximum(se, 0.5) ** 2
        rows.append(dict(case=r["case"], intervals=len(I), direct_ppm=float(np.sum(w * p) / np.sum(w)),
                         direct_median_ppm=float(np.median(p)),
                         sd_between_ppm=float(np.std(p, ddof=1)) if len(p) > 1 else np.nan,
                         sd_expected_ppm=float(np.sqrt(np.mean(se ** 2))) if len(p) > 1 else np.nan,
                         rate_pc_ecg=r["clocks"]["ecg"]["rate_pc_ppm"], src_ecg=r["clocks"]["ecg"]["source"],
                         rate_pc_eeg=r["clocks"]["eeg1"]["rate_pc_ppm"], src_eeg=r["clocks"]["eeg1"]["source"],
                         ecg_event_s=r["clocks"]["ecg"]["median_event_s"]))
    P, I = pd.DataFrame(rows), pd.DataFrame(ints)
    def pct(s):
        s = pd.Series(s, dtype=float).dropna()
        return [round(float(x), 2) for x in np.percentile(s, [10, 25, 50, 75, 90])] + [int(s.size)] if s.size else None
    step = P.ecg_event_s.fillna(0) <= -0.4
    multi = P.intervals >= 2
    out = {"recordings_read": len(R), "errors": sum(1 for r in R if "error" in r),
           "with_direct_rate": len(P), "intervals": len(I),
           "direct_ppm": {"all": pct(P.direct_ppm), "snuadc_step_mode": pct(P.direct_ppm[step]),
                          "no_snuadc_step_correction": pct(P.direct_ppm[~step])},
           "interval_se_ppm_median": round(float(I.se_ppm.median()), 2) if len(I) else None,
           "within_recording": {"recordings_2plus_intervals": int(multi.sum()),
                                "sd_between_intervals_ppm": pct(P.sd_between_ppm[multi]),
                                "sd_expected_from_slope_errors_ppm": pct(P.sd_expected_ppm[multi])},
           "pilot_reference": {"direct_median_ppm": -72, "cases": 191}}
    if a.pc_rates:
        d = Path(a.pc_rates)
        s = pd.read_csv(d / "snuadc_reset_drift_per_recording.csv")[["case", "ppm"]].rename(columns={"ppm": "snu_pc"})
        b = pd.read_csv(d / "bis_drift_per_recording.csv")[["case", "ppm"]].rename(columns={"ppm": "bis_pc"})
        C = P.merge(s, on="case").merge(b, on="case")
        C["closure"] = C.direct_ppm + (C.bis_pc - C.snu_pc)
        out["closure_vs_correction_steps"] = {"recordings": len(C), "closure_ppm": pct(C.closure),
                                              "abs_median": round(float(C.closure.abs().median()), 2) if len(C) else None}
    if a.keys:
        keys = {}
        for line in open(a.keys, encoding="utf-8"):
            r = json.loads(line); p = r.get("ports") or {}
            if all(p.get(k) for k in ("SNUADC", "Solar8000", "Primus", "BIS")):
                keys[r["case"]] = " · ".join(p[k] for k in ("SNUADC", "Solar8000", "Primus", "BIS"))
        P["key"] = P.case.map(keys)
        g = P.dropna(subset=["key"]).groupby("key").direct_ppm
        K = pd.DataFrame({"n": g.size(), "median": g.median()}).query("n >= 5")
        med = P.dropna(subset=["key"]).groupby("key").direct_ppm.transform("median")
        x = P.dropna(subset=["key"])
        out["by_pc_key"] = {"keys_with_5": int(len(K)), "key_medians_ppm": pct(K["median"]),
                            "within_3ppm_of_key_median": round(float((abs(x.direct_ppm - med) <= 3)[x.key.isin(K.index)].mean()), 3) if len(K) else None,
                            "within_3ppm_of_release_median": round(float((abs(x.direct_ppm - x.direct_ppm.median()) <= 3).mean()), 3)}
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "summary.json").write_text(json.dumps(out, indent=1))
        P.to_csv(Path(a.out) / "direct_rate_per_recording.csv", index=False)
        I.to_csv(Path(a.out) / "direct_rate_per_interval.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scan"); s.add_argument("--root", required=True); s.add_argument("--cases", required=True)
    s.add_argument("--out", required=True); s.add_argument("--time-budget", type=float, default=0)
    g = sp.add_parser("agg"); g.add_argument("parts", nargs="+"); g.add_argument("--pc-rates"); g.add_argument("--keys")
    g.add_argument("--out")
    a = ap.parse_args()
    scan(a) if a.cmd == "scan" else agg(a)


if __name__ == "__main__":
    main()
