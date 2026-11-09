"""SNUADC: the rate bound for recordings with no resynchronisation (H.-C. Lee, 9 Nov 2026).

Vital Recorder counts SNUADC samples from an anchor taken at arrival. When the counted time runs more than 0.5 s
ahead of the recorder's clock, it discards the block just read and re-anchors (a step); when it falls behind by a
sample period or more, it inserts blank samples (frequent small corrections). So a recording with no correction of
either kind ran faster than the recorder's clock by less than 0.5 s over its whole length: |rate| < 0.5 s / T.

This checks the borrowed step-mode rates (setup, else port, else '*'; the v3 rates file) that sa_corrected model4
gives such recordings against that bound, and compares the ECG~EEG drift left by the burst check (burst_validation_m4b
per_recording.csv) between recordings whose borrowed rate breaks the bound and those whose rate does not. The
"implied" rate is the borrowed rate less the drift left (3.6 ms/h per ppm; a positive drift left is over-correction).
The capped estimate replaces a breaking rate by the bound and shifts the drift left accordingly: an estimate, not a
rerun. T is the recording's SNUADC duration (snuadc_mode.json dur_h).

  python examples/snuadc_rate_bound.py --modes $W/results/sync_full/snuadc_mode.json \
      --step-rates $W/port_rates/snuadc_step_rates_v3.json \
      --burst $W/burst_validation_m4b/per_recording.csv --out $W/port_rates/snuadc_rate_bound.json
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st

BOUND_S = 0.5
MS_H_PER_PPM = 3.6


def q(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(p * len(xs)))], 2) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", required=True); ap.add_argument("--step-rates", required=True)
    ap.add_argument("--burst", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    modes = json.load(open(a.modes)); rates = json.load(open(a.step_rates)); cases = rates.get("cases", {})
    res = {"bound_s": BOUND_S, "source": "H.-C. Lee, personal communication, 9 Nov 2026"}
    none = {}
    for c, v in modes.items():
        if v.get("steps", 0) == 0 and v.get("small", 0) == 0:
            port = v.get("port")
            if c in cases: r, tier = cases[c], "setup"
            elif port in rates: r, tier = rates[port], "port"
            else: r, tier = rates["*"], "star"
            b = BOUND_S / (v["dur_h"] * 3600) * 1e6
            none[c] = {"rate": abs(r), "tier": tier, "dur_h": v["dur_h"], "bound": b, "breaks": abs(r) > b}
    res["no_correction_recordings"] = len(none)
    res["by_tier"] = {t: {"n": sum(x["tier"] == t for x in none.values()),
                          "breaks": sum(x["tier"] == t and x["breaks"] for x in none.values())}
                      for t in ("setup", "port", "star")}
    res["breaks_total"] = sum(x["breaks"] for x in none.values())
    res["implied_drift_s"] = {"median": q([x["rate"] * x["dur_h"] * 3.6e-3 for x in none.values()], .5),
                              "breaking_over_1s": sum(x["rate"] * x["dur_h"] * 3.6e-3 > 1 for x in none.values())}
    res["bound_ppm"] = {"median": q([x["bound"] for x in none.values()], .5), "p10": q([x["bound"] for x in none.values()], .1)}
    res["dur_h_median"] = {"no_correction": q([x["dur_h"] for x in none.values()], .5),
                           "steps": q([v["dur_h"] for v in modes.values() if v.get("steps", 0) > 0], .5)}
    grp = {}
    for r in csv.DictReader(open(a.burst)):
        try: s = float(r["slope_ms_h"]); er = abs(float(r["ecg_rate"]))
        except (TypeError, ValueError): continue
        x = none.get(r["case"])
        if x and r["ecg_source"] == "given":
            g = "no_correction_breaks" if x["breaks"] else "no_correction_within"
            capped = s - (er - x["bound"]) * MS_H_PER_PPM if x["breaks"] else s
            grp.setdefault(g, []).append({"s": s, "imp_ratio": (er - s / MS_H_PER_PPM) / x["bound"], "capped": capped,
                                          "eeg_teeth": r["eeg_source"] == "teeth"})
        elif r["ecg_source"] == "teeth":
            grp.setdefault("whole_teeth", []).append({"s": s})
    res["burst_drift_left_ms_h"] = {}
    for g, L in grp.items():
        d = {"n": len(L), "median": q([x["s"] for x in L], .5), "median_abs": q([abs(x["s"]) for x in L], .5),
             "p90_abs": q([abs(x["s"]) for x in L], .9)}
        if "imp_ratio" in L[0]:
            d["implied_rate_over_bound"] = {"p25": q([x["imp_ratio"] for x in L], .25), "median": q([x["imp_ratio"] for x in L], .5),
                                            "p75": q([x["imp_ratio"] for x in L], .75)}
            d["capped_estimate"] = {"median": q([x["capped"] for x in L], .5), "median_abs": q([abs(x["capped"]) for x in L], .5),
                                    "p90_abs": q([abs(x["capped"]) for x in L], .9)}
            d["eeg_whole_teeth"] = sum(x["eeg_teeth"] for x in L)
        res["burst_drift_left_ms_h"][g] = d
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
