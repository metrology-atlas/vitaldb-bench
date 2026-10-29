# SPDX-License-Identifier: MIT
"""Summarise the whole-release delta scan (release_*.jsonl) per track: prevalence of the 240 Hz line and of the
Primus lines, the commonest change-pattern lines, value ranges, BIS grid, equal-delta runs against the smoothed null.

Usage: python tools/waveform_deltas_summary.py   (in the folder holding release_*.jsonl; writes release_summary.json)
"""
import json, glob, collections
import numpy as np
R = [json.loads(l) for f in sorted(glob.glob("release_*.jsonl")) for l in open(f)]
print("recordings", len(R), "with any waveform", sum(1 for r in R if r.get("tracks")))
T = collections.defaultdict(list)
for r in R:
    for k, v in (r.get("tracks") or {}).items():
        T[k].append((r["case"], v))
def has(lines, lo, hi, s=4.0): return any(lo <= f <= hi and st >= s for f, st in lines)
out = {}
for k in sorted(T, key=lambda k: -len(T[k])):
    vs = [v for _, v in T[k]]
    o = {"recordings": len(vs)}
    zl = [v.get("zero_delta_lines") or [] for v in vs]
    o["line_240"] = sum(has(l, 239.9, 240.05) for l in zl)
    f240 = [f for l in zl for f, st in l if 239.9 <= f <= 240.05 and st >= 4]
    if f240: o["f240_p5_50_95"] = np.percentile(f240, [5, 50, 95]).round(3).tolist()
    # commonest lines (rounded to 0.1 Hz) among the 3 strongest
    c = collections.Counter(round(f, 1) for l in zl for f, st in l if st >= 4)
    o["common_lines"] = [[f, n] for f, n in c.most_common(8)]
    o["zero_share_median"] = float(np.median([v["zero_share"] for v in vs]))
    o["sandwiched_median"] = float(np.median([v["sandwiched_zero_share"] for v in vs]))
    o["vmin_vmax_common"] = collections.Counter((v["vmin"], v["vmax"]) for v in vs).most_common(3)
    o["at_rail_median"] = float(np.median([v["at_vmin"] + v["at_vmax"] for v in vs]))
    if "equal_step2plus_pairs_share" in vs[0]:
        a = np.array([[v["equal_step2plus_pairs_share"], v["equal_step2plus_pairs_share_smoothed_null"]] for v in vs if "equal_step2plus_pairs_share" in v])
        o["equal_pairs_vs_null_median"] = np.median(a, 0).round(4).tolist(); o["equal_pairs_above_null"] = int((a[:, 0] > a[:, 1]).sum())
    if "lattice_step" in vs[0]:
        o["lattice_step"] = collections.Counter(v.get("lattice_step") for v in vs).most_common(4)
        o["lattice_share_median"] = float(np.median([v.get("lattice_share") or 0 for v in vs]))
        o["vmin_min_vmax_max"] = [min(v["vmin"] for v in vs), max(v["vmax"] for v in vs)]
    if k.startswith("Primus"):
        o["co2_12_5"] = sum(has(l, 12.4, 12.7) for l in zl); o["co2_25"] = sum(has(l, 24.9, 25.15) for l in zl)
        o["awp_16_6"] = sum(has(l, 16.5, 16.7) for l in zl); o["awp_29_3"] = sum(has(l, 29.2, 29.4) for l in zl)
    out[k] = o
json.dump(out, open("release_summary.json", "w"), indent=1, default=str)
for k, o in out.items(): print(k, json.dumps(o, default=str)[:900])
