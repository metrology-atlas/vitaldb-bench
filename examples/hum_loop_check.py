"""Can the mains hum close loops in the clock network? A diagnostic before building it into tools/clock_edges.py
(11 Nov 2026).

The hum route to the SNUADC: x[snuadc] = line + g_setup, where line = f_grid - f_SNUADC from the SNUADC waveforms'
hum (tools/mains_hum.py agg, per_recording table, snu_line) and g_setup = the median over the recording's setup (PC
key) of g = r - line in other recordings whose own rate r comes from whole teeth (two corrections in one stretch),
leaving the recording itself out, at least --min-setup recordings. Sign as the network: x = how much the PC gains on
the clock (ppm). No constant calibrated on the BIS (hum_snuadc_rate.py's k), so the BIS loop is not circular. r, and
so the route, is on the SNUADC's stamps (the reset arithmetic and the reset loss, ~1 + ~2.6 ppm, are not applied):
test 1 measures that offset where resets exist.

Tests, against the clock network's edge table (tools/clock_edges.py):
  1 hum - resets           recordings with a reset edge (snuadc_stamps-pc)
  2 hum - solar            recordings with a Solar8000 edge (snuadc-pc), with and without resets
  3 hum - BIS route        BIS route = bis_steps - bursts (x[bis]-x[pc] - (x[bis]-x[snuadc])), with and without resets
  4 the 245 BIS + bursts + solar loop recordings: how many have hum, and in those, hum - solar, hum - BIS route and
    solar - BIS route
Each: n, median, p25, p75, robust SD (1.4826 MAD).

    python examples/hum_loop_check.py --hum W/pc_grid_check/agg_check/per_recording_t1.csv \
        --edges W/clock_network/r6_inputs/edges_r6.csv --out OUT.json
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def stats(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if not v.size:
        return {"n": 0}
    med = float(np.median(v))
    return {"n": int(v.size), "median": round(med, 1), "p25": round(float(np.percentile(v, 25)), 1),
            "p75": round(float(np.percentile(v, 75)), 1), "robust_sd": round(float(1.4826 * np.median(np.abs(v - med))), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hum", required=True)
    ap.add_argument("--edges", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-setup", type=int, default=10)
    a = ap.parse_args()
    H = {}
    for r in csv.DictReader(open(a.hum)):
        f = lambda k: float(r[k]) if r.get(k) not in (None, "") else None
        H[int(r["case"])] = {"line": f("snu_line"), "r": f("snu_rate"), "teeth": f("snu_teeth") or 0,
                             "events": f("snu_events") or 0, "key": r["key"]}
    g_by_key = defaultdict(list)
    for c, h in H.items():
        if h["line"] is not None and h["r"] is not None and h["events"] and h["teeth"] >= 1:
            g_by_key[h["key"]].append((c, h["r"] - h["line"]))
    hum = {}
    for c, h in H.items():
        if h["line"] is None:
            continue
        others = [g for cc, g in g_by_key.get(h["key"], []) if cc != c]
        if len(others) >= a.min_setup:
            hum[c] = h["line"] + float(np.median(others))
    E = defaultdict(dict)
    for r in csv.DictReader(open(a.edges)):
        E[int(r["recording"])][r["method"]] = float(r["value"])
    has = lambda c, m: m in E.get(c, {})
    bis_route = lambda c: E[c]["bis_steps"] - E[c]["bursts"] if has(c, "bis_steps") and has(c, "bursts") else None
    res = {"recordings_with_hum_line": sum(h["line"] is not None for h in H.values()),
           "setups_with_g": sum(len(v) >= a.min_setup for v in g_by_key.values()),
           "recordings_with_hum_route": len(hum)}
    res["1_hum_minus_resets"] = stats(hum[c] - E[c]["resets"] for c in hum if has(c, "resets"))
    for lab, sel in (("with_resets", True), ("without_resets", False)):
        res[f"2_hum_minus_solar_{lab}"] = stats(hum[c] - E[c]["solar"] for c in hum if has(c, "solar") and has(c, "resets") == sel)
        res[f"3_hum_minus_bis_route_{lab}"] = stats(hum[c] - bis_route(c) for c in hum
                                                     if bis_route(c) is not None and has(c, "resets") == sel)
    loop = [c for c in E if has(c, "bis_steps") and has(c, "bursts") and has(c, "solar")]
    lh = [c for c in loop if c in hum]
    res["4_solar_bis_loop"] = {"recordings": len(loop), "with_hum": len(lh),
                               "hum_minus_solar": stats(hum[c] - E[c]["solar"] for c in lh),
                               "hum_minus_bis_route": stats(hum[c] - bis_route(c) for c in lh),
                               "solar_minus_bis_route": stats(E[c]["solar"] - bis_route(c) for c in lh),
                               "solar_minus_bis_route_all_loop": stats(E[c]["solar"] - bis_route(c) for c in loop)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
