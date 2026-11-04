# SPDX-License-Identifier: MIT
"""Tier I of the rebuilt sawtooth correction, by setup (metrology-atlas study ppg-sawtooth): the recordings with
one-frame slips but no fitted sawtooth. Are they sawteeth with too few confirmed slips, or isolated slips?

    python tools/tier_i_check.py --fit $W/slip_rebuild/full/fit.jsonl --teeth $W/slip_rebuild/full/tiers/teeth.csv \
        --rates $W/sawtooth_stats/recordings.csv --out $W/slip_rebuild/full/tier_i

Per recording whose slips are all tier I (no sawtooth fitted; slips kept because the shape, or the arterial line,
confirmed them): its SNUADC port and correction mode, the slips per hour and their directions, the arterial verdict at
each slip (where there is an arterial line), and the gaps between consecutive slips the same way against the setup's
tooth (the median tooth of the setup's sawtooth recordings, sawtooth_stats): a gap within 5% of a whole number (1-6)
of teeth is "on the setup's period". A recording with three or more slips the same way and most of its gaps on the
setup's period is a sawtooth the fit missed; one whose slips are spread off the period, or that has one or two, is
isolated slips. By setup: recordings, the share of the setup's recordings in tier I, and these measures. Writes
recordings.csv and summary.json.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("fit", "teeth", "rates", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    teeth = defaultdict(list)
    for row in csv.DictReader(open(a.teeth)):
        teeth[int(row["case"])].append(row["tier"])
    L_setup = defaultdict(list)
    for row in csv.DictReader(open(a.rates)):
        if row["direction"] == "up" and row["L_s"]:
            L_setup[(row["port"], row["mode"])].append(float(row["L_s"]))
    L_med = {k: float(np.median(v)) for k, v in L_setup.items() if len(v) >= 10}
    rows, all_by_setup = [], Counter()
    GAPS = {"gaps": 0, "on_whole_teeth": 0, "on_half_teeth_control": 0, "gap_in_teeth": []}
    for line in open(a.fit):
        r = json.loads(line)
        R = r.get("real")
        if not R:
            continue
        setup = (r.get("port") or "?", r.get("mode") or "?")
        all_by_setup[setup] += 1
        if set(teeth.get(r["case"], [])) != {"I"}:
            continue
        hours = (R["end"] - R["start"]) / 3600
        sl = sorted(R["slips"], key=lambda s: s["t"])
        ev = {}
        for src in R.get("predicted", []) + R.get("off_grid", []):
            if src.get("evidence") is not None:
                ev[round(src["t"], 1)] = src["evidence"].get("arterial")
        art = [ev.get(round(s["t"], 1)) for s in sl]
        signs = [s.get("sign", 1) for s in sl]
        main_sign = 1 if signs.count(1) >= signs.count(-1) else -1
        same = [s["t"] for s in sl if s.get("sign", 1) == main_sign]
        L = L_med.get(setup)
        if L and len(same) >= 2:
            for gap in np.diff(same):
                q = gap / L
                k_int, k_half = np.round(q), np.round(q - 0.5) + 0.5
                GAPS["gaps"] += 1
                GAPS["on_whole_teeth"] += int(1 <= k_int <= 6 and abs(q - k_int) <= 0.05 * k_int)
                GAPS["on_half_teeth_control"] += int(1.5 <= k_half <= 6.5 and abs(q - k_half) <= 0.05 * k_half)
                GAPS["gap_in_teeth"].append(round(float(q), 3))
        on = None
        if L and len(same) >= 2:
            g = np.diff(same) / L
            k = np.round(g)
            on = float(np.mean((k >= 1) & (k <= 6) & (np.abs(g - k) <= 0.05 * np.maximum(k, 1))))
        kind = ("missed sawtooth" if len(same) >= 3 and on is not None and on >= 0.5 else
                "isolated, 1-2 slips" if len(sl) <= 2 else "isolated, off the period" if on is not None else
                "isolated, setup period unknown")
        rows.append({"case": r["case"], "port": setup[0], "mode": setup[1], "hours": round(hours, 2), "slips": len(sl),
                     "slips_per_hour": round(len(sl) / hours, 2) if hours > 0 else None, "up": signs.count(1),
                     "down": signs.count(-1), "art_slip": art.count("slip"), "art_neither": art.count("neither"),
                     "art_none": art.count(None), "setup_tooth_s": round(L, 1) if L else None,
                     "gaps_on_period": round(on, 2) if on is not None else None, "fit_confirmed": (R.get("fit") or {}).get("confirmed"),
                     "kind": kind})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "recordings.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by = defaultdict(list)
    for x in rows:
        by[(x["port"], x["mode"])].append(x)
    S = {"fit": a.fit, "tier_i_recordings": len(rows), "kind": dict(Counter(x["kind"] for x in rows)),
         "slips": sum(x["slips"] for x in rows),
         "slips_per_recording": dict(Counter(min(x["slips"], 6) for x in sorted(rows, key=lambda x: x["slips"]))),
         "arterial_at_slips": {"slip": sum(x["art_slip"] for x in rows), "neither": sum(x["art_neither"] for x in rows),
                               "no arterial line": sum(x["art_none"] for x in rows)},
         "direction": {"all up": sum(1 for x in rows if x["down"] == 0), "all down": sum(1 for x in rows if x["up"] == 0),
                       "mixed": sum(1 for x in rows if x["up"] and x["down"])},
         "by_setup": {}}
    for k, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        if len(v) < 10:
            continue
        S["by_setup"][f"{k[0]} {k[1]}"] = {
            "tier_i": len(v), "recordings": all_by_setup[k], "share": round(len(v) / all_by_setup[k], 2),
            "setup_tooth_min": round(L_med[k] / 60, 1) if k in L_med else None,
            "kind": dict(Counter(x["kind"] for x in v)),
            "slips_per_hour_median": round(float(np.median([x["slips_per_hour"] for x in v if x["slips_per_hour"]])), 2),
            "share_up": round(sum(x["up"] for x in v) / max(1, sum(x["slips"] for x in v)), 2),
            "arterial_slip_share": round(sum(x["art_slip"] for x in v) / max(1, sum(x["art_slip"] + x["art_neither"] for x in v)), 2)
            if sum(x["art_slip"] + x["art_neither"] for x in v) else None}
    gq = np.array(GAPS.pop("gap_in_teeth"))
    GAPS["gap_in_teeth_quartiles"] = [round(float(v), 2) for v in np.percentile(gq, [25, 50, 75])] if gq.size else None
    S["gaps_against_setup_tooth"] = GAPS
    (out / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps({k: S[k] for k in ("tier_i_recordings", "kind", "slips", "slips_per_recording", "arterial_at_slips", "direction",
                                         "gaps_against_setup_tooth")}, indent=1))
    for k, v in S["by_setup"].items():
        print(k, v)


if __name__ == "__main__":
    main()
