# SPDX-License-Identifier: MIT
"""Each device port's typical clock rate against the recorder PC, from whole teeth: the fallback rate the stamp
model gives a track whose own stamps hold no whole tooth. This machine only; the output is aggregates (no case ids).

    python tools/port_rates.py CENSUS_PARTS... --ports ports.jsonl --out-snuadc snuadc_step_rates.json \
        --out-bis bis_port_rates.json [--min-n 10] [--summary summary.json]

CENSUS_PARTS: tools/clock_correction_periods.py output (physiosync.stamp_clock per track: rate_ppm, events,
event_intervals_s = the whole teeth, median_event_s). ports.jsonl: each recording's device ports (the setup).

A track's rate counts only if it comes from at least one whole tooth (two corrections in one stretch): with fewer,
stamp_clock's rate is a line fitted through a one-step staircase, not a rate (examples/sim_stamp_clock_onecorr.py;
physiosync 0.3.4 rate_measured). SNUADC: step-mode tracks only (median correction -0.4 s or larger), as the stamp
model uses the rate for step-mode tracks. BIS: every track with a whole tooth. Per port, the median over tracks with
at least --min-n; "*" is the median over all counted tracks. Writes the two JSON files the stamp model reads
(port -> ppm) and, with --summary, the counts behind each value.

Replaces the hand-made $W/results/pat/snuadc_step_rates.json (11 Oct) and bis_port_rates.json (19 Oct), which had no
saved generating script (9 Nov 2026).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("parts", nargs="+")
    ap.add_argument("--ports", required=True)
    ap.add_argument("--out-snuadc", required=True)
    ap.add_argument("--out-bis", required=True)
    ap.add_argument("--min-n", type=int, default=10)
    ap.add_argument("--summary")
    a = ap.parse_args()
    ports = {}
    for l in open(a.ports):
        d = json.loads(l)
        ports[d["case"]] = {k: str(v).replace("\x00", "") for k, v in d["ports"].items()}
    rates = {"SNUADC": defaultdict(list), "BIS": defaultdict(list)}
    skipped = defaultdict(int)
    for f in a.parts:
        for l in open(f):
            d = json.loads(l)
            for dev, track in (("SNUADC", "SNUADC/ECG_II"), ("BIS", "BIS/EEG1_WAV")):
                t = d.get(track)
                if not isinstance(t, dict) or t.get("rate_ppm") is None:
                    continue
                if not t.get("event_intervals_s"):
                    skipped[f"{dev} no whole tooth"] += 1
                    continue
                if dev == "SNUADC" and not (t.get("median_event_s") is not None and t["median_event_s"] <= -0.4):
                    skipped["SNUADC not step mode"] += 1
                    continue
                port = ports.get(d["case"], {}).get(dev)
                if port:
                    rates[dev][port].append(float(t["rate_ppm"]))
    summ = {"min_n": a.min_n, "skipped": dict(skipped)}
    for dev, out in (("SNUADC", a.out_snuadc), ("BIS", a.out_bis)):
        by = rates[dev]
        res = {p: round(float(np.median(v)), 1) for p, v in sorted(by.items()) if len(v) >= a.min_n}
        allv = [x for v in by.values() for x in v]
        res["*"] = round(float(np.median(allv)), 1)
        json.dump(res, open(out, "w"), indent=1)
        summ[dev] = {"tracks": len(allv), "ports": {p: {"n": len(v), "median_ppm": round(float(np.median(v)), 1),
                                                       "iqr_ppm": np.round(np.percentile(v, [25, 75]), 1).tolist()}
                                                   for p, v in sorted(by.items(), key=lambda kv: -len(kv[1]))}}
    if a.summary:
        json.dump(summ, open(a.summary, "w"), indent=1)
    print(json.dumps({k: (v if k in ("min_n", "skipped") else {"tracks": v["tracks"]}) for k, v in summ.items()}))


if __name__ == "__main__":
    main()
