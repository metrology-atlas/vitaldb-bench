# SPDX-License-Identifier: MIT
"""Each device port's typical clock rate against the recorder PC, from whole teeth: the fallback rate the stamp
model gives a track whose own stamps hold no whole tooth. This machine only: the port rates are aggregates, but with
--setup-min both files also hold a per-recording table, which stays on the analysis machine.

    python tools/port_rates.py CENSUS_PARTS... --ports ports.jsonl --out-snuadc snuadc_step_rates.json \
        --out-bis bis_port_rates.json [--min-n 10] [--setup-min 5] [--snuadc-ports-from FILE] [--summary summary.json]

CENSUS_PARTS: tools/clock_correction_periods.py output (physiosync.stamp_clock per track: rate_ppm, events,
event_intervals_s = the whole teeth, median_event_s). ports.jsonl: each recording's device ports (the setup).

A track's rate counts only if it comes from at least one whole tooth (two corrections in one stretch): with fewer,
stamp_clock's rate is a line fitted through a one-step staircase, not a rate (examples/sim_stamp_clock_onecorr.py;
physiosync 0.3.4 rate_measured). SNUADC: step-mode tracks only (median correction -0.4 s or larger), as the stamp
model uses the rate for step-mode tracks. BIS: every track with a whole tooth. Per port, the median over tracks with
at least --min-n; "*" is the median over all counted tracks. Writes the two JSON files the stamp model reads
(port -> ppm) and, with --summary, the counts behind each value. With --setup-min N (> 0) each file also gets
"cases": every recording whose setup (its device ports) has at least N counted tracks of that device, mapped to that
setup's median, which tools/fallback_rate.py gives in place of the port rate (validated: examples/setup_rate_check.py).

--snuadc-ports-from FILE keeps the SNUADC port and "*" values of an earlier file instead of the whole-teeth medians.
A track falls back to its port only when its setup has too few whole teeth, and whole teeth need fast drift, so those
stations are the slow ones, while whole-teeth port medians are dominated by fast stations (COM4: -77.4 ppm against
the 11 Oct file's -65.4). In the whole-release burst validation (9 Nov, model4) the new port values left more drift
in the 941 recordings that fall back to a port (median 72-106 ms/h against 60-78 with the old values), so the
adopted file keeps the old port values with the new setup table ($W/port_rates/m3_m4_compare.json).

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
    ap.add_argument("--snuadc-ports-from", default=None, help="keep the SNUADC port and '*' values of this file")
    ap.add_argument("--setup-min", type=int, default=0, help="also write per-recording setup rates (setups with at "
                    "least this many whole-teeth step-mode tracks); 0 = off")
    ap.add_argument("--summary")
    a = ap.parse_args()
    ports = {}
    for l in open(a.ports):
        d = json.loads(l)
        ports[d["case"]] = {k: str(v).replace("\x00", "") for k, v in d["ports"].items()}
    rates = {"SNUADC": defaultdict(list), "BIS": defaultdict(list)}
    setup_key = lambda c: " ".join(f"{k}:{v}" for k, v in sorted(ports.get(c, {}).items()))
    by_setup = {"SNUADC": defaultdict(list), "BIS": defaultdict(list)}
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
                if d["case"] in ports:
                    by_setup[dev][setup_key(d["case"])].append(float(t["rate_ppm"]))
    summ = {"min_n": a.min_n, "skipped": dict(skipped)}
    for dev, out in (("SNUADC", a.out_snuadc), ("BIS", a.out_bis)):
        by = rates[dev]
        res = {p: round(float(np.median(v)), 1) for p, v in sorted(by.items()) if len(v) >= a.min_n}
        allv = [x for v in by.values() for x in v]
        res["*"] = round(float(np.median(allv)), 1)
        if dev == "SNUADC" and a.snuadc_ports_from:
            res = {k: v for k, v in json.loads(open(a.snuadc_ports_from).read()).items() if k != "cases"}
            summ["snuadc_ports_from"] = a.snuadc_ports_from
        if a.setup_min > 0:
            med = {k: round(float(np.median(v)), 1) for k, v in by_setup[dev].items() if len(v) >= a.setup_min}
            res["cases"] = {str(c): med[setup_key(c)] for c in sorted(ports) if setup_key(c) in med}
            summ.setdefault("setups", {})[dev] = {"with_rate": len(med), "cases_covered": len(res["cases"])}
        json.dump(res, open(out, "w"), indent=1)
        summ[dev] = {"tracks": len(allv), "ports": {p: {"n": len(v), "median_ppm": round(float(np.median(v)), 1),
                                                       "iqr_ppm": np.round(np.percentile(v, [25, 75]), 1).tolist()}
                                                   for p, v in sorted(by.items(), key=lambda kv: -len(kv[1]))}}
    if a.summary:
        json.dump(summ, open(a.summary, "w"), indent=1)
    print(json.dumps({k: (v if k in ("min_n", "skipped", "setups", "snuadc_ports_from") else {"tracks": v["tracks"]}) for k, v in summ.items()}))


if __name__ == "__main__":
    main()
