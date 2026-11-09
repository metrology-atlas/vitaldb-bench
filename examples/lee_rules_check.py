"""BIS and Primus stamping rules (H.-C. Lee, 9 Nov 2026) against the whole-release correction census.

BIS: each EEG packet (0.125 s) is placed 0.125 s after the last while the sequence is consecutive and the result stays
within 2 s of arrival minus 1 s; otherwise it is re-anchored at arrival minus 1 s. If the 2 s rule made the
corrections, they would be about 2 s; if sequence breaks made them, they are the drift since the last anchor
(tenths of a second). And a BIS track with no correction could only break the 2 s bound beyond 2 s / rate hours.
Primus waveforms: counted at 1/62.5 s from an anchor, re-anchored only when the count and arrival differ by more than
5 s; the device's own gap is about 21 s every 7,230 s. Tabulates the stamp steps of Primus/AWP by size.

  python examples/lee_rules_check.py --census "$W/correction_census/part_*.jsonl" --out $W/port_rates/lee_rules_check.json
"""
import argparse
import collections
import glob
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--bis-rate-ppm", type=float, default=25.0, help="a typical BIS rate (atlas bis-rate)")
    a = ap.parse_args()
    bev, bbig, bnoev, prim, psmall, bn, pn = [], [], [], [], collections.Counter(), 0, 0
    for f in glob.glob(a.census):
        for line in open(f):
            r = json.loads(line); b = r.get("BIS/EEG1_WAV"); p = r.get("Primus/AWP")
            if b:
                bn += 1
                if b.get("median_event_s") is not None:
                    bev.append(b["median_event_s"])
                bbig += b.get("big_step_s") or []
                if not b.get("events"):
                    bnoev.append(b["hours"])
            if p:
                pn += 1; prim += p.get("big_step_s") or []
                for x in (p.get("small_step_ms") or [])[:1]:
                    psmall[round(x)] += 1
    bev, bbig, h, prim = map(np.array, (bev, bbig, bnoev, prim))
    q = lambda x, ps: [round(float(v), 3) for v in np.percentile(x, ps)]
    out = {"source": "H.-C. Lee, personal communication, 9 Nov 2026",
           "bis": {"tracks": bn, "with_corrections": int(bev.size),
                   "median_correction_s_p5_p50_p95": q(bev, [5, 50, 95]), "largest_median_correction_s": round(float(bev.max()), 3),
                   "no_correction_tracks": int(h.size), "no_correction_hours_p50_p90_max": q(h, [50, 90, 100]),
                   "hours_to_reach_2s_at_rate": round(2 / (a.bis_rate_ppm * 1e-6) / 3600, 1)},
           "primus_awp": {"tracks": pn, "big_steps": int(prim.size),
                          "device_gap_19_22s": int(np.sum((prim > 19) & (prim < 22))),
                          "other_over_5s": int(np.sum((np.abs(prim) > 5) & ~((prim > 19) & (prim < 22)))),
                          "between_1_and_5s": int(np.sum((np.abs(prim) > 1) & (np.abs(prim) <= 5))),
                          "up_to_1s": int(np.sum(np.abs(prim) <= 1)), "negative": int(np.sum(prim < 0)),
                          "negative_over_5s": int(np.sum(prim < -5)),
                          "first_small_step_ms_top5": psmall.most_common(5)}}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
