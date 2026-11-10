"""Compare two sets of numeric delay labels (tools/numeric_delay_labels.py outputs), track by track (11 Nov 2026).

For the stage-3 rerun of tools/numeric_delay_variability.py on the v4 SNUADC rates: per track, the rows in each set,
the cases in both, the median delay (good rows), the per-case change (median, p10, p90, and how many move by more
than 0.1 s), and quality changes; for the HR delay groups, the counts and the old -> new transitions.

    python examples/delay_labels_compare.py OLD_DIR NEW_DIR --out compare.json
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def rows(d):
    out = defaultdict(dict)
    for r in csv.DictReader(open(Path(d) / "numeric_delay_labels.csv")):
        out[r["track"]][int(r["case"])] = r
    return out


def groups(d):
    return {int(r["case"]): r for r in csv.DictReader(open(Path(d) / "hr_delay_group_labels.csv"))}


def pct(v, q):
    return round(float(np.percentile(v, q)), 3) if len(v) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    A, B = rows(a.old), rows(a.new)
    res = {"old": a.old, "new": a.new, "tracks": {}}
    for t in sorted(set(A) | set(B)):
        x, y = A.get(t, {}), B.get(t, {})
        both = sorted(set(x) & set(y))
        dx = lambda r: float(r["delay_from_window_end_s"])
        ch = np.array([dx(y[c]) - dx(x[c]) for c in both])
        gx = [dx(r) for r in x.values() if r["quality"] == "good"]
        gy = [dx(r) for r in y.values() if r["quality"] == "good"]
        res["tracks"][t] = {
            "rows_old": len(x), "rows_new": len(y), "cases_both": len(both),
            "median_good_old": pct(gx, 50), "median_good_new": pct(gy, 50),
            "change_median": pct(ch, 50), "change_p10": pct(ch, 10), "change_p90": pct(ch, 90),
            "moved_over_0.1s": int(np.sum(np.abs(ch) > 0.1)) if len(ch) else 0,
            "quality_changes": dict(Counter(f"{x[c]['quality']}->{y[c]['quality']}" for c in both
                                            if x[c]["quality"] != y[c]["quality"])),
        }
    G, H = groups(a.old), groups(a.new)
    res["hr_groups"] = {"old": dict(Counter(r["group"] for r in G.values())),
                        "new": dict(Counter(r["group"] for r in H.values())),
                        "transitions": dict(Counter(f"{G[c]['group']}->{H[c]['group']}" for c in set(G) & set(H)
                                                    if G[c]["group"] != H[c]["group"]))}
    Path(a.out).write_text(json.dumps(res, indent=1))
    for t, v in res["tracks"].items():
        print(t, v)
    print("HR groups", res["hr_groups"])


if __name__ == "__main__":
    main()
