"""Stage 3, batch 2: compare the v4-rate rerun with the 28-29 Oct release run (10 Nov 2026).

Compares two JSON results (readout.json of numeric_derivation_readout.py, or summary.json of spo2_event_agg.py)
leaf by leaf and writes every difference. Lists longer than --max-list (curves, per-recording arrays) are compared
only by length and median. Family tables (lists of [name, count]) are compared as name -> count.

  python examples/stage3_batch2_compare.py OLD.json NEW.json --out DIFF.json
"""
import argparse
import json

import numpy as np


def is_table(v):
    return isinstance(v, list) and v and all(isinstance(x, list) and len(x) == 2 and isinstance(x[0], str) for x in v)


def walk(a, b, path, out, max_list):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append({"path": f"{path}/{k}", "old": a.get(k, "<absent>") if k in a else "<absent>",
                            "new": b.get(k, "<absent>") if k in b else "<absent>"})
            else:
                walk(a[k], b[k], f"{path}/{k}", out, max_list)
    elif is_table(a) and is_table(b):
        walk(dict(a), dict(b), path, out, max_list)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) > max_list or len(b) > max_list:
            num = lambda v: [x for x in v if isinstance(x, (int, float))]
            ma = float(np.median(num(a))) if num(a) else None
            mb = float(np.median(num(b))) if num(b) else None
            if len(a) != len(b) or ma != mb:
                out.append({"path": path, "old": {"len": len(a), "median": ma}, "new": {"len": len(b), "median": mb}})
        elif len(a) != len(b):
            out.append({"path": path, "old": a, "new": b})
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]", out, max_list)
    elif a != b:
        out.append({"path": path, "old": a, "new": b})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-list", type=int, default=12)
    a = ap.parse_args()
    old, new = json.load(open(a.old)), json.load(open(a.new))
    diffs = []
    walk(old, new, "", diffs, a.max_list)
    json.dump({"old": a.old, "new": a.new, "n_differences": len(diffs), "differences": diffs}, open(a.out, "w"), indent=1)
    print(f"{len(diffs)} differences -> {a.out}")


if __name__ == "__main__":
    main()
