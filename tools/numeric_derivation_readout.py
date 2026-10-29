"""Release readout of numeric_derivation_pilot.py runs (one JSON line per recording).

Per numeric track: the winning algorithm family and its share, the window it describes (beats |n or seconds |t),
truncation against rounding, the delay d from the window's end to the stamp, the value offset b, the remaining
scatter s, and how decisive the win is (log-likelihood margin of the best over the second-best candidate).

    python tools/numeric_derivation_readout.py "work/overnight/numeric/part_*.jsonl" --out readout.json

Used for the atlas study numerics-derivation-release (whole release, 6,388 recordings, 28-29 Oct 2026).
"""
import argparse, collections, glob, json

import numpy as np

TRACKS = ["HR", "PLETH_HR", "ART_SBP", "ART_MBP", "ART_DBP", "CVP", "ETCO2", "RR_CO2", "PIP", "PEEP"]


def pc(x, p=(10, 25, 50, 75, 90)):
    return np.round(np.percentile(x, p), 2).tolist() if len(x) else None


def readout(records):
    out = {}
    for tr in TRACKS:
        rows = [r["tracks"][tr] for r in records
                if tr in (r.get("tracks") or {}) and (r["tracks"][tr] or {}).get("best")]
        if not rows:
            continue
        fam, quant, win = collections.Counter(), collections.Counter(), collections.defaultdict(list)
        d, s, b, margin, fr = [], [], [], [], []
        for t in rows:
            bst = t["best"]
            f, unit, w = bst["cand"].rsplit("|", 2)
            fam[f + "|" + unit] += 1
            win[f + "|" + unit].append(float(w))
            quant[bst["quant"]] += 1
            d.append(bst["d"]); s.append(bst["s"]); b.append(bst.get("b", 0))
            top = t.get("top") or []
            if len(top) > 1:
                margin.append(top[0]["ll"] - top[1]["ll"])
            fl = [v["ll"] for k, v in (t.get("families") or {}).items() if k.endswith("|floor")]
            ro = [v["ll"] for k, v in (t.get("families") or {}).items() if k.endswith("|round")]
            if fl and ro:
                fr.append(max(fl) - max(ro))
        out[tr] = {
            "cases": len(rows),
            "family": fam.most_common(),
            "window_by_family": {k: {"n": len(v), "p10_25_50_75_90": pc(v),
                                     "mode": collections.Counter(v).most_common(3)}
                                 for k, v in win.items() if len(v) >= 20},
            "quant": quant.most_common(),
            "delay_s": pc(d), "bias": pc(b), "scatter_s": pc(s),
            "ll_margin_top1_vs_top2": pc(margin),
            "gap_median_s": float(np.nanmedian([t.get("gap_median", np.nan) for t in rows])),
            "ll_floor_minus_round": pc(fr),
            "floor_better_share": round(float(np.mean(np.array(fr) > 0)), 3) if fr else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("parts", help="glob of part_*.jsonl files from numeric_derivation_pilot.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = [json.loads(l) for f in sorted(glob.glob(a.parts)) for l in open(f) if l.strip()]
    out = readout(recs)
    json.dump(out, open(a.out, "w"), indent=1)
    for k, o in out.items():
        print(f"{k:9s} cases {o['cases']:5d} delay {o['delay_s']} floor>round {o['floor_better_share']}")
        print("          family", o["family"][:3])


if __name__ == "__main__":
    main()
