# SPDX-License-Identifier: MIT
"""The fitted tooth of injected sawteeth against the injected one, by how the tooth was found and whether its creep is
corrected (metrology-atlas study ppg-sawtooth). Compares fits of tools/slip_rebuild.py on the same recordings.

    python tools/tooth_check.py --fit v25=$W/slip_rebuild/full2/fit.jsonl --fit v26=$W/slip_rebuild/fit_v26.jsonl \
        [--cases CASES.json] [--anchored-in v25] --out $W/slip_rebuild/tooth_check_v26

Per recording with an injected sawtooth (upward, the injected direction) fitted as a sawtooth by a fit: the injected
tooth, the fitted one, its source (broad, prior, anchored ...), its creep (corrected, or reported: tooth from two
confirmed slips, v26), and whether it is right (within 2% of the injected tooth). --anchored-in NAME keeps the
recordings whose injected sawtooth that fit found by the anchored search. Per fit, a table: source x creep -> right /
wrong; and, on the real beats, the sawtooth recordings by source and creep. Writes recordings.csv and summary.json.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

RIGHT = 0.02


def load(path: str, keep: set | None) -> dict:
    out = {}
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if keep is None or r["case"] in keep:
                out[r["case"]] = r
    return out


def tooth(r: dict) -> dict | None:
    I, T = r.get("inj") or {}, r.get("inj_truth") or {}
    f = I.get("fit") or {}
    if not (I.get("sawtooth") and f.get("L") and I.get("sign", 1) > 0 and T.get("period_s")):
        return None
    return {"L": f["L"], "true": T["period_s"], "source": f.get("source") or "?", "creep": f.get("creep") or "corrected",
            "right": abs(f["L"] / T["period_s"] - 1) <= RIGHT, "half": (f.get("half_check") or {}).get("outcome")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", action="append", required=True, help="NAME=PATH, repeatable")
    ap.add_argument("--cases")
    ap.add_argument("--anchored-in")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    keep = set(json.loads(Path(a.cases).read_text())["cases"]) if a.cases else None
    fits = {k: load(v, keep) for k, v in (x.split("=", 1) for x in a.fit)}
    cases = set.intersection(*(set(f) for f in fits.values()))
    if a.anchored_in:
        cases = {c for c in cases if (tooth(fits[a.anchored_in][c]) or {}).get("source", "").startswith("anchored")}
    rows, S = [], {"fits": {k: v for k, v in (x.split("=", 1) for x in a.fit)}, "recordings": len(cases), "by_fit": {}}
    for c in sorted(cases):
        row = {"case": c}
        for k, f in fits.items():
            t = tooth(f[c])
            row[f"{k}_true_s"] = round(t["true"], 1) if t else None
            row.update({f"{k}_L_s": round(t["L"], 1) if t else None, f"{k}_source": t["source"] if t else "no sawtooth",
                        f"{k}_creep": t["creep"] if t else None, f"{k}_right": t["right"] if t else None,
                        f"{k}_midpoint": t["half"] if t else None})
        rows.append(row)
    for k, f in fits.items():
        tab = defaultdict(Counter)
        for c in cases:
            t = tooth(f[c])
            if t:
                tab[f"{t['source'].split(' (')[0]} / creep {t['creep']}"]["right" if t["right"] else "wrong"] += 1
            else:
                tab["no sawtooth"]["-"] += 1
        real = Counter()
        for r in f.values():
            R = r.get("real") or {}
            if R.get("sawtooth"):
                fr = R.get("fit") or {}
                real[f"{fr.get('source') or '?'} / creep {fr.get('creep') or 'corrected'}"] += 1
        S["by_fit"][k] = {"injected": {s: dict(v) for s, v in sorted(tab.items())},
                          "wrong_tooth_creep_corrected": sum(v["wrong"] for s, v in tab.items() if s.endswith("corrected")),
                          "real_sawtooth_recordings": dict(real.most_common())}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out / "recordings.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
