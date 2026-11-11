"""Share of a recording's Solar8000 HR values the adopted rule reproduces exactly (11 Nov 2026).

Atlas parameter hr-derivation-exact (study hr-derivation). Reads the parts of tools/hr_derivation.py: per recording,
cands[RULE] = [delay, log-likelihood, exact share]; reports the exact share over recordings.

    python examples/hr_rule_exact.py $W/examples/hr_derivation_v4/part_*.jsonl --rule "rmean|8|15" --out OUT.json
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="+")
    ap.add_argument("--rule", default="rmean|8|15")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    v = [r["cands"][a.rule][2] for p in a.parts for r in map(json.loads, open(p, encoding="utf-8"))
         if r.get("cands") and a.rule in r["cands"]]
    q = np.percentile(np.array(v) * 100, [10, 25, 50, 75, 90])
    out = {"rule": a.rule, "recordings": len(v), "exact_pct_p10_25_50_75_90": [round(float(x), 1) for x in q]}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
