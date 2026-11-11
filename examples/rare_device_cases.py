"""Cases for rare_device_sync.py and vigilance_window.py, from the device ports of each recording (11 Nov 2026).

The 13 Oct runs (results/pat/rare_device_sync_summary.json, 997 cases; vigilance_window_summary.json, 63) saved no
case list. A recording is taken if its setup table names one of the rarer haemodynamic monitors; this gives back
997 and 63 exactly.

    python examples/rare_device_cases.py $W/setups/ports.jsonl --out DIR   (writes DIR/cases.txt, DIR/cases_vigilance.txt)
"""
import argparse
import json
from collections import Counter
from pathlib import Path

RARE = {"CardioQ", "EV1000", "EV-1000", "Vigileo", "Vigilance"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ports")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.ports, encoding="utf-8") if l.strip()]
    rare = sorted(r["case"] for r in rows if RARE & set(r["ports"]))
    vig = sorted(r["case"] for r in rows if "Vigilance" in r["ports"])
    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    (o / "cases.txt").write_text(",".join(map(str, rare)) + "\n")
    (o / "cases_vigilance.txt").write_text(",".join(map(str, vig)) + "\n")
    n = Counter(d for r in rows for d in r["ports"] if d in RARE)
    print(json.dumps({"rare": len(rare), "vigilance": len(vig), "by_device": dict(sorted(n.items()))}))


if __name__ == "__main__":
    main()
