"""Recording pauses: the moments every device's stamps jump forward together (11 Nov 2026).

Atlas parameter recording-pause (study recorder-pc-steps). Reads the saved output of tools/pc_clock_steps.py and keeps
the PC-step events where the SNUADC stamps jump forward by more than --min-jump s and the BIS, where present, jumps by
the same within 1 s (the selection of tools/adhoc/audit_low_checks.py, which only prints; this writes it).

    python examples/recording_pause.py $W/results/clock/pc_steps_0.jsonl $W/results/clock/pc_steps_1.jsonl --out OUT.json
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="+")
    ap.add_argument("--min-jump", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [r for p in a.parts for r in map(json.loads, open(p, encoding="utf-8"))]
    ev = [(r["case"], e) for r in rows for e in r.get("events", [])
          if e.get("snuadc_jump_s") is not None and e["snuadc_jump_s"] > a.min_jump
          and (e.get("bis_jump_s") is None or abs(e["bis_jump_s"] - e["snuadc_jump_s"]) <= 1)]
    j = np.array([e["snuadc_jump_s"] for _, e in ev])
    files = {c for c, _ in ev}
    out = {"files_scanned": len(rows), "min_jump_s": a.min_jump, "pauses": len(ev), "files_with_pause": len(files),
           "share_of_files": round(len(files) / len(rows), 4) if rows else None,
           "jump_s_p10_25_50_75_90": [round(float(x), 2) for x in np.percentile(j, [10, 25, 50, 75, 90])]}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
