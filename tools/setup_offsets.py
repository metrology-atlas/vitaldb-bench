# SPDX-License-Identifier: MIT
"""One hardware delay per recording setup, from the per-recording records of a first pass.

    python tools/setup_offsets.py --records DIR --setups units.csv --adopted 0.27 --out setup_offsets.json

Reads each record's raw PAT level (median PAT, sawtooth removed, no offset) and setup, and writes
physiosync.setup_offsets: per setup (>= 3 recordings) the median level minus the adopted mean PAT, else the
median over all recordings. Levels outside 0.4-1.0 s (beats paired one beat out) are left out.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from physiosync import setup_offsets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True)
    ap.add_argument("--setups", required=True)
    ap.add_argument("--adopted", type=float, default=0.27)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    setup_of = {}
    for r in csv.DictReader(open(a.setups)):
        if r["snuadc_drift_ppm"]:
            setup_of[r["case"]] = r["snuadc_port"] + (" step" if float(r["snuadc_drift_ppm"]) < -20 else " small")
    levels = {}
    for f in Path(a.records).glob("*.json"):
        if f.stem.isdigit():
            rec = json.loads(f.read_text())
            levels[f.stem] = rec["offset"].get("raw_level_s")
    so = setup_offsets(levels, setup_of, adopted_pat_s=a.adopted, plausible_s=(0.4, 1.0))
    Path(a.out).write_text(json.dumps(so, indent=1))
    print(json.dumps({k: {"n": v["n"], "offset_ms": round(v["offset_s"] * 1000, 1),
                          "level_iqr_ms": [round(x * 1000, 1) for x in v["iqr_s"]]} for k, v in so["setups"].items()}, indent=1))
    print("all", round(so["all"]["offset_s"] * 1000, 1), so["all"]["n"])


if __name__ == "__main__":
    main()
