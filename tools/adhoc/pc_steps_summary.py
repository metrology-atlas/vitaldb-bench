# SPDX-License-Identifier: MIT
"""Simultaneous delays of the PC-stamped numerics in the PC-step scan (metrology-atlas study recorder-pc-steps).
Ad hoc: reads the saved output of tools/pc_clock_steps.py.

    python tools/adhoc/pc_steps_summary.py $W/results/clock/pc_steps_0.jsonl $W/results/clock/pc_steps_1.jsonl

Events (both Solar8000 HR and Primus ETCO2 late by more than the median interval, within the scan's window) are
counted by size: small (both excesses 0.5-1.0 s) with the SNUADC and BIS waveform stamps not jumping (under 0.1 s),
and large (both 1 s or more) with no SNUADC jump, split by whether the two excesses agree within 0.5 s. The large
equal ones are listed. Prints only.
"""
from __future__ import annotations

import json
import sys


def main() -> None:
    E = [(r["case"], e) for p in sys.argv[1:] for r in map(json.loads, open(p)) for e in r.get("events", [])]
    small = [(c, e) for c, e in E if 0.5 <= e["solar_excess_s"] < 1.0 and 0.5 <= e["primus_excess_s"] < 1.0]
    quiet = [(c, e) for c, e in small if abs(e.get("snuadc_jump_s") or 0) < 0.1 and abs(e.get("bis_jump_s") or 0) < 0.1]
    large = [(c, e) for c, e in E if e["solar_excess_s"] >= 1 and e["primus_excess_s"] >= 1 and e.get("snuadc_jump_s") == 0.0]
    equal = [(c, e) for c, e in large if abs(e["solar_excess_s"] - e["primus_excess_s"]) <= 0.5]
    print(f"events {len(E)} in {len({c for c, _ in E})} files")
    print(f"small (0.5-1.0 s, both): {len(small)} in {len({c for c, _ in small})} files; "
          f"no SNUADC or BIS jump: {len(quiet)} in {len({c for c, _ in quiet})} files")
    print(f"large (>= 1 s, both, no SNUADC jump): {len(large)} in {len({c for c, _ in large})} files; "
          f"equal within 0.5 s: {len(equal)}")
    for c, e in sorted(equal):
        print(f"  case {c} t {e['t']} solar {e['solar_excess_s']} primus {e['primus_excess_s']}")


if __name__ == "__main__":
    main()
