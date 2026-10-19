# SPDX-License-Identifier: MIT
"""Which Primus AWP pauses are connection breaks: the CO2 waveform stops too. Per case, this machine; aggregates leave.

    python tools/primus_co2_gaps.py scan --root R --cases 1,2,3 --out part.jsonl
    python tools/primus_co2_gaps.py agg part*.jsonl

Reads only Primus/AWP and Primus/CO2. AWP pauses are stamp gaps over 5 s, timed and zeroed exactly as in
tools/primus_sync_accuracy.py (the first sample after the gap, minus the first AWP sample), so the lists line up.
A pause is a break when the CO2 stamps also have a gap over 1 s within 30 s of it (as in tools/primus_pause_steps.py),
or when the recording has no CO2. At breaks the Primus stamps step by up to seconds; at the other pauses they keep
their timing (atlas study primus-pause-steps).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pat import sample_times  # noqa: E402

AWP_GAP_S, CO2_GAP_S, NEAR_S = 5.0, 1.0, 30.0


def scan_case(c: int, root: str) -> dict:
    from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
    v = read_vital(Path(root) / "vital_files" / f"{c:04d}.vital", track_names={"Primus/AWP", "Primus/CO2"})
    aw, co2 = v.tracks.get("Primus/AWP"), v.tracks.get("Primus/CO2")
    if aw is None or not aw.record_times.size:
        return {"case": c, "skip": "no AWP"}
    ta = sample_times(aw)
    z = float(ta[0])
    g = np.diff(ta)
    ends = ta[1:][g > AWP_GAP_S]
    if co2 is None or not co2.record_times.size:
        brk = [True] * ends.size
        has_co2 = False
    else:
        tc = sample_times(co2)
        cg = tc[1:][np.diff(tc) > CO2_GAP_S]
        brk = [bool(np.any(np.abs(cg - e) < NEAR_S)) for e in ends] if cg.size else [False] * ends.size
        has_co2 = True
    return {"case": c, "z": z, "pauses": (ends - z).round(1).tolist(), "co2_break": brk, "has_co2": has_co2}


def scan(a) -> None:
    out = Path(a.out)
    done = {json.loads(x)["case"] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    with open(out, "a", encoding="utf-8") as fh:
        for c in [int(x) for x in a.cases.split(",") if x]:
            if c in done:
                continue
            t1 = time.time()
            try:
                r = scan_case(c, a.root)
            except Exception as err:  # noqa: BLE001
                r = {"case": c, "error": f"{type(err).__name__}: {err}"[:300]}
            r["secs"] = round(time.time() - t1, 2)
            fh.write(json.dumps(r) + "\n")
            fh.flush()


def agg(a) -> None:
    R = [json.loads(x) for p in a.parts for x in open(p, encoding="utf-8") if x.strip()]
    ok = [r for r in R if "pauses" in r]
    n_p = sum(len(r["pauses"]) for r in ok)
    n_b = sum(sum(r["co2_break"]) for r in ok)
    print(json.dumps({"recordings_read": len(R), "with_awp": len(ok), "errors": sum("error" in r for r in R),
                      "pauses": n_p, "breaks": n_b, "share_breaks": round(n_b / max(n_p, 1), 3),
                      "recordings_without_co2": sum(not r["has_co2"] for r in ok)}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scan"); s.add_argument("--root", required=True); s.add_argument("--cases", required=True)
    s.add_argument("--out", required=True)
    g = sp.add_parser("agg"); g.add_argument("parts", nargs="+")
    a = ap.parse_args()
    scan(a) if a.cmd == "scan" else agg(a)


if __name__ == "__main__":
    main()
