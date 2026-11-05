# SPDX-License-Identifier: MIT
"""Regression of the physiosync port (physiosync.slips) against the vitaldb-bench prototype (tools/slip_rebuild.py):
both run on the same recordings' beats and quality; their slips, outcomes, tooth and drift must agree.

    python tools/port_regress.py --scan $W/sawtooth_char/scan_full --quality $W/slip_rebuild/quality \
        --cases $W/slip_rebuild/cases.json --rates $W/sawtooth_stats/recordings.csv \
        --modes $W/results/sync_full/snuadc_mode_label.json --rule $W/label_quality_final/summary.json \
        --out $W/slip_rebuild/port_regress.jsonl [--part 0 --of 2 --time-budget 150]

Per recording (real beats, and injected ones where present): equal when the sawtooth flag, direction, the slips (same
count, times within 1 ms, same kinds and signs), the predicted outcomes in order, the tooth (within 1e-6 s), the
reported drift, and (slip_rebuild v26, physiosync 0.3.1) whether the creep is corrected or reported and the midpoint
judgement agree. Writes one row per recording; the summary is printed by --summary.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import slip_rebuild as proto  # noqa: E402
from physiosync.slips import SlipOptions, find_slips  # noqa: E402

FIDS = ("foot", "up", "peak", "down50")


def compare(a: dict, b: dict) -> list[str]:
    d = []
    if a.get("sawtooth") != b.get("sawtooth"):
        d.append("sawtooth")
    fa, fb = a.get("fit") or {}, b.get("fit") or {}
    if fa.get("creep") != fb.get("creep") or (fa.get("half_check") or {}).get("outcome") != (fb.get("half_check") or {}).get("outcome"):
        d.append("creep")
    if a.get("direction") != b.get("direction"):
        d.append("direction")
    sa, sb = a.get("slips", []), b.get("slips", [])
    if len(sa) != len(sb):
        d.append(f"slips {len(sa)} vs {len(sb)}")
    else:
        for x, y in zip(sa, sb):
            if abs(x["t"] - y["t"]) > 1e-3 or x.get("kind") != y.get("kind") or x.get("sign", 1) != y.get("sign", 1):
                d.append("slip differs")
                break
    if [p["outcome"] for p in a.get("predicted", [])] != [p["outcome"] for p in b.get("predicted", [])]:
        d.append("outcomes")
    La, Lb = (a.get("fit") or {}).get("L"), (b.get("fit") or {}).get("L")
    if (La is None) != (Lb is None) or (La is not None and abs(La - Lb) > 1e-6):
        d.append("tooth")
    da, db = (a.get("drift") or {}).get("ppm"), (b.get("drift") or {}).get("ppm")
    if (da is None) != (db is None) or (da is not None and abs(da - db) > 1e-6):
        d.append("drift")
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("scan", "quality", "cases", "rates", "modes", "rule", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--time-budget", type=float, default=1e9)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if a.summary:
        rows = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
        from collections import Counter
        diffs = Counter(d for r in rows for k in ("real", "inj") for d in r.get(k, []))
        print(json.dumps({"recordings": len(rows), "equal": sum(1 for r in rows if not r.get("real") and not r.get("inj")),
                          "errors": sum(1 for r in rows if "error" in r), "differences": dict(diffs)}, indent=1))
        return
    rule = json.loads(Path(a.rule).read_text())["rule"]["params"]
    modes = json.loads(Path(a.modes).read_text())
    rates = proto.load_rates(a.rates, modes)
    cases = [c for i, c in enumerate(json.loads(Path(a.cases).read_text())["cases"]) if i % a.of == a.part]
    done = {json.loads(x)["case"] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    t0 = time.time()
    with out.open("a") as fh:
        for c in cases:
            if c in done or time.time() - t0 > a.time_budget:
                continue
            row = {"case": c}
            try:
                z = np.load(Path(a.scan) / f"{c:04d}.npz")
                Qz = np.load(Path(a.quality) / f"{c:04d}.npz")
                pr = proto.prior_range(rates, str(z["port"]) or "?", modes.get(str(c), "?"), c)
                po = types.SimpleNamespace(r="template", resp="on", gate="on", prior="setup", rule_params=rule,
                                           grid_veto="clear", evidence="on", anchored="on", saw_rule="two confirmed",
                                           anchor_check="on", prior_by_sign=pr)
                so = SlipOptions(rule_params=rule, prior_by_sign=pr)
                for key, pre in (("real", ""), ("inj", "inj_")):
                    if pre and ("inj_truth" not in z.files or "inj_r_ref" not in Qz.files):
                        continue
                    Q = {k[len(pre):]: Qz[k] for k in Qz.files if k.startswith(pre) and (pre or not k.startswith("inj_"))}
                    F = {j: z[pre + j] for j in FIDS}
                    A = proto.find_slips(z[pre + "r_t"], F, Q, pre, po, z[pre + "art_t"])
                    B = find_slips(z[pre + "r_t"], F, Q, art_t=z[pre + "art_t"], opts=so)
                    row[key] = compare(A, B)
            except Exception as ex:  # noqa: BLE001
                row["error"] = f"{type(ex).__name__}: {ex}"[:300]
            fh.write(json.dumps(row) + "\n")
            fh.flush()
    print(json.dumps({"part": a.part, "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
