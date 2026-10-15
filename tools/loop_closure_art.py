# SPDX-License-Identifier: MIT
"""Loop-closure test for arterial pressure measured three ways: SNUADC ART, EV1000 ART_MBP and CardioQ ABP.
Per case, this machine only; aggregates leave.

    python tools/loop_closure_art.py scan --root R --cases 1,2 --units units.csv --step-rates rates.json --out out.jsonl
    python tools/loop_closure_art.py agg out*.jsonl --summary summary.json

Three links of the timing graph meet in one loop:

    ES  EV1000 stamp minus SNUADC stamp-model time   (EV1000 MAP against 1 s SNUADC MAP; link N)
    CS  CardioQ stamp minus SNUADC stamp-model time  (beat matching within re-anchored stretches; link M)
    EC  EV1000 stamp minus CardioQ stamp             (EV1000 MAP against 1 s CardioQ MAP on its own stamps)

If every link is right, ES = EC + CS. Per 30-min EV1000 block (step 15 min) the closure residual is
ES - (EC + CS_block), where CS_block is the median CardioQ lag over the beat-matched windows inside the block.
EC is measured independently of SNUADC: it never uses the SNUADC samples or the stamp model.
The methods are those of tools/rare_device_sync.py (numeric_vs, cardioq).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import rare_device_sync as rds  # noqa: E402

BLOCK = 1800.0
MIN_R_EV = 0.9
MIN_R_CQ = 0.8


def scan_case(root, c, u, sr):
    v = rds.read_vital(Path(root) / "vital_files" / f"{c:04d}.vital",
                       track_names={"SNUADC/ART", "EV1000/ART_MBP", "CardioQ/ABP"})
    art, ev, cq = (v.tracks.get(n) for n in ("SNUADC/ART", "EV1000/ART_MBP", "CardioQ/ABP"))
    if any(t is None or not t.record_times.size for t in (art, ev, cq)):
        return {"case": c, "skip": "not all three"}
    ta, _ = rds.snu_times(art, u, sr)
    xa = np.asarray(art.physical(), float)[: ta.size]
    t0 = float(np.floor(ta[0]))
    n = int(ta[-1] - t0) + 1
    mapp = rds.per_second(ta, xa, t0, n)
    mapp[(mapp < 20) | (mapp > 200)] = np.nan
    es = rds.numeric_vs(v, "EV1000/ART_MBP", t0 + np.arange(n) + 0.5, mapp, 20, 120, block=BLOCK)
    cs = rds.cardioq(v, ta, xa, float(art.srate))
    tq = rds.sample_times(cq)
    xq = np.asarray(cq.physical(), float)[: tq.size]
    q0 = float(np.floor(tq[0]))
    nq = int(tq[-1] - q0) + 1
    mq = rds.per_second(tq, xq, q0, nq)
    mq[(mq < 20) | (mq > 200)] = np.nan
    ec = rds.numeric_vs(v, "EV1000/ART_MBP", q0 + np.arange(nq) + 0.5, mq, 20, 120, block=BLOCK)
    if not (es and "windows" in es and ec and "windows" in ec and cs and "windows" in cs):
        return {"case": c, "skip": "an arm failed"}
    csw = np.array([w for w in cs["windows"] if w[3] >= MIN_R_CQ], float)
    ecd = {w[0]: w for w in ec["windows"]}
    rows = []
    for w in es["windows"]:
        tb, l_es, r_es = w[0], w[1], w[2]
        e = ecd.get(tb)
        if e is None or r_es < MIN_R_EV or e[2] < MIN_R_EV or not csw.size:
            continue
        m = (csw[:, 0] >= tb - BLOCK / 2) & (csw[:, 0] < tb + BLOCK / 2)
        if m.sum() < 3:
            continue
        l_cs = float(np.median(csw[m, 1]))
        spread = float(np.subtract(*np.percentile(csw[m, 1], [75, 25])))
        rows.append({"t": tb, "es": l_es, "ec": e[1], "cs": round(l_cs, 3), "cs_iqr": round(spread, 3),
                     "resid": round(l_es - (e[1] + l_cs), 3), "r_es": r_es, "r_ec": e[2], "cq_windows": int(m.sum())})
    return {"case": c, "blocks": rows}


def agg(paths, out):
    R = [json.loads(l) for p in paths for l in open(p)]
    B = [b for r in R for b in r.get("blocks", [])]
    res = np.array([b["resid"] for b in B])
    cases = [r for r in R if r.get("blocks")]
    per_case = np.array([np.median([b["resid"] for b in r["blocks"]]) for r in cases]) if cases else np.array([])
    S = {"cases_scanned": len(R), "cases_with_blocks": len(cases), "blocks": int(res.size),
         "skipped": {k: sum(1 for r in R if r.get("skip") == k) for k in {r.get("skip") for r in R if r.get("skip")}}}
    if res.size:
        S |= {"resid_s_p10_25_50_75_90": np.percentile(res, [10, 25, 50, 75, 90]).round(2).tolist(),
              "abs_resid_s_p50_p90": np.percentile(np.abs(res), [50, 90]).round(2).tolist(),
              "within_1s": round(float(np.mean(np.abs(res) <= 1.0)), 3),
              "within_2s": round(float(np.mean(np.abs(res) <= 2.0)), 3),
              "case_median_resid_s": sorted(np.round(per_case, 2).tolist()),
              "arms_s_p50": {k: round(float(np.median([b[k] for b in B])), 2) for k in ("es", "ec", "cs")},
              "cs_iqr_within_block_p50": round(float(np.median([b["cs_iqr"] for b in B])), 2)}
    Path(out).write_text(json.dumps(S, indent=1))
    print(json.dumps(S, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["scan", "agg"])
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--root")
    ap.add_argument("--cases")
    ap.add_argument("--units")
    ap.add_argument("--step-rates")
    ap.add_argument("--out")
    ap.add_argument("--summary")
    a = ap.parse_args()
    if a.mode == "agg":
        agg(a.inputs, a.summary)
        return
    U = {int(r["case"]): r for r in csv.DictReader(open(a.units))}
    sr = json.loads(Path(a.step_rates).read_text())
    done = {json.loads(l)["case"] for l in open(a.out)} if Path(a.out).exists() else set()
    for c in [int(x) for x in a.cases.split(",") if x]:
        if c in done:
            continue
        try:
            r = scan_case(a.root, c, U.get(c, {}), sr)
        except Exception as e:  # noqa: BLE001
            r = {"case": c, "error": repr(e)[:300]}
        with open(a.out, "a") as fh:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
