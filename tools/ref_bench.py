# SPDX-License-Identifier: MIT
"""Does the arterial pressure reference improve the ECG-PPG sawtooth correction? Aggregates only.

    python tools/ref_bench.py --beats DIR --art DIR2 --priors setup_priors.json --out out.jsonl [--part 0 --of 3]

For each case with a usable arterial line (``art_export``, reference on at least half the beats):
real: the fit without and with the reference (setup prior if known, all tiers reported): time per
tier, kept / physiological / confirmed slips. Semi-synthetic: cases whose real fit (without
reference) shows no sawtooth get a known sawtooth added to the PPG only (the arterial times untouched),
then are fitted without and with the reference; per tier the error against the original PAT (see
``tier_bench``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from physiosync import fit_ecg_ppg_delay
from tier_bench import FRAME, RATES, TIERS, med15, tier_of


def errors(mm, r_t, p_syn, pat0, saw) -> dict:
    corr = mm.correct(p_syn) - r_t
    err = corr - pat0
    err = err - np.median(err)
    raw = saw - np.median(saw)
    tiers = tier_of(mm, r_t)
    out = {}
    for t in TIERS:
        sel = tiers == t
        if sel.sum() < 45:
            continue
        e15, r15 = med15(err[sel]) * 1000, med15(raw[sel]) * 1000
        out[t] = {"seconds": float(np.sum(np.diff(r_t)[sel[:-1]])), "rms_ms": float(np.sqrt(np.mean(e15**2))),
                  "over10_frac": float(np.mean(np.abs(e15) > 10)), "raw_rms_ms": float(np.sqrt(np.mean(r15**2)))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--beats", required=True)
    ap.add_argument("--art", required=True)
    ap.add_argument("--priors", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--time-budget", type=float, default=150)
    a = ap.parse_args()
    import time
    t_start = time.time()
    done = {json.loads(x)["case"] for x in open(a.out).read().splitlines() if x.strip()} if Path(a.out).exists() else set()
    pri = {int(k): v for k, v in json.loads(Path(a.priors).read_text()).items()}
    files = [f for i, f in enumerate(sorted(Path(a.art).glob("*.npz"))) if i % a.of == a.part]
    with open(a.out, "a") as fh:
        for f in files:
            c = int(f.stem)
            if c in done:
                continue
            if time.time() - t_start > a.time_budget:
                break
            z = np.load(Path(a.beats) / f.name)
            r_t, p_t = z["r_t"], z["p_t"]
            a_t = np.load(f)["a_t"]
            res: dict = {"case": c, "ref_coverage": float(np.isfinite(a_t).mean())}
            if res["ref_coverage"] < 0.5:
                res["skip"] = "reference on fewer than half the beats"
                fh.write(json.dumps(res) + "\n")
                continue
            try:
                for tag, ref in (("noref", None), ("ref", a_t)):
                    m = fit_ecg_ppg_delay(r_t, p_t, adopted_pat_s=0.27, frame_s=FRAME, matched=True,
                                          rate_prior_ppm=pri.get(c), apply_min_tier="C", ref_t=ref)
                    res[tag] = {"tier_s": m.qc["tier_time_s"], "kept": m.qc["n_kept"], "inserted": m.qc["n_inserted"],
                                "rejected": m.qc["n_rejected"], "physiological": len(m.qc.get("physiological_t", [])),
                                "confirmed": len(m.qc.get("confirmed_by_reference_t", [])),
                                "tiers": m.tooth_tier}
                if res["noref"]["tiers"] == ["N"]:
                    rng = np.random.default_rng(c)
                    pat0 = p_t - r_t
                    syn = []
                    for _ in range(a.reps):
                        rate, phase = float(rng.choice(RATES)), float(rng.uniform(0, 1))
                        creep = rate * 1e-6 * (r_t - r_t[0]) + phase * FRAME
                        saw = FRAME * np.floor(creep / FRAME) - creep
                        p_syn = p_t + saw
                        row = {"rate": rate}
                        for tag, ref in (("noref", None), ("ref", a_t)):
                            mm = fit_ecg_ppg_delay(r_t, p_syn, adopted_pat_s=0.27, frame_s=FRAME, matched=True,
                                                   apply_min_tier="C", ref_t=ref)
                            row[tag] = errors(mm, r_t, p_syn, pat0, saw)
                        syn.append(row)
                    res["synthetic"] = syn
            except Exception as ex:  # noqa: BLE001
                res["error"] = f"{type(ex).__name__}: {ex}"
            fh.write(json.dumps(res) + "\n")
            fh.flush()
    print(json.dumps({"part": a.part, "done": sum(1 for _ in open(a.out))}))


if __name__ == "__main__":
    main()
