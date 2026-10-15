# SPDX-License-Identifier: MIT
"""Confidence tiers of the ECG-PPG sawtooth correction: coverage on real cases and error on semi-synthetic
cases with a known sawtooth. Aggregates only.

    python tools/tier_bench.py --beats DIR --priors setup_priors.json --out out.jsonl [--part 0 --of 3]

Real: every exported case is fitted (setup prior if known, apply_min_tier C so all tiers are reported);
the time in each tier is recorded.

Semi-synthetic: cases whose real fit shows no sawtooth (tier N) carry real physiology and detection noise.
A frame-slip sawtooth (frame 1/30 s, rate from the setups seen in VitalDB, random phase) is added to
their PPG times and the fit is run again, with and without the setup rate as a prior. Each beat is
classified by the tier of its tooth; per tier: the error of the corrected PAT against the original
(sawtooth-free) PAT, after removing the overall median, at the level of 15-beat medians.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from physiosync import fit_ecg_ppg_delay

FRAME = 1 / 30
RATES = (20.0, 27.0, 45.0, 51.0)
TIERS = ("A", "B", "C", "N", "U")


def tier_of(m, t: np.ndarray) -> np.ndarray:
    edges = np.asarray(m.tooth_edges)
    k = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(m.tooth_tier) - 1)
    return np.asarray(m.tooth_tier)[k]


def med15(x: np.ndarray) -> np.ndarray:
    n = x.size // 15
    return np.median(x[: 15 * n].reshape(n, 15), axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--beats", required=True)
    ap.add_argument("--priors", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    pri = {int(k): v for k, v in json.loads(Path(a.priors).read_text()).items()}
    files = sorted(Path(a.beats).glob("*.npz"))
    files = [f for i, f in enumerate(files) if i % a.of == a.part]
    with open(a.out, "a") as fh:
        for f in files:
            c = int(f.stem)
            z = np.load(f)
            r_t, p_t = z["r_t"], z["p_t"]
            res: dict = {"case": c}
            try:
                m = fit_ecg_ppg_delay(r_t, p_t, adopted_pat_s=0.27, frame_s=FRAME, matched=True,
                                      rate_prior_ppm=pri.get(c), apply_min_tier="C")
                res["real_tier_s"] = m.qc["tier_time_s"]
                res["real_intervals"] = [{k: q[k] for k in ("start", "end", "tier", "reason")} for q in m.quality_intervals()]
                res["real_intervals_t0"] = float(m.t_start)
                if m.tooth_tier == ["N"]:
                    rng = np.random.default_rng(c)
                    syn = []
                    pat0 = p_t - r_t
                    for rep in range(a.reps):
                        rate = float(rng.choice(RATES))
                        phase = float(rng.uniform(0, 1))
                        creep = rate * 1e-6 * (r_t - r_t[0]) + phase * FRAME
                        saw = FRAME * np.floor(creep / FRAME) - creep
                        p_syn = p_t + saw
                        for use_prior in (False, True):
                            mm = fit_ecg_ppg_delay(r_t, p_syn, adopted_pat_s=0.27, frame_s=FRAME, matched=True,
                                                   rate_prior_ppm=rate if use_prior else None, apply_min_tier="C")
                            corr = mm.correct(p_syn) - r_t
                            err = corr - pat0
                            err = err - np.median(err)
                            raw = saw - np.median(saw)
                            tiers = tier_of(mm, r_t)
                            row = {"rate": rate, "prior": use_prior, "tiers": {}}
                            for t in TIERS:
                                sel = tiers == t
                                if sel.sum() < 45:
                                    continue
                                e15, r15 = med15(err[sel]) * 1000, med15(raw[sel]) * 1000
                                row["tiers"][t] = {"beats": int(sel.sum()), "seconds": float(np.sum(np.diff(r_t)[sel[:-1]])),
                                                   "rms_ms": float(np.sqrt(np.mean(e15**2))),
                                                   "p95_ms": float(np.quantile(np.abs(e15), 0.95)),
                                                   "over10_frac": float(np.mean(np.abs(e15) > 10)),
                                                   "raw_rms_ms": float(np.sqrt(np.mean(r15**2)))}
                            syn.append(row)
                    res["synthetic"] = syn
            except Exception as ex:  # noqa: BLE001
                res["error"] = f"{type(ex).__name__}: {ex}"
            fh.write(json.dumps(res) + "\n")
            fh.flush()
    print(json.dumps({"part": a.part, "done": sum(1 for _ in open(a.out))}))


if __name__ == "__main__":
    main()
