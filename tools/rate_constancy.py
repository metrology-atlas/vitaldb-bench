# SPDX-License-Identifier: MIT
"""Is the sawtooth's slope constant per setup, per case, or only per tooth? Could adjacent teeth share one label?

    python tools/rate_constancy.py --records DIR --priors P --out summary.json [--tol-s 10]

Reads the measurement records (measurement_export: DIR/<case>.json). For each case, the runs of consecutive
full teeth whose slips were detected (not inserted or predicted) are tested against three models of the
slip times t_k:

* constant rate:  t_k = t_0 + k L                     (one slope and one phase for the whole run)
* drifting rate:  t_k = t_0 + k L + k^2 q             (the slope changes slowly and smoothly)
* per tooth:      every interval free (what physiosync does now)

and split into merge groups: the longest stretches of teeth whose slips a drifting-rate model places within
``--tol-s`` (a slip misplaced by s seconds leaves a whole-frame error for about s seconds). A residual far
outside the others marks a phase jump. Across cases, the recordings' rates are compared within each setup
(the setup's rate in --priors identifies it). Prints and writes aggregates only.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def groups(t: np.ndarray, tol: float) -> list[tuple[int, int]]:
    """Greedy: extend a group while a quadratic (>= 4 slips) or linear fit keeps every slip within tol."""
    out, i = [], 0
    while i < t.size:
        j = min(i + 2, t.size)
        while j < t.size:
            k = np.arange(j + 1 - i)
            deg = 2 if k.size >= 4 else 1
            r = t[i : j + 1] - np.polyval(np.polyfit(k, t[i : j + 1], deg), k)
            if np.max(np.abs(r)) > tol:
                break
            j += 1
        out.append((i, j))
        i = max(i + 1, j)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True)
    ap.add_argument("--priors", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol-s", type=float, default=10.0)
    a = ap.parse_args()
    pri = {int(k): v for k, v in json.loads(Path(a.priors).read_text()).items()}
    per_case, by_setup = [], defaultdict(list)
    teeth_total = teeth_in_runs = labels_now = labels_merged = 0
    for f in sorted(p for p in Path(a.records).glob("*.json") if p.stem.isdigit()):
        rec = json.loads(f.read_text())
        c = int(rec.get("extra", {}).get("case", f.stem))
        segs = rec["segments"]
        labels_now += len(segs)
        F = rec["clocks"]["ppg_frame_s"]
        # runs of consecutive slips that were detected (both ends of each tooth), from teeth of tier A or B
        slips, runs = [], []
        for s in segs:
            teeth_total += 1
            ok = (s["tier"] in ("A", "B") and s["slip_at_start"] and s["slip_at_end"]
                  and (s.get("slip_evidence") or {}).get("start", "").startswith("detected")
                  and (s.get("slip_evidence") or {}).get("end", "").startswith("detected"))
            if ok:
                if not slips or abs(slips[-1] - s["start"]) > 1e-6:
                    if len(slips) >= 4:
                        runs.append(np.array(slips))
                    slips = [s["start"]]
                slips.append(s["end"])
            else:
                if len(slips) >= 4:
                    runs.append(np.array(slips))
                slips = []
        if len(slips) >= 4:
            runs.append(np.array(slips))
        if not runs:
            labels_merged += len(segs)
            continue
        res = {"case": c, "prior_ppm": pri.get(c), "runs": []}
        n_groups_case, teeth_case = 0, 0
        for t in runs:
            k = np.arange(t.size)
            L = np.diff(t)
            rates = F / L * 1e6
            r1 = t - np.polyval(np.polyfit(k, t, 1), k)
            r2 = t - np.polyval(np.polyfit(k, t, 2), k) if t.size >= 4 else r1
            g = groups(t, a.tol_s)
            hours = (t[-1] - t[0]) / 3600
            drift = float(np.polyfit((t[:-1] + t[1:]) / 2 / 3600, rates, 1)[0]) if rates.size >= 3 else None
            res["runs"].append({"teeth": int(L.size), "hours": round(hours, 2), "rate_median_ppm": round(float(np.median(rates)), 2),
                                "rate_cv": round(float(np.std(rates) / np.mean(rates)), 4),
                                "drift_ppm_per_h": None if drift is None else round(drift, 3),
                                "const_rate_max_s": round(float(np.max(np.abs(r1))), 1),
                                "drift_model_max_s": round(float(np.max(np.abs(r2))), 1),
                                "merge_groups": len(g)})
            n_groups_case += len(g)
            teeth_case += int(L.size)
            teeth_in_runs += int(L.size)
        labels_merged += len(segs) - teeth_case + n_groups_case
        if pri.get(c):
            by_setup[round(pri[c], 1)].append(float(np.median([r["rate_median_ppm"] for r in res["runs"]])))
        per_case.append(res)
    runs = [r for c in per_case for r in c["runs"]]

    def q(x: list[float]) -> list[float]:
        return [round(float(v), 3) for v in np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9])] if x else []

    summary = {
        "cases_with_runs": len(per_case), "runs": len(runs), "teeth_in_runs": teeth_in_runs, "teeth_total": teeth_total,
        "rate_cv_within_run_q": q([r["rate_cv"] for r in runs]),
        "drift_ppm_per_h_q": q([r["drift_ppm_per_h"] for r in runs if r["drift_ppm_per_h"] is not None]),
        "const_rate_max_slip_error_s_q": q([r["const_rate_max_s"] for r in runs]),
        "drift_model_max_slip_error_s_q": q([r["drift_model_max_s"] for r in runs]),
        "share_runs_constant_rate_within_tol": round(float(np.mean([r["const_rate_max_s"] <= a.tol_s for r in runs])), 3) if runs else None,
        "share_runs_drift_model_within_tol": round(float(np.mean([r["drift_model_max_s"] <= a.tol_s for r in runs])), 3) if runs else None,
        "labels_per_tooth": labels_now, "labels_if_merged": labels_merged,
        "between_case_rate_by_setup": {str(k): {"cases": len(v), "median": round(float(np.median(v)), 2),
                                                 "iqr": [round(float(x), 2) for x in np.quantile(v, [0.25, 0.75])]}
                                       for k, v in sorted(by_setup.items()) if len(v) >= 3},
        "tol_s": a.tol_s,
    }
    Path(a.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
