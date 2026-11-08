# SPDX-License-Identifier: MIT
"""Where PLETH_HR's window sits: the two routes, term by term, on the same recordings.

    python examples/pr_window_routes.py --hr-pr $W/examples/hr_vs_pulse_rate --pleth $W/examples/pleth_hr_derivation_phase \
        --labels $W/results/pat --out $W/pr_window_routes [--poll 2.0] [--pat 0.27]

Route 1, from the comparison with heart rate (examples/hr_vs_pulse_rate.py, its scan parts only; per-lag sufficient
statistics). The lag L at which HR and PLETH_HR changes agree best (pooled within each HR delay group, read from the
whole-record lags by a parabola, as that example does) is the difference of the two windows' centres before their
stamps, so PLETH_HR's centre = L + c_HR, with c_HR = d_HR + E[phi] + w_HR / 2:
  d_HR    the recording's HR delay label (numeric_delay_labels.csv, 1 Nov, 8-interval rule; quality good), the
          shortest delay in the delay fits' uniform-phase convention;
  E[phi]  the mean wait for the recorder's poll, half the record gap (poll / 2);
  w_HR    8 R-R intervals, taken as 8 x the recording's median pulse interval (the PLETH_HR derivation's parts).
Its time base is the heartbeat that HR counts (R peaks).

Route 2, from the PLETH_HR derivation (tools/pleth_hr_derivation.py, phase run): c_PR = d_PR + gap / 2 + w_PR / 2,
d_PR the recording's uniform-phase delay for rmean|n|10|round, w_PR = 10 x its median pulse interval. Its time base is
the monitor's pulse at the finger (pulses moved earlier by the plethysmogram's hardware delay), which comes a
pulse-arrival time (PAT) after the R peak; so route 1 should exceed route 2 by about PAT.

Both routes on the recordings that have both, by HR delay group; each term's mean is written out, with route 1's lag
bootstrapped over recordings, and both routes by thirds of heart rate. Also route 1 as first computed (31 Oct: old labels' delay, w_HR = the labels' 6 s, no
poll term), for comparison. Aggregates only: writes OUT/summary.json and prints it.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hr_vs_pulse_rate import RECORD_S, _corr  # noqa: E402

KEY = "rmean|n|10|round"


def read_parts(folder: Path) -> dict:
    rows = {}
    for p in sorted(folder.glob("part_*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["case"]] = r
    return rows


def step_peak(sub: list) -> float:
    """The lag of best agreement from the whole-record lags (2k s), by a parabola through the highest and its
    neighbours (as examples/hr_vs_pulse_rate.py)."""
    rec = list(range(-8, 9))
    c = []
    for k in rec:
        t = np.zeros(6)
        for r in sub:
            x = r["lags"].get(str(float(RECORD_S * k)))
            if x:
                t += np.array(x["ch"], float)
        c.append(_corr(t))
    c = np.array(c)
    i = int(np.nanargmax(c))
    y0, y1, y2 = c[i - 1], c[i], c[i + 1]
    return RECORD_S * (rec[i] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hr-pr", type=Path, required=True)
    ap.add_argument("--pleth", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--poll", type=float, default=2.0)
    ap.add_argument("--pat", type=float, default=0.27, help="pulse-arrival time, R peak to finger (atlas adopted-pat)")
    a = ap.parse_args()

    hp = {c: r for c, r in read_parts(a.hr_pr).items() if r.get("lags")}
    ph = {c: r for c, r in read_parts(a.pleth).items() if r.get("phase") and KEY in r["phase"]}
    dh, grp = {}, {}
    for r in csv.DictReader((a.labels / "numeric_delay_labels.csv").open()):
        if r["track"] == "Solar8000/HR" and r.get("quality") == "good":
            dh[int(r["case"])] = float(r["delay_from_window_end_s"])
    for r in csv.DictReader((a.labels / "hr_delay_group_labels.csv").open()):
        grp[int(r["case"])] = r["group"]

    rng = np.random.default_rng(0)
    out = {"poll_s": a.poll, "pat_s": a.pat, "recordings": {"hr_pr": len(hp), "pleth_phase": len(ph),
                                                            "hr_labels_good": len(dh)}, "groups": {}}
    for g in ("early", "late"):
        cases = sorted(c for c in hp if grp.get(c) == g and c in dh and c in ph)
        if len(cases) < 50:
            continue
        sub = [hp[c] for c in cases]
        L = step_peak(sub)
        boot = [step_peak([sub[i] for i in rng.integers(0, len(sub), len(sub))]) for _ in range(50)]
        iv = np.array([ph[c]["median_interval_s"] for c in cases])
        d_hr = np.array([dh[c] for c in cases])
        d_pr = np.array([ph[c]["phase"][KEY][0] for c in cases])
        gap = np.array([ph[c]["median_gap_s"] for c in cases])
        terms1 = {"lag_s": L, "hr_delay_s": float(d_hr.mean()), "poll_wait_s": a.poll / 2,
                  "hr_half_window_s": float((8 * iv / 2).mean())}
        r1 = sum(terms1.values())
        terms2 = {"pr_delay_s": float(d_pr.mean()), "poll_wait_s": float((gap / 2).mean()),
                  "pr_half_window_s": float((10 * iv / 2).mean())}
        r2 = sum(terms2.values())
        out["groups"][g] = {
            "recordings": len(cases), "median_pulse_interval_s": round(float(np.median(iv)), 3),
            "route1_terms": {k: round(v, 3) for k, v in terms1.items()},
            "route1_lag_boot_p2.5_p97.5": [round(float(np.percentile(boot, 2.5)), 2), round(float(np.percentile(boot, 97.5)), 2)],
            "route1_centre_s": round(r1, 2),
            "route2_terms": {k: round(v, 3) for k, v in terms2.items()},
            "route2_centre_s": round(r2, 2),
            "route1_minus_route2_s": round(r1 - r2, 2),
            "expected_from_pat_s": a.pat,
            "left_over_s": round(r1 - r2 - a.pat, 2),
            "route1_as_first_computed_s": round(L + float(d_hr.mean()) + 3.0, 2),
        }
        # by heart rate: thirds of the recordings by median pulse interval. A left-over that is the same in each third
        # is a delay; one that grows with the interval is a window counted in beats
        strata = []
        for lo, hi in zip(np.quantile(iv, [0, 1 / 3, 2 / 3]), np.quantile(iv, [1 / 3, 2 / 3, 1])):
            m = (iv >= lo) & (iv <= hi)
            ss = [sub[i] for i in np.flatnonzero(m)]
            Ls = step_peak(ss)
            r1s = Ls + float(d_hr[m].mean()) + a.poll / 2 + float((8 * iv[m] / 2).mean())
            r2s = float((d_pr[m] + gap[m] / 2 + 10 * iv[m] / 2).mean())
            strata.append({"interval_s": [round(float(lo), 3), round(float(hi), 3)], "recordings": int(m.sum()),
                           "mean_interval_s": round(float(iv[m].mean()), 3), "route1_centre_s": round(r1s, 2),
                           "route2_centre_s": round(r2s, 2), "left_over_s": round(r1s - r2s - a.pat, 2)})
        out["groups"][g]["by_rate"] = strata
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
