# SPDX-License-Identifier: MIT
"""Figure line-access-summary and the access-match table (metrology-atlas study lab-timestamp-meaning).

    python tools/line_access_summary.py inputs --labels $W/results/pat/art_line_labels.csv \
        --bench $W/ppg_correction_check/cases266.json --out $W/line_access
    (then the WSL pipeline: beats_export, art_export, line_access_scan, lab_draw_labels; see the study's method)
    python tools/line_access_summary.py figure --scan '$W/line_access/scan_*.jsonl' \
        --labels $W/line_access/art_line_labels.csv --times $W/line_access/lab_time_labels.csv --out $W/line_access

``inputs`` writes the case lists: cases_art.json, the cases the first access scan labelled (from its saved
art_line_labels.csv), and cases_bench.json, the benchmark stretches for the arterial-side steps.

``figure`` reads the scan's per-case lines (line_access_scan.py, with its span_rel_s) and:

* left: minutes from the start of the latest access to each lab draw (results within 60 s grouped), within
  15 min, with the share of draws that have one against the scan's control (draw times shifted at random, 20
  circular shifts per case);
* right: seconds from each arterial-side timing step (ECG->ART moves by > 10 ms while ECG->PPG does not, on the
  benchmark stretches) to the nearest access, against random times in the same stretches;
* the table: per panel (every sample group of lab_time_labels.csv, matched or not: blood gas, window 15 min;
  central lab, window 30 min), the share of sample groups with an access starting 0..window before them, against
  the same share for the group times shifted circularly within the case's scanned span (20 shifts, seed 0), and the
  share above chance = 1 - control / matched. The matched share is also read from the labels' own `matched` column
  (lab_draw_labels.py) as a check.

Writes summary.json and line-access-summary.png.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

WIN = 900.0


def inputs(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = sorted(pd.read_csv(a.labels, usecols=["case"])["case"].unique().tolist())
    (out / "cases_art.json").write_text(json.dumps({"cases": [int(c) for c in cases],
                                                     "note": "cases labelled by the first access scan (" + Path(a.labels).name + ")"}))
    b = json.loads(Path(a.bench).read_text())["cases"]
    (out / "cases_bench.json").write_text(json.dumps({"cases": b, "note": "benchmark stretches (" + Path(a.bench).name + ")"}))
    print(json.dumps({"cases_art": len(cases), "cases_bench": len(b)}))


def matched_share(starts: np.ndarray, t: np.ndarray, window: float) -> np.ndarray:
    """Whether an access starts 0..window s before each time."""
    i = np.searchsorted(starts, t, side="right") - 1
    ok = i >= 0
    lag = np.full(t.size, np.inf)
    lag[ok] = t[ok] - starts[i[ok]]
    return lag <= window


def figure(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for f in sorted(glob.glob(a.scan)) for x in open(f) if x.strip()]
    rows = list({r["case"]: r for r in rows}.values())
    ok = [r for r in rows if "draws" in r]
    lags = np.array([x for r in ok for x in r["lags_s"]]) / 60
    n_draws = sum(r["draws"] for r in ok)
    n_match = sum(r["draws_matched"] for r in ok)
    ctrl = float(np.average([r["control_match_share"] for r in ok if r["draws"]], weights=[r["draws"] for r in ok if r["draws"]]))
    st = [r["stretch"] for r in ok if r.get("stretch")]
    s_acc = np.array([x for s in st for x in s["step_to_access_s"]])
    r_acc = np.array([x for s in st for x in s["random_to_access_s"]])
    # table: per panel, matched against circular shifts within the case's scanned span
    lab = pd.read_csv(a.labels)
    acc = lab[lab.label == "art_line:access"]
    starts = {c: np.sort(g.start_rel_s.to_numpy(float)) for c, g in acc.groupby("case")}
    span = {r["case"]: r["span_rel_s"] for r in ok if r.get("span_rel_s")}
    groups = pd.read_csv(a.times).drop_duplicates("draw_id")
    rng = np.random.default_rng(0)
    table = {}
    for panel, window in (("blood_gas", 900.0), ("central", 1800.0)):
        g = groups[groups.panel == panel]
        labelled = float(g["matched"].astype(str).str.lower().eq("true").mean()) if len(g) else np.nan
        m_real, m_ctrl, n = 0, [], 0
        for c, gg in g.groupby("case"):
            if c not in span:
                continue
            lo, hi = span[c]
            t = gg.lab_rel_s.to_numpy(float)
            t = t[(t >= lo) & (t <= hi)]
            if not t.size:
                continue
            s_ = starts.get(c, np.empty(0))
            n += t.size
            m_real += int(matched_share(s_, t, window).sum())
            L = hi - lo
            for _ in range(20):
                sh = (t - lo + rng.uniform(0, L)) % L + lo
                m_ctrl.append(matched_share(s_, np.sort(sh), window).mean() * t.size)
        mr = m_real / n if n else np.nan
        mc = float(np.sum(m_ctrl) / (20 * n)) if n else np.nan
        table[panel] = {"groups_all": int(len(g)), "labels_matched_share": round(labelled, 3),
                        "groups": n, "matched": m_real, "matched_share": round(mr, 3), "control_share": round(mc, 3),
                        "above_chance": round(1 - mc / mr, 3) if mr else None, "window_s": window}
    S = {"scan_cases": len(rows), "cases_with_draws": len(ok), "draws": n_draws, "draws_matched_15min": n_match,
         "matched_share": round(n_match / n_draws, 3), "control_share": round(ctrl, 3),
         "lag_min_p10_50_90": [round(float(x), 1) for x in np.percentile(lags, [10, 50, 90])],
         "stretches": len(st), "arterial_steps": int(s_acc.size),
         "steps_within_60s": round(float(np.mean(s_acc <= 60)), 3) if s_acc.size else None,
         "random_within_60s": round(float(np.mean(r_acc <= 60)), 3) if r_acc.size else None,
         "table": table}
    (out / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps(S, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.2))
    ax[0].hist(lags, bins=np.arange(0, 15.5, 0.5), color="C0")
    ax[0].set_xlabel("minutes from the start of the line access to the lab_data time")
    ax[0].set_ylabel("draws")
    ax[0].set_title(f"{n_match} of {n_draws} lab draws follow an access within 15 min ({100 * n_match / n_draws:.1f}%; "
                    f"random times {100 * ctrl:.0f}%)", fontsize=9)
    if s_acc.size:
        b = np.arange(0, 630, 30)
        ax[1].hist(np.clip(s_acc, 0, 600), bins=b, weights=np.full(s_acc.size, 1 / s_acc.size), color="C2", alpha=0.7,
                   label=f"arterial-side timing steps (n={s_acc.size})")
        ax[1].hist(np.clip(r_acc, 0, 600), bins=b, weights=np.full(r_acc.size, 1 / r_acc.size), color="0.6", alpha=0.6,
                   label="random times")
        ax[1].set_title(f"timing steps on the arterial side cluster at line accesses ({100 * S['steps_within_60s']:.0f}% within "
                        f"60 s; random {100 * S['random_within_60s']:.0f}%)", fontsize=9)
        ax[1].legend(fontsize=8)
    ax[1].set_xlabel("seconds to the nearest line access (600 or more in the last bin)")
    ax[1].set_ylabel("share")
    fig.tight_layout()
    fig.savefig(out / "line-access-summary.png", dpi=100)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inputs")
    p.add_argument("--labels", required=True)
    p.add_argument("--bench", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("figure")
    p.add_argument("--scan", required=True, help="glob of the scan's jsonl parts")
    p.add_argument("--labels", required=True)
    p.add_argument("--times", required=True, help="lab_time_labels.csv: every sample group, matched or not")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"inputs": inputs, "figure": figure}[a.cmd](a)


if __name__ == "__main__":
    main()
