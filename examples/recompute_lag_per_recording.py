# SPDX-License-Identifier: MIT
"""The recompute test's HR lag, recording by recording: is the pooled ~1 s one shift for all, or a mixture of
recordings on the right poll and recordings one poll (2 s) off?

    python examples/recompute_lag_per_recording.py PARTS... --groups hr_delay_group_labels.csv \
        --labels numeric_delay_labels.csv --out DIR

Reads the per-recording output of tools/numeric_recompute_lag.py (its HR change statistics per lag) and reads each
recording's own lag of best agreement with the tool's peak() on that recording alone. Writes OUT/summary.json: the
distribution of per-recording lags (overall and by HR delay group), the share near 0, near one record (-2 s) and
elsewhere, and the per-recording lag against the recording's fitted HR delay. Aggregates only.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from numeric_recompute_lag import peak  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parts", nargs="+")
    ap.add_argument("--groups", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = {}
    for p in a.parts:
        for l in Path(p).read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                rows[r["case"]] = r
    grp = {int(r["case"]): r["group"] for r in csv.DictReader(open(a.groups))}
    dhr = {int(r["case"]): float(r["delay_from_window_end_s"]) for r in csv.DictReader(open(a.labels))
           if r["track"] == "Solar8000/HR" and r["quality"] == "good"}
    lag, g, d = [], [], []
    for c, r in rows.items():
        if not r.get("hr"):
            continue
        try:
            L = peak([r], "hr")
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue
        if not np.isfinite(L):
            continue
        lag.append(L)
        g.append(grp.get(c, "none"))
        d.append(dhr.get(c, np.nan))
    lag, g, d = np.array(lag), np.array(g), np.array(d)

    def desc(x):
        x = x[np.isfinite(x)]
        h, e = np.histogram(np.clip(x, -6, 4), bins=np.arange(-6, 4.01, 0.5))
        return {"n": int(x.size), "median": round(float(np.median(x)), 3),
                "share_within_0.5s_of_0": round(float(np.mean(np.abs(x) <= 0.5)), 3),
                "share_within_0.5s_of_-2": round(float(np.mean(np.abs(x + 2) <= 0.5)), 3),
                "share_below_-2.5": round(float(np.mean(x < -2.5)), 3),
                "hist_edges": [round(float(v), 1) for v in e], "hist": [int(v) for v in h]}

    S = {"parts": a.parts, "all": desc(lag), "by_group": {k: desc(lag[g == k]) for k in ("early", "late", "uncertain")}}
    fin = np.isfinite(d)
    if fin.sum() > 10:
        S["lag_vs_fitted_delay"] = {}
        for lo, hi in ((-1, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 9)):
            k = fin & (d >= lo) & (d < hi)
            if k.sum() >= 20:
                S["lag_vs_fitted_delay"][f"{lo}-{hi}s"] = {"n": int(k.sum()), "median_lag": round(float(np.median(lag[k])), 3),
                                                           "share_near_-2": round(float(np.mean(np.abs(lag[k] + 2) <= 0.5)), 3)}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps({k: (v if k != "parts" else len(v)) for k, v in S.items()}, indent=1))


if __name__ == "__main__":
    main()
