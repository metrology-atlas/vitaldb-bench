# SPDX-License-Identifier: MIT
"""Summarise the rates a resync-timed TimeNet build estimated, per waveform track (aggregates only).

    python -m vitaldb_bench.timing_report --timenet DIR --out timing_report.json

For each track: samples by rate source (resync / track / nominal), and the sample-weighted distribution
of the estimated rate error in ppm, overall and per case (a device's clock should be consistent within
a case, so a wide within-case spread suggests the estimate is noise rather than drift).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from vitaldb_bench.stores import TimeNetStore


def wquantiles(x: np.ndarray, w: np.ndarray, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> list[float]:
    """Weighted quantiles: the smallest value whose cumulative weight share reaches each q.

    Args:
        x: Values.
        w: Non-negative weights, one per value (here: samples per signal).
        qs: Quantiles in [0, 1].

    Returns:
        One value of ``x`` per quantile.

    Examples:
        >>> import numpy as np
        >>> wquantiles(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 8.0]), (0.5,))
        [3.0]
    """
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return [float(x[o][min(len(x) - 1, np.searchsorted(cw, q))]) for q in qs]


def main() -> None:
    """Command-line entry point: ``python -m vitaldb_bench.timing_report --timenet DIR --out FILE``."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timenet", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    store = TimeNetStore(a.timenet)
    by_track: dict = defaultdict(lambda: {"samples": defaultdict(int), "ppm": [], "w": [], "case_spread": [], "case_median": []})
    for caseid in store.caseids:
        per_case: dict = defaultdict(lambda: ([], []))
        for s in store.record(caseid).signals:
            md = s.metadata
            if "rate_source" not in md:
                continue
            g = by_track[md["vital_track"]]
            g["samples"][md["rate_source"]] += s.n_values
            if md["rate_source"] == "resync":
                g["ppm"].append(md["ppm_vs_nominal"])
                g["w"].append(s.n_values)
                per_case[md["vital_track"]][0].append(md["ppm_vs_nominal"])
                per_case[md["vital_track"]][1].append(s.n_values)
        for name, (p, w) in per_case.items():
            if len(p) > 1:
                q = wquantiles(np.array(p), np.array(w, dtype=float), (0.1, 0.5, 0.9))
                by_track[name]["case_spread"].append(q[2] - q[0])
            by_track[name]["case_median"].append(wquantiles(np.array(p), np.array(w, dtype=float), (0.5,))[0])
    out = {}
    for name, g in sorted(by_track.items()):
        total = sum(g["samples"].values())
        entry = {"samples": total, "share_by_source": {k: v / total for k, v in g["samples"].items()}}
        if g["ppm"]:
            entry["ppm_weighted_q05_q25_q50_q75_q95"] = wquantiles(np.array(g["ppm"]), np.array(g["w"], dtype=float))
            entry["case_medians_ppm_q10_q50_q90"] = np.quantile(g["case_median"], [0.1, 0.5, 0.9]).tolist()
            if g["case_spread"]:
                entry["within_case_spread_ppm_median"] = float(np.median(g["case_spread"]))
            entry["cases_with_resync"] = len(g["case_median"])
        out[name] = entry
    Path(a.out).write_text(json.dumps(out, indent=1))
    for name, e in out.items():
        if e.get("cases_with_resync"):
            print(name, json.dumps({k: (np.round(v, 2).tolist() if isinstance(v, list) else v) for k, v in e.items()}))


if __name__ == "__main__":
    main()
