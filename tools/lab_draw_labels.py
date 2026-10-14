# SPDX-License-Identifier: MIT
"""Estimated blood-draw times for VitalDB lab samples, from arterial-line accesses. This machine only.

    python tools/lab_draw_labels.py --lab lab_data.csv --art-labels art_line_labels.csv \
        --out-draw lab_draw_labels.csv --out-time lab_time_labels.csv [--window 900 --window-central 1800 --group 60]

What VitalDB's lab_data ``dt`` marks (13 October 2026): a per-specimen timestamp, NOT the result time and
NOT the draw time. Every analyte of a panel shares it to the second; panels run on different analysers from
one draw are stamped seconds apart in no fixed order; and, timed against the arterial-line access of the
draw, central-lab panels (CBC, chemistry, coagulation) are stamped about 10 min or more after it and blood
gases 3-12 min after it (two modes, about 4 and 9 min). So ``dt`` is best read as the specimen being logged
at the lab (central lab) or the point-of-care result (blood gas). The result itself is known only after a
turnaround that the dataset does not record: a lab value must not be used at ``dt`` (the availability rule).
See the drift document, "What the lab timestamp marks".

Results within --group seconds of the first of a group are one specimen group. Its panel is ``blood_gas``
(pH, gases, lactate, base excess and no central-lab marker), ``central`` (CBC, chemistry or coagulation
markers and no blood gas), ``mixed`` (both) or ``other`` (electrolytes only, etc.). Each group is matched to
the latest arterial-line access (art_line:access, from tools/line_access_scan.py) that starts
0..window seconds before it: --window for blood gases, --window-central for every other panel, because
central-lab specimens are logged later. As lineaccess.match_times does:

    lab_draw:single     one candidate access in the window
    lab_draw:multiple   several candidates; the latest is used
    (no lab_draw label) no access in the window

Every group also gets a ``lab_time:<panel>`` instant at its timestamp, with the same draw_id and
``available_at = lab_time_plus_turnaround`` (the value is known at the timestamp plus an unknown turnaround,
never earlier). Only groups inside the scanned ART recording (first label start to last label end + the
panel's window) are kept. A match is evidence, not proof (77% of groups had an access within 15 min against
26% for random times). Test names only; the values belong to a lab import.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

DRAW_FIELDS = ["case", "label", "start_s", "end_s", "draw_id", "panel", "lab_time_s", "lab_rel_s", "lag_s",
               "window_s", "n_candidates", "confidence", "tests", "n_tests", "access_kinds", "access_max_mmhg",
               "access_start_rel_s"]
TIME_FIELDS = ["case", "label", "start_s", "end_s", "draw_id", "panel", "lab_rel_s", "tests", "n_tests", "matched",
               "confidence", "lag_s", "available_at"]
# when a lab value could first be known: its timestamp plus a turnaround the dataset does not record
AVAILABLE_AT_LAB = "lab_time_plus_turnaround"

BLOOD_GAS = {"ph", "po2", "pco2", "hco3", "sao2", "lac", "be"}
CENTRAL = {"wbc", "plt", "hb", "ptinr", "ptsec", "pt%", "aptt", "fib", "alb", "bun", "cr", "gfr", "tprot", "ast",
           "alt", "tbil", "crp", "esr", "ammo", "ccr"}


def panel(tests: set[str]) -> str:
    """blood_gas, central, mixed or other, from the analytes sharing one timestamp group."""
    gas, central = bool(tests & BLOOD_GAS), bool(tests & CENTRAL)
    return "mixed" if gas and central else "blood_gas" if gas else "central" if central else "other"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lab", required=True)
    ap.add_argument("--art-labels", required=True)
    ap.add_argument("--out-draw", required=True)
    ap.add_argument("--out-time", required=True)
    ap.add_argument("--window", type=float, default=900.0, help="match window for blood gases (s)")
    ap.add_argument("--window-central", type=float, default=1800.0,
                    help="match window for central-lab and other panels (s); they are logged later")
    ap.add_argument("--group", type=float, default=60.0)
    a = ap.parse_args()
    acc = defaultdict(list)
    span: dict[int, list[float]] = {}
    origin: dict[int, float] = {}
    with open(a.art_labels, newline="") as fh:
        for r in csv.DictReader(fh):
            c = int(r["case"])
            s, sr, er = float(r["start_s"]), float(r["start_rel_s"]), float(r["end_rel_s"])
            origin[c] = s - sr
            lo, hi = span.get(c, [sr, er])
            span[c] = [min(lo, sr), max(hi, er)]
            if r["label"] == "art_line:access":
                acc[c].append((sr, er, r["kinds"], r["max_mmhg"]))
    labs = defaultdict(list)
    with open(a.lab, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            c = int(r["caseid"])
            if c in origin:
                labs[c].append((float(r["dt"]), r["name"]))
    n = {"draws": 0, "single": 0, "multiple": 0, "none": 0}
    reach = max(a.window, a.window_central)
    with open(a.out_draw, "w", newline="") as fd, open(a.out_time, "w", newline="") as fr:
        wd = csv.DictWriter(fd, fieldnames=DRAW_FIELDS)
        wr = csv.DictWriter(fr, fieldnames=TIME_FIELDS)
        wd.writeheader()
        wr.writeheader()
        for c in sorted(labs):
            lo, hi = span[c]
            rows = sorted(x for x in labs[c] if lo <= x[0] <= hi + reach)
            groups: list[list[tuple[float, str]]] = []
            for dt, name in rows:
                if not groups or dt - groups[-1][0][0] > a.group:
                    groups.append([])
                groups[-1].append((dt, name))
            for gi, g in enumerate(groups):
                t_res = g[0][0]      # the group's lab timestamp (see the module docstring)
                tests = sorted({nm for _, nm in g})
                pan = panel(set(tests))
                win = a.window if pan == "blood_gas" else a.window_central
                if t_res > hi + win:
                    continue            # beyond the ART recording plus this panel's window
                cand = [x for x in acc[c] if 0 <= t_res - x[0] <= win]
                conf = "none" if not cand else ("single" if len(cand) == 1 else "multiple")
                n["draws"] += 1
                n[conf] += 1
                draw_id = f"{c}-{gi}"
                lag = ""
                if cand:
                    best = max(cand, key=lambda x: x[0])
                    lag = round(t_res - best[0], 1)
                    wd.writerow({"case": c, "label": f"lab_draw:{conf}", "start_s": round(best[0] + origin[c], 3),
                                 "end_s": round(best[1] + origin[c], 3), "draw_id": draw_id, "panel": pan,
                                 "lab_time_s": round(t_res + origin[c], 3), "lab_rel_s": t_res, "lag_s": lag,
                                 "window_s": win, "n_candidates": len(cand), "confidence": conf, "tests": ";".join(tests),
                                 "n_tests": len(tests), "access_kinds": best[2], "access_max_mmhg": best[3],
                                 "access_start_rel_s": best[0]})
                ts = round(t_res + origin[c], 3)
                wr.writerow({"case": c, "label": f"lab_time:{pan}", "start_s": ts, "end_s": ts, "draw_id": draw_id,
                             "panel": pan, "lab_rel_s": t_res, "tests": ";".join(tests), "n_tests": len(tests),
                             "matched": bool(cand), "confidence": conf, "lag_s": lag,
                             "available_at": AVAILABLE_AT_LAB})
    print(n)


if __name__ == "__main__":
    main()
