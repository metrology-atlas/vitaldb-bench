# SPDX-License-Identifier: MIT
"""Per-case BIS index smoothing label for the AtriumDB label pass (link S). This machine only.

    python tools/bis_smoothing_labels.py --scan f*.jsonl --root R --devices va/p?.jsonl --out RESULTS \
        --summary summary.json

Input: the per-case scan of tools/bis_index_delay.py (lag kernel of BIS/BIS against its own EEG, split into a
trailing-boxcar fast part of length L from lag d, a slow tail and a flat floor). Output, in RESULTS:

bis_smoothing_labels.csv
    One row per scanned case with a BIS index, spanning the case (release case start 2100-01-01 00:00 UTC to
    caseend from clinical_data.csv), on the BIS/BIS measure:
      group        ``15s``  good fit and fast part 13-17 s long (the 15 s smoothing setting)
                   ``long`` good fit and fast part >= 28 s (a longer setting; the method reads long windows long)
                   ``uncertain`` otherwise (poor fit, or a fast part of 18-27 s or under 13 s)
      prior_group  ``group``, except that an uncertain case on a PC whose good fits are <= 5% ``15s`` (at least
                   20 good fits) takes ``long`` from its PC; marked by prior_source = ``pc``
      window_s (the EEG the fast part averages, L + 1 s: the boxcar's L one-second feature stamps, each
      from a 2 s EEG window), delay_from_window_end_s (from the end of that EEG to the index's stamp: the
      boxcar's first lag d, since each feature is stamped at its window's end; scans made before 13 October
      stamped features at the window centre and need d - 1), half_weight_age_s (how old, from the stamp to
      the end of the EEG concerned, the index's inputs are once half its weight is counted), the raw fit
      (fast_L_s, fast_d_s: lags between feature stamps),
      fit_r2, quality (``good``: R2 >= 0.5 and at most 10% of the kernel at negative lags; else ``weak``),
      pc_fast15_share and pc_good_fits for the case's PC (ports of Solar8000, SNUADC, Primus and BIS).
      delay_check: ``negative`` where the fast part would start before the EEG window ends (impossible under
      the availability rule); such a fit is treated as poor (``weak``, group ``uncertain``).
Label name: bis_smoothing:<group>.

Availability: the kernel is fitted over the whole case and the PC prior pools other cases, so the label is
known only at the case's end (``available_at`` = ``case_end``); it must not be model input earlier in the case.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from timenet_connectors.datasets.vitaldb.vitaldb.timing_model import MODEL_VERSION  # noqa: E402

CASE_START = 4102444800.0  # 2100-01-01 00:00 UTC, the release's case start
METHOD = "bis_index_delay rank-one lag kernel, boxcar + slow tail + floor"
GOOD_R2, MAX_NEG = 0.5, 0.1
PC_MIN, PC_NEVER = 20, 0.05


def delay_from_end(r: dict) -> float | None:
    """The fast part's delay from the end of its EEG window (scans before 13 October were centre-stamped)."""
    d = r.get("fast_d")
    return None if d is None else d - (0 if r.get("feature_anchor") == "window_end" else 1)


def classify(r: dict) -> tuple[str, str]:
    neg = (delay_from_end(r) or 0) < 0
    good = "fast_L" in r and r.get("r2", 0) >= GOOD_R2 and r.get("neg_mass", 1) <= MAX_NEG and not neg
    if not good:
        return "uncertain", "weak"
    L = r["fast_L"]
    if 13 <= L <= 17:
        return "15s", "good"
    if L >= 28:
        return "long", "good"
    return "uncertain", "good"


def pc_key(dev: dict) -> str:
    d = {k: v[1] for k, v in dev.items()}
    return "|".join(f"{k}:{d.get(k, '-')}" for k in ("Solar8000", "SNUADC", "Primus", "BIS"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", nargs="+", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--devices", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    a = ap.parse_args()
    R = {}
    for p in a.scan:
        for line in open(p):
            r = json.loads(line)
            if "centroid" in r:
                R[r["case"]] = r
    pc = {}
    for p in a.devices:
        for line in open(p):
            e = json.loads(line)
            pc[e["case"]] = pc_key(e.get("devices", {}))
    caseend = {}
    with open(Path(a.root) / "clinical_data.csv", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            caseend[int(r["caseid"])] = (float(r["casestart"] or 0), float(r["caseend"]))

    cls = {c: classify(r) for c, r in R.items()}
    by_pc = defaultdict(Counter)
    for c, (g, q) in cls.items():
        if q == "good" and c in pc:
            by_pc[pc[c]]["good"] += 1
            by_pc[pc[c]]["15s"] += g == "15s"

    rows = []
    for c in sorted(R):
        if c not in caseend:
            continue
        r, (g, q) = R[c], cls[c]
        st = by_pc.get(pc.get(c), Counter())
        share = st["15s"] / st["good"] if st["good"] else None
        prior, src = g, "own"
        if g == "uncertain" and st["good"] >= PC_MIN and share is not None and share <= PC_NEVER:
            prior, src = "long", "pc"
        s0, s1 = caseend[c]
        L, d = r.get("fast_L"), r.get("fast_d")
        # features stamped at their window's end (scans from 13 October) or at its centre (earlier: one second less)
        shift = 0 if r.get("feature_anchor") == "window_end" else 1
        rows.append({"case": c, "label": f"bis_smoothing:{g}", "start_s": repr(CASE_START + s0),
                     "end_s": repr(CASE_START + s1), "group": g, "prior_group": prior, "prior_source": src,
                     "window_s": None if L is None else L + 1, "delay_from_window_end_s": None if d is None else d - shift,
                     "half_weight_age_s": None if r.get("q50") is None else r["q50"] - shift,
                     "fast_L_s": L, "fast_d_s": d, "fit_r2": r.get("r2"), "quality": q, "available_at": "case_end",
                     "delay_check": "negative" if (delay_from_end(r) or 0) < 0 else "ok",
                     "pc_fast15_share": None if share is None else round(share, 3), "pc_good_fits": st["good"],
                     "model_version": MODEL_VERSION, "method": METHOD})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "bis_smoothing_labels.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    S = {"rows": len(rows), "model_version": MODEL_VERSION, "groups": dict(Counter(r["group"] for r in rows)),
         "prior_groups": dict(Counter(r["prior_group"] for r in rows)),
         "prior_from_pc": sum(r["prior_source"] == "pc" for r in rows),
         "quality": dict(Counter(r["quality"] for r in rows)),
         "pcs_never_15s": sum(1 for k, v in by_pc.items() if v["good"] >= PC_MIN and v["15s"] / v["good"] <= PC_NEVER)}
    Path(a.summary).write_text(json.dumps(S, indent=1))
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
