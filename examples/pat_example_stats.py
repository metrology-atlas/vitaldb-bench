# SPDX-License-Identifier: MIT
"""The numbers the atlas page examples/pulse-arrival-time quotes, from a run of examples/pulse_arrival_time.py.

    python examples/pat_example_stats.py OUT [--compare OUT_DETECT] [--case 5975] --json OUT/page_stats.json

OUT holds summary.json and windows/<case>.npz. Cases enter as in the example's figures (usable(): ran, and pulses
paired with their own beat). Per-case spreads are the example's own (robust SD of the 30 s maximum-likelihood window
estimates, in ms); "better" / "worse" mean by more than 2 ms. --compare pairs the cases with a second run on the same
windows (PAT_METHOD=detect) and compares each with its own stored ("raw") value.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pulse_arrival_time import robust_sd, usable  # noqa: E402

BY = 2.0


def r1(x):
    return None if x is None else round(float(x), 1)


def qs(v, p=(5, 50, 95)):
    return [r1(x) for x in np.percentile(np.asarray(v, float), p)] if len(v) else None


def load(out: Path) -> tuple[dict, list[dict]]:
    s = json.loads((out / "summary.json").read_text())
    return s, [v for v in s.values() if usable(v)]


def spreads(ok: list[dict], key: str) -> dict:
    c = [v for v in ok if v.get(key)]
    raw, cor = np.array([v[key]["raw"] for v in c]), np.array([v[key]["cor"] for v in c])
    return {"cases": len(c), "raw_median": r1(np.median(raw)) if c else None, "cor_median": r1(np.median(cor)) if c else None,
            "cor_iqr": qs(cor, (25, 75)), "better": int(np.sum(raw - cor > BY)), "worse": int(np.sum(cor - raw > BY))}


def teeth(ok: list[dict]) -> dict:
    """Cases by their teeth: any corrected (A, B or C); else isolated slips (I); no slip (N); only unresolved (U)."""
    t = [v.get("tiers", "") for v in ok]
    corrected = [any(c in "ABC" for c in x) for x in t]
    isolated = [not k and "I" in x for k, x in zip(corrected, t)]
    no_slip = [x in ("", "N") for x in t]
    unresolved = [bool(x) and set(x) <= set("U") for x in t]
    return {"corrected": sum(corrected), "isolated_only": sum(isolated), "no_slip": sum(no_slip),
            "unresolved_only": sum(unresolved),
            "other": len(t) - sum(corrected) - sum(isolated) - sum(no_slip) - sum(unresolved)}


def pooled_ab(out: Path, ok: list[dict]) -> dict:
    """Finger minus arterial PAT, each case centred, pooled over the 30 s windows of the cases with an arterial line
    and a tooth in tier A or B: robust SD as stored and with the sawtooth removed (as the cohort figure's panel C)."""
    gap = {"raw": [], "saw": []}
    n = 0
    for v in ok:
        if "minus_art_sd_ms" not in v or not any(t in "AB" for t in v.get("tiers", "")):
            continue
        z = np.load(out / "windows" / f"{v['case']}.npz")
        if not all(k + "_k" in z.files for k in ("raw", "saw", "art")):
            continue
        n += 1
        for k in gap:
            _, iu, iv = np.intersect1d(z[k + "_k"], z["art_k"], return_indices=True)
            d = (z[k][iu] - z["art"][iv]) * 1000
            gap[k].extend((d - np.median(d)).tolist())
    return {"cases": n, **{k: r1(robust_sd(g)) if g else None for k, g in gap.items()}}


def stats(out: Path, case: int | None) -> dict:
    s, ok = load(out)
    S = {"run": str(out), "cases": len(s), "usable": len(ok),
         "skipped": dict(Counter(v["skip"] for v in s.values() if "skip" in v)),
         "correction": dict(Counter(v.get("correction") for v in ok))}
    mis = [v["raw_median_ms"] for v in s.values() if "within_case_sd_ms" in v and not usable(v)]
    S["mispaired"] = {"cases": len(mis), "raw_median_ms": [r1(min(mis)), r1(max(mis))] if mis else None}
    S["arterial_line"] = {"all": sum(bool(v.get("arterial_line")) for v in s.values()),
                          "usable": sum(bool(v.get("arterial_line")) for v in ok)}
    S["slips_per_hour_median"] = r1(np.median([v["slips"] / v["minutes"] * 60 for v in ok]))
    S["cases_with_corrected_tooth"] = sum(any(t in "ABC" for t in v.get("tiers", "")) for v in ok)
    S["cases_with_ab_tooth"] = sum(any(t in "AB" for t in v.get("tiers", "")) for v in ok)
    S["teeth"] = teeth(ok)
    S["stored_case_median_ms"] = [r1(min(v["raw_median_ms"] for v in ok)), r1(max(v["raw_median_ms"] for v in ok))]
    S["minus_art_pooled_ab"] = pooled_ab(out, ok)
    S["offset_ms"] = qs([v["offset_ms"] for v in ok], (10, 50, 90))
    S["case_median_corrected_ms"] = qs([v["raw_median_ms"] - v["offset_ms"] for v in ok], (25, 50, 75))
    S["within_case_sd_ms"] = spreads(ok, "within_case_sd_ms")
    S["minus_art_sd_ms"] = spreads(ok, "minus_art_sd_ms")
    # windows: values, distance from the case median, and the arterial pairs
    W = {k: [] for k in ("raw", "cor", "art")}
    far, n_win, pairs = 0, 0, {"raw": [], "cor": []}
    per_case = {"raw": [], "cor": []}
    for v in ok:
        z = np.load(out / "windows" / f"{v['case']}.npz")
        for k in W:
            if k in z.files and len(z[k]):
                W[k].append(z[k] * 1000)
        far += int(np.sum(np.abs(z["raw"] - np.median(z["raw"])) * 1000 > 40))
        n_win += len(z["raw"])
        if "art_k" in z.files and len(z["art"]) and v.get("minus_art_sd_ms"):
            for k in pairs:
                _, iu, iv = np.intersect1d(z[k + "_k"], z["art_k"], return_indices=True)
                xy = np.c_[z["art"][iv], z[k][iu]] * 1000
                pairs[k].append(xy)
                per_case[k].append(np.median(xy, axis=0))
    S["windows"] = {}
    for k, v in W.items():
        y = np.concatenate(v) if v else np.array([])
        S["windows"][k] = {"windows": int(y.size), "p5_p50_p95_ms": qs(y), "negative": round(float(np.mean(y < 0)), 3)
                           if y.size else None}
    S["windows_beyond_40ms_of_case_median"] = round(far / n_win, 3) if n_win else None
    S["vs_art"] = {}
    for k in pairs:
        if not pairs[k]:
            continue
        xy, m = np.vstack(pairs[k]), np.array(per_case[k])
        slope, _ = np.polyfit(m[:, 0], m[:, 1], 1)
        S["vs_art"][k] = {"pairs": int(len(xy)), "cases": int(len(m)),
                          "finger_minus_art_median_ms": r1(np.median(xy[:, 1] - xy[:, 0])),
                          "across_cases_slope": round(float(slope), 2),
                          "across_cases_r": round(float(np.corrcoef(m[:, 0], m[:, 1])[0, 1]), 2)}
    if case is not None and str(case) in s:
        v = s[str(case)]
        S["case"] = {k: v.get(k) for k in ("case", "minutes", "slips", "tiers", "correction", "offset_ms",
                                            "within_case_sd_ms", "minus_art_sd_ms", "raw_median_ms")}
        z = np.load(out / "windows" / f"{case}.npz")
        if "art_k" in z.files:
            _, iu, iv = np.intersect1d(z["cor_k"], z["art_k"], return_indices=True)
            S["case"]["cor_minus_art_median_ms"] = r1(np.median(z["cor"][iu] - z["art"][iv]) * 1000)
    return S


def compare(a: Path, b: Path) -> dict:
    sa, _ = load(a)
    sb, _ = load(b)
    c = [k for k in sa if k in sb and usable(sa[k]) and usable(sb[k])]
    C = {"cases": len(c), "level_difference_ms_max": r1(max(abs(sa[k]["offset_ms"] - sb[k]["offset_ms"]) for k in c)),
         "b_teeth": teeth([sb[k] for k in c])}
    for key in ("within_case_sd_ms", "minus_art_sd_ms"):
        cc = [k for k in c if sa[k].get(key) and sb[k].get(key)]
        ca = np.array([sa[k][key]["cor"] for k in cc])
        cb = np.array([sb[k][key]["cor"] for k in cc])
        C[key] = {"cases": len(cc), "a": spreads([sa[k] for k in cc], key), "b": spreads([sb[k] for k in cc], key),
                  "a_better_than_b": int(np.sum(cb - ca > BY)), "b_better_than_a": int(np.sum(ca - cb > BY))}
    return C


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--compare", type=Path)
    ap.add_argument("--case", type=int)
    ap.add_argument("--json", type=Path, required=True)
    a = ap.parse_args()
    S = stats(a.out, a.case)
    if a.compare:
        S["compare"] = {"with": str(a.compare), **compare(a.out, a.compare)}
    a.json.write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
