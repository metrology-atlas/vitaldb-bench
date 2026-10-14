# SPDX-License-Identifier: MIT
"""Aggregate tools/numeric_derivation_pilot.py output across cases (aggregates only).

    python tools/numeric_derivation_agg.py out1.jsonl [...] --summary summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np

ORDER = ["HR", "ART_SBP", "ART_MBP", "ART_DBP", "CVP", "ETCO2", "RR_CO2", "PIP", "PEEP"]


def q(x, p=(25, 50, 75), nd=2):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return [round(float(v), nd) for v in np.percentile(x, p)] if x.size else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--summary", required=True)
    a = ap.parse_args()
    rows, errors, seen = [], [], set()
    for f in a.files:
        for l in open(f):
            r = json.loads(l)
            if r["case"] in seen:
                continue
            seen.add(r["case"])
            (errors if "error" in r else rows).append(r)
    S = {"cases": len(rows), "errors": [e["error"][:120] for e in errors], "tracks": {}}
    for name in ORDER:
        T = [r["tracks"][name] for r in rows if name in r.get("tracks", {})]
        if not T:
            continue
        fam = Counter()
        win = defaultdict(list)
        d, b, s, dll_wt, dll_fn = [], [], [], [], []
        for t in T:
            fn, wt, w = t["best"]["cand"].split("|")
            fam[f"{fn}|{wt}"] += 1
            win[f"{fn}|{wt}"].append(float(w))
            d.append(t["best"]["d"])
            b.append(t["best"]["b"])
            s.append(t["best"]["s"])
            # log-lik per record: best of the other window type, best of other functions
            n = t["n_fit"]
            other = [v["ll"] for k, v in t["families"].items() if k.split("|")[1] != wt]
            if other:
                dll_wt.append((t["ll_best_stage1"] - max(other)) / n)
            oth = [v["ll"] for k, v in t["families"].items() if k.split("|")[0] != fn]
            if oth:
                dll_fn.append((t["ll_best_stage1"] - max(oth)) / n)
        out = {"n_cases": len(T), "records_median": int(np.median([t["n"] for t in T])),
               "gap_median_s": round(float(np.median([t["gap_median"] for t in T])), 2),
               "q": Counter(t["q"] for t in T).most_common(2),
               "best_family": fam.most_common(), "window_by_family": {k: sorted(v) for k, v in win.items()},
               "delay_min_s_iqr": q(d), "delay_mid_s_iqr": q([x + t["gap_median"] / 2 for x, t in zip(d, T)]),
               "bias_iqr": q(b), "sigma_iqr": q(s, nd=3),
               "dll_per_record_vs_other_window_type_iqr": q(dll_wt, nd=4),
               "dll_per_record_vs_other_function_iqr": q(dll_fn, nd=4)}
        rf = [t["round_minus_floor_ll"] for t in T if "round_minus_floor_ll" in t]
        if rf:
            out["round_minus_floor"] = {"floor_better_cases": int(sum(x < 0 for x in rf)), "of": len(rf),
                                        "dll_per_record_iqr": q([x / t["n_fit"] for x, t in zip(rf, T)], nd=3)}
        # delay by interval tertile
        tert = {}
        for wt in ("n", "t"):
            sl = []
            for t in T:
                tt = (t.get("tertiles") or {}).get(wt)
                if not tt or None in tt["by_tertile"]:
                    continue
                lo, hi = tt["by_tertile"][0], tt["by_tertile"][2]
                di = hi["interval_s"] - lo["interval_s"]
                if di > 0.02:
                    sl.append((hi["d"] - lo["d"]) / di)
            if sl:
                tert[wt] = {"cases": len(sl), "delay_change_per_s_interval_iqr": q(sl)}
        out["delay_vs_interval"] = tert
        # repeats
        R = np.zeros((5, 2))
        for t in T:
            R += np.asarray(t["repeats_by_pred_change_q"]["n_rep"], float)
        out["repeat_fraction_by_pred_change_q"] = {
            str(e): (round(r[1] / r[0], 3) if r[0] else None, int(r[0]))
            for e, r in zip(T[0]["repeats_by_pred_change_q"]["edges"], R)}
        H = [t["hold"] for t in T if "hold" in t]
        if H:
            hu = defaultdict(list)
            for h in H:
                for u in h["U"]:
                    hu[u["U"]].append(u["delta_bic"])
            out["hold_delta_bic"] = {str(U): {"median": round(float(np.median(v)), 1), "cases_gt10": int(sum(x > 10 for x in v)),
                                              "of": len(v)} for U, v in sorted(hu.items())}
            out["hold_best_U"] = Counter(max(h["U"], key=lambda u: u["delta_bic"])["U"] if max(u["delta_bic"] for u in h["U"]) > 10
                                         else "none" for h in H).most_common()
        P = [t["primus_drift"] for t in T if "primus_drift" in t]
        if P:
            out["primus_drift_best_ppm"] = Counter(max(p, key=lambda x: x["ll"])["ppm"] for p in P).most_common()
            out["primus_drift_dll_115_minus_0_iqr"] = q([[x["ll"] for x in p if x["ppm"] == 115][0] -
                                                         [x["ll"] for x in p if x["ppm"] == 0][0] for p in P])
        S["tracks"][name] = out
    json.dump(S, open(a.summary, "w"), indent=1)
    print(json.dumps(S, indent=1)[:20000])


if __name__ == "__main__":
    main()
