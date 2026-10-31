# SPDX-License-Identifier: MIT
"""Method comparison: the theatre blood-gas analyser against the central laboratory, on the same draw.

    python tools/lab_bg_vs_central.py --root R --labels RESULTS --out DIR [--window 900]

Both analysers report sodium, potassium, glucose and haematocrit. Each lab specimen group in VitalDB's lab table is
a panel (blood_gas: pH and gases; central: CBC, chemistry or coagulation) with one timestamp per specimen
(tools/lab_draw_labels.py, lab_time_labels.csv). Pairs, one blood-gas group with one central group of the same case:

  same access   both groups matched to the same arterial-line access (lab_draw_labels.csv): the same draw, as far as
                the release can tell;
  within T      the central group stamped 0..T s after the blood gas (default 15 min), with no other blood gas in
                between: most are one draw, since central panels are stamped about 10 min after a blood gas. Groups
                are built from lab_data.csv for every case (results within 60 s are one group).

Per analyte, blood gas minus central: the mean difference and limits of agreement, the difference by the mean of the
two (Bland-Altman bins), the SD between patients of their mean difference and within a patient, and the time from
the blood gas's stamp to the central one. Writes DIR/summary.json (aggregates only) and DIR/lab_bg_vs_central.png.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ANALYTES = {"na": ("Sodium", "mmol/L", 1.0), "k": ("Potassium", "mmol/L", 0.2), "gluc": ("Glucose", "mg/dL", 10.0),
            "hct": ("Haematocrit", "%", 2.0)}


def read_groups(labels):
    """Specimen groups: (case, lab_rel_s, panel, draw_id) and each draw_id's matched access start."""
    groups = [dict(case=int(r["case"]), t=float(r["lab_rel_s"]), panel=r["panel"], draw=r["draw_id"])
              for r in csv.DictReader(open(Path(labels) / "lab_time_labels.csv"))]
    access = {r["draw_id"]: (int(r["case"]), r["start_s"]) for r in csv.DictReader(open(Path(labels) / "lab_draw_labels.csv"))}
    return groups, access


def read_lab(root):
    """(case, dt, name) -> result for the analytes compared."""
    lab = {}
    with open(Path(root) / "lab_data.csv", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["name"] in ANALYTES:
                try:
                    lab[(int(r["caseid"]), float(r["dt"]), r["name"])] = float(r["result"])
                except ValueError:
                    pass
    return lab


CENTRAL = {"wbc", "plt", "hb", "cr", "bun", "alb", "ast", "alt", "tbil", "tprot", "cl", "gfr", "aptt", "ptinr"}
GROUP_S = 60.0


def groups_from_lab(root):
    """Specimen groups straight from lab_data.csv, every case: results stamped within 60 s of the group's first are
    one group; blood_gas if it has pH and no central-lab marker, central if it has one and no pH."""
    by = defaultdict(list)
    with open(Path(root) / "lab_data.csv", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            by[int(r["caseid"])].append((float(r["dt"]), r["name"]))
    out = []
    for c, rows in by.items():
        rows.sort()
        cur, t0 = [], None
        for t, n in rows + [(float("inf"), "")]:
            if t0 is None or t - t0 > GROUP_S:
                if cur:
                    names = {x for _, x in cur}
                    bg, ce = "ph" in names, bool(names & CENTRAL)
                    panel = "blood_gas" if bg and not ce else "central" if ce and not bg else "mixed" if bg else "other"
                    out.append(dict(case=c, t=cur[0][0], panel=panel, draw=None, times={x: tt for tt, x in cur}))
                cur, t0 = [], t
            cur.append((t, n))
    return out


def pairs(groups, access, window):
    by_case = defaultdict(list)
    for g in groups:
        if g["panel"] in ("blood_gas", "central"):
            by_case[g["case"]].append(g)
    same, timed = [], []
    for c, gs in by_case.items():
        gs.sort(key=lambda g: g["t"])
        bgs = [g for g in gs if g["panel"] == "blood_gas"]
        cen = [g for g in gs if g["panel"] == "central"]
        for b in bgs:
            ab = access.get(b["draw"])
            for z in cen:
                if ab and access.get(z["draw"]) == ab:
                    same.append((c, b["t"], z["t"]))
            nxt = [x["t"] for x in bgs if x["t"] > b["t"]]
            limit = min([b["t"] + window] + nxt)
            for z in cen:
                if b["t"] <= z["t"] <= limit:
                    timed.append((c, b, z) if "times" in b else (c, b["t"], z["t"]))
                    break
    return same, timed


def compare(prs, lab, name):
    case, d, a, dt = [], [], [], []
    for c, tb, tz in prs:
        if isinstance(tb, dict):        # groups from lab_data: each analyte at its own time in the group
            vb = lab.get((c, tb["times"].get(name), name)); vz = lab.get((c, tz["times"].get(name), name))
            tb, tz = tb["t"], tz["t"]
        else:
            vb, vz = lab.get((c, tb, name)), lab.get((c, tz, name))
        if vb is None or vz is None:
            continue
        case.append(c); d.append(vb - vz); a.append((vb + vz) / 2); dt.append(tz - tb)
    if len(d) < 20:
        return None
    case, d, a, dt = np.array(case), np.array(d), np.array(a), np.array(dt)
    uc = [c for c in np.unique(case) if (case == c).sum() >= 2]
    pm = np.array([d[case == c].mean() for c in uc])
    within = np.concatenate([d[case == c] - d[case == c].mean() for c in uc]) if uc else np.array([])
    step = ANALYTES[name][2]
    edges = np.quantile(a, np.linspace(0, 1, 6))
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (a >= lo) & (a <= hi)
        if m.sum() >= 10:
            bins.append({"from": round(float(lo), 2), "to": round(float(hi), 2), "pairs": int(m.sum()),
                         "mean_diff": round(float(d[m].mean()), 3), "sd_diff": round(float(d[m].std()), 3)})
    return {"pairs": int(d.size), "patients": int(np.unique(case).size), "mean_diff": round(float(d.mean()), 3),
            "sd_diff": round(float(d.std()), 3),
            "loa": [round(float(d.mean() - 1.96 * d.std()), 2), round(float(d.mean() + 1.96 * d.std()), 2)],
            "median_diff": round(float(np.median(d)), 3),
            "robust_sd": round(float(1.4826 * np.median(np.abs(d - np.median(d)))), 3),
            "p2.5_p97.5": [round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2)],
            "patients_with_2_plus": len(uc),
            "between_patient_sd": round(float(pm.std()), 3) if len(uc) >= 10 else None,
            "within_patient_sd": round(float(within.std()), 3) if within.size >= 20 else None,
            "stamp_gap_min": [round(float(np.percentile(dt, p)) / 60, 1) for p in (10, 50, 90)],
            "by_level": bins, "_d": d, "_a": a}


def figure(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [n for n in ANALYTES if res.get(n)]
    fig, ax = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.2), squeeze=False)
    for a_, n in zip(ax[0], names):
        r = res[n]
        a_.scatter(r["_a"], r["_d"], s=4, alpha=0.3, color="#3f51b5", lw=0)
        for y in (r["mean_diff"], *r["loa"]):
            a_.axhline(y, color="#e65100", lw=0.8, ls="-" if y == r["mean_diff"] else ":")
        a_.axhline(0, color="0.5", lw=0.6)
        a_.set_title(f"{ANALYTES[n][0]} ({r['pairs']} pairs)", fontsize=9, loc="left")
        a_.set_xlabel(f"mean of the two ({ANALYTES[n][1]})", fontsize=8)
        a_.spines[["top", "right"]].set_visible(False)
    ax[0][0].set_ylabel("blood gas minus central lab")
    fig.suptitle("Blood-gas analyser against the central laboratory, same draw; orange: mean and 95% limits of agreement",
                 fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "lab_bg_vs_central.png", dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels", required=True, help="folder with lab_time_labels.csv and lab_draw_labels.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=float, default=900.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    groups, access = read_groups(a.labels)
    lab = read_lab(a.root)
    same, _ = pairs(groups, access, a.window)
    _, timed = pairs(groups_from_lab(a.root), {}, a.window)      # every case, not only those with a scanned line
    summ = {"groups": len(groups), "pairs_same_access": len(same), "pairs_within_window": len(timed), "window_s": a.window}
    fig_res = {}
    for key, prs in (("same_access", same), ("within_window", timed)):
        summ[key] = {}
        for n in ANALYTES:
            r = compare(prs, lab, n)
            if r:
                if key == "within_window":
                    fig_res[n] = r
                summ[key][n] = {k: v for k, v in r.items() if not k.startswith("_")}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    figure(fig_res, out)


if __name__ == "__main__":
    main()
