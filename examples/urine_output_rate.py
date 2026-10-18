"""Worked example: intraoperative urine output in mL/kg/h as a probability distribution.

    python examples/urine_output_rate.py --resolution <vitaldb-resolution checkout> --root R --out OUT
        [--draws 4000] [--figure CASE ...]

Urine output, weight and anaesthesia duration are each recorded with their own rounding: urine output heaped on
50 and 100 mL, weight on whole kilograms and on 0.1 or 0.05 kg scales, the anaesthesia start and end on 5- and
10-minute clock marks. For every case with a positive urine output, a weight and both anaesthesia times, the
example draws each true value from the vitaldb-resolution library (``Registry.case_distribution``, the same draws
for every input of a case being independent), forms

    rate = urine output / weight / (anaesthesia duration in hours)

per draw, and summarises each case by the median, a 95% interval and P(rate < 0.5 mL/kg/h). Each case is then
classed as confidently below (P > 0.95), confidently above (P < 0.05) or uncertain, against the single rate
computed from the recorded numbers.

Cases with no urine output recorded, or a recorded 0, are left out: the zero rule is not yet in the library.

Writes OUT/summary.json (aggregates only), OUT/cases.csv (per case; stays on this machine) and figures:
OUT/uo_examples.png (the cases given with --figure) and OUT/uo_cohort.png. A case id appears in a figure only
because the example discusses that case.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THRESHOLD = 0.5   # mL/kg/h


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resolution", required=True, type=Path, help="vitaldb-resolution checkout (the pinned commit)")
    ap.add_argument("--root", required=True, type=Path, help="VitalDB release folder")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--figure", nargs="*", type=int, default=[])
    a = ap.parse_args()
    sys.path.insert(0, str(a.resolution))
    import vitaldb_resolution as vr

    a.out.mkdir(parents=True, exist_ok=True)
    reg = vr.load()
    clin = pd.read_csv(a.root / "clinical_data.csv")
    use = clin[(clin.intraop_uo > 0) & clin.weight.notna() & clin.anestart.notna() & clin.aneend.notna()]
    rng = np.random.default_rng(0)
    rows, keep = [], {}
    for r in use.to_dict("records"):
        d = {v: reg.case_distribution(r, v) for v in ("intraop_uo", "weight", "ane_duration")}
        if any(x.status != "ok" or not x.components for x in d.values()):
            continue
        u = d["intraop_uo"].sample(a.draws, rng)
        w = d["weight"].sample(a.draws, rng)
        t = d["ane_duration"].sample(a.draws, rng) / 60.0
        ok = (w > 0) & (t > 0)
        rate = u[ok] / w[ok] / t[ok]
        naive = r["intraop_uo"] / r["weight"] / ((r["aneend"] - r["anestart"]) / 3600.0)
        p = float(np.mean(rate < THRESHOLD))
        lo, med, hi = np.percentile(rate, [2.5, 50, 97.5])
        cls = "below" if p > 0.95 else ("above" if p < 0.05 else "uncertain")
        rows.append({"case": int(r["caseid"]), "naive": naive, "median": med, "lo": lo, "hi": hi, "p_below": p,
                     "class": cls, "naive_below": naive < THRESHOLD,
                     "rel_width": (hi - lo) / med if med > 0 else np.nan})
        if int(r["caseid"]) in a.figure:
            keep[int(r["caseid"])] = (rate, naive, r)
    df = pd.DataFrame(rows)
    df.to_csv(a.out / "cases.csv", index=False)
    near = df[(df.naive > 0.35) & (df.naive < 0.7)]
    summ = {
        "cases": len(df), "draws": a.draws, "threshold": THRESHOLD,
        "naive_below": int(df.naive_below.sum()),
        "classes": df["class"].value_counts().to_dict(),
        "uncertain_share": round(float((df["class"] == "uncertain").mean()), 4),
        "uncertain_among_naive_0.35_0.7": [int((near["class"] == "uncertain").sum()), len(near)],
        "naive_vs_distribution": {
            "naive_below_but_not_confident": int((df.naive_below & (df["class"] != "below")).sum()),
            "naive_above_but_not_confident": int((~df.naive_below & (df["class"] != "above")).sum())},
        "rel_width_95_median": round(float(df.rel_width.median()), 3),
        "rel_width_95_p10_p90": [round(float(df.rel_width.quantile(.1)), 3), round(float(df.rel_width.quantile(.9)), 3)],
        "median_minus_naive_median": round(float((df["median"] - df.naive).median()), 4),
    }
    (a.out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    figures(df, keep, a.out)


def figures(df: pd.DataFrame, keep: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if keep:
        n = len(keep)
        fig, ax = plt.subplots(1, n, figsize=(2.6 * n, 3.0), squeeze=False)
        for axi, (c, (rate, naive, r)) in zip(ax[0], sorted(keep.items(), key=lambda x: np.median(x[1][0]))):
            lo, hi = np.percentile(rate, [0.5, 99.5])
            lo, hi = min(lo, THRESHOLD - 0.05), max(hi, THRESHOLD + 0.05)
            axi.hist(rate[(rate >= lo) & (rate <= hi)], bins=np.linspace(lo, hi, 70), color="#3f51b5", alpha=0.75,
                     density=True)
            axi.axvline(THRESHOLD, color="#e65100", lw=1.2)
            axi.axvline(naive, color="0.2", lw=1, ls="--")
            p = np.mean(rate < THRESHOLD)
            axi.set_title(f"case {c}: P(< 0.5) = {p:.2f}\n{r['intraop_uo']:g} mL, {r['weight']:g} kg, "
                          f"{(r['aneend'] - r['anestart']) / 60:.0f} min", fontsize=8, loc="left")
            axi.set_xlabel("mL/kg/h", fontsize=8)
            axi.set_yticks([])
            axi.spines[["top", "right", "left"]].set_visible(False)
        fig.suptitle("Urine output rate: distribution (blue), the recorded numbers' rate (dashed), 0.5 mL/kg/h (orange)",
                     fontsize=8, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out / "uo_examples.png", dpi=140)
        plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.0))
    x = df.naive.clip(upper=3)
    col = df["class"].map({"below": "#e65100", "above": "#3f51b5", "uncertain": "#9e9e9e"})
    ax[0].scatter(x, df.p_below, s=4, c=col, alpha=0.6, linewidths=0)
    ax[0].axvline(THRESHOLD, color="0.4", lw=0.8, ls=":")
    ax[0].set_xlabel("rate from the recorded numbers (mL/kg/h, cut at 3)")
    ax[0].set_ylabel("P(true rate < 0.5)")
    ax[0].set_xlim(0, 1.5)
    ax[1].hist(df.rel_width.clip(upper=1), bins=50, color="#3f51b5", alpha=0.8)
    ax[1].set_xlabel("95% interval width / median")
    ax[1].set_ylabel("cases")
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    counts = df["class"].value_counts()
    fig.suptitle(f"{len(df):,} cases: {counts.get('below', 0)} confidently below 0.5, {counts.get('uncertain', 0)} "
                 f"uncertain, {counts.get('above', 0)} confidently above", fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "uo_cohort.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
