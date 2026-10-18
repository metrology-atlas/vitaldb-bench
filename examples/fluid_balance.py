"""Worked example: intraoperative fluid balance as a probability distribution.

    python examples/fluid_balance.py --resolution <vitaldb-resolution checkout> --root R --out OUT
        [--draws 4000] [--figure CASE ...]

    balance = crystalloid + colloid - estimated blood loss - urine output      (mL)

Each term is a charted volume with its own heaping (crystalloid on 100 and 50 mL, blood loss and urine output on
50 and 100 mL, colloid on container sizes). For every case with crystalloid, blood loss and urine output recorded
and no blood products (whose volume per unit is not in the release), the example draws each true volume from the
vitaldb-resolution library, sums per draw, and summarises each case by the median and a 95% interval. A recorded
0 (no colloid, say) counts as exactly 0. The question it answers is how much of the spread in a computed balance
comes from how the volumes were written down.

Writes OUT/summary.json (aggregates only), OUT/cases.csv (per case; stays on this machine), OUT/fb_examples.png
(the cases given with --figure) and OUT/fb_cohort.png. A case id appears in a figure only because the example
discusses that case.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TERMS = (("intraop_crystalloid", +1), ("intraop_colloid", +1), ("intraop_ebl", -1), ("intraop_uo", -1))


def draws(reg, row: dict, var: str, n: int, rng) -> np.ndarray | None:
    v = row.get(var)
    if v == 0:
        return np.zeros(n)
    d = reg.case_distribution(row, var)
    if d.status != "ok" or not d.components:
        return None
    return d.sample(n, rng)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resolution", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--figure", nargs="*", type=int, default=[])
    a = ap.parse_args()
    sys.path.insert(0, str(a.resolution))
    import vitaldb_resolution as vr

    a.out.mkdir(parents=True, exist_ok=True)
    reg = vr.load()
    clin = pd.read_csv(a.root / "clinical_data.csv")
    use = clin[clin.intraop_crystalloid.notna() & clin.intraop_ebl.notna() & clin.intraop_uo.notna()
               & (clin.intraop_rbc.fillna(0) == 0) & (clin.intraop_ffp.fillna(0) == 0)]
    rng = np.random.default_rng(0)
    rows, keep, dropped = [], {}, 0
    for r in use.to_dict("records"):
        parts = {v: draws(reg, r, v, a.draws, rng) for v, _ in TERMS}
        if any(x is None for x in parts.values()):
            dropped += 1
            continue
        bal = sum(s * parts[v] for v, s in TERMS)
        naive = sum(s * (r[v] if pd.notna(r[v]) else 0.0) for v, s in TERMS)
        lo, med, hi = np.percentile(bal, [2.5, 50, 97.5])
        sds = {v: float(np.std(parts[v])) for v, _ in TERMS}
        rows.append({"case": int(r["caseid"]), "naive": naive, "median": med, "lo": lo, "hi": hi, "width": hi - lo,
                     **{f"sd_{v.split('_', 1)[1]}": s for v, s in sds.items()}})
        if int(r["caseid"]) in a.figure:
            keep[int(r["caseid"])] = (bal, naive, r)
    df = pd.DataFrame(rows)
    df.to_csv(a.out / "cases.csv", index=False)
    share = {k: float((df[k] ** 2 / sum(df[c] ** 2 for c in df.columns if c.startswith("sd_"))).median())
             for k in df.columns if k.startswith("sd_")}
    summ = {"cases": len(df), "not_computable": dropped, "draws": a.draws,
            "width95_ml": {"median": round(float(df.width.median()), 1),
                           "p10": round(float(df.width.quantile(.1)), 1), "p90": round(float(df.width.quantile(.9)), 1)},
            "median_minus_naive_ml": round(float((df["median"] - df.naive).median()), 1),
            "naive_balance_ml_median": round(float(df.naive.median()), 1),
            "variance_share_median": {k[3:]: round(v, 3) for k, v in share.items()},
            "sign_uncertain": int(((df.lo < 0) & (df.hi > 0)).sum())}
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
        for axi, (c, (bal, naive, r)) in zip(ax[0], sorted(keep.items())):
            lo, hi = np.percentile(bal, [0.5, 99.5])
            axi.hist(bal[(bal >= lo) & (bal <= hi)], bins=60, color="#3f51b5", alpha=0.75, density=True)
            axi.axvline(naive, color="0.2", lw=1, ls="--")
            l95, h95 = np.percentile(bal, [2.5, 97.5])
            axi.set_title(f"case {c}: 95% {l95:.0f} to {h95:.0f} mL\n"
                          f"in {r['intraop_crystalloid']:g} + {r['intraop_colloid']:g}, "
                          f"out {r['intraop_ebl']:g} + {r['intraop_uo']:g}", fontsize=8, loc="left")
            axi.set_xlabel("balance (mL)", fontsize=8)
            axi.set_yticks([])
            axi.spines[["top", "right", "left"]].set_visible(False)
        fig.suptitle("Fluid balance: distribution (blue) and the recorded numbers' balance (dashed)", fontsize=8,
                     x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out / "fb_examples.png", dpi=140)
        plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.0))
    ax[0].scatter(df.naive.clip(-2000, 6000), df.width, s=4, alpha=0.5, color="#3f51b5", linewidths=0)
    ax[0].set_xlabel("balance from the recorded numbers (mL)")
    ax[0].set_ylabel("95% interval width (mL)")
    ax[1].hist(df.width.clip(upper=400), bins=60, color="#3f51b5", alpha=0.8)
    ax[1].set_xlabel("95% interval width (mL, cut at 400)")
    ax[1].set_ylabel("cases")
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{len(df):,} cases without blood products: median 95% width {df.width.median():.0f} mL",
                 fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fb_cohort.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
