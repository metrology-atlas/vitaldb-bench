"""Worked example: a regression that carries rounding uncertainty through.

    python examples/regression_with_rounding.py --resolution <vitaldb-resolution checkout> --root R --out OUT
        [--draws 200]

Is lower intraoperative urine output associated with acute kidney injury after surgery, and how sure are we?
The exposure, urine output in mL/kg/h, is formed from three rounded inputs (urine output heaped on 50 and 100 mL,
weight, and anaesthesia times on 5- and 10-minute marks); see examples/urine_output_rate.py. Preoperative
creatinine has its own rounding. Rounded inputs act like measurement error: they can flatten a slope (regression
dilution) and make the naive confidence interval too narrow.

Outcome: acute kidney injury by the KDIGO creatinine criterion within 48 h of the end of surgery: the highest
postoperative creatinine at least 0.3 mg/dL above the preoperative value, or at least 1.5 times it. Postoperative
creatinine is reported to 0.01 mg/dL and is used as recorded.

Three fits, each two ways: once on the recorded numbers (naive), and once per draw from the true-value
distributions of the vitaldb-resolution library, combined by Rubin's rules as in multiple imputation:
  A. logistic: AKI ~ urine output rate (per 1 mL/kg/h, rate capped at 5)
  B. logistic: AKI ~ oliguria (rate < 0.5 mL/kg/h)
  C. linear:   log(highest postoperative / preoperative creatinine) ~ urine output rate
The fits are unadjusted on purpose: the example is about carrying the rounding through, not about the clinical
question. Writes OUT/summary.json (aggregates only), OUT/cases.csv (per case; stays local) and
OUT/regression_rounding.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THRESHOLD, CAP, WINDOW_H = 0.5, 5.0, 48.0


def logit_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Slope and its variance from a one-covariate logistic regression (IRLS)."""
    X = np.column_stack([np.ones_like(x), x])
    b = np.zeros(2)
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-10:
            break
    cov = np.linalg.inv(H)
    return float(b[1]), float(cov[1, 1])


def ols_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    X = np.column_stack([np.ones_like(x), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    s2 = r @ r / (len(y) - 2)
    return float(b[1]), float(s2 * np.linalg.inv(X.T @ X)[1, 1])


def rubin(est: np.ndarray, var: np.ndarray) -> dict:
    m = len(est)
    qbar, ubar, bvar = est.mean(), var.mean(), est.var(ddof=1)
    t = ubar + (1 + 1 / m) * bvar
    return {"estimate": qbar, "se": float(np.sqrt(t)), "lo": qbar - 1.96 * np.sqrt(t), "hi": qbar + 1.96 * np.sqrt(t),
            "within_var": ubar, "between_var": float(bvar),
            "fraction_from_rounding": float((1 + 1 / m) * bvar / t)}


def naive_row(est: float, var: float) -> dict:
    se = np.sqrt(var)
    return {"estimate": est, "se": float(se), "lo": est - 1.96 * se, "hi": est + 1.96 * se}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resolution", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--draws", type=int, default=200)
    a = ap.parse_args()
    sys.path.insert(0, str(a.resolution))
    import vitaldb_resolution as vr

    a.out.mkdir(parents=True, exist_ok=True)
    reg = vr.load()
    clin = pd.read_csv(a.root / "clinical_data.csv")
    lab = pd.read_csv(a.root / "lab_data.csv")
    cr = lab[lab.name == "cr"].merge(clin[["caseid", "opend"]], on="caseid")
    post = cr[(cr.dt > cr.opend) & (cr.dt <= cr.opend + WINDOW_H * 3600)].groupby("caseid").result.max()
    use = clin[(clin.intraop_uo > 0) & clin.weight.notna() & clin.anestart.notna() & clin.aneend.notna()
               & (clin.preop_cr > 0) & clin.caseid.isin(post.index)]
    rng = np.random.default_rng(0)
    M = a.draws
    rates, pre, ids, naive_rate, naive_pre, naive_uo = [], [], [], [], [], []
    for r in use.to_dict("records"):
        d = {v: reg.case_distribution(r, v) for v in ("intraop_uo", "weight", "ane_duration", "preop_cr")}
        if any(x.status != "ok" or not x.components for x in d.values()):
            continue
        u, w = d["intraop_uo"].sample(M, rng), d["weight"].sample(M, rng)
        t = d["ane_duration"].sample(M, rng) / 60.0
        if np.any(w <= 0) or np.any(t <= 0):
            continue
        rates.append(u / w / t)
        pre.append(d["preop_cr"].sample(M, rng))
        ids.append(int(r["caseid"]))
        naive_rate.append(r["intraop_uo"] / r["weight"] / ((r["aneend"] - r["anestart"]) / 3600.0))
        naive_pre.append(r["preop_cr"])
        naive_uo.append(r["intraop_uo"])
    R, P = np.array(rates), np.array(pre)                      # cases x draws
    ids = np.array(ids)
    postcr = post.loc[ids].to_numpy()
    nr, npre, nuo = np.array(naive_rate), np.array(naive_pre), np.array(naive_uo)

    def aki(precr):
        return ((postcr[:, None] if precr.ndim == 2 else postcr) - precr >= 0.3) | \
               ((postcr[:, None] if precr.ndim == 2 else postcr) >= 1.5 * precr)

    akiN, akiD = aki(npre).astype(float), aki(P).astype(float)
    fits = {"A_logit_rate": [], "B_logit_oliguria": [], "C_linear_logratio": []}
    for m in range(M):
        x, y = np.minimum(R[:, m], CAP), akiD[:, m]
        fits["A_logit_rate"].append(logit_fit(x, y))
        fits["B_logit_oliguria"].append(logit_fit((R[:, m] < THRESHOLD).astype(float), y))
        fits["C_linear_logratio"].append(ols_fit(x, np.log(postcr / P[:, m])))
    naive = {"A_logit_rate": logit_fit(np.minimum(nr, CAP), akiN),
             "B_logit_oliguria": logit_fit((nr < THRESHOLD).astype(float), akiN),
             "C_linear_logratio": ols_fit(np.minimum(nr, CAP), np.log(postcr / npre))}
    res = {}
    for k, v in fits.items():
        e, s = np.array(v).T
        rb, nv = rubin(e, s), naive_row(*naive[k])
        res[k] = {"naive": {kk: round(vv, 4) for kk, vv in nv.items()},
                  "distribution": {kk: round(vv, 4) for kk, vv in rb.items()},
                  "dilution_ratio_naive_over_distribution": round(nv["estimate"] / rb["estimate"], 3),
                  "se_ratio_distribution_over_naive": round(rb["se"] / nv["se"], 3),
                  "draw_estimates_p2.5_p97.5": [round(float(np.percentile(e, 2.5)), 4),
                                                round(float(np.percentile(e, 97.5)), 4)]}
        if k.startswith("B"):
            for side in ("naive", "distribution"):
                for x in ("estimate", "lo", "hi"):
                    res[k][side]["OR_" + x] = round(float(np.exp(res[k][side][x])), 3)
        if k.startswith("A"):
            for side in ("naive", "distribution"):
                for x in ("estimate", "lo", "hi"):
                    res[k][side]["OR_per_mLkgh_" + x] = round(float(np.exp(res[k][side][x])), 3)
    sweep = []
    dur = np.array([(r_["aneend"] - r_["anestart"]) / 60 for r_ in use.set_index("caseid").loc[ids].reset_index().to_dict("records")])
    for th in (0.2, 0.3, 0.5, 0.75, 1.0, 1.5):
        pd_ = (R < th).mean(0)
        within = pd_.mean() * (1 - pd_.mean()) / len(ids)
        between = (1 + 1 / M) * pd_.var(ddof=1)
        est = np.array([logit_fit((R[:, m] < th).astype(float), akiD[:, m]) for m in range(M)])
        rb, (ne, nv) = rubin(est[:, 0], est[:, 1]), logit_fit((nr < th).astype(float), akiN)
        sweep.append({"threshold": th, "prevalence_naive": round(float(np.mean(nr < th)), 4),
                      "prevalence_distribution": round(float(pd_.mean()), 4),
                      "prevalence_rounding_share_of_variance": round(float(between / (within + between)), 3),
                      "cases_crossing": round(float(np.mean((R < th).min(1) != (R < th).max(1))), 4),
                      "or_naive": round(float(np.exp(ne)), 3), "or_distribution": round(float(np.exp(rb["estimate"])), 3),
                      "or_rounding_share_of_variance": round(rb["fraction_from_rounding"], 3),
                      "or_se_ratio": round(rb["se"] / float(np.sqrt(nv)), 3)})
    sub = {}
    for name, m in (("anaesthesia 90-180 min", (dur >= 90) & (dur < 180)), ("anaesthesia 180 min or more", dur >= 180),
                    ("urine output 100 mL or less", nuo <= 100),
                    ("round 50 or 100 mL, 200 mL or less", (nuo % 50 == 0) & (nuo <= 200))):
        pd_ = (R[m] < THRESHOLD).mean(0)
        within, between = pd_.mean() * (1 - pd_.mean()) / m.sum(), (1 + 1 / M) * pd_.var(ddof=1)
        sub[name] = {"cases": int(m.sum()), "cases_crossing": round(float(np.mean((R[m] < THRESHOLD).min(1) != (R[m] < THRESHOLD).max(1))), 4),
                     "prevalence_rounding_share_of_variance": round(float(between / (within + between)), 3)}
    olig = R < THRESHOLD
    summ = {"cases": int(len(ids)), "draws": M, "aki_naive": int(akiN.sum()),
            "aki_share_naive": round(float(akiN.mean()), 4),
            "aki_cases_changing_across_draws": int(((akiD.min(1) == 0) & (akiD.max(1) == 1)).sum()),
            "oliguric_naive": int((nr < THRESHOLD).sum()),
            "oliguria_changing_across_draws": int(((olig.min(1) == 0) & (olig.max(1) == 1)).sum()),
            "fits": res, "threshold_sweep": sweep, "subgroups_at_0.5": sub}
    pd.DataFrame({"case": ids, "naive_rate": nr, "rate_median": np.median(R, 1), "p_oliguric": olig.mean(1),
                  "preop_cr": npre, "postop_cr_max": postcr, "aki_naive": akiN, "p_aki": akiD.mean(1)}
                 ).to_csv(a.out / "cases.csv", index=False)
    (a.out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    np.savez_compressed(a.out / "draw_fits.npz", **{k: np.array(v) for k, v in fits.items()},
                        **{"naive_" + k: np.array(v) for k, v in naive.items()})
    figure(fits, naive, summ, a.out)


def figure(fits: dict, naive: dict, summ: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    titles = {"A_logit_rate": ("A. AKI per 1 mL/kg/h (odds ratio)", True),
              "B_logit_oliguria": ("B. AKI if oliguric, < 0.5 mL/kg/h (odds ratio)", True),
              "C_linear_logratio": ("C. log creatinine ratio per 1 mL/kg/h", False)}
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.1))
    for axi, (k, (title, expo)) in zip(ax, titles.items()):
        e, s = np.array(fits[k]).T
        f = np.exp if expo else (lambda z: z)
        rb, (ne, nvar) = summ["fits"][k]["distribution"], naive[k]
        axi.hist(f(e), bins=30, color="#3f51b5", alpha=0.35, label="one fit per draw")
        y0 = axi.get_ylim()[1]
        axi.errorbar([f(rb["estimate"])], [y0 * 0.85], xerr=[[f(rb["estimate"]) - f(rb["lo"])], [f(rb["hi"]) - f(rb["estimate"])]],
                     fmt="o", color="#3f51b5", capsize=3, label="combined (Rubin), 95% CI")
        nlo, nhi = ne - 1.96 * np.sqrt(nvar), ne + 1.96 * np.sqrt(nvar)
        axi.errorbar([f(ne)], [y0 * 0.65], xerr=[[f(ne) - f(nlo)], [f(nhi) - f(ne)]], fmt="s", color="0.25",
                     capsize=3, label="recorded numbers, 95% CI")
        axi.set_title(title, fontsize=9, loc="left")
        axi.set_yticks([])
        axi.spines[["top", "right", "left"]].set_visible(False)
    ax[0].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0, 0.6))
    fig.suptitle(f"{summ['cases']:,} cases, {summ['aki_naive']} with AKI by creatinine within 48 h; "
                 f"{summ['draws']} draws from the true-value distributions", fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "regression_rounding.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
