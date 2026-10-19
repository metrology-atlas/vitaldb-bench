# SPDX-License-Identifier: MIT
"""Worked example: oliguria labels for a prediction model, when the label comes from rounded numbers.

    python examples/oliguria_labels.py --resolution <vitaldb-resolution checkout> --root R --out OUT [--draws 200]

A prediction model needs one label per case: here, intraoperative oliguria (urine output below 0.5 mL/kg/h over
the anaesthetic). The label is computed from three rounded numbers (examples/urine_output_rate.py), so for some
cases the recorded numbers fall on one side of the threshold while the true rate may lie on the other. Per case,
the vitaldb-resolution library gives P(oliguric) from draws of the true values.

1. Labels: the recorded label; P(oliguric); the expected share of recorded labels that are wrong.
2. The ceiling: score each case by its true rate (one draw) and evaluate against the recorded labels. Even this
   perfect predictor of the true rate falls short of AUC 1 against labels from rounded numbers.
3. A simple model (logistic regression on preoperative and case-type variables, 5-fold cross-validated) trained
   on the recorded labels (hard) or on P(oliguric) (soft: each case twice, weighted P and 1 - P), and evaluated
   against the recorded labels and against labels drawn from the distributions.

Writes OUT/summary.json (aggregates only), OUT/cases.csv (per case; stays local) and OUT/oliguria_labels.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

THRESHOLD = 0.5
NUM = ["age", "height", "bmi", "asa", "preop_hb", "preop_alb", "preop_bun", "preop_cr", "preop_na", "preop_k",
       "preop_gluc", "preop_htn", "preop_dm", "emop"]
CAT = ["sex", "department", "approach", "ane_type", "optype", "position"]


def model():
    pre = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM),
                             ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                                   OneHotEncoder(handle_unknown="ignore", min_frequency=20)), CAT)])
    return make_pipeline(pre, LogisticRegression(C=0.3, max_iter=2000))


def oof(X, y_hard, p_soft, soft: bool, seed=0) -> np.ndarray:
    """Out-of-fold predicted probabilities, 5 folds stratified on the recorded label."""
    out = np.zeros(len(X))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y_hard):
        m = model()
        if soft:
            Xt = pd.concat([X.iloc[tr], X.iloc[tr]])
            yt = np.r_[np.ones(tr.size), np.zeros(tr.size)]
            w = np.r_[p_soft[tr], 1 - p_soft[tr]]
            m.fit(Xt, yt, logisticregression__sample_weight=w)
        else:
            m.fit(X.iloc[tr], y_hard[tr])
        out[te] = m.predict_proba(X.iloc[te])[:, 1]
    return out


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
    use = clin[(clin.intraop_uo > 0) & clin.weight.notna() & clin.anestart.notna() & clin.aneend.notna()]
    rng = np.random.default_rng(0)
    M = a.draws
    R, keep, naive = [], [], []
    for r in use.to_dict("records"):
        d = {v: reg.case_distribution(r, v) for v in ("intraop_uo", "weight", "ane_duration")}
        if any(x.status != "ok" or not x.components for x in d.values()):
            continue
        w, t = d["weight"].sample(M, rng), d["ane_duration"].sample(M, rng) / 60.0
        if np.any(w <= 0) or np.any(t <= 0):
            continue
        R.append(d["intraop_uo"].sample(M, rng) / w / t)
        keep.append(r)
        naive.append(r["intraop_uo"] / r["weight"] / ((r["aneend"] - r["anestart"]) / 3600.0))
    R, naive = np.array(R), np.array(naive)
    df = pd.DataFrame(keep)
    df["age"] = df["age"].replace(">89", "90")           # the release caps age as ">89"
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    X = df[NUM + CAT]
    y = (naive < THRESHOLD).astype(int)
    D = (R < THRESHOLD)                                   # cases x draws: the label under each draw
    p = D.mean(1)
    wrong = np.where(y == 1, 1 - p, p)                    # P(recorded label is wrong)
    uncertain = (p > 0.05) & (p < 0.95)
    round_small = (df.intraop_uo % 50 == 0).to_numpy() & (df.intraop_uo <= 200).to_numpy()

    # 2. the ceiling: the true rate as the score, against recorded labels and against the truth's own labels
    ceil_rec = np.array([roc_auc_score(y, -R[:, m]) for m in range(M)])
    acc_rec = np.array([np.mean((R[:, m] < THRESHOLD) == y) for m in range(M)])
    # 2b. models of every quality: the true rate plus noise on the log scale, scored against true and recorded labels
    quality = []
    rng3 = np.random.default_rng(2)
    for sd in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0):
        at, ar, ct, cr_ = [], [], [], []
        for m in range(0, M, max(M // 50, 1)):
            z = np.log(R[:, m]) + sd * rng3.standard_normal(len(y))
            at.append(roc_auc_score(D[:, m], -z)); ar.append(roc_auc_score(y, -z))
            ct.append(np.mean((z < np.log(THRESHOLD)) == D[:, m])); cr_.append(np.mean((z < np.log(THRESHOLD)) == y))
        quality.append({"noise_sd_log": sd, "auc_true": round(float(np.mean(at)), 4), "auc_recorded": round(float(np.mean(ar)), 4),
                        "accuracy_true": round(float(np.mean(ct)), 4), "accuracy_recorded": round(float(np.mean(cr_)), 4)})
    # 3. the model
    s_hard, s_soft = oof(X, y, p, soft=False), oof(X, y, p, soft=True)
    res = {}
    for name, s in (("hard", s_hard), ("soft", s_soft)):
        auc_draws = np.array([roc_auc_score(D[:, m], s) for m in range(M)])
        res[name] = {"auc_vs_recorded": round(float(roc_auc_score(y, s)), 4),
                     "auc_vs_true_labels_mean": round(float(auc_draws.mean()), 4),
                     "auc_vs_true_labels_p2.5_p97.5": [round(float(np.percentile(auc_draws, q)), 4) for q in (2.5, 97.5)],
                     "auc_vs_recorded_certain_cases_only": round(float(roc_auc_score(y[~uncertain], s[~uncertain])), 4)}
    # bootstrap SE of the AUC from sampling cases, to compare with the spread from the labels
    bs = []
    rng2 = np.random.default_rng(1)
    for _ in range(200):
        i = rng2.integers(0, len(y), len(y))
        bs.append(roc_auc_score(y[i], s_hard[i]))
    summ = {"cases": int(len(y)), "draws": M, "threshold": THRESHOLD,
            "oliguric_recorded": int(y.sum()), "oliguric_expected": round(float(p.sum()), 1),
            "uncertain_labels": int(uncertain.sum()), "uncertain_share": round(float(uncertain.mean()), 4),
            "expected_wrong_labels": round(float(wrong.sum()), 1), "expected_wrong_share": round(float(wrong.mean()), 4),
            "round_small_volume_cases": int(round_small.sum()),
            "uncertain_share_round_small": round(float(uncertain[round_small].mean()), 4),
            "uncertain_share_other": round(float(uncertain[~round_small].mean()), 4),
            "ceiling": {"auc_true_rate_vs_recorded_labels_mean": round(float(ceil_rec.mean()), 4),
                        "accuracy_true_label_vs_recorded_mean": round(float(acc_rec.mean()), 4)},
            "model": res, "quality_sweep": quality,
            "model_auc_bootstrap_sd_cases": round(float(np.std(bs)), 4),
            "label_draw_sd_of_auc_hard": round(float(np.std([roc_auc_score(D[:, m], s_hard) for m in range(M)])), 4)}
    pd.DataFrame({"case": df.caseid, "naive_rate": naive, "label": y, "p_oliguric": p, "uncertain": uncertain,
                  "score_hard": s_hard, "score_soft": s_soft}).to_csv(a.out / "cases.csv", index=False)
    (a.out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    figure(naive, p, y, round_small, summ, a.out)


def figure(naive, p, y, round_small, summ, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3), gridspec_kw={"width_ratios": [1.2, 1]})
    bins = np.linspace(0, 1, 21)
    for lab, col, name in ((1, "#e65100", "recorded: oliguric"), (0, "#3f51b5", "recorded: not oliguric")):
        ax[0].hist(p[(y == lab) & (p > 0) & (p < 1)], bins=bins, color=col, alpha=0.6, label=name)
    ax[0].set_xlabel("P(oliguric) from the true-value distributions (cases with 0 < P < 1)")
    ax[0].set_ylabel("cases")
    ax[0].legend(fontsize=7, frameon=False)
    ax[0].set_title(f"A. {summ['uncertain_labels']:,} of {summ['cases']:,} labels uncertain "
                    f"({summ['uncertain_share']:.0%}); {summ['expected_wrong_labels']:.0f} expected wrong", fontsize=9, loc="left")
    q = summ["quality_sweep"]
    et = 100 - np.array([x["accuracy_true"] for x in q]) * 100
    er = 100 - np.array([x["accuracy_recorded"] for x in q]) * 100
    k = et <= 20
    ax[1].plot(et[k], er[k], "o-", color="#3f51b5", ms=4, label="the true rate with noise, as a model")
    ax[1].plot([0, 20], [0, 20], color="0.6", lw=0.8, ls="--", label="if the labels had no rounding")
    ax[1].set_xlim(-0.5, 20); ax[1].set_ylim(-0.5, 20)
    ax[1].set_xlabel("error rate against the true labels (%)")
    ax[1].set_ylabel("error rate against recorded labels (%)")
    ax[1].legend(fontsize=7, frameon=False, loc="upper left")
    ax[1].annotate(f"a perfect model: {er[0]:.1f}% 'wrong'", (et[0], er[0]), xytext=(12, -4),
                   textcoords="offset points", fontsize=7.5)
    ax[1].set_title("B. Recorded labels set a floor under the measured error", fontsize=9, loc="left")
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "oliguria_labels.png", dpi=140)
    plt.close(fig)
    return
    rows = [("perfect predictor\nof the true rate", summ["ceiling"]["auc_true_rate_vs_recorded_labels_mean"], None),
            ("model, hard labels", summ["model"]["hard"]["auc_vs_recorded"], summ["model"]["hard"]["auc_vs_true_labels_mean"]),
            ("model, soft labels", summ["model"]["soft"]["auc_vs_recorded"], summ["model"]["soft"]["auc_vs_true_labels_mean"])]
    yy = np.arange(len(rows))[::-1]
    for yi, (name, rec, tru) in zip(yy, rows):
        ax[1].plot(rec, yi, "s", color="0.25")
        if tru is not None:
            ax[1].plot(tru, yi, "o", color="#3f51b5")
        ax[1].text(0.502, yi + 0.18, name, fontsize=7.5, va="bottom")
    ax[1].plot([], [], "s", color="0.25", label="scored against recorded labels")
    ax[1].plot([], [], "o", color="#3f51b5", label="scored against true labels (mean of draws)")
    ax[1].set_xlim(0.5, 1.0)
    ax[1].set_ylim(-0.6, len(rows) - 0.2)
    ax[1].set_yticks([])
    ax[1].set_xlabel("AUC for oliguria")
    ax[1].legend(fontsize=7, frameon=False, loc="lower right")
    ax[1].set_title("B. What the labels allow a model to score", fontsize=9, loc="left")
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    ax[1].spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "oliguria_labels.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
