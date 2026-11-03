# SPDX-License-Identifier: MIT
"""Drift units: per-case device drift rates fall into units (boards or monitors), part of the spread is shared by
the two devices of a case. Numbers and figure for the metrology-atlas study drift-units.

    python tools/drift_units.py --drift $W/results/sync_full/full.drift.csv \
        --modes $W/results/sync_full/snuadc_mode_label.json --out $W/drift_units

Inputs: per-case drift rates from the clock-correction scan (syncanalyze.py's full.drift.csv: case, device type,
serial port, drift in ppm) and each case's SNUADC correction mode (snuadc_mode_label.json).

* **Cases.** SNUADC with a drift between -90 and -40 ppm (the step-mode range: the small-correction mode carries
  no drift of its own; the modes found are reported); BIS with |drift| < 80 ppm.
* **Units.** A Gaussian mixture with one shared (tied) width, fitted for 1-25 components (10 starts each); the BIC
  picks the number. Each case's unit is its most probable component, with that probability as its confidence.
* **Setups.** Cases with both devices, grouped by the pair of serial ports; per pair with at least 5 cases, the
  spread of each device's drift.
* **Shared term.** In each setup with at least 20 cases, a robust covariance (minimum covariance determinant) of
  the two devices' drifts gives the shared SD (sqrt of the covariance, where positive) and each device's own SD
  (sqrt of its variance less the covariance).
* **Case order.** Within each setup, the absolute drift difference between cases adjacent in case id against
  random pairs of the same setup (one-sided permutation test, 2,000 shuffles).

Writes OUT/units.csv (one row per case), OUT/summary.json and OUT/drift-units.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
from sklearn.mixture import GaussianMixture

SNU_RANGE = (-90.0, -40.0)
BIS_ABS = 80.0


def fit_units(x: np.ndarray, kmax: int = 25) -> dict:
    X = x[:, None]
    fits = []
    for k in range(1, kmax + 1):
        g = GaussianMixture(k, covariance_type="tied", n_init=10, random_state=0).fit(X)
        fits.append((g.bic(X), k, g))
    bic, k, g = min(fits, key=lambda t: t[0])
    order = np.argsort(g.means_[:, 0])
    rank = np.empty(k, int)
    rank[order] = np.arange(k)
    p = g.predict_proba(X)
    lab = rank[np.argmax(p, 1)]
    return {"k": k, "bic": {str(kk): round(float(b), 1) for b, kk, _ in fits}, "width_ppm": float(np.sqrt(g.covariances_[0, 0])),
            "means": [float(g.means_[i, 0]) for i in order], "weights": [float(g.weights_[i]) for i in order],
            "label": lab, "conf": p.max(1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drift", required=True)
    ap.add_argument("--modes", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(a.drift)
    mode = json.loads(Path(a.modes).read_text())
    s = d[d["device_type"] == "SNUADC"].copy()
    s["mode"] = s["case"].astype(str).map(mode)
    s = s[(s["drift_ppm"] > SNU_RANGE[0]) & (s["drift_ppm"] < SNU_RANGE[1])]
    b = d[(d["device_type"] == "BIS") & (d["drift_ppm"].abs() < BIS_ABS)].copy()
    S = {"inputs": {"drift": a.drift, "modes": a.modes}, "cases": {"SNUADC": int(len(s)), "BIS": int(len(b))},
         "snuadc_modes": {str(k): int(v) for k, v in s["mode"].value_counts().items()}}
    res = {}
    for name, t in (("SNUADC", s), ("BIS", b)):
        f = fit_units(t["drift_ppm"].to_numpy())
        t["unit"] = [f"U{i + 1}" for i in f["label"]]
        t["unit_mean"] = [f["means"][i] for i in f["label"]]
        t["conf"] = f["conf"]
        res[name] = t
        spread = t.groupby("unit")["drift_ppm"].std()
        S[name] = {"units": f["k"], "width_ppm": round(f["width_ppm"], 3), "means_ppm": [round(m, 1) for m in f["means"]],
                   "weights": [round(w, 3) for w in f["weights"]], "bic": f["bic"],
                   "confident_over_0_9": round(float((t["conf"] > 0.9).mean()), 3),
                   "sd_within_unit_ppm": [round(float(spread.min()), 2), round(float(spread.max()), 2)],
                   "bic_within_2_of_best": sorted(int(k) for k, v in f["bic"].items() if v - min(f["bic"].values()) <= 10)}
    both = res["SNUADC"][["case", "port", "drift_ppm", "unit", "unit_mean", "conf"]].merge(
        res["BIS"][["case", "port", "drift_ppm", "unit", "unit_mean", "conf"]], on="case", suffixes=("_snuadc", "_bis"))
    both["setup"] = both["port_snuadc"].astype(str) + " · " + both["port_bis"].astype(str)
    S["both"] = int(len(both))
    g = both.groupby("setup")
    setups = {k: v for k, v in g if len(v) >= 5}
    S["setups_5plus"] = len(setups)
    S["setup_spread_sd_ppm"] = {k: {"n": int(len(v)), "snuadc": round(float(v["drift_ppm_snuadc"].std()), 2),
                                     "bis": round(float(v["drift_ppm_bis"].std()), 2)} for k, v in setups.items()}
    shared = {}
    for k, v in setups.items():
        if len(v) < 20:
            continue
        r = v[["drift_ppm_snuadc", "drift_ppm_bis"]].to_numpy()
        C = MinCovDet(random_state=0).fit(r).covariance_
        c = C[0, 1]
        shared[k] = {"n": int(len(v)), "shared_sd": round(float(np.sqrt(c)), 2) if c > 0 else 0.0,
                     "snuadc_own_sd": round(float(np.sqrt(max(C[0, 0] - max(c, 0), 0))), 2),
                     "bis_own_sd": round(float(np.sqrt(max(C[1, 1] - max(c, 0), 0))), 2)}
    S["shared_term"] = shared
    rng = np.random.default_rng(1)
    adj, rnd = [], []
    for k, v in setups.items():
        for col in ("drift_ppm_snuadc", "drift_ppm_bis"):
            x = v.sort_values("case")[col].to_numpy()
            if x.size < 3:
                continue
            adj.append(np.abs(np.diff(x)))
            rnd.append(x)
    obs = float(np.mean(np.concatenate(adj)))
    null = []
    for _ in range(2000):
        null.append(np.mean(np.concatenate([np.abs(np.diff(rng.permutation(x))) for x in rnd])))
    S["case_order"] = {"adjacent_mean_abs_diff_ppm": round(obs, 3), "shuffled_mean": round(float(np.mean(null)), 3),
                       "p_adjacent_more_alike": round(float(np.mean(np.array(null) <= obs)), 4)}
    units = pd.concat([res["SNUADC"].assign(device="SNUADC"), res["BIS"].assign(device="BIS")])[
        ["case", "device", "port", "drift_ppm", "unit", "unit_mean", "conf"]]
    units.to_csv(out / "units.csv", index=False)
    (out / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    # figure (as the atlas's drift-units.png): each device's drift histogram with its fitted units, and SNUADC
    # against BIS drift per case coloured by the pair of ports
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(11, 10))
    for i, (name, lo, hi, title) in enumerate((("SNUADC", -90, -40, "SNUADC (step-correction mode)"), ("BIS", 0, 80, "BIS"))):
        t = res[name]
        bw = 0.25
        ax[i].hist(t["drift_ppm"], bins=np.arange(lo, hi + bw, bw), color="#9aa7b8")
        xs = np.linspace(lo, hi, 2000)
        w = S[name]["width_ppm"]
        for u_, (m, wt) in enumerate(zip(S[name]["means_ppm"], S[name]["weights"])):
            y = len(t) * bw * wt * np.exp(-0.5 * ((xs - m) / w) ** 2) / (w * np.sqrt(2 * np.pi))
            ax[i].plot(xs, y, color="#2b5ea7", lw=1.2)
            ax[i].text(m, y.max(), f"U{u_ + 1}", color="#2b5ea7", fontsize=7, ha="center", va="bottom")
        ax[i].set_xlim(lo, hi)
        ax[i].set_ylabel("cases")
        ax[i].set_title(f"{title}: drift per case (n={len(t)}) with {S[name]['units']} fitted units, "
                        f"shared width {w:.2f} ppm (1 sd)", fontsize=10)
    top = sorted(setups.items(), key=lambda kv: -len(kv[1]))[:8]
    named = {k for k, _ in top}
    rest = both[~both["setup"].isin(named)]
    for j, (k, v) in enumerate(top):
        sp_, bp_ = k.split(" · ")
        ax[2].scatter(v["drift_ppm_snuadc"], v["drift_ppm_bis"], s=10, color=plt.cm.tab10(j), label=f"SNUADC {sp_} / BIS {bp_} ({len(v)})")
    ax[2].scatter(rest["drift_ppm_snuadc"], rest["drift_ppm_bis"], s=8, color="0.7", label=f"other port pairs ({len(rest)})")
    ax[2].set_xlim(-90, -40)
    ax[2].set_ylim(0, 70)
    ax[2].set_xlabel("SNUADC drift, ppm")
    ax[2].set_ylabel("BIS drift, ppm")
    ax[2].set_title(f"Per case, SNUADC against BIS drift (n={len(both)}), coloured by the pair of ports the two devices used", fontsize=10)
    ax[2].legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "drift-units.png", dpi=110)
    print(json.dumps({k: v for k, v in S.items() if k not in ("setup_spread_sd_ppm",)}, indent=1)[:4000])


if __name__ == "__main__":
    main()
