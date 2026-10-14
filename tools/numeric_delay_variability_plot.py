# SPDX-License-Identifier: MIT
"""Figure for tools/numeric_delay_variability.py. Reads the per-case scan files on this machine and writes a PNG.

    python tools/numeric_delay_variability_plot.py --scan part*.jsonl --hrwin hw*.jsonl --devices va/p?.jsonl --out fig.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TRACKS = ["HR", "ART_SBP", "ART_MBP", "ART_DBP", "CVP", "ETCO2", "PIP", "RR_CO2"]
LABEL = {"HR": "HR", "ART_SBP": "ART SBP", "ART_MBP": "ART MBP", "ART_DBP": "ART DBP", "CVP": "CVP",
         "ETCO2": "ETCO2", "PIP": "PIP", "RR_CO2": "RR (CO2)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", nargs="+")
    ap.add_argument("--hrwin", nargs="*")
    ap.add_argument("--devices", nargs="*")
    ap.add_argument("--out")
    a = ap.parse_args()
    case_d, block_dev = defaultdict(list), defaultdict(list)
    hr_case = {}
    for p in a.scan:
        for l in open(p):
            r = json.loads(l)
            for k, t in r.get("tracks", {}).items():
                if t["edge"]:
                    continue
                case_d[k].append(t["d"])
                b = np.array([x["d"] for x in t["blocks"] if not x["edge"]])
                if b.size >= 2:
                    block_dev[k].append(float(b.std(ddof=1) / np.sqrt(b.size)))
                if k == "HR":
                    hr_case[r["case"]] = t["d"]
    port = {}
    for p in a.devices or []:
        for l in open(p):
            r = json.loads(l)
            port[r["case"]] = (r.get("devices", {}).get("Solar8000") or [None, None])[1]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(len(TRACKS))
    for i, k in enumerate(TRACKS):
        d = np.array(case_d[k])
        rng = np.random.default_rng(i)
        bd = rng.normal(0.0, np.array(block_dev[k]))
        if not d.size:
            continue
        p10, p25, p50, p75, p90 = np.percentile(d, [10, 25, 50, 75, 90])
        ax[0].plot([i - 0.12, i - 0.12], [p10, p90], color="#2a6fdb", lw=1.5)
        ax[0].plot([i - 0.12, i - 0.12], [p25, p75], color="#2a6fdb", lw=6, solid_capstyle="butt")
        ax[0].plot(i - 0.12, p50, "o", color="white", mec="#2a6fdb", ms=6, zorder=3)
        w10, w25, w75, w90 = np.percentile(bd, [10, 25, 75, 90]) + p50
        ax[0].plot([i + 0.12, i + 0.12], [w10, w90], color="#e8743b", lw=1.5)
        ax[0].plot([i + 0.12, i + 0.12], [w25, w75], color="#e8743b", lw=6, solid_capstyle="butt")
    ax[0].plot([], [], color="#2a6fdb", lw=6, label="observed: delay per case (IQR, p10-p90)")
    ax[0].plot([], [], color="#e8743b", lw=6, label="expected if every case had the same delay\n(within-case block scatter / \u221ablocks)")
    ax[0].set_xticks(x, [LABEL[k] for k in TRACKS])
    ax[0].axhline(0, color="0.6", lw=0.8)
    ax[0].set_ylabel("delay from the window end (s)")
    ax[0].set_title("Case delays spread far more than fixed delays would", loc="left", fontsize=11)
    ax[0].legend(loc="upper left", fontsize=8, frameon=False)
    ax[0].grid(axis="y", color="0.9")

    groups = defaultdict(list)
    for c, d in hr_case.items():
        groups[port.get(c)].append(d)
    big = [k for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1])) if len(v) >= 40 and k]  # cases without a port left out
    bins = np.arange(0, 7.01, 0.25)
    frac = {k: float(np.mean(np.array(groups[k]) >= 3.1)) for k in big}
    cats = [("late (>80% of cases)", [k for k in big if frac[k] > 0.8], "#7b5ea7"),
            ("early (<20% late)", [k for k in big if frac[k] < 0.2], "#2fa37c"),
            ("mixed", [k for k in big if 0.2 <= frac[k] <= 0.8], "#b0b0b0")]
    bottom = np.zeros(bins.size - 1)
    for name, ks, col in cats:
        if not ks:
            continue
        h, _ = np.histogram(np.concatenate([groups[k] for k in ks]), bins=bins)
        ax[1].bar(bins[:-1], h, width=0.25, align="edge", bottom=bottom, color=col, alpha=0.85,
                  label=f"port {', '.join(ks)}: {name}")
        bottom += h
    ax[1].set_xlabel("HR delay from the end of a 6 s median window (s)")
    ax[1].set_ylabel("cases")
    ax[1].set_title("HR: two groups about one 2 s record apart", loc="left", fontsize=11)
    ax[1].set_ylim(0, bottom.max() * 1.45)
    ax[1].legend(fontsize=7.5, frameon=False, loc="upper left")
    ax[1].grid(axis="y", color="0.9")
    n = len(hr_case)
    fig.suptitle(f"Numeric delays with the algorithm held fixed ({n} cases with HR; 30-min blocks)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)


if __name__ == "__main__":
    main()
