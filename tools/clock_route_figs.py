# SPDX-License-Identifier: MIT
"""Figures for two clock routes: the SNUADC rate through the Solar8000 numerics, and the Primus step at each AWP pause.

    python tools/clock_route_figs.py --work W --out DIR

Aggregates only. Inputs: W/solar_snuadc_clock/agg/counts_combined.csv (tools/solar_snuadc_clock.py agg),
W/solar_snuadc_clock/reset_rates.csv (the reset-interval rates per recording, from clock_network/edges_v3.csv),
W/results/sync_full/snuadc_mode.json (recording lengths) and
W/primus_pause_steps/agg/steps.jsonl with summary.json (tools/primus_pause_steps.py agg).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, GREY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#9b9a94", "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def solar(W: Path, out: Path):
    """Median Solar8000 rate by recording length: the estimate shrinks toward 0 in short recordings."""
    c = pd.read_csv(W / "solar_snuadc_clock/agg/counts_combined.csv")
    r = pd.read_csv(W / "solar_snuadc_clock/reset_rates.csv")
    modes = json.loads((W / "results/sync_full/snuadc_mode.json").read_text())
    c["hours"] = c["case"].map(lambda k: (modes.get(str(k)) or {}).get("dur_h"))
    c["resets"] = c["case"].isin(r["case"])
    edges = [1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8]
    mids = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])]
    groups = [("Step mode, two or more resets", (c["mode"] == "step") & c["resets"], BLUE),
              ("Step mode, no whole reset interval", (c["mode"] == "step") & ~c["resets"], ORANGE),
              ("No correction seen", c["mode"] == "none", GREY)]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    for name, m, col in groups:
        g = c[m]
        ys, xs, ns = [], [], []
        for a, b, x in zip(edges[:-1], edges[1:], mids):
            v = g.loc[(g["hours"] >= a) & (g["hours"] < b), "ppm"]
            if len(v) >= 15:
                xs.append(x); ys.append(v.median()); ns.append(len(v))
        ax.plot(xs, ys, "o-", color=col, lw=1.6, ms=4, label=f"{name} ({m.sum():,})")
    ax.axhline(-63, color=MUTED, ls="--", lw=1)
    ax.text(7.9, -60, "step-mode rate from the resets (−63)", fontsize=7.5, color=MUTED, ha="right", va="bottom")
    ax.axhline(0, color=GRID, lw=1)
    ax.set_xlabel("recording length (h)", fontsize=8.5, color=MUTED)
    ax.set_ylabel("Solar8000 route, median (ppm)", fontsize=8.5, color=MUTED)
    ax.set_ylim(-100, 20)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    style(ax)
    ax.grid(axis="y", color=GRID, lw=0.8)
    fig.tight_layout()
    fig.savefig(out / "solar-snuadc-rate.png", dpi=150, facecolor="white")


def pauses(W: Path, out: Path):
    s = [json.loads(x) for x in open(W / "primus_pause_steps/agg/steps.jsonl")]
    summ = json.loads((W / "primus_pause_steps/agg/summary.json").read_text())
    ctrl = summ["control_s_p5_10_25_50_75_90_95"]
    co2 = np.array([x["step_s"] for x in s if x["co2_gap"]]) * 1000
    no = np.array([x["step_s"] for x in s if not x["co2_gap"]]) * 1000
    bins = np.linspace(-2000, 2000, 81)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.axvspan(ctrl[1] * 1000, ctrl[5] * 1000, color=GRID, alpha=0.8, lw=0)
    ax.text(ctrl[5] * 1000 + 20, 0.93, "controls, p10–p90\n(no pause)", transform=ax.get_xaxis_transform(),
            fontsize=7.5, color=MUTED, va="top")
    for v, col, lab in ((no, BLUE, f"CO2 carries on ({len(no):,} pauses)"), (co2, ORANGE, f"CO2 stops too ({len(co2):,})")):
        ax.hist(np.clip(v, -1999, 1999), bins=bins, histtype="step", lw=1.8, color=col, label=lab,
                weights=np.full(v.size, 100 / v.size))
    ax.set_xlabel("Primus stamp step at the AWP pause, against the SNUADC (ms)", fontsize=8.5, color=MUTED)
    ax.set_ylabel("% of pauses", fontsize=8.5, color=MUTED)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "primus-pause-steps.png", dpi=150, facecolor="white")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    solar(Path(a.work), out)
    pauses(Path(a.work), out)


if __name__ == "__main__":
    main()
