# SPDX-License-Identifier: MIT
"""Redraw the clock re-synchronisation figures for the drift document, from the per-case rows that
``syncanalyze.py --stage collect`` cached (this machine only) and its per-case drift table.

    python tools/sync_figures.py --rows DIR/full_rows --drift full.drift.csv --out OUTDIR

Writes, without the empty panels of the first drafts (CardioQ and Primus have almost no usable drift):

    sync_correction_jumps.png   jump at every device-level correction candidate (SNUADC, BIS, CardioQ)
    sync_drift_within_case.png  each correction's drift minus its case's drift (SNUADC, BIS)
    sync_drift_pairs.png        SNUADC against BIS drift in the same case
    sync_snuadc_drift_zoom.png  SNUADC drift per case, -90 to +30 ppm, with its two correction modes
    sync_join_tests.png         continuity across the join for corrections, other boundaries and controls
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PREFERRED = {"SNUADC": "SNUADC/ECG_II", "BIS": "BIS/EEG1_WAV"}
BEAT = ["SNUADC/ECG_II", "SNUADC/ART", "SNUADC/PLETH"]
STEP = ["BIS/EEG1_WAV", "Primus/CO2"]
C = {"SNUADC": "#2a6fdb", "BIS": "#e8743b", "CardioQ": "#7b5ea7", "ctl": "#6b6b6b", "ev": "#2a6fdb", "oth": "#e8743b"}
NOTE = "ppm = jump / time since the previous correction; negative = device clock fast against the recording PC"


def load(rows_dir: Path):
    rows, controls, seen = [], [], set()
    for f in sorted(rows_dir.glob("rows_*.jsonl")):
        for line in f.open():
            e = json.loads(line)
            if e["case"] in seen:
                continue
            seen.add(e["case"])
            rows += e["rows"]
            controls += e["controls"]
    pref = {}
    for r in rows:
        key = (r["case"], r["device"])
        best = PREFERRED.get(r["device"])
        cur = pref.get(key)
        if cur is None or (r["track"] == best and cur != best) or (cur != best and r["track"] < cur):
            pref[key] = r["track"]
    dev_rows = [r for r in rows if pref[(r["case"], r["device"])] == r["track"]]
    within = defaultdict(list)
    by = defaultdict(list)
    for r in dev_rows:
        if r["drift_ok"]:
            by[(r["case"], r["device"], r["type"])].append(r)
    for (_, _, typ), v in by.items():
        ppm = np.array([x["ppm"] for x in v])
        w = np.array([x["span_s"] for x in v])
        o = np.argsort(ppm)
        med = float(ppm[o][np.searchsorted(np.cumsum(w[o]) / w.sum(), 0.5)])
        if ppm.size >= 3:
            within[typ].extend((ppm - med).tolist())
    return rows, controls, dev_rows, within, len(seen)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9")


def fig_jumps(dev_rows, n_cases, out):
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.2))
    for ax, t, rng, step in zip(axes, ("SNUADC", "BIS", "CardioQ"), ((-0.7, 0.3), (-0.2, 0.6), (-2, 2)), (0.005, 0.005, 0.02),
                                strict=True):
        v = np.array([r["jump_s"] for r in dev_rows if r["type"] == t and r["event"]])
        inr = (v >= rng[0]) & (v <= rng[1])
        ax.hist(v[inr], bins=np.arange(rng[0], rng[1] + step, step), color=C[t])
        extra = f"; {int((~inr).sum()):,} outside the axis" if (~inr).any() else ""
        ax.set_title(f"{t}: jump at each device-level correction (n = {v.size:,}{extra})", loc="left", fontsize=10)
        ax.set_xlim(*rng)
        style(ax)
    axes[0].annotate("step mode: the recorder\nmoves the stamps back ~0.5 s", xy=(-0.5, 0), xytext=(-0.68, axes[0].get_ylim()[1] * 0.55),
                     fontsize=8, color="0.3")
    axes[0].annotate("small mode: corrections\nof a few ms", xy=(0.0, 0), xytext=(0.08, axes[0].get_ylim()[1] * 0.55), fontsize=8,
                     color="0.3")
    axes[-1].set_xlabel("jump at the boundary (s); positive = stamps move later")
    p = sum(1 for r in dev_rows if r["type"] == "Primus" and r["event"])
    fig.suptitle(f"What a clock correction looks like ({n_cases:,} cases; Primus: only {p} candidates, not shown)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "sync_correction_jumps.png", dpi=130)
    plt.close(fig)


def fig_within(within, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    for ax, t in zip(axes, ("SNUADC", "BIS"), strict=True):
        w = np.array(within.get(t, []))
        inside = np.mean(np.abs(w) <= 1) * 100 if w.size else 0
        ax.hist(np.clip(w, -5, 5), bins=np.arange(-5, 5.05, 0.1), color=C[t])
        ax.set_title(f"{t}: {w.size} corrections in cases with ≥ 3; {inside:.0f}% within ±1 ppm", loc="left", fontsize=10)
        ax.set_xlabel("correction's drift minus its case's drift (ppm, clipped at ±5)")
        style(ax)
    fig.suptitle("Drift is stable within a case", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "sync_drift_within_case.png", dpi=130)
    plt.close(fig)


def fig_pairs(drift, out):
    s = {r["case"]: float(r["drift_ppm"]) for r in drift if r["device_type"] == "SNUADC"}
    b = {r["case"]: float(r["drift_ppm"]) for r in drift if r["device_type"] == "BIS"}
    cs = sorted(set(s) & set(b))
    x = np.array([s[c] for c in cs])
    y = np.array([b[c] for c in cs])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    step = x < -20
    ax.scatter(x[step], y[step], s=10, alpha=0.6, color=C["SNUADC"], label=f"SNUADC in step mode (n = {step.sum()})")
    ax.scatter(x[~step], y[~step], s=10, alpha=0.6, color=C["BIS"], label=f"SNUADC in small mode (n = {(~step).sum()})")
    ax.set_xlim(-100, 30)
    ax.set_ylim(0, 75)
    ax.set_xlabel("SNUADC drift in the case (ppm)")
    ax.set_ylabel("BIS drift in the same case (ppm)")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    style(ax)
    ax.set_title(f"SNUADC and BIS drift in the same case ({len(cs)} cases)",
                 loc="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "sync_drift_pairs.png", dpi=130)
    plt.close(fig)


def fig_zoom(drift, out):
    v = np.array([float(r["drift_ppm"]) for r in drift if r["device_type"] == "SNUADC"])
    fig, ax = plt.subplots(figsize=(10, 3.8))
    shown = (v >= -90) & (v <= 30)
    ax.hist(v[shown], bins=np.arange(-90, 30.25, 0.25), color=C["SNUADC"])
    ax.axvspan(-90, -40, color="0.95", zorder=0)
    ax.text(-88, ax.get_ylim()[1] * 0.9, "step mode: clusters at about −81, −68 and −60 ppm", fontsize=8, color="0.3")
    ax.text(2, ax.get_ylim()[1] * 0.9, "small mode: 0 to +20 ppm", fontsize=8, color="0.3")
    ax.set_xlabel(NOTE, fontsize=8)
    ax.set_ylabel("cases")
    style(ax)
    ax.set_title(f"SNUADC drift per case, 0.25 ppm bins ({shown.sum()} of {v.size} cases between −90 and +30 ppm)",
                 loc="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "sync_snuadc_drift_zoom.png", dpi=130)
    plt.close(fig)


def fig_join(rows, controls, out):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    panels = [*BEAT, *STEP]
    for ax, tr in zip(axes.flat, panels, strict=False):
        if tr in BEAT:
            key, bins, lab = "z_butt", np.linspace(-15, 40, 111), "beat interval across the join, sides butted (robust z)"
        else:
            key, bins, lab = "step_pct", np.linspace(0, 1, 51), "step across the join (percentile of steps inside)"
        series = (([c.get(key) for c in controls if c["track"] == tr], "controls (no boundary)", C["ctl"]),
                  ([r.get(key) for r in rows if r["track"] == tr and r["event"]], "correction candidates", C["ev"]),
                  ([r.get(key) for r in rows if r["track"] == tr and not r["event"] and abs(r["jump_s"]) <= 10],
                   "other boundaries ≤ 10 s", C["oth"]))
        for v, name, col in series:
            v = [x for x in v if x is not None]
            if v:
                ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins, histtype="step", density=True, color=col, lw=1.3,
                        label=f"{name} (n = {len(v):,})")
        ax.set_title(tr, loc="left", fontsize=10)
        ax.set_xlabel(lab, fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        style(ax)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0, 0.7, "Beat panels: with the two sides butted together,\nz near 0 means nothing is missing at the join.\n"
                       "Values beyond the axis are piled at its ends.\nStep panels: a percentile near 1 is a jump in\nthe signal itself.",
                       fontsize=9, color="0.25", va="top")
    fig.suptitle("Continuity across the join, by kind of boundary", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "sync_join_tests.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--drift", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    drift = list(csv.DictReader(open(a.drift)))
    fig_pairs(drift, out)
    fig_zoom(drift, out)
    rows, controls, dev_rows, within, n = load(Path(a.rows))
    fig_jumps(dev_rows, n, out)
    fig_within(within, out)
    fig_join(rows, controls, out)
    print("ok", n)


if __name__ == "__main__":
    main()
