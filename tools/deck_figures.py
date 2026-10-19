# SPDX-License-Identifier: MIT
"""Slide-sized figures of the sawtooth corrections, large type, one message each.

    python tools/deck_figures.py --root R --sa sa.jsonl --out DIR

sa.jsonl: tools/sa_corrected.py with --variants stamps,model3 for the cases drawn (2580, 4644). Draws
  DIR/deck_clock_sawtooth.png   the EEG-minus-ECG burst offset on the recorded stamps and on the stamp model;
  DIR/deck_ppg_sawtooth.png     ECG-to-PLETH PAT, as stored and with the PLETH sawtooth removed (6241);
  DIR/deck_pat_correction.png   the PAT example's two correction steps against the arterial PAT (5975).
Case ids appear because each figure is about that case.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GREY, BLUE, ORANGE, GREEN, INK = "#9AA0A6", "#3F51B5", "#E65100", "#2E7D32", "#1B2333"
plt.rcParams.update({"font.size": 20, "axes.titlesize": 22, "axes.labelsize": 20, "xtick.labelsize": 18,
                     "ytick.labelsize": 18, "legend.fontsize": 18, "axes.edgecolor": "#6A7180", "text.color": INK,
                     "axes.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False})


def good(rows):
    a = np.array(rows, float).reshape(-1, 5)
    return a[(a[:, 2] >= 0.3) & (a[:, 2] - a[:, 3] >= 0.15) & (a[:, 1] > -2500) & (a[:, 1] < 500)]


def clock_sawtooth(sa, out):
    rows = {json.loads(l)["case"]: json.loads(l) for l in Path(sa).read_text().splitlines() if l.strip()}
    cases = [c for c in (2580, 4644) if c in rows]
    fig, ax = plt.subplots(1, len(cases), figsize=(16, 5.4), sharey=True)
    for a, c in zip(np.atleast_1d(ax), cases):
        r = rows[c]
        s, m = good(r["stamps"]["burst"]["ecg~eeg1"]), good(r["model3"]["burst"]["ecg~eeg1"])
        a.plot(s[:, 0] / 3600, s[:, 1], "o", ms=6, color=GREY, label="recorded stamps")
        a.plot(m[:, 0] / 3600, m[:, 1], "o", ms=6, color=BLUE, label="stamp model")
        sd = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
        a.set_title(f"case {c}: spread {sd(s[:, 1]):.0f} → {sd(m[:, 1]):.0f} ms", loc="left")
        a.set_xlabel("hours")
    np.atleast_1d(ax)[0].set_ylabel("EEG minus ECG (ms)")
    h, l = np.atleast_1d(ax)[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=2, markerscale=1.6)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "deck_clock_sawtooth.png", dpi=120)
    plt.close(fig)


def pat_case(root, case, minutes):
    import pulse_arrival_time as P
    res, data = P.one_case(Path(root), case, minutes)
    if data is None:
        raise SystemExit(f"case {case}: {res}")
    return res, data


def ppg_sawtooth(root, out, case=6241, minutes=90):
    res, d = pat_case(root, case, minutes)
    fig, ax = plt.subplots(2, 1, figsize=(16, 6.4), sharex=True)
    for a, (k, col, name) in zip(ax, (("raw", GREY, "ECG to PLETH, as stored"), ("saw", BLUE, "PLETH sawtooth removed"))):
        _, t, y = d["win"][k]
        a.plot((t - d["t0"]) / 60, y * 1000, "o", ms=5, color=col)
        mid = np.median(y) * 1000
        a.set_ylim(mid - 55, mid + 55)
        for s in d["slips"]:
            a.axvline(s / 60, color="#C9CDD3", lw=1.2, zorder=0)
        a.set_ylabel("PAT (ms)")
        a.set_title(name, loc="left", color=col if k == "saw" else INK)
    ax[1].set_xlabel(f"minutes (case {case}; grey lines: the {len(d['slips'])} frame slips, each a 33 ms jump)")
    fig.tight_layout()
    fig.savefig(out / "deck_ppg_sawtooth.png", dpi=120)
    plt.close(fig)


def pat_correction(root, out):
    res, d = pat_case(root, 5975, 60)
    fig, ax = plt.subplots(figsize=(16, 5.6))
    lab = {"raw": ("ECG to finger, as stored", GREY), "saw": ("step 1: sawtooth removed", BLUE),
           "cor": ("step 2: device delay removed (385 ms)", ORANGE), "art": ("ECG to arterial line (reference)", GREEN)}
    for k in ("raw", "saw", "cor", "art"):
        if k in d["win"]:
            _, t, y = d["win"][k]
            ax.plot((t - d["t0"]) / 60, y * 1000, color=lab[k][1], lw=3, label=lab[k][0])
    for s in d["slips"]:
        ax.axvline(s / 60, color="#C9CDD3", lw=1, ls=":", zorder=0)
    ax.set_ylim(0, 760)
    ax.set_xlabel("minutes")
    ax.set_ylabel("pulse arrival time (ms)")
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.62))
    fig.tight_layout()
    fig.savefig(out / "deck_pat_correction.png", dpi=120)
    plt.close(fig)


def uo_rounding(root, resolution, out):
    """Recorded intraoperative urine output (heaped on 50 and 100 mL) and the true-volume distributions behind a
    recorded 100 mL and a recorded 160 mL (vitaldb-resolution)."""
    import pandas as pd
    sys.path.insert(0, str(resolution))
    import vitaldb_resolution as vr
    reg = vr.load()
    clin = pd.read_csv(Path(root) / "clinical_data.csv")
    uo = clin.intraop_uo[(clin.intraop_uo > 0) & (clin.intraop_uo <= 1000)]
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1.25, 1]})
    bins = np.arange(0, 1001, 10) - 5
    h, e = np.histogram(uo, bins=bins)
    x = e[:-1] + 5
    col = [ORANGE if v % 100 == 0 else (BLUE if v % 50 == 0 else GREY) for v in x]
    ax[0].bar(x, h, width=9, color=col)
    ax[0].set_xlim(0, 1000)
    ax[0].set_xlabel("recorded urine output (mL), 10 mL bins")
    ax[0].set_ylabel("cases")
    ax[0].set_title(f"{np.mean(uo % 100 == 0):.0%} on 100 mL, {np.mean((uo % 50 == 0) & (uo % 100 != 0)):.0%} on 50 mL", loc="left")
    rng = np.random.default_rng(0)
    base = clin[(clin.intraop_uo == 100) & clin.anestart.notna() & clin.aneend.notna()].iloc[0].to_dict()
    for rec, colr in ((100, ORANGE), (160, BLUE)):
        row = dict(base, intraop_uo=rec)
        d = reg.case_distribution(row, "intraop_uo")
        s = d.sample(40000, rng)
        lo, hi = np.percentile(s, [2.5, 97.5])
        ax[1].hist(s, bins=np.arange(40, 221, 2), density=True, color=colr, alpha=0.75,
                   label=f"recorded {rec} mL: 95% {lo:.0f}–{hi:.0f} mL")
    ax[1].set_xlabel("true urine output (mL)")
    ax[1].set_yticks([])
    ax[1].spines["left"].set_visible(False)
    ax[1].legend(frameon=False, loc="lower right", fontsize=17)
    ax[1].set_title("What a recorded value means", loc="left")
    fig.tight_layout()
    fig.savefig(out / "deck_uo_rounding.png", dpi=120)
    plt.close(fig)


def pat_bp_ruffolo(parts, out):
    """Beat-paired PAT-SBP Pearson r over an hour per recording (examples/pat_vs_bp.py scan output), naive against
    corrected, drawn as Ruffolo et al. 2025 (Physiol Meas 46:115006) appendix B, figure 10: PDF and CDF."""
    rows = {}
    for p_ in parts:
        for l in Path(p_).read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                rows[r["case"]] = r
    ok = [r for r in rows.values() if "beat" in r and r["sbp_sd"] >= 3]
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.6))
    bins = np.linspace(-1, 1, 41)
    for k, col, name in (("naive", ORANGE, "naive pairing"), ("saw", BLUE, "corrected")):
        x = np.array([r["beat"][k]["sbp"] for r in ok if r["beat"][k]["sbp"] is not None])
        lab = f"{name}: mean {x.mean():+.2f}, {np.mean(x < 0):.0%} below 0"
        ax[0].hist(x, bins=bins, histtype="step", lw=3, color=col, density=True, label=lab)
        xs = np.sort(x)
        ax[1].plot(xs, np.arange(1, xs.size + 1) / xs.size, lw=3, color=col, label=lab)
    for a in ax:
        a.axvline(0, color="#6A7180", lw=1)
        a.set_xlabel("Pearson r, PAT against SBP, one hour per recording")
        a.set_xlim(-1, 1)
    ax[0].set_yticks([])
    ax[0].spines["left"].set_visible(False)
    ax[0].set_title("Distribution", loc="left")
    ax[1].set_title("Cumulative", loc="left")
    ax[1].set_ylabel("share of recordings")
    ax[1].legend(frameon=False, loc="lower right", fontsize=17)
    fig.tight_layout()
    fig.savefig(out / "deck_pat_bp.png", dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--sa", required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    clock_sawtooth(a.sa, a.out)
    ppg_sawtooth(a.root, a.out)
    pat_correction(a.root, a.out)


if __name__ == "__main__":
    main()
