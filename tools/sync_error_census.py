# SPDX-License-Identifier: MIT
"""Timing error over the whole release, before and after synchronisation, for every link a shared signal can check.

    python tools/sync_error_census.py --work W --out FIG.png [--summary OUT.json]

Each link is checked by a signal that two clocks both record; the check never enters the correction it tests.
  BIS - SNUADC     electrocautery and muscle bursts in the ECG and the EEG (10-min windows; good windows only).
                   Before: the EEG-minus-ECG lag on the recorded stamps, i.e. the error of reading the two devices'
                   stamps as one clock. After: the lag on the stamp model, minus the constant of the recording's
                   SNUADC mode (taken over all recordings).
                   Inputs: W/burst_validation_stamps (tools/run_burst_validation.sh stamps ...) and
                   W/burst_validation_m3.
  Primus - SNUADC  the breath cycle in airway pressure and central venous pressure (10-min windows).
                   Before: the lag on both devices' recorded stamps, minus the recording's own median (the
                   physiological airway-to-CVP delay is unknown, so this before is a lower bound). After: each
                   window predicted from all the others (one slope and one offset per stretch between AWP pauses).
                   Inputs: W/primus_sync_raw (tools/primus_sync_accuracy.py scan --arms cvp_raw) and
                   W/primus_sync_accuracy (arm cvp).
  Primus - PC      the PIP numeric, stamped by the PC, against the airway-pressure breath it describes (30-min
                   blocks). Before: the block delay minus the recording's median. After: leave-one-out as above.
                   Input: W/primus_numeric_clock.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

GOOD = (0.3, 0.15)
MODE_CONST_S = {"step": -0.804, "small": -0.660, "none": -0.766, "mixed": -0.698}
BEFORE, AFTER, INK, MUTED, GRID = "#9b9a94", "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"


def parts(d):
    for p in sorted(glob.glob(str(Path(d) / "part_*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            if line.strip():
                yield json.loads(line)


def good_lags(win):
    return np.array([w[1] / 1000 for w in win if w[2] >= GOOD[0] and w[2] - w[3] >= GOOD[1] and -2500 < w[1] < 2500])


def bursts(W, modes):
    before, after, n = [], [], set()
    for r in parts(W / "burst_validation_stamps"):
        lag = good_lags((r.get("stamps") or {}).get("burst", {}).get("ecg~eeg1", []))
        if lag.size >= 5:
            before.append(lag); n.add(r["case"])
    for r in parts(W / "burst_validation_m3"):
        lag = good_lags((r.get("model3") or {}).get("burst", {}).get("ecg~eeg1", []))
        if lag.size >= 5:
            after.append(lag - MODE_CONST_S.get(modes.get(r["case"], "none"), -0.766)); n.add(r["case"])
    cat = lambda x: np.abs(np.concatenate(x)) if x else np.empty(0)
    return cat(before), cat(after), len(n)


def stretch_loo(t, y, pauses):
    st = np.searchsorted(np.asarray(pauses, float), t)
    u = np.unique(st)
    X = np.column_stack([t] + [(st == s).astype(float) for s in u])
    out = []
    for i in range(t.size):
        m = np.ones(t.size, bool); m[i] = False
        if (st[m] == st[i]).sum() < 2:
            continue
        cols = np.any(X[m] != 0, axis=0)
        b = np.linalg.lstsq(X[m][:, cols], y[m], rcond=None)[0]
        out.append(float(X[i, cols] @ b - y[i]))
    return out


def primus(W):
    before, after, n = [], [], set()
    for r in parts(W / "primus_sync_raw"):
        A = (r.get("arms") or {}).get("cvp_raw")
        if A and len(A["lag"]) >= 6:
            y = np.array(A["lag"]); before.append(np.abs(y - np.median(y))); n.add(r["case"])
    for r in parts(W / "primus_sync_accuracy"):
        A = (r.get("arms") or {}).get("cvp")
        if A and len(A["t"]) >= 6:
            after.append(np.abs(stretch_loo(np.array(A["t"]) / 3600, np.array(A["lag"]), np.array(r["pauses"]) / 3600)))
            n.add(r["case"])
    cat = lambda x: np.concatenate(x) if x else np.empty(0)
    return cat(before), cat(after), len(n)


def pip(W):
    before, after, n = [], [], 0
    for r in parts(W / "primus_numeric_clock"):
        B = np.array([b for b in r.get("blocks", []) if b[2] >= 0.5], float)
        if len(B) < 4:
            continue
        before.append(np.abs(B[:, 1] - np.median(B[:, 1])))
        after.append(np.abs(stretch_loo(B[:, 0] / 3600, B[:, 1], np.array(r.get("pauses") or []) / 3600)))
        n += 1
    cat = lambda x: np.concatenate(x) if x else np.empty(0)
    return cat(before), cat(after), n


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary")
    ap.add_argument("--linear", action="store_true", help="linear x axes, each panel to the 98th percentile of before")
    a = ap.parse_args()
    W = Path(a.work)
    modes = {int(k): v for k, v in json.loads((W / "results/sync_full/snuadc_mode_label.json").read_text()).items()}
    links = [("BIS – SNUADC", "ECG–EEG bursts", *bursts(W, modes), "before: the recorded stamps read as one clock"),
             ("Primus – SNUADC", "breath cycle, AWP–CVP", *primus(W), "before: recorded stamps, each recording's own offset removed"),
             ("Primus – PC", "PIP numerics, PC-stamped", *pip(W), "before: each recording's own offset removed")]
    links = [L for L in links if L[2].size and L[3].size]
    fig, axes = plt.subplots(1, len(links), figsize=(4.4 * len(links), 3.8), squeeze=False)
    bins = np.logspace(0, np.log10(5000), 46)
    summ = {}
    for ax, (name, sig, b, af, n, note) in zip(axes[0], links):
        b_ms, a_ms = np.clip(b * 1000, 1, 5000), np.clip(af * 1000, 1, 5000)
        if a.linear:
            xmax = float(np.percentile(np.abs(b) * 1000, 98))
            bins = np.linspace(0, xmax, 51)
            b_ms, a_ms = np.clip(np.abs(b) * 1000, 0, xmax), np.clip(np.abs(af) * 1000, 0, xmax)
        ax.hist(b_ms, bins=bins, color=BEFORE, alpha=0.85, label="as recorded", weights=np.full(b_ms.size, 100 / b_ms.size))
        ax.hist(a_ms, bins=bins, histtype="step", lw=2, color=AFTER, label="synchronised", weights=np.full(a_ms.size, 100 / a_ms.size))
        mb, ma = float(np.median(b) * 1000), float(np.median(af) * 1000)
        for m, c in ((mb, BEFORE), (ma, AFTER)):
            ax.axvline(m, color=c, lw=1, ls="--")
        if a.linear:
            ax.set_xlim(0, xmax)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f} ms"))
        else:
            ax.set_xscale("log")
            ax.set_xticks([1, 10, 100, 1000])
            ax.set_xticklabels(["≤1 ms", "10 ms", "100 ms", "1 s"])
        ax.set_title(f"{name}: median {mb:,.0f} ms → {ma:,.0f} ms\n{sig} · {n:,} recordings", fontsize=9, loc="left",
                     color=INK)
        ax.set_xlabel(f"timing error per window (|error|)\n{note}", fontsize=7.5, color=MUTED)
        style(ax)
        summ[name] = {"recordings": n, "windows_before": int(b.size), "windows_after": int(af.size),
                      "median_ms_before": round(mb, 1), "median_ms_after": round(ma, 1),
                      "p90_ms_before": round(float(np.percentile(b, 90) * 1000), 1),
                      "p90_ms_after": round(float(np.percentile(af, 90) * 1000), 1)}
    axes[0][0].set_ylabel("% of windows", fontsize=8, color=MUTED)
    axes[0][0].legend(fontsize=7.5, frameon=False, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150, facecolor="white")
    print(json.dumps(summ, indent=1, ensure_ascii=False))
    if a.summary:
        Path(a.summary).write_text(json.dumps(summ, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
