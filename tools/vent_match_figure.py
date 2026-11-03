# SPDX-License-Identifier: MIT
"""Figure vent-breath-matching (metrology-atlas study ventilator-breath-matching): the Primus-to-SNUADC lag from
CVP after breath matching, per case. Replaces tools/adhoc/vent_match_plot.py (same rules).

    python tools/vent_sync_pilot.py --root $R --cases 4644,4817,3148,750,2443,1403,3706,5780,2838,1936,3146,3549,2172,629 \
        --units $W/results/sync_full/units.csv --step-rates $W/results/pat/snuadc_step_rates.json \
        --out $W/vent_breath_matching/vent14.jsonl
    python tools/vent_match_figure.py --scan $W/vent_breath_matching/vent14.jsonl \
        --cases 4644,4817,3148,750,2443,1403,3706,5780,2838,1936,3146,3549,2172,629 --out $W/vent_breath_matching

Per case, the good CVP windows of vent_sync_pilot (split-half agreement >= 0.8, precision < 100 ms). Breath
matching: a block is confident when its best breath offset scores >= 0.3 and beats the next by >= 0.15; each Primus
stretch takes one offset, the confident blocks' vote weighted by their margin; windows within 15 min of a confident
block with that offset are confirmed (filled) and shifted by it; the others are left as measured (open). Windows
whose offset was not zero are also drawn before the shift (red crosses). Red dashed lines: Primus stamp jumps
(stretch starts). Writes summary.json (per case: windows, confirmed, shifted, stretches) and vent-breath-matching.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def conf(b) -> bool:
    return b[3] >= 0.3 and b[3] - b[4] >= 0.15


def matched(r):
    g = np.array([x for x in r["rows"]["cvp"] if x[2] >= 0.8 and x[3] is not None and x[3] < 100], dtype=float)
    if not g.size:
        return None
    sst = np.array(r["stretch_starts_s"])
    seg = np.searchsorted(sst, g[:, 0])
    lag = g[:, 1].copy()
    ver = np.zeros(len(g), bool)
    fixed = np.zeros(len(g), bool)
    votes: dict = {}
    for b in r.get("match_cvp", []):
        if conf(b):
            votes.setdefault(b[1], {}).setdefault(b[2], 0.0)
            votes[b[1]][b[2]] += b[3] - b[4]
    kst = {s_: max(v, key=v.get) for s_, v in votes.items()}
    for b in r.get("match_cvp", []):
        if not conf(b) or b[2] != kst.get(b[1]):
            continue
        m = (seg == b[1]) & (np.abs(g[:, 0] - b[0]) <= 900)
        lag[m] += b[6]
        ver[m] = True
        fixed[m] |= b[2] != 0
    return g, lag, ver, fixed, sst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rs = {json.loads(x)["case"]: json.loads(x) for x in Path(a.scan).read_text().splitlines() if x.strip()}
    cases = [int(x) for x in a.cases.split(",") if x]
    S = {}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = (len(cases) + 1) // 2
    fig, ax = plt.subplots(rows, 2, figsize=(18, 2.6 * rows), squeeze=False)
    for axx, c in zip(ax.flat, cases):
        m = matched(rs[c])
        if m is None:
            axx.set_title(f"case {c}: no good CVP windows", fontsize=9)
            S[str(c)] = {"windows": 0}
            continue
        g, lag, ver, fixed, sst = m
        S[str(c)] = {"windows": int(len(g)), "confirmed": int(ver.sum()), "shifted": int(fixed.sum()),
                     "stretches": int(len(sst)), "lag_ms_median": round(float(np.median(lag)), 1)}
        axx.plot(g[~ver, 0] / 3600, lag[~ver], "o", mfc="none", color="C0", ms=4, label="breath identity unconfirmed")
        axx.plot(g[ver, 0] / 3600, lag[ver], "o", color="C0", ms=4, label="confirmed by breath matching")
        if fixed.any():
            axx.plot(g[fixed, 0] / 3600, g[fixed, 1], "x", color="C3", ms=5, label="before fix")
        for s in sst:
            axx.axvline(s / 3600, color="r", lw=0.6, ls="--")
        axx.set_title(f"case {c}", fontsize=9)
        axx.set_ylabel("Primus→SNUADC lag (ms)")
        axx.set_xlabel("hours")
    for axx in ax.flat[len(cases):]:
        axx.set_visible(False)
    handles: dict = {}
    for axx in ax.flat:
        for h, lab in zip(*axx.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    ax.flat[0].legend(handles.values(), handles.keys(), fontsize=7)
    fig.suptitle("Ventilator (Primus) lag from CVP after breath matching; red dashed = Primus stamp jumps", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "vent-breath-matching.png", dpi=65)
    tot = {k: sum(v.get(k, 0) for v in S.values()) for k in ("windows", "confirmed", "shifted")}
    (out / "summary.json").write_text(json.dumps({"total": tot, "cases": S}, indent=1) + "\n")
    print(json.dumps({"total": tot, "cases": S}))


if __name__ == "__main__":
    main()
