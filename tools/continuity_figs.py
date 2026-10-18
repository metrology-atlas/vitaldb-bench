# SPDX-License-Identifier: MIT
"""Continuity figures from the collected tables (tools/signal_events_collect.py). Aggregates only.

    python tools/continuity_figs.py TABLES OUTDIR

survey_gaps_ccdf.png      N(gap > x) against x, log-log, per waveform
survey_segments_ccdf.png  N(segment > x): all segments, interior segments (a gap at both ends), and the
                          Kaplan-Meier estimate with segments that touch the recording's start or end as censored
survey_gap_vs_since.png   gap duration against time since the previous gap (the length of the segment before it)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

TRACKS = ["SNUADC__ECG_II", "SNUADC__ECG_V5", "SNUADC__PLETH", "SNUADC__ART", "SNUADC__CVP", "SNUADC__FEM",
          "Primus__CO2", "Primus__AWP", "BIS__EEG1_WAV", "CardioQ__FLOW"]


def ccdf(x):
    x = np.sort(np.asarray(x, float))
    return x, np.arange(x.size, 0, -1)          # N(>= x_i)


def km(d, event):
    """Kaplan-Meier survival S(t) at the sorted distinct durations; event False = censored."""
    o = np.argsort(d)
    d, e = np.asarray(d)[o], np.asarray(event)[o]
    t, idx = np.unique(d, return_index=True)
    n_at = d.size - idx
    ev = np.add.reduceat(e.astype(int), idx)
    s = np.cumprod(1 - ev / n_at)
    return t, s


def main() -> None:
    tab, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    G = pd.concat([pd.read_parquet(p) for p in sorted(tab.glob("gaps_*.parquet"))])
    S = pd.concat([pd.read_parquet(p) for p in sorted(tab.glob("segs_*.parquet"))])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def grid():
        fig, axs = plt.subplots(5, 2, figsize=(11, 13), sharex=True)
        return fig, axs

    marks = ((0.2, "0.2 s"), (1, "1 s"), (60, "1 min"), (3600, "1 h"))
    fig, axs = grid()
    for a_, tr in zip(axs.flat, TRACKS):
        g = G[G.track == tr]
        x, n = ccdf(g.d[g.d > 0])
        a_.step(x, n, where="post", color="#08519c", lw=1.2)
        a_.set_xscale("log"); a_.set_yscale("log")
        a_.set_title(f"{tr.replace('__', '/')}: {len(g):,} gaps in {g.case.nunique():,} cases", fontsize=8.5, loc="left")
        a_.set_ylabel("gaps longer than x", fontsize=8)
        for v, lab in marks:
            a_.axvline(v, color="#bbb", lw=.6, ls=":")
        a_.spines[["top", "right"]].set_visible(False)
        a_.tick_params(labelsize=7.5)
    for a_ in axs[-1]:
        a_.set_xlabel("gap duration x (s); dotted at 0.2 s, 1 s, 1 min, 1 h", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "survey_gaps_ccdf.png", dpi=140); plt.close(fig)

    fig, axs = grid()
    for a_, tr in zip(axs.flat, TRACKS):
        s = S[S.track == tr]
        s = s[s.d > 0]
        x, n = ccdf(s.d)
        a_.step(x, n, where="post", color="#bdbdbd", lw=1.2, label=f"all ({len(s):,})")
        inner = s[~s["first"] & ~s["last"]]
        if len(inner):
            x, n = ccdf(inner.d)
            a_.step(x, n, where="post", color="#08519c", lw=1.2, label=f"interior: gap at both ends ({len(inner):,})")
        ev = (~s["first"] & ~s["last"]).to_numpy()
        if ev.any():
            t, sv = km(s.d.to_numpy(), ev)
            a_.step(t, sv * len(s), where="post", color="#e6550d", lw=1.2, label="Kaplan-Meier, edge segments censored (scaled to all)")
        a_.set_xscale("log"); a_.set_yscale("log")
        a_.set_title(f"{tr.replace('__', '/')}: {s.case.nunique():,} cases", fontsize=8.5, loc="left")
        a_.set_ylabel("segments longer than x", fontsize=8)
        for v, lab in marks:
            a_.axvline(v, color="#bbb", lw=.6, ls=":")
        a_.legend(frameon=False, fontsize=6.5, loc="lower left")
        a_.spines[["top", "right"]].set_visible(False)
        a_.tick_params(labelsize=7.5)
    for a_ in axs[-1]:
        a_.set_xlabel("continuous segment duration x (s)", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "survey_segments_ccdf.png", dpi=140); plt.close(fig)

    # gap duration against the time since the previous gap (= the segment just before the gap)
    fig, axs = grid()
    ex = np.logspace(-3, 5, 81)
    ey = np.logspace(-2, 5, 71)
    for a_, tr in zip(axs.flat, TRACKS):
        s = S[S.track == tr].sort_values(["case", "t"])
        g = G[G.track == tr].sort_values(["case", "t"])
        # each gap starts where a segment ends: match the gap to the segment that ends at its start
        s = s.assign(end=s.t + s.d)
        m = pd.merge_asof(g.sort_values("t"), s.sort_values("end")[["case", "end", "d", "first"]].rename(columns={"d": "before"}),
                          left_on="t", right_on="end", by="case", direction="nearest", tolerance=0.01)
        m = m.dropna(subset=["before"])
        h, _, _ = np.histogram2d(m.d.clip(1e-3, 9e4), m.before.clip(1e-2, 9e4), bins=[ex, ey])
        im = a_.pcolormesh(ex, ey, np.log10(h.T + 0), cmap="Blues", shading="auto")
        a_.set_xscale("log"); a_.set_yscale("log")
        a_.set_title(f"{tr.replace('__', '/')}: {len(m):,} gaps", fontsize=8.5, loc="left")
        a_.set_ylabel("time since previous gap (s)", fontsize=8)
        a_.tick_params(labelsize=7.5)
        fig.colorbar(im, ax=a_, pad=0.01).set_label("log10 gaps", fontsize=7)
    for a_ in axs[-1]:
        a_.set_xlabel("gap duration (s)", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "survey_gap_vs_since.png", dpi=140); plt.close(fig)
    print("ok")


if __name__ == "__main__":
    main()
