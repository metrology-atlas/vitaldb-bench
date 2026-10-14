# SPDX-License-Identifier: MIT
"""Plot the SpO2 step-event pilot summary (tools/spo2_event_agg.py output; aggregates only).

    python tools/spo2_event_plot.py summary.json out.png
"""

from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

G = np.arange(-20, 61)
C = {"pleth_only": "#c0392b", "cautery": "#7d5ba6", "control": "0.45", "flat": "#1f5fa8", "low": "#e08a1e"}


def box(ax, y, p, col, label):
    p10, p25, p50, p75, p90 = p
    ax.plot([p10, p90], [y, y], color=col, lw=1.5)
    ax.add_patch(plt.Rectangle((p25, y - 0.18), p75 - p25, 0.36, color=col, alpha=0.35))
    ax.plot([p50, p50], [y - 0.22, y + 0.22], color=col, lw=2.5)
    ax.text(-2.5, y + 0.3, f"{label}: median {p50:.1f} s (IQR {p25:.1f} to {p75:.1f})", va="bottom", fontsize=9)


def main():
    S = json.load(open(sys.argv[1]))
    E = S["episodes"]
    fig, axs = plt.subplots(2, 2, figsize=(15, 9.5))
    # A dropout by duration
    ax = axs[0, 0]
    w = 0.38
    for k, (typ, off) in enumerate((("flat", -w / 2), ("low", w / 2))):
        e = E.get(f"spo2_{typ}")
        bd = [b for b in e["by_duration"] if b["n"]]
        x = np.arange(len(bd)) + off
        drop = [b["dropout"] / b["n"] for b in bd]
        held = [b["held_const_no_dropout"] / b["n"] for b in bd]
        ax.bar(x, drop, w, color=C[typ], label=f"{'probe off (flat PLETH)' if typ == 'flat' else 'low perfusion'}: SpO2 blanked")
        ax.bar(x, held, w, bottom=drop, color=C[typ], alpha=0.35, hatch="//",
               label=f"{'probe off' if typ == 'flat' else 'low perfusion'}: SpO2 frozen (no change)")
        for xi, b in zip(x, bd):
            ax.text(xi, 1.02, str(b["n"]), ha="center", fontsize=8.5)
    labs = [f"{b['dur'][0]}-{b['dur'][1]} s" if b["dur"][1] else f">{b['dur'][0]} s" for b in E["spo2_flat"]["by_duration"] if b["n"]]
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs)
    ax.set_ylim(0, 1.45)
    ax.set_xlabel("duration of pulse loss in PLETH (numbers: episodes, probe off / low perfusion)")
    ax.set_ylabel("fraction of episodes")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.set_title(f"A  Pulse loss: what SpO2 does ({E['total'] - E['recorder_gap']} episodes, "
                 f"{E['recorder_gap']} recorder gaps excluded)", loc="left", fontsize=11)
    # B timing
    ax = axs[0, 1]
    rows = [("spo2_flat", "hold_s", "flat", "probe off: last SpO2 record after pulse loss"),
            ("spo2_low", "hold_s", "low", "low perfusion: last record after pulse loss"),
            ("spo2_flat", "resume_s", "flat", "probe off: first SpO2 after pulse return"),
            ("spo2_low", "resume_s", "low", "low perfusion: first SpO2 after pulse return")]
    for i, (k, f, typ, lab) in enumerate(rows):
        p = E[k][f]
        box(ax, len(rows) - 1 - i, [min(x, 30) for x in p], C[typ], lab)
    ax.set_ylim(-0.5, len(rows) - 0.1)
    ax.axvline(0, color="0.5", lw=1)
    ax.set_yticks([])
    ax.set_xlim(-3, 32)
    ax.set_xlabel("s relative to the PLETH edge (box IQR, whiskers p10-p90; clipped at 30 s)")
    ax.set_title("B  Blanking and re-acquisition timing (dropout episodes)", loc="left", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    # C, D mean curves
    for ax, key, unit, ttl in ((axs[1, 0], "spo2", "% SpO2", "C  SpO2 around artefact onsets"),
                               (axs[1, 1], "phr", "bpm", "D  PLETH_HR around artefact onsets")):
        for typ, lab in (("pleth_only", "PLETH-only artefact (motion / HF)"), ("cautery", "electrocautery (ECG)"),
                         ("control", "random control")):
            s = S[typ][key]
            m = np.array([np.nan if z is None else z for z in s["mean_curve"]], float)
            r = s["responded"] / s["n"]
            ax.plot(G, m, color=C[typ], lw=2 if typ != "control" else 1.4,
                    label=f"{lab}: n={s['n']}, deviation >= 2 in {100 * r:.0f}%")
        ax.axvline(0, color="0.5", lw=1)
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_xlabel("s from artefact onset in the recorded PLETH")
        ax.set_ylabel(f"mean change from pre-event median ({unit})")
        ax.set_title(ttl, loc="left", fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left" if key == "spo2" else "upper right")
        ax.set_xlim(-20, 60)
    f = S.get("pleth_hr_fit")
    if f:
        fam = f["best_family"][0]
        axs[1, 1].text(0.99, 0.03, f"PLETH_HR algorithm fit ({f['n']} cases): {fam[0].replace('|n', ' of N beats').replace('|t', ' over T s')} "
                                   f"in {fam[1]}, N/T median {np.median(f['windows']):.0f};\ndelay (mid) median {f['delay_mid_s'][2]:.1f} s "
                                   f"(IQR {f['delay_mid_s'][1]:.1f}-{f['delay_mid_s'][3]:.1f}); rounded in {f['n'] - f['floor_better']}/{f['n']}; "
                                   f"no held updates in {dict(f['hold_best_U']).get('none', 0)}/{f['n']}",
                      transform=axs[1, 1].transAxes, ha="right", va="bottom", fontsize=8.3,
                      bbox=dict(facecolor="white", edgecolor="0.7"))
    fig.suptitle(f"SpO2 and PLETH_HR responses to step events in the PLETH ({S['cases_used']} cases). "
                 "Times are relative to the recorded PLETH (itself ~0.4 s behind the finger).", x=0.01, ha="left", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(sys.argv[2], dpi=110)


if __name__ == "__main__":
    main()
