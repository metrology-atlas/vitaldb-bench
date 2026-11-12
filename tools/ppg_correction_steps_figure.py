# SPDX-License-Identifier: MIT
"""Paper figure: the PLETH sawtooth correction step by step (metrology-atlas-paper, Section 4.2).

Reads saved outputs only:
  - the per-recording output of tools/ppg_fix_check.py --method rebuilt (check_rebuilt6.jsonl, as the atlas figure
    ppg-correction-check-rebuilt), for one recording's series (ECG R peak to PPG foot, 15-beat medians, measured and corrected), its slips and, across recordings, the
    PAT step at each modelled slip measured and corrected (as tools/ppg_sawtooth_figures.py check, panel d).
  - for panel e, the recording's beat times from the full-release scan (BEATS/<case>.npz: R peaks, PLETH steepest
    upstroke, arterial upstroke) and its rebuilt fit (FIT, fit.jsonl): the ECG-to-arterial delay, and the
    ECG-to-PPG delay uncorrected and corrected (minus physiosync's delay_at, the released correction, as
    tools/correction_vs_art.py applies it), each as 15-beat medians, on one axis from 0 ms.

    PYTHONPATH=physiosync python tools/ppg_correction_steps_figure.py \
        --scan $W/ppg_correction_check/check_rebuilt6.jsonl --beats $W/sawtooth_char/scan_full \
        --fit $W/slip_rebuild/full3/fit.jsonl --case 6241 --out $W/ppg_correction_steps

Writes OUT/ppg-correction-steps.pdf, OUT/ppg-correction-steps.png and OUT/summary.json (every number drawn).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, MUTED, GRID = "#1b2333", "#5a6170", "#e4e3df"
RAW, COR, FIT = "#2a78d6", "#eb6834", "#1b2333"


def half_up(x: float, d: int = 1) -> str:
    """Round half away from zero on the value as written (13.95 -> 14.0), as the paper's facts do."""
    from decimal import ROUND_HALF_UP, Decimal
    return str(Decimal(str(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def rows(paths):
    for p in paths:
        for x in Path(p).read_text().splitlines():
            if x.strip():
                yield json.loads(x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", required=True)
    ap.add_argument("--beats", required=True, help="folder of the full-release scan, <case>.npz")
    ap.add_argument("--fit", required=True, help="the rebuilt fit, fit.jsonl")
    ap.add_argument("--chain-delay", required=True,
                    help="the atlas parameter of the PLETH's fixed delay (parameters/pleth-chain-delay.yaml)")
    ap.add_argument("--case", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ok = [r for r in rows([a.scan]) if "series" in r]
    r = next(x for x in ok if x["case"] == a.case)
    t, raw, cor = (np.array(v, float) for v in r["series"])
    m = t / 60
    model = raw - cor                                  # what the correction removes: the fixed offset and the sawtooth
    cor_shown = cor - np.median(cor) + np.median(raw)  # the corrected series at the measured median, for comparison
    slips = np.array(r["slips_t"], float) / 60
    st_raw = np.array([s for x in ok for s in x["step_at_slips_raw_ms"]], float)
    st_cor = np.array([s for x in ok for s in x["step_at_slips_corr_ms"]], float)

    S = {"case": a.case, "minutes": r["minutes"], "beats": r["beats"], "slips_corrected": r["n_kept"],
         "placed_within_interval": r.get("n_uncertain"), "offset_ms": r["offset_ms"],
         "p2p_raw_ms": r["p2p_raw_ms"], "p2p_corr_ms": r["p2p_corr_ms"],
         "creep_ppm_median": round(float(np.median([x for x, s in zip(r["rates_ppm"], r["sources"]) if s == "tooth"])), 1),
         "step_recordings": len(ok), "steps": int(st_raw.size),
         "step_raw_ms_median": round(float(np.median(st_raw)), 1),
         "step_corr_ms_median": round(float(np.median(st_cor)), 1)}
    # panel e data
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from correction_vs_art import FRAME  # noqa: E402
    from physiosync.slips import delay_at  # noqa: E402
    z = np.load(Path(a.beats) / f"{a.case}.npz")
    R = None
    with open(a.fit) as fh:
        for line in fh:
            if line.startswith(f'{{"case": {a.case},'):
                R = json.loads(line)["real"]
                break
    rt, up, at = z["r_t"], z["up"], z["art_t"]
    okb = np.isfinite(rt) & np.isfinite(up) & np.isfinite(at)
    rt, up, at = rt[okb], up[okb], at[okb]

    def med15(x):
        n = x.size // 15
        return np.median(x[:n * 15].reshape(n, 15), axis=1) * 1000

    e_art = med15(at - rt)
    e_raw = med15(up - rt)
    e_cor = med15(up - rt - delay_at(up, R, FRAME))
    # the correction's zero is a convention (the sawtooth model counts from the recording's first tooth), so the
    # corrected delays are placed at the measured median, as in panel c; their spread is what the correction sets
    e_cor_shift = float(np.median(e_raw) - np.median(e_cor))
    # and the monitor's fixed delay removed too (one value for the release, the atlas's pleth-chain-delay), so the
    # corrected finger delay sits where the pulse reaches the finger, beside the arterial pulse
    import yaml
    chain_ms = float(yaml.safe_load(Path(a.chain_delay).read_text(encoding="utf-8"))["value"])
    e_cor = e_cor + e_cor_shift - chain_ms

    def rsd(v):
        return round(float(np.subtract(*np.percentile(v, [75, 25])) / 1.349), 2)

    S.update({"e_beats": int(okb.sum()), "e_medians": int(e_art.size),
              "e_sd_ms": {"art": rsd(e_art), "raw": rsd(e_raw), "cor": rsd(e_cor)},
              "e_median_ms": {"art": round(float(np.median(e_art)), 1), "raw": round(float(np.median(e_raw)), 1),
                              "cor": round(float(np.median(e_cor)), 1)},
              "e_cor_shift_ms": round(e_cor_shift, 1), "e_chain_delay_ms": chain_ms})
    (out / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps(S, indent=1))

    # panel b: the fitted sawtooth alone. Each tooth (between the fit's tooth edges) is a straight line at that
    # tooth's own rate (rates_ppm; the delay falls at the rate, then steps back at the slip), placed by least squares
    # on what the correction removes (measured minus corrected); teeth shorter than a minute (the ramp of a slip
    # placed only within an interval) are left undrawn. The dots are the measured delay less the slow trend of the
    # corrected delay (a running median over 31 points), so they hold the sawtooth and the noise, not the physiology.
    k = 15
    trend = np.array([np.median(cor[max(0, i - k):i + k + 1]) for i in range(cor.size)])
    detr = raw - trend
    level = np.median(detr)
    teeth = []
    E = np.array(r["tooth_edges_t"], float)
    for (ea, eb), rate in zip(zip(E[:-1], E[1:]), r["rates_ppm"]):
        sel = (t >= ea) & (t < eb)
        if eb - ea < 60 or sel.sum() < 3:
            continue
        slope = -rate * 1e-3                                  # ppm = us/s, as ms/s
        c0 = float(np.mean(model[sel] - slope * (t[sel] - ea)))
        teeth.append((ea, eb, slope, c0))
    S["teeth_drawn"] = len(teeth)
    S["teeth_rate_ppm"] = [round(-sl * 1e3, 2) for *_, sl, _ in teeth]
    (out / "summary.json").write_text(json.dumps(S, indent=1) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
                         "axes.spines.top": False, "axes.spines.right": False})
    # a-c full width on one time axis; d and e side by side in a fourth row
    fig = plt.figure(figsize=(6.32, 6.9))   # drawn at the text width, so type prints at its stated size
    g = fig.add_gridspec(3, 1, left=0.095, right=0.975, top=0.955, bottom=0.33, hspace=0.62)
    axs = [fig.add_subplot(g[k]) for k in range(3)]
    gd = fig.add_gridspec(1, 3, left=0.095, right=0.975, top=0.22, bottom=0.065, wspace=0.30,
                          width_ratios=[0.8, 1.6, 0.5])
    axd = fig.add_subplot(gd[0])
    # e: one axis from 0 ms, no break: the arterial delay, the corrected finger delay and the finger delay as
    # recorded (which still holds the monitor's fixed delay)
    axe = fig.add_subplot(gd[1])
    lo, hi = np.floor(min(raw.min(), cor_shown.min()) / 10) * 10, np.ceil(max(raw.max(), cor_shown.max()) / 10) * 10
    for ax in axs[1:]:
        ax.sharex(axs[0])
    for ax in axs:
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    for ax in (axs[0], axs[2]):
        ax.set_ylim(lo, hi)
        ax.set_ylabel("ECG to PPG (ms)")
    ax = axs[0]
    ax.plot(m, raw, ".", ms=2, color=RAW)
    ax.set_title("a. Find: the delay creeps, then jumps one frame", loc="left", fontsize=8, color=INK)
    ax = axs[1]
    ax.plot(m, detr - level, ".", ms=2, color=RAW, alpha=0.45)
    for ea, eb, sl, c0 in teeth:
        x = np.array([ea, eb])
        ax.plot(x / 60, c0 + sl * (x - ea) - np.median(model) , color=FIT, lw=1.1)
    for x in slips:
        ax.axvline(x, color=MUTED, lw=0.6, ls=(0, (2, 2)))
    ax.set_ylim(-35, 35)
    ax.set_ylabel("less its slow\ntrend (ms)")
    ax.set_title(f"b. Measure: {S['slips_corrected']} slips; the sawtooth fitted (black, "
                 f"{half_up(S['creep_ppm_median'])} ppm)", loc="left", fontsize=8, color=INK)
    ax = axs[2]
    ax.plot(m, raw, ".", ms=2, color=RAW, alpha=0.25)
    ax.plot(m, cor_shown, ".", ms=2, color=COR)
    ax.set_title("c. Undo: corrected (orange); what remains is physiology", loc="left", fontsize=8, color=INK)
    ax.set_xlabel(f"minutes (recording {a.case})")
    xl = (m[0] - 2, m[-1] + 2)
    for ax in axs:
        ax.set_xlim(*xl)
    for ax in axs[:2]:
        plt.setp(ax.get_xticklabels(), visible=False)
    for ax in axs:
        ax.yaxis.set_label_coords(-0.058, 0.5)

    # d: the step at each slip, measured and corrected; each distribution labelled at its peak (no legend to
    # cover the bars)
    ax = axd
    b = np.arange(-20, 51, 2)
    hr, _, _ = ax.hist(np.clip(st_raw, -20, 50), bins=b, color=RAW, alpha=0.75)
    hc, _, _ = ax.hist(np.clip(st_cor, -20, 50), bins=b, color=COR, alpha=0.75)
    top = max(hr.max(), hc.max())
    ax.set_ylim(0, top * 1.42)
    mid = (b[:-1] + b[1:]) / 2
    for h, txt, col in ((hc, f"corrected\nmedian\n{half_up(S['step_corr_ms_median'])} ms", COR),
                        (hr, f"measured\nmedian\n{half_up(S['step_raw_ms_median'])} ms", RAW)):
        i = int(np.argmax(h))
        ax.text(mid[i], h[i] + top * 0.04, txt.replace("-", "\u2212"), ha="center", va="bottom", fontsize=6.3,
                color=col, linespacing=1.1)
    ax.set_xlabel("step at each slip (ms)")
    ax.set_ylabel("slips")
    ax.set_title(f"d. Test: steps at {S['steps']} slips\n({S['step_recordings']} recordings)", loc="left",
                 fontsize=8, color=INK)

    COLS = {"art": ("#1baf7a", "ECG to arterial"), "raw": (RAW, "ECG to PPG,\nas recorded"),
            "cor": (COR, "ECG to PPG,\ncorrected")}
    hi_e = float(np.ceil((np.percentile(e_raw, 99.5) + 20) / 50) * 50)
    for key, v in (("art", e_art), ("cor", e_cor), ("raw", e_raw)):
        col, lab = COLS[key]
        axe.hist(np.clip(v, 0, hi_e), bins=np.arange(0, hi_e + 0.1, 3), histtype="step", lw=1.2, color=col,
                 label=f"{lab}\n(robust SD {half_up(S['e_sd_ms'][key])} ms)")
    axe.set_xlim(0, hi_e)
    axe.set_xlabel("delay from the R peak (ms)")
    axe.set_ylabel("15-beat medians")
    axe.set_title(f"e. Test: the arterial line\n(recording {a.case})", loc="left", fontsize=8, color=INK)
    axe.legend(frameon=False, fontsize=6.3, loc="upper left", bbox_to_anchor=(1.03, 1.0), handlelength=1.4,
               labelspacing=0.9)

    fig.savefig(out / "ppg-correction-steps.pdf")
    fig.savefig(out / "ppg-correction-steps.png", dpi=200)


if __name__ == "__main__":
    main()
