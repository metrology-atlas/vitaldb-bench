# SPDX-License-Identifier: MIT
"""Figure ppg-slip-detector: the slip detector of metrology-atlas study ppg-sawtooth-slips, step by step, on
synthetic data (illustrative; no recording is read).

    python tools/slip_concept_figure.py --out $W/sawtooth_char/figures

Panels: (a) the sawtooth in the ECG->PPG delay, a one-frame jump at each slip and a creep between; (b) the three
PLETH fiducials per pulse; (c) every foot, upstroke and peak in time order, a slip falling inside one beat; (d) the
step test, a sharp one-frame step against no step and against the same change over 7 beats; (e) the arterial check,
ECG->PPG and ART->PPG step while ECG->ART does not; (f) placement, fiducials before and after the slip with one
outlier, the posterior over the slip's position and its 95% bracket. The numbers in the drawing (tooth length,
noise, offsets) are chosen for legibility, not measured. Writes ppg-slip-detector.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

F = 1000 / 30          # one PLETH frame, ms
PPG, ART, EA, INK, MUTED = "#3b4fb8", "#e8590c", "#2f9e44", "#222222", "#777777"


def tag(ax, s):
    ax.text(-0.12, 1.06, s, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


def draw(out: Path) -> None:
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#999999", "axes.labelcolor": INK, "xtick.color": "#555555",
                         "ytick.color": "#555555", "axes.titlesize": 9, "axes.titleweight": "bold",
                         "axes.titlelocation": "left"})
    rng = np.random.default_rng(3)
    fig = plt.figure(figsize=(11, 7.6))
    gs = fig.add_gridspec(3, 6, hspace=0.75, wspace=1.1, height_ratios=[1, 1.05, 1.05])

    # (a) what a slip is: the delay creeps, then jumps one frame
    ax = fig.add_subplot(gs[0, :4]); tag(ax, "a")
    t = np.linspace(0, 60, 3000)
    L = 14.0
    slips = np.arange(5, 60, L)
    k = np.searchsorted(slips, t, side="right")
    D = F * (k - (t - 5) / L) + rng.normal(0, 2.5, t.size) + 6 * np.sin(t / 9)
    ax.plot(t, D, color=PPG, lw=0.8)
    for s in slips:
        ax.axvline(s, color=MUTED, lw=0.6, ls=":")
    ax.annotate("slip: the delay jumps\nby one 33 ms frame", xy=(slips[1], 20), xytext=(slips[1] + 2.2, 52),
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.7), fontsize=8)
    ax.annotate("creep: the oximeter's frame clock and\nthe recorder's sample clock differ slightly", xy=(40.5, 4), xytext=(42.5, -40),
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.7), fontsize=8)
    ax.set(xlabel="minutes", ylabel="ECG→PPG delay\n(ms, about its median)", ylim=(-55, 70), xlim=(0, 60))
    ax.set_title("What a slip is: a sawtooth in the ECG→PPG delay")

    # (b) three fiducials per pulse
    ax = fig.add_subplot(gs[0, 4:]); tag(ax, "b")
    tt = np.linspace(0, 0.9, 400)
    p = np.where(tt < 0.15, np.sin(np.pi / 2 * tt / 0.15) ** 2, np.exp(-(tt - 0.15) / 0.3))
    ax.plot(tt * 1000, p, color=PPG, lw=1.6)
    for x, lab in ((0.0, "foot"), (0.075, "upstroke"), (0.15, "peak")):
        y = np.interp(x, tt, p)
        ax.plot(x * 1000, y, "o", ms=6, color="white", mec=INK, mew=1.2, zorder=3)
        ax.text(x * 1000 + 25, y - 0.02, lab, fontsize=8, va="top" if lab != "peak" else "bottom")
    ax.set(xlabel="ms after the pulse foot", yticks=[], ylabel="PLETH")
    ax.set_title("Three fiducials per pulse")

    # (c) the fine series: a slip is a clean step between two fiducials
    ax = fig.add_subplot(gs[1, :3]); tag(ax, "c")
    nb = 30
    beat = np.repeat(np.arange(nb), 3) + np.tile([0, 0.33, 0.66], nb)
    slip_at = 15.2                                      # between the foot and the upstroke of beat 15
    lvl = np.where(beat > slip_at, F, 0) + rng.normal(0, 3.0, beat.size) + 0.4 * beat
    kinds = np.tile(["o", "s", "^"], nb)
    for m, lab in (("o", "foot"), ("s", "upstroke"), ("^", "peak")):
        sel = kinds == m
        ax.plot(beat[sel], lvl[sel], m, ms=4, color=PPG, mfc="white" if m != "s" else PPG, mew=0.9, label=lab, ls="none")
    ax.axvline(slip_at, color=INK, lw=0.8, ls="--")
    ax.text(slip_at + 0.5, 22, "the slip falls inside beat 15:\nits foot is before it, its\nupstroke and peak after", fontsize=7.5, va="top")
    ax.annotate("", xy=(31, F + 12), xytext=(31, 12), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
    ax.text(31.5, F / 2 + 12, "one frame,\n33 ms", fontsize=8, va="center")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", handletextpad=0.2)
    ax.set(xlabel="beat", ylabel="fiducial delay, levelled (ms)", ylim=(-15, 60), xlim=(-1, 36))
    ax.set_title("Every foot, upstroke and peak in time order")

    # (d) the step test: sharp one-frame step vs no step vs a 7-beat ramp
    ax = fig.add_subplot(gs[1, 3:]); tag(ax, "d")
    x = np.arange(-15, 15) + 0.5
    y = np.where(x > 0, F, 0) + rng.normal(0, 3.5, x.size)
    ax.plot(x, y, "o", ms=3.5, color=PPG, mfc="white", mew=0.9)
    ax.plot(x, np.polyval(np.polyfit(x, y, 1), x), color=MUTED, lw=1.4, ls="-")
    ax.plot(x, np.where(x > 0, F, 0), color=INK, lw=1.8)
    ax.plot(x, np.clip((x + 3.5) / 7, 0, 1) * F, color=MUTED, lw=1.4, ls="--")
    ax.text(-14.5, 22, "no step (one line):\nrejected", color=MUTED, fontsize=7.5)
    ax.text(1.2, 6, "one-frame step:\nkept", color=INK, fontsize=7.5, fontweight="bold")
    ax.text(-9.5, 33, "same change over\n7 beats (physiology)", color=MUTED, fontsize=7.5)
    ax.set(xlabel="beats from the candidate (15 each side)", ylabel="delay (ms)")
    ax.set_title("Is it a sharp step of exactly one frame?")

    # (e) the arterial check
    ax = fig.add_subplot(gs[2, :3]); tag(ax, "e")
    xb = np.arange(-25, 25)
    for i, (lab, col, step, off) in enumerate((("ECG→PPG", PPG, F, 100), ("ART→PPG", ART, F, 55), ("ECG→ART", EA, 0, 10))):
        yy = off + np.where(xb >= 0, step, 0) + rng.normal(0, 2.5, xb.size)
        ax.plot(xb, yy, color=col, lw=1.3)
        ax.text(26, yy[-6:].mean(), lab, color=col, fontsize=8, va="center", fontweight="bold")
    ax.axvline(0, color=INK, lw=0.7, ls=":")
    ax.text(-24, 205, "PPG moves, the arterial pulse does not:\na slip in the oximeter, not physiology", fontsize=7.5, va="top")
    ax.set(xlabel="beats from the step", ylabel="delay (ms, offset for display)", yticks=[], xlim=(-26, 36), ylim=(-5, 215))
    ax.set_title("Confirmed against the arterial line")

    # (f) placement
    ax = fig.add_subplot(gs[2, 3:]); tag(ax, "f")
    pos = np.array([-2.9, -2.6, -2.3, -1.95, -1.6, -1.35, -0.9, -0.62, -0.35, 0.2, 0.45, 0.8, 1.1, 1.4, 1.7, 2.1, 2.45, 2.8])
    side = np.where(pos < 0, 0, 1)
    side[7] = 1                                         # an outlier: tolerated by the robust likelihood
    ax.scatter(pos, side * 0.95, s=22, c=[PPG if s_ else "white" for s_ in side], edgecolors=PPG, zorder=3)
    ax.text(-3.1, 1.03, "after the slip (moved one frame)", fontsize=7.5)
    ax.text(-3.1, -0.11, "before", fontsize=7.5)
    g = np.linspace(-3, 3, 400)
    post = np.exp(-0.5 * ((g + 0.12) / 0.16) ** 2) + 0.12 * np.exp(-0.5 * ((g - 0.5) / 0.12) ** 2)
    ax.fill_between(g, 0.3, 0.3 + 0.5 * post / post.max(), color=INK, alpha=0.18, lw=0)
    ax.plot(g, 0.3 + 0.5 * post / post.max(), color=INK, lw=1)
    ax.plot([-0.4, 0.62], [0.2, 0.2], color=INK, lw=4, solid_capstyle="butt")
    ax.text(0.75, 0.2, "95% bracket", fontsize=7.5, va="center")
    ax.text(0.75, 0.62, "posterior over where\nthe slip falls", fontsize=7.5)
    ax.set(xlabel="seconds from the most likely position", yticks=[], ylim=(-0.18, 1.12))
    ax.set_title("Placed between two fiducials, with a bracket")

    fig.text(0.01, 0.005, "Illustrative: synthetic data, not a recording.", fontsize=7.5, color=MUTED)
    fig.savefig(out / "ppg-slip-detector.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    draw(a.out)
    print(a.out / "ppg-slip-detector.png")


if __name__ == "__main__":
    main()
