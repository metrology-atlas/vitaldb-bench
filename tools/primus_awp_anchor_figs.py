# SPDX-License-Identifier: MIT
"""Figures for the atlas study primus-awp-anchoring.

    python tools/primus_awp_anchor_figs.py --work $W --root $R --out DIR
        [--modes M --ports P --step-rates S]   (defaults: $W/results/sync_full/..., $W/results/pat/...)

Writes three figures into DIR, their panels stacked vertically for the study page's column:
  primus-awp-breath-ambiguity.png  why breath matching can sit a breath off, from two recordings read here:
      A the breath-triggered averages of airway pressure and CVP in one 10-minute window;
      B their shape correlation against lag: peaks one breath period apart, nearly equal;
      C a pause in ventilation: the breathing envelopes of airway pressure and CVP stop and restart together, the
        CVP's shifted by the lag the pause gives, which is unique.
  primus-awp-offsets-by-rate.png   offsets at the start of each stretch, by the ventilator's set rate: as matched
      (with a second peak one breath above the main one) and after the breath rule (saved outputs only).
  primus-awp-correction.png        A per-recording offsets before and after; B leave-one-out accuracy of each step
      of the rule; C steps of the offset at AWP pauses, by gap kind (saved outputs only).
Inputs: $W/primus_awp_anchor/{breaths,gapsteps}.json, $W/primus_breath_rule/{stretch_k,summary}.json,
$W/primus_numbers/per_recording{,_uncorrected}.json, and for figure 1 the recordings named in EXAMPLES.
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
from scipy import ndimage  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fallback_rate import load_step_rates  # noqa: E402

BLUE, ORANGE, INK, MUTED, GRID, PALE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#f6c9b5"
# recording, window centre (s from the first AWP sample) for A-B; recording, pause stop (s) for C
EXAMPLES = {"profile": (136, 4491.9), "pause": (200, 6586.8)}


def style(ax, ygrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    if ygrid:
        ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def title(ax, t):
    ax.set_title(t, fontsize=9, loc="left", color=INK)


def fig_ambiguity(a, out):
    import ppv_by_breath as P
    from pat import sample_times
    from primus_breath_id import CORE, TAUS
    from primus_sync_accuracy import lowpass_mean
    from primus_vent_events import DT, envelope
    from vitaldb_reader.vital_file import read_vital
    from vent_sync_pilot import G, binned, trig_avg
    modes = {int(k): v for k, v in json.loads(Path(a.modes).read_text()).items()}
    ports = {int(k): (v or {}).get("port") for k, v in json.loads(Path(a.ports).read_text()).items()}
    rates = load_step_rates(a.step_rates)

    def load(c):
        v = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={"Primus/AWP", "SNUADC/CVP"})
        aw, cv = v.tracks["Primus/AWP"], v.tracks["SNUADC/CVP"]
        ta = sample_times(aw); xa = np.nan_to_num(aw.physical().astype(float))[: ta.size]
        tc = P.model_times(cv, ports.get(c), modes.get(c), rates.for_case(c)); xc = cv.physical().astype(float)[: tc.size]
        return ta, xa, tc, xc, P.breath_onsets(ta, xa, float(aw.srate)), float(cv.srate)

    fig, ax = plt.subplots(3, 1, figsize=(7, 10.5))
    c, mid = EXAMPLES["profile"]
    ta, xa, tc, xc, on, fs = load(c)
    z = ta[0]; t0 = float(np.floor(min(ta[0], tc[0])))
    n = int((max(ta[-1], tc[-1]) - t0) / G) + 1
    ok = np.isfinite(xc)
    awp50, cvp50 = binned(ta, xa, t0, G, n), binned(tc[ok], xc[ok], t0, G, n)
    o = on[(on >= z + mid - 300) & (on < z + mid + 300)]
    T = float(np.median(np.diff(o)))
    aa, cc = trig_avg(awp50, t0, o, TAUS), trig_avg(cvp50, t0, o, TAUS)
    m = (TAUS >= -2) & (TAUS <= 12)
    ax[0].plot(TAUS[m], aa[m], color=BLUE, lw=1.6, label="airway pressure (Primus)")
    ax2 = ax[0].twinx()
    ax2.plot(TAUS[m], cc[m], color=ORANGE, lw=1.6, label="CVP (SNUADC)")
    ax2.tick_params(colors=MUTED, labelsize=8); ax2.spines["top"].set_visible(False); ax2.spines["right"].set_color(MUTED)
    ax[0].set_xlabel("time after the breath's onset on the AWP stamps (s)", fontsize=8, color=MUTED)
    ax[0].set_ylabel("airway pressure, mean (mbar)", fontsize=8, color=BLUE)
    ax2.set_ylabel("CVP, mean change (mmHg)", fontsize=8, color=ORANGE)
    title(ax[0], f"A. Averaged over the breaths of 10 minutes\nbreath period {T:.1f} s: every breath looks alike")
    style(ax[0])
    a_ = ndimage.uniform_filter1d(np.nan_to_num(aa), 10); c_ = ndimage.uniform_filter1d(np.nan_to_num(cc), 10)
    Ls = np.arange(int(-3 / G), int(8 / G) + 1); rs = []
    x = a_[CORE] - a_[CORE].mean()
    for L in Ls:
        y = c_[CORE + L] - c_[CORE + L].mean()
        rs.append(float((x * y).sum() / np.sqrt((x ** 2).sum() * (y ** 2).sum())))
    rs = np.array(rs); lag = Ls * G
    ax[1].plot(lag, rs, color=INK, lw=1.4)
    pk = [i for i in range(1, rs.size - 1) if rs[i] >= rs[i - 1] and rs[i] >= rs[i + 1] and rs[i] > 0.5]
    for i in sorted(pk, key=lambda i: -rs[i])[:2]:
        ax[1].plot(lag[i], rs[i], "o", color=ORANGE, ms=6)
        ax[1].annotate(f"{lag[i]:.1f} s, r = {rs[i]:.3f}", (lag[i], rs[i]), xytext=(0, 8), textcoords="offset points",
                       ha="center", fontsize=8, color=INK)
    ax[1].set_xlabel("lag tried: CVP time minus AWP stamp (s)", fontsize=8, color=MUTED)
    ax[1].set_ylabel("shape correlation", fontsize=8, color=MUTED)
    ax[1].set_ylim(-1.05, 1.25)
    title(ax[1], "B. Matching the shapes: two lags one breath\napart fit almost equally well")
    style(ax[1])
    c, stop = EXAMPLES["pause"]
    ta, xa, tc, xc, on, fs = load(c)
    z = ta[0]; t0 = float(np.floor(min(ta[0], tc[0])))
    xcl = lowpass_mean(np.nan_to_num(xc, nan=float(np.nanmedian(xc))), fs)
    n = int((max(ta[-1], tc[-1]) - t0) / DT) + 1
    ea, ec = envelope(binned(ta, xa, t0, DT, n)), envelope(binned(tc, xcl, t0, DT, n))
    i0 = int(round((z + stop - t0) / DT)); w = int(150 / DT)
    lo_s, hi_s = -40, 150
    tt = (np.arange(i0 - w, i0 + w) - i0) * DT
    norm = lambda e: (e - np.nanpercentile(e, 5)) / (np.nanpercentile(e, 95) - np.nanpercentile(e, 5))  # noqa: E731
    seg_a = norm(ea[i0 - w:i0 + w])
    best, bl = -2, 0.0
    for L in range(int(-3 / DT), int(12 / DT) + 1):
        b = ec[i0 - w + L:i0 + w + L]
        r = np.corrcoef(seg_a, b)[0, 1]
        if r > best:
            best, bl = r, L * DT
    seg_cs = norm(ec[i0 - w + int(round(bl / DT)):i0 + w + int(round(bl / DT))])
    nxt = on[on > z + stop + 1][0] - z - stop
    ax[2].axvspan(0, nxt, color=GRID, lw=0)
    ax[2].plot(tt, seg_a, color=BLUE, lw=1.6, label="airway pressure")
    ax[2].plot(tt, seg_cs, color=ORANGE, lw=1.6, label=f"CVP, moved back by\nthe pause's lag ({bl:.1f} s)")
    ax[2].text(nxt / 2, 1.12, "no breaths", ha="center", fontsize=8, color=MUTED)
    ax[2].set_xlabel("time from the last breath before the pause (s)", fontsize=8, color=MUTED)
    ax[2].set_ylabel("envelope (scaled)", fontsize=8, color=MUTED)
    ax[2].set_xlim(lo_s, hi_s); ax[2].set_ylim(-0.1, 1.25)
    ax[2].legend(fontsize=7, frameon=False, labelcolor=INK, loc="lower left", bbox_to_anchor=(0.23, 0.12),
                 title="breathing envelopes", title_fontsize=7)
    title(ax[2], f"C. A pause in ventilation is not periodic:\nonly one lag lines the two up (r = {best:.2f})")
    style(ax[2])
    fig.tight_layout()
    fig.savefig(out / "primus-awp-breath-ambiguity.png", dpi=150, facecolor="white")
    plt.close(fig)


def fig_rates(a, out):
    W = Path(a.work)
    B = json.loads((W / "primus_awp_anchor" / "breaths.json").read_text())["stretches"]
    K = {(x["case"], x["stretch"], x["arm"]): x for x in json.loads((W / "primus_breath_rule" / "stretch_k.json").read_text())}
    m0 = json.loads((W / "primus_breath_rule" / "summary.json").read_text())["main_mode_s"]
    rows = [(s_["at_start"], s_["T"], K.get((r["case"], s_["i"], r["arm"]), {}).get("corrected_start_s"))
            for r in B for s_ in r["stretches"] if s_.get("T")]
    rates = (18, 16, 14, 12)
    fig, ax = plt.subplots(len(rates), 1, figsize=(7, 11), sharex=True)
    bins = np.arange(-2, 9.01, 0.25)
    for k, r_ in enumerate(rates):
        sel = [x for x in rows if round(60 / x[1]) == r_]
        T = 60 / r_
        before = [x[0] for x in sel]; after = [x[2] for x in sel if x[2] is not None]
        inr = lambda v: [x for x in v if -2 <= x <= 9]  # noqa: E731
        ax[k].hist(inr(before), bins=bins, color=PALE, edgecolor="white", lw=0.4, label="as matched")
        ax[k].hist(inr(after), bins=bins, histtype="step", color=BLUE, lw=1.6, label="after the breath rule")
        ax[k].axvline(m0, color=INK, lw=0.8)
        ax[k].axvline(m0 + T, color=ORANGE, lw=1.2, ls="--")
        ymax = ax[k].get_ylim()[1] * 1.12
        ax[k].set_ylim(0, ymax)
        ax[k].text(m0 - 0.15, ymax * 0.97, "main\nmode", fontsize=7, color=INK, ha="right", va="top")
        ax[k].text(m0 + T + 0.15, ymax * 0.97, f"one breath\nlater (+{T:.2f} s)", fontsize=7, color=ORANGE, va="top")
        outn = len(before) - len(inr(before))
        title(ax[k], f"{'ABCD'[k]}. Set rate {r_}/min\n{len(sel):,} stretches" + (f" ({outn} outside −2 to 9 s)" if outn else ""))
        ax[k].set_ylabel("stretches", fontsize=8, color=MUTED)
        style(ax[k])
    ax[-1].set_xlabel("offset at the stretch start (s)", fontsize=8, color=MUTED)
    ax[-1].legend(fontsize=7, frameon=False, labelcolor=INK, loc="center right")
    fig.tight_layout()
    fig.savefig(out / "primus-awp-offsets-by-rate.png", dpi=150, facecolor="white")
    plt.close(fig)


def fig_correction(a, out):
    W = Path(a.work)
    before = json.loads((W / "primus_numbers" / "per_recording_uncorrected.json").read_text())["lag_median_s"]
    after = json.loads((W / "primus_numbers" / "per_recording.json").read_text())["lag_median_s"]
    S = json.loads((W / "primus_breath_rule" / "summary.json").read_text())["leave_one_out"]
    G = json.loads((W / "primus_awp_anchor" / "gapsteps.json").read_text())
    fig, ax = plt.subplots(3, 1, figsize=(7, 10.5))
    bins = np.arange(-8, 2.01, 0.25)
    b_, a_ = -np.array(before), -np.array(after)
    nout = int(np.sum((b_ < -8) | (b_ > 2)) + np.sum((a_ < -8) | (a_ > 2)))
    ax[0].hist(b_[(b_ >= -8) & (b_ <= 2)], bins=bins, color=PALE, edgecolor="white", lw=0.4,
               label=f"as matched (median {np.median(b_):.2f} s)".replace("-", "\u2212"))
    ax[0].hist(a_[(a_ >= -8) & (a_ <= 2)], bins=bins, histtype="step", color=BLUE, lw=1.8,
               label=f"each stretch on its right\nbreath (median {np.median(a_):.2f} s)".replace("-", "\u2212"))
    ax[0].set_xlabel("AWP stamp minus the breath on CVP, per recording (s)", fontsize=8, color=MUTED)
    ax[0].set_ylabel("recordings", fontsize=8, color=MUTED)
    ax[0].legend(fontsize=7, frameon=False, labelcolor=INK, loc="center left")
    title(ax[0], f"A. The AWP offset, {len(after):,} recordings" + (f"\n({nout} values outside −8 to 2 s not shown)" if nout else ""))
    style(ax[0])
    labels = ["trust the matching", "carried from a\nneighbour", "nearest the\nmain mode"]
    vals = [S["baseline_k0_right_share"], S["step2_right_share"], S["step3_right_share"]]
    ns = [S["stretches_with_own_cvp_events"], S["step2_reached"], S["step3_reached"]]
    ax[1].bar(range(3), [100 * v for v in vals], color=[PALE, BLUE, BLUE], width=0.6)
    for i, (v, n_) in enumerate(zip(vals, ns)):
        ax[1].text(i, 100 * v + 2, f"{100 * v:.0f}%\n(n = {n_})", ha="center", fontsize=8, color=INK)
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels(labels, fontsize=8, color=INK)
    ax[1].set_ylim(0, 112); ax[1].set_ylabel("right breath (%)", fontsize=8, color=MUTED)
    title(ax[1], f"B. Leaving out the {ns[0]} stretches\nwhose own pauses fix the breath")
    style(ax[1])
    bins = np.arange(-3, 3.01, 0.1)
    for flag, col, lab in ((True, BLUE, "gap of whole records: the 0.48 s grid carries on"),
                           (False, ORANGE, "other gap: a new anchor")):
        v = np.array([x["step"] for x in G if x["whole"] == flag])
        ax[2].hist(v[np.abs(v) <= 3], bins=bins, density=True, histtype="step", lw=1.8, color=col, label=f"{lab} ({v.size:,})")
    ax[2].set_xlabel("step of the offset at an AWP pause (s)", fontsize=8, color=MUTED)
    ax[2].set_ylabel("density", fontsize=8, color=MUTED)
    ax[2].set_ylim(0, 2.6)
    ax[2].legend(fontsize=7, frameon=False, labelcolor=INK, loc="upper left")
    title(ax[2], "C. The stamp anchor holds or resets\n(steps beyond ±3 s not shown)")
    style(ax[2])
    fig.tight_layout()
    fig.savefig(out / "primus-awp-correction.png", dpi=150, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--modes"); ap.add_argument("--ports"); ap.add_argument("--step-rates")
    ap.add_argument("--only", choices=("ambiguity", "rates", "correction"))
    a = ap.parse_args()
    W = Path(a.work)
    a.modes = a.modes or str(W / "results" / "sync_full" / "snuadc_mode_label.json")
    a.ports = a.ports or str(W / "results" / "sync_full" / "snuadc_mode.json")
    a.step_rates = a.step_rates or str(W / "results" / "pat" / "snuadc_step_rates.json")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name, f in (("ambiguity", fig_ambiguity), ("rates", fig_rates), ("correction", fig_correction)):
        if a.only in (None, name):
            f(a, out)


if __name__ == "__main__":
    main()
