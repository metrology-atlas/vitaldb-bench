# SPDX-License-Identifier: MIT
"""What synchronising the clocks does, in one recording: timing errors before and after, as time series.

    python tools/sync_demo.py --root R --case 1216 --out FIG.png [--json OUT.json]

Three clocks against the SNUADC stamp-model timeline (the recorder PC's timeline, carried by the SNUADC):
  SNUADC  its record stamps minus the stamp model (the sawtooth of drift and half-second resets);
  BIS     its EEG stamps minus the stamp model plus the step-mode EEG-ECG constant (-0.804 s, from other recordings);
  Primus  its airway-pressure stamps' drift: the AWP-to-CVP lag fitted with one slope and one offset per stretch
          between AWP pauses, minus its median (the physiological airway-to-CVP delay is unknown and left out).
Two checks between clocks, from the signals themselves:
  BIS vs SNUADC     the EEG-minus-ECG lag of shared electrocautery and muscle bursts, per 10-minute window, on the
                    recorded stamps and on the synchronised times (independent of the correction: the model uses
                    only the stamps and a constant from other recordings);
  Primus vs SNUADC  the AWP-to-CVP lag per 10-minute window, on the recorded stamps, and after synchronisation as the
                    leave-one-out error of the fit (each window predicted from all the others).
Left column before, right column after, on the same scale.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ppv_by_breath as P  # noqa: E402
from pat import sample_times  # noqa: E402
from physiosync import corrected_times, stamp_clock  # noqa: E402
from shared_artifacts import BAND, DT, burst_probe, envelope  # noqa: E402
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital  # noqa: E402

EEG_ECG_STEP_MODE_S = -0.804
LOSS_S = 0.0177
BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
GOOD = (0.3, 0.15)


def model(tr, sign, loss=0.0):
    st = tr.record_offsets[: tr.record_times.size]
    clk = stamp_clock(tr.record_times, st, float(tr.srate), correction_sign=sign, event_loss_s=loss)
    t_model = corrected_times(clk, st, tr.record_sizes[: st.size])
    return clk, t_model


def burst_lags(t_ecg, x_ecg, fs_e, t_eeg, x_eeg, fs_b):
    t0 = float(np.floor(min(t_ecg[0], t_eeg[0])))
    n = int((max(t_ecg[-1], t_eeg[-1]) - t0) / DT) + 1
    env = {"ecg": envelope(t_ecg, x_ecg, fs_e, BAND["ecg"], t0, n), "eeg1": envelope(t_eeg, x_eeg, fs_b, BAND["eeg1"], t0, n)}
    w = np.array([r for r in burst_probe(env, 600, 300, 20).get("ecg~eeg1", [])
                  if r[2] >= GOOD[0] and r[2] - r[3] >= GOOD[1] and -2500 < r[1] < 2500])
    return (w[:, 0] + t0, w[:, 1] / 1000) if len(w) else (np.empty(0), np.empty(0))


def stretch_fit(t, y, pauses):
    st = np.searchsorted(pauses, t)
    u = np.unique(st)
    X = np.column_stack([t] + [(st == s).astype(float) for s in u])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    loo = []
    for i in range(t.size):
        m = np.ones(t.size, bool); m[i] = False
        if (st[m] == st[i]).sum() < 2:
            loo.append(np.nan); continue
        cols = np.any(X[m] != 0, axis=0)
        b = np.linalg.lstsq(X[m][:, cols], y[m], rcond=None)[0]
        loo.append(float(X[i, cols] @ b - y[i]))
    return beta, st, u, np.array(loo)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--case", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    v = read_vital(Path(a.root) / "vital_files" / f"{a.case:04d}.vital",
                   track_names={"SNUADC/ECG_II", "SNUADC/CVP", "BIS/EEG1_WAV", "Primus/AWP"})
    ecg, cvp, eeg, awp = (v.tracks[n] for n in ("SNUADC/ECG_II", "SNUADC/CVP", "BIS/EEG1_WAV", "Primus/AWP"))
    z = float(ecg.record_times[0])
    # each clock against the stamp-model timeline
    ce, te_m = model(ecg, -1, LOSS_S)
    cb, tb_m = model(eeg, +1)
    _, tc_m = model(cvp, -1, LOSS_S)
    te_r, tb_r, tc_r = sample_times(ecg), sample_times(eeg), sample_times(cvp)
    tb_s = tb_m - EEG_ECG_STEP_MODE_S                       # BIS synchronised: model plus the mode's constant
    xe = np.nan_to_num(ecg.physical().astype(np.float32))[: te_r.size]
    xb = np.nan_to_num(eeg.physical().astype(np.float32))[: tb_r.size]
    xc = cvp.physical().astype(np.float64)[: tc_r.size]
    snu_err = (te_r - te_m[: te_r.size])[:: ecg.record_sizes[0]]
    snu_t = te_m[: te_r.size][:: ecg.record_sizes[0]]
    bis_err = (tb_r - tb_s[: tb_r.size])[:: 128]
    bis_t = tb_s[: tb_r.size][:: 128]
    # bursts before (recorded stamps) and after (synchronised)
    lb_t, lb = burst_lags(te_r, xe, float(ecg.srate), tb_r, xb, float(eeg.srate))
    la_t, la = burst_lags(te_m[: te_r.size], xe, float(ecg.srate), tb_s[: tb_r.size], xb, float(eeg.srate))
    # Primus: AWP-to-CVP lag on recorded stamps (before) and on the stamp model (fit for after)
    ta = sample_times(awp)
    xa = np.nan_to_num(awp.physical().astype(np.float64))[: ta.size]
    on = P.breath_onsets(ta, xa, float(awp.srate))
    pa = ta[1:][np.diff(ta) > 5]
    pr_t, pr = P.cvp_lag(ta, xa, tc_r, xc, on)
    pm_t, pm = P.cvp_lag(ta, xa, tc_m[: tc_r.size], xc, on)
    beta, st, u, loo = stretch_fit(pm_t - z, pm, pa - z)
    off = dict(zip(u, beta[1:]))
    grid = np.linspace(pm_t[0], pm_t[-1], 2000)
    gst = np.searchsorted(pa, grid)
    fit_line = np.array([beta[0] * (g - z) + off.get(s, np.nan) for g, s in zip(grid, gst)])
    phys = float(np.nanmedian(pm))
    prim_err = fit_line - phys
    # figure
    h = lambda t: (np.asarray(t) - z) / 3600
    fig, ax = plt.subplots(3, 2, figsize=(12, 8.2), sharex=True)
    lim = [max(0.6, 1.1 * np.nanmax(np.abs(np.r_[snu_err, bis_err, prim_err])))] * 2
    ax[0, 0].plot(h(snu_t), snu_err, color=BLUE, lw=1.2, label="SNUADC record stamps")
    ax[0, 0].plot(h(bis_t), bis_err, color=ORANGE, lw=1.2, label="BIS EEG stamps")
    ax[0, 0].plot(h(grid), prim_err, color=AQUA, lw=1.2, label="Primus airway-pressure stamps")
    for k, (c, w, lab) in enumerate([(BLUE, 0.003, "SNUADC ±3 ms"), (ORANGE, 0.042, "BIS ±42 ms"),
                                     (AQUA, float(np.nanmedian(np.abs(loo))), "Primus")]):
        ax[0, 1].fill_between(h([z, max(snu_t[-1], bis_t[-1], grid[-1])]), -w, w, color=c, alpha=0.25, lw=0)
    ax[0, 1].text(0.02, 0.92, "typical error left: SNUADC about 3 ms, BIS 42 ms, Primus "
                  f"{1000 * np.nanmedian(np.abs(loo)):.0f} ms (bands)", transform=ax[0, 1].transAxes, fontsize=8, color=INK)
    ax[0, 0].legend(fontsize=7, frameon=False, loc="lower left", labelcolor=INK)
    for j in (0, 1):
        ax[0, j].set_ylim(-lim[0], lim[0])
    ax[0, 0].set_ylabel("clock time minus\nsynchronised time (s)", fontsize=8, color=MUTED)
    # BIS vs SNUADC
    yl = max(1.0, 1.1 * np.nanmax(np.abs(lb))) if lb.size else 1.0
    ax[1, 0].plot(h(lb_t), lb, "o", ms=4, color=ORANGE, mec="white", mew=0.6)
    ax[1, 1].plot(h(la_t), la, "o", ms=4, color=ORANGE, mec="white", mew=0.6)
    ax[1, 0].set_ylabel("EEG minus ECG, same\nburst (s)", fontsize=8, color=MUTED)
    ax[1, 1].text(0.02, 0.92, f"spread after: {1000 * np.nanmedian(np.abs(la - np.median(la))):.0f} ms (median |deviation|)"
                  if la.size else "", transform=ax[1, 1].transAxes, fontsize=8, color=INK)
    for j in (0, 1):
        ax[1, j].set_ylim(-yl, yl)
        ax[1, j].axhline(0, color=MUTED, lw=0.6)
    # Primus vs SNUADC
    yp = 1.1 * max(np.nanmax(np.abs(pr - np.nanmedian(pr))), 0.5) if pr.size else 1.0
    ax[2, 0].plot(h(pr_t), pr - np.nanmedian(pr), "o", ms=4, color=AQUA, mec="white", mew=0.6)
    ax[2, 1].plot(h(pm_t), loo, "o", ms=4, color=AQUA, mec="white", mew=0.6)
    ax[2, 0].set_ylabel("airway-to-CVP lag,\nminus its median (s)", fontsize=8, color=MUTED)
    ax[2, 1].text(0.02, 0.92, f"spread after: {1000 * np.nanmedian(np.abs(loo)):.0f} ms (median |leave-one-out error|)",
                  transform=ax[2, 1].transAxes, fontsize=8, color=INK)
    for j in (0, 1):
        ax[2, j].set_ylim(-yp, yp)
        ax[2, j].axhline(0, color=MUTED, lw=0.6)
        ax[2, j].set_xlabel("hours into the recording", fontsize=8, color=MUTED)
    for p in pa:
        for i in range(3):
            for j in (0, 1):
                ax[i, j].axvline(h(p), color=AQUA, lw=0.5, ls=":", alpha=0.6)
    titles = [["A. Each clock against the synchronised timeline, as recorded", "B. After synchronisation"],
              ["C. BIS against SNUADC, from shared bursts, as recorded", "D. The same bursts, after synchronisation"],
              ["E. Primus against SNUADC, from the breath cycle, as recorded", "F. After synchronisation"]]
    for i in range(3):
        for j in (0, 1):
            ax[i, j].set_title(titles[i][j], fontsize=9, loc="left", color=INK)
            for s in ("top", "right"):
                ax[i, j].spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax[i, j].spines[s].set_color(MUTED)
            ax[i, j].tick_params(colors=MUTED, labelsize=8)
            ax[i, j].grid(axis="y", color=GRID, lw=0.6)
            ax[i, j].set_axisbelow(True)
    fig.suptitle(f"Recording {a.case}: what synchronising the clocks does (dotted lines: Primus airway-pressure pauses)",
                 fontsize=10, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(a.out, dpi=150, facecolor="white")
    summ = {"case": a.case, "snuadc_rate_ppm": ce["rate_ppm"], "bis_rate_ppm": cb["rate_ppm"],
            "primus_slope_ppm": beta[0] * 1e6, "burst_windows": [int(lb.size), int(la.size)],
            "burst_abs_dev_ms_before_after": [float(1000 * np.median(np.abs(lb - np.median(lb)))) if lb.size else None,
                                              float(1000 * np.median(np.abs(la - np.median(la)))) if la.size else None],
            "burst_median_ms_after": float(1000 * np.median(la)) if la.size else None,
            "primus_windows": int(pm.size), "primus_loo_abs_ms": float(1000 * np.nanmedian(np.abs(loo))),
            "primus_raw_abs_dev_ms": float(1000 * np.nanmedian(np.abs(pr - np.nanmedian(pr)))) if pr.size else None}
    print(json.dumps(summ))
    if a.json:
        Path(a.json).write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
