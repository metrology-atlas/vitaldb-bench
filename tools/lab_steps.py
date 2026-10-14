# SPDX-License-Identifier: MIT
"""Do arterial-line events and timing steps line up with blood-draw times? One case, this machine only.

    python tools/lab_steps.py --root R --case 3255 --out out.json --fig fig.png

Whole case, split at breaks in the samples. Per beat: R-peak (ECG_II), PPG steepest upstroke (PLETH),
ART steepest upstroke (as art_export). Then:

1. Line events from the ART waveform itself (2 s windows): flat (pulse pressure < 5 mmHg), high (median
   > 200 mmHg, e.g. a pressurised flush), low (median < 15 mmHg, e.g. open to air / zeroing). Merged
   into events with start and duration.
2. Timing steps (median of 15 beats after minus before, |step| > 10 ms, at least 60 s apart) in ECG->ART,
   ECG->PPG and ART->PPG. A step in ECG->ART and ART->PPG of opposite sign, with ECG->PPG unchanged, is the
   arterial side moving; one in ECG->PPG and ART->PPG with ECG->ART unchanged is the PPG side.
3. Lab draws from lab_data.csv (results within 60 s grouped into one draw; times from the case start).
   For each draw: the nearest line event and arterial-side step, before and after. Control: the same
   with the draw times moved at random (1,000 circular shifts), for the share of draws with an event in
   the window ``--window`` minutes (default -15 to +5) around them.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "physiosync"))
import itertools

from pat import ECG, PPG, r_peaks, sample_times
from pat_res import ecg_subsample, ppg_onsets
from physiosync import match_beats
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital

warnings.filterwarnings("ignore")
ART = "SNUADC/ART"
FS = 500.0


def segments(t: np.ndarray, min_s: float = 300.0) -> list[tuple[int, int]]:
    gap = np.flatnonzero(np.abs(np.diff(t) - 1 / FS) > 0.01 / FS)
    b = np.r_[0, gap + 1, t.size]
    return [(int(x), int(y)) for x, y in itertools.pairwise(b) if y - x >= min_s * FS]


def art_upstrokes(ta: np.ndarray, x: np.ndarray, r_t: np.ndarray) -> np.ndarray:
    w = int(10 * FS)
    nw = x.size // w
    X = x[: nw * w].reshape(nw, w)
    med = np.median(X, axis=1)
    pp = np.quantile(X, 0.95, axis=1) - np.quantile(X, 0.05, axis=1)
    ok_w = (med > 30) & (med < 200) & (pp > 15) & (pp < 150)
    ok_s = np.r_[np.repeat(ok_w, w), np.zeros(x.size - nw * w, bool)]
    sos = signal.butter(2, (0.5, 10), btype="bandpass", fs=FS, output="sos")
    d = np.gradient(signal.sosfiltfilt(sos, np.nan_to_num(x)))
    a_t = np.full(r_t.size, np.nan)
    for k, rt in enumerate(r_t):
        i_a, i_b = int(np.searchsorted(ta, rt + 0.05)), int(np.searchsorted(ta, rt + 0.45))
        if i_a < 2 or i_b + 2 >= x.size or not ok_s[i_a:i_b].all():
            continue
        m = i_a + int(np.argmax(d[i_a:i_b]))
        if d[m] <= 0 or m in (i_a, i_b - 1):
            continue
        den = d[m - 1] - 2 * d[m] + d[m + 1]
        off = 0.5 * (d[m - 1] - d[m + 1]) / den if den != 0 else 0.0
        a_t[k] = ta[m] + float(np.clip(off, -0.5, 0.5)) / FS
    return a_t


def line_events(ta: np.ndarray, x: np.ndarray) -> list[dict]:
    w = int(2 * FS)
    nw = x.size // w
    X = x[: nw * w].reshape(nw, w)
    med = np.median(X, axis=1)
    pp = np.quantile(X, 0.95, axis=1) - np.quantile(X, 0.05, axis=1)
    kind = np.where(med > 200, "high", np.where(med < 15, "low", np.where(pp < 5, "flat", "")))
    out: list[dict] = []
    for i, k in enumerate(kind):
        if not k:
            continue
        t0 = float(ta[i * w])
        if out and t0 - out[-1]["end"] <= 4.0:
            out[-1]["end"] = t0 + 2.0
            out[-1]["kinds"] = sorted(set(out[-1]["kinds"]) | {str(k)})
        else:
            out.append({"start": t0, "end": t0 + 2.0, "kinds": [str(k)]})
    return out


def steps(t: np.ndarray, y: np.ndarray, thr: float = 0.010, sep: float = 60.0, k: int = 15) -> np.ndarray:
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    if y.size < 3 * k:
        return np.empty((0, 2))
    w = np.lib.stride_tricks.sliding_window_view(y, k)
    med = np.median(w, axis=1)
    s = np.full(y.size, np.nan)
    s[k : y.size - k + 1] = med[k:] - med[:-k]
    tt = np.r_[t[0], 0.5 * (t[1:] + t[:-1])]
    # breaks between beats more than 10 s apart are not steps
    s[np.r_[False, np.diff(t) > 10]] = np.nan
    cand = np.flatnonzero(np.abs(np.nan_to_num(s)) > thr)
    picked: list[int] = []
    for i in cand[np.argsort(-np.abs(s[cand]))]:
        if all(abs(tt[i] - tt[j]) >= sep for j in picked):
            picked.append(int(i))
    picked.sort()
    return np.array([[tt[i], s[i]] for i in picked]).reshape(-1, 2)


def step_at(t: np.ndarray, y: np.ndarray, x: float, k: int = 15) -> float:
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    i = int(np.searchsorted(t, x))
    if i < k or i + k > t.size:
        return float("nan")
    return float(np.median(y[i : i + k]) - np.median(y[i - k : i]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--case", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig", default=None)
    ap.add_argument("--window", type=float, nargs=2, default=(-15.0, 5.0))
    a = ap.parse_args()
    root = Path(a.root)
    v = read_vital(root / "vital_files" / f"{a.case:04d}.vital", track_names={ECG, PPG, ART})
    t0 = v.origin
    e, p, ar = v.tracks[ECG], v.tracks[PPG], v.tracks[ART]
    te, tp, ta = sample_times(e), sample_times(p), sample_times(ar)
    xe, xa = e.physical().astype(np.float64), ar.physical().astype(np.float64)
    R, P = [], []
    for s0, s1 in segments(te):
        rp = r_peaks(xe[s0:s1], FS)
        rt = te[s0] + ecg_subsample(xe[s0:s1], rp, FS) / FS
        pa, pb = int(np.searchsorted(tp, te[s0])), int(np.searchsorted(tp, te[s1 - 1]))
        _, up_f = ppg_onsets(np.asarray(p.values[pa:pb]), FS)
        pt = tp[pa] + up_f / FS
        i, j, _ = match_beats(rt, pt)
        R.append(rt[i])
        P.append(pt[j])
    r_t, p_t = np.concatenate(R), np.concatenate(P)
    a_t = art_upstrokes(ta, xa, r_t)
    ea, ep, apg = a_t - r_t, p_t - r_t, p_t - a_t
    ev = line_events(ta, xa)
    st = {name: steps(r_t, y) for name, y in (("ecg_art", ea), ("ecg_ppg", ep), ("art_ppg", apg))}
    # classify each step: which side moved
    rows = []
    for name, arr in st.items():
        for x, s in arr:
            sz = {n: step_at(r_t, y, x) for n, y in (("ecg_art", ea), ("ecg_ppg", ep), ("art_ppg", apg))}
            side = ("arterial" if abs(sz["ecg_art"]) > 0.010 and abs(sz["ecg_ppg"]) <= 0.010
                    else "ppg" if abs(sz["ecg_ppg"]) > 0.010 and abs(sz["ecg_art"]) <= 0.010
                    else "both" if abs(sz["ecg_art"]) > 0.010 else "unclear")
            rows.append({"t": round(float(x - t0), 1), "found_in": name, "side": side,
                         **{k: round(v_ * 1000, 1) for k, v_ in sz.items()}})
    rows.sort(key=lambda r: r["t"])
    art_steps = sorted({r["t"] for r in rows if r["side"] in ("arterial", "both")})
    lab = pd.read_csv(root / "lab_data.csv")
    dts = np.sort(lab.loc[lab.caseid == a.case, "dt"].to_numpy(dtype=float))
    span = float(r_t[-1] - t0)
    dts = dts[(dts >= r_t[0] - t0) & (dts <= span)]
    draws: list[float] = []
    for d in dts:
        if not draws or d - draws[-1] > 60:
            draws.append(float(d))
    ev_t = np.array([x["start"] - t0 for x in ev])
    lo, hi = a.window[0] * 60, a.window[1] * 60

    def share(times: np.ndarray, marks: np.ndarray) -> float:
        if not len(times) or not len(marks):
            return 0.0
        return float(np.mean([np.any((marks - d >= lo) & (marks - d <= hi)) for d in times]))

    per_draw = []
    for d in draws:
        dev = ev_t - d
        dst = np.array(art_steps) - d
        near_ev = dev[(dev >= lo) & (dev <= hi)]
        near_st = dst[(dst >= lo) & (dst <= hi)]
        per_draw.append({"draw_min": round(d / 60, 1),
                         "line_events_min": [round(float(x) / 60, 1) for x in near_ev],
                         "line_event_kinds": [ev[int(i)]["kinds"] for i in np.flatnonzero((dev >= lo) & (dev <= hi))],
                         "line_event_s": [round(ev[int(i)]["end"] - ev[int(i)]["start"]) for i in np.flatnonzero((dev >= lo) & (dev <= hi))],
                         "arterial_steps_min": [round(float(x) / 60, 1) for x in near_st]})
    rng = np.random.default_rng(0)
    t_lo, t_hi = float(r_t[0] - t0), span
    ctrl_ev, ctrl_st = [], []
    for _ in range(1000):
        sh = (np.array(draws) - t_lo + rng.uniform(0, t_hi - t_lo)) % (t_hi - t_lo) + t_lo
        ctrl_ev.append(share(sh, ev_t))
        ctrl_st.append(share(sh, np.array(art_steps)))
    obs_ev, obs_st = share(np.array(draws), ev_t), share(np.array(draws), np.array(art_steps))
    res = {
        "case": a.case, "hours": round(span / 3600, 2), "beats": int(r_t.size),
        "art_coverage": round(float(np.isfinite(a_t).mean()), 3), "draws": len(draws), "window_min": list(a.window),
        "line_events": len(ev), "line_event_kinds": {k: sum(k in x["kinds"] for x in ev) for k in ("flat", "high", "low")},
        "steps_by_side": {s: sum(r["side"] == s for r in rows) for s in ("arterial", "ppg", "both", "unclear")},
        "share_draws_with_line_event": round(obs_ev, 3),
        "control_line_event": {"mean": round(float(np.mean(ctrl_ev)), 3), "p_ge": round(float(np.mean(np.array(ctrl_ev) >= obs_ev)), 4)},
        "share_draws_with_arterial_step": round(obs_st, 3),
        "control_arterial_step": {"mean": round(float(np.mean(ctrl_st)), 3), "p_ge": round(float(np.mean(np.array(ctrl_st) >= obs_st)), 4)},
        "per_draw": per_draw, "steps": rows,
        "line_events_list": [{"start_min": round((x["start"] - t0) / 60, 2), "s": round(x["end"] - x["start"]), "kinds": x["kinds"]} for x in ev],
    }
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in list(res)[:13]}))
    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(3, 1, 1)
        for y, lab_, col in ((ea, "ECG→ART", "tab:green"), (ep - np.nanmedian(ep) + np.nanmedian(ea), "ECG→PPG (shifted)", "0.5"),
                             (apg - np.nanmedian(apg) + np.nanmedian(ea) - 0.1, "ART→PPG (shifted)", "tab:red")):
            tm = (r_t - t0) / 60
            ok = np.isfinite(y)
            n = ok.sum() // 15
            ax.plot(tm[ok][: 15 * n].reshape(n, 15).mean(1), 1000 * np.median(y[ok][: 15 * n].reshape(n, 15), 1), ".", ms=2,
                    color=col, label=lab_)
        for d in draws:
            ax.axvline(d / 60, color="tab:blue", lw=0.8)
        for x in ev:
            ax.axvspan((x["start"] - t0) / 60, (x["end"] - t0) / 60 + 0.3, color="tab:orange", alpha=0.5)
        ax.set_ylabel("ms (15-beat medians)")
        ax.set_title(f"case {a.case}: blue = lab draw times, orange = ART line events (flat / high / low)", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        sel = [d for d in draws][:6]
        for i, d in enumerate(sel):
            ax2 = fig.add_subplot(3, 3, 4 + i)
            m = (ta - t0 >= d + lo) & (ta - t0 <= d + hi)
            ax2.plot((ta[m][::25] - t0 - d) / 60, xa[m][::25], lw=0.3, color="tab:green")
            ax2.axvline(0, color="tab:blue", lw=0.8)
            ax2.set_title(f"draw at {d / 60:.0f} min: ART (mmHg)", fontsize=8)
            ax2.set_ylim(-20, 320)
            ax2.tick_params(labelsize=7)
        fig.supxlabel("top: minutes from case start; small panels: minutes from the lab time", fontsize=8)
        fig.tight_layout()
        fig.savefig(a.fig, dpi=110)


if __name__ == "__main__":
    main()
