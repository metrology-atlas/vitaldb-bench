"""Worked example: pulse arrival time (PAT) in two correction steps, smoothed by maximum likelihood.

    python examples/pulse_arrival_time.py --root R --cases 1 2 3 --out OUT [--minutes 60] [--figure CASE ...]
    python examples/pulse_arrival_time.py --out OUT --cohort-figure      # after the cases have run

Per case, on up to ``--minutes`` of SNUADC ECG lead II and PLETH (and arterial pressure where there is a line),
starting 10 minutes into their overlap:

1. **Fiducials.** R peaks in ECG lead II; the steepest upstroke after each R peak in the PLETH (and in the arterial
   pressure), band-passed 0.5-10 Hz with parabolic sub-sample refinement (as ``tools/pat.py``).
2. **Step 1, sawtooth removal.** ``physiosync.fit_ecg_ppg_delay`` fits the PLETH's non-physiological delay
   D(t) = offset + sawtooth(t): a creep at a fixed rate, reset by one 30 Hz frame at each slip (arterial
   upstrokes, where present, as the reference for the slips). Removing only the sawtooth part leaves the PAT at
   its stored level.
3. **Step 2, global delay.** Subtract one device delay for every recording, 0.445 s: the stored delay's median over
   the release (0.715 s, tools/pat_offset_census.py) minus an adopted mean finger PAT of 0.27 s. Differences between
   patients are kept; the absolute level rests on the adopted 0.27 s.
4. **Maximum-likelihood smoothing** (after Ruffolo et al. 2025). For each 30 s window the latent PAT is the value
   that maximises the likelihood of the window's beats under the recording's per-beat error model
   (``physiosync.pat_measurement_model``: the PLETH fiducial's frame rounding or jitter and the ECG's sample
   rounding), evaluated by Monte Carlo from ``physiosync.error_samples``, with a 2% uniform floor for
   mis-detected beats. Where the fiducial is frame-locked the error is a 33 ms plateau, and the maximum-likelihood
   value is the centre of the plateau the beats agree on, where a median is pulled by the rounding.

Writes OUT/summary.json (per case: beats, slips, tiers, offset, within-case spread at each step; no signal
values), OUT/windows/<case>.npz (30 s window estimates; kept on this machine) and, for ``--figure`` cases,
OUT/pat_<case>.png. ``--cohort-figure`` draws OUT/pat_cohort.png from the windows of all cases run. A case id
appears in a figure only because the example discusses that case.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from pat import r_peaks, sample_times  # noqa: E402
from physiosync import error_samples, fiducial_error_model, fit_ecg_ppg_delay, pat_measurement_model  # noqa: E402
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital  # noqa: E402

ECG, PPG, ART = "SNUADC/ECG_II", "SNUADC/PLETH", "SNUADC/ART"
FS = 500.0
ADOPTED = 0.27      # adopted population mean finger PAT (anaesthetised adults; atlas study ppg-offset-reference)
# One device delay for every recording (PAT offset census, tools/pat_offset_census.py, 18 Oct 2026: 5,536
# recordings; stored ECG->PLETH delay 715 ms in the median recording, one peak, no clusters by setup or case order).
# Global delay = stored median minus the adopted finger PAT: 0.715 - 0.270 = 0.445 s.
STORED_MEDIAN = 0.715
GLOBAL_DELAY = STORED_MEDIAN - ADOPTED
WIN = 30.0          # s per smoothing window
COLOURS = {"raw": "#9e9e9e", "saw": "#3f51b5", "cor": "#e65100", "art": "#2e7d32"}


def upstrokes(t, x, r_t, lo, hi):
    """Steepest upstroke between ``lo`` and ``hi`` s after each R peak: (time, sample index); NaN/-1 if none."""
    sos = signal.butter(2, (0.5, 10), btype="bandpass", fs=FS, output="sos")
    d = np.gradient(signal.sosfiltfilt(sos, x))
    out, idx = np.full(r_t.size, np.nan), np.full(r_t.size, -1)
    for k, rt in enumerate(r_t):
        a, b = np.searchsorted(t, rt + lo), np.searchsorted(t, rt + hi)
        if b - a < 10 or a < 2 or b > d.size - 2:
            continue
        m = a + int(np.argmax(d[a:b]))
        if d[m] <= 0:
            continue
        den = d[m - 1] - 2 * d[m] + d[m + 1]
        out[k] = t[m] + float(np.clip(0.5 * (d[m - 1] - d[m + 1]) / den if den else 0.0, -0.5, 0.5)) / FS
        idx[k] = m
    return out, idx


def error_density(rec: dict, n: int = 40000, seed: int = 0):
    """Per-beat observed-minus-latent PAT density (fiducial rounding or jitter), by Monte Carlo."""
    e = error_samples(rec, np.zeros(1), n, rng=np.random.default_rng(seed), components=("quantisation",))[:, 0]
    edges = np.arange(-0.1, 0.1 + 1e-9, 0.00025)
    h, _ = np.histogram(e, bins=edges, density=True)
    h = np.convolve(h, np.ones(3) / 3, mode="same")
    return (edges[:-1] + edges[1:]) / 2, h


def ml_windows(t, y, dens, t0, floor=0.02, span=0.6):
    """Maximum-likelihood latent PAT per 30 s window from ``t0``; returns window index, centre and estimate (s)."""
    ex, eh = dens
    out_k, out_t, out_y = [], [], []
    good = np.isfinite(y)
    t, y = t[good], y[good]
    k = np.floor((t - t0) / WIN).astype(int)
    for w in np.unique(k):
        yy = y[k == w]
        if yy.size < 15:
            continue
        c = np.median(yy)
        grid = c + np.arange(-0.05, 0.05 + 1e-9, 0.00025)
        res = yy[None, :] - grid[:, None]
        f = np.interp(res, ex, eh, left=0.0, right=0.0)
        ll = np.log((1 - floor) * f + floor / span).sum(axis=1)
        out_k.append(w)
        out_t.append(t[k == w].mean())
        out_y.append(grid[int(np.argmax(ll))])
    return np.array(out_k), np.array(out_t), np.array(out_y)


def paired(u, v):
    """Differences of two window series on the windows they share."""
    _, iu, iv = np.intersect1d(u[0], v[0], return_indices=True)
    return u[2][iu] - v[2][iv]


def robust_sd(x):
    x = np.asarray(x)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def one_case(root: Path, case: int, minutes: float):
    v = read_vital(root / "vital_files" / f"{case:04d}.vital", track_names={ECG, PPG, ART})
    tr = {k: v.tracks.get(k) for k in (ECG, PPG, ART)}
    if any(tr[k] is None or not tr[k].record_times.size or tr[k].srate != FS for k in (ECG, PPG)):
        return {"case": case, "skip": "no 500 Hz ECG lead II and PLETH"}, None
    use = [ECG, PPG] + ([ART] if tr[ART] is not None and tr[ART].record_times.size and tr[ART].srate == FS else [])
    ts = {k: sample_times(tr[k]) for k in use}
    t0 = max(ts[k][0] for k in (ECG, PPG)) + 600
    t1 = min(min(ts[k][-1] for k in (ECG, PPG)), t0 + 60 * minutes)
    if t1 - t0 < 600:
        return {"case": case, "skip": "less than 10 minutes of overlap"}, None
    seg = {}
    for k in use:
        a, b = np.searchsorted(ts[k], t0), np.searchsorted(ts[k], t1)
        seg[k] = (ts[k][a:b], np.nan_to_num(tr[k].physical()[a:b].astype(float)))
    te, xe = seg[ECG]
    r_t = te[r_peaks(xe, FS)]
    p_t, p_i = upstrokes(*seg[PPG], r_t, 0.10, 0.80)
    a_t = upstrokes(*seg[ART], r_t, 0.05, 0.45)[0] if ART in seg else np.full(r_t.size, np.nan)
    ok = np.isfinite(p_t)
    # an arterial line counts only if most beats have a plausible arterial upstroke
    has_art = ART in seg and np.mean(np.isfinite(a_t) & ok) > 0.7
    if ok.sum() < 500:
        return {"case": case, "skip": "too few beats with a PLETH upstroke"}, None
    r, p, ix = r_t[ok], p_t[ok], p_i[ok]
    a = a_t[ok] if has_art else None
    m = fit_ecg_ppg_delay(r, p, adopted_pat_s=ADOPTED, frame_s=1 / 30, matched=True,
                          ref_t=a if has_art else None)
    saw = m.shift(p) - m.offset_s                     # the sawtooth part of the delay
    # step 2: one global delay for every recording, so differences in PAT between patients are kept
    pat = {"raw": p - r, "saw": p - saw - r, "cor": p - saw - r - GLOBAL_DELAY}
    rec = pat_measurement_model(m, ecg_fs=FS, ppg_fs=FS, r_t=r, ppg_t=p)
    noise = [s["beat_noise_s"] for s in rec["segments"] if s.get("beat_noise_s")]
    ppg_err = fiducial_error_model(ix, FS, frame_samples=FS / 30, pat_noise_s=float(np.median(noise)) if noise else None,
                                   other_sd_s=0.002 / np.sqrt(12), ibi_range=(150, 1000))
    rec = pat_measurement_model(m, ecg_fs=FS, ppg_fs=FS, ppg_error=ppg_err, r_t=r, ppg_t=p)
    dens = error_density(rec)
    win = {k: ml_windows(r, y, dens, t0) for k, y in pat.items()}
    if has_art:
        win["art"] = ml_windows(r, a - r, dens, t0)
    med = {k: float(np.median(y)) for k, y in pat.items()}
    res = {"case": case, "minutes": round((t1 - t0) / 60, 1), "beats": int(ok.sum()), "arterial_line": bool(has_art),
           "slips": len(m.slips_t), "tiers": "".join(m.tooth_tier), "ppg_error": ppg_err.get("rounding"),
           "offset_ms": round((med["raw"] - med["cor"]) * 1000, 1),
           "within_case_sd_ms": {k: round(robust_sd(win[k][2]) * 1000, 2) for k in pat},
           "raw_median_ms": round(med["raw"] * 1000, 1)}
    if has_art:
        res["minus_art_sd_ms"] = {k: round(robust_sd(paired(win[k], win["art"])) * 1000, 2) for k in pat}
    return res, {"t0": t0, "win": win, "slips": [s - t0 for s in m.slips_t]}


MISPAIRED_BELOW_MS = 600   # stored median below this: pulses paired one beat out (fast heart rate or doubled R peaks)


def usable(v: dict) -> bool:
    """A case enters the cohort figures if it ran and its pulses were paired with the right beat. The stored
    ECG->PLETH delay is about 0.7 s in every recording (PAT offset census), longer than one RR interval at fast heart
    rates, where this example's nearest-upstroke search pairs a pulse with the next beat and the stored median comes
    out near 0.45-0.55 s."""
    return "within_case_sd_ms" in v and v.get("raw_median_ms", 0) >= MISPAIRED_BELOW_MS


def figure(case: int, series: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w, t0 = series["win"], series["t0"]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    lines = [("raw", "ECG → finger (PLETH), as stored"), ("saw", "ECG → finger, step 1: sawtooth removed"),
             ("cor", f"ECG → finger, step 2: global delay removed ({GLOBAL_DELAY * 1000:.0f} ms)")]
    if "art" in w:
        lines.append(("art", "ECG → arterial line (ART): no correction needed"))
    for k, lab in lines:
        _, x, y = w[k]
        ax.plot((x - t0) / 60, y * 1000, lw=1.0, color=COLOURS[k], label=lab)
    for s in series["slips"]:
        ax.axvline(s / 60, color="0.75", lw=0.6, ls=":")
    ax.set_ylim(0, None)
    ax.set_ylabel("pulse arrival time (ms)")
    ax.set_xlabel("minutes from the start of the stretch")
    ax.set_title(f"Case {case}: 30 s maximum-likelihood estimates; dotted lines, frame slips in the PLETH",
                 fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def cohort_figure(out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = json.loads((out / "summary.json").read_text())
    ok = [v for v in s.values() if usable(v)]
    dev = {"raw": [], "saw": []}
    for v in ok:
        z = np.load(out / "windows" / f"{v['case']}.npz")
        for k in dev:
            y = z[k]
            dev[k].extend(((y - np.median(y)) * 1000).tolist())
    # where there is an arterial line and a tooth was corrected with confidence: finger minus arterial PAT,
    # each case centred, on the 30 s windows both series have (saved as equal-length arrays per window)
    gap = {"raw": [], "saw": []}
    for v in ok:
        if "minus_art_sd_ms" not in v or not any(t in "AB" for t in v["tiers"]):
            continue
        z = np.load(out / "windows" / f"{v['case']}.npz")
        kw = {k: z[k + "_k"] for k in ("raw", "saw", "art") if k + "_k" in z}
        if len(kw) < 3:
            continue
        for k in gap:
            _, iu, iv = np.intersect1d(kw[k], kw["art"], return_indices=True)
            d = (z[k][iu] - z["art"][iv]) * 1000
            gap[k].extend((d - np.median(d)).tolist())
    def inside(x):
        x = np.asarray(x)
        return x[np.abs(x) <= 40]
    beyond = {k: int(np.sum(np.abs(dev[k]) > 40)) for k in dev}
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))
    bins = np.arange(-40, 40.5, 1)
    ax[0].hist(inside(dev["raw"]), bins=bins, color=COLOURS["raw"], alpha=0.8, label="as stored")
    ax[0].hist(inside(dev["saw"]), bins=bins, histtype="step", color=COLOURS["saw"], lw=1.4,
               label="step 1: sawtooth removed")
    ax[0].set_xlabel("PAT minus the case's median (ms), 30 s windows")
    ax[0].set_ylabel("windows")
    ax[0].legend(frameon=False, fontsize=7)
    off = [v["offset_ms"] for v in ok]
    raw_med = [v["raw_median_ms"] for v in ok]
    if gap["raw"]:
        ax[2].hist(inside(gap["raw"]), bins=bins, color=COLOURS["raw"], alpha=0.8, label="as stored")
        ax[2].hist(inside(gap["saw"]), bins=bins, histtype="step", color=COLOURS["saw"], lw=1.4,
                   label="step 1: sawtooth removed")
        ax[2].set_xlabel("finger minus arterial PAT, centred (ms)")
        ax[2].set_ylabel("windows")
        ax[2].legend(frameon=False, fontsize=7)
        ax[2].set_title("arterial line, tooth corrected (A/B)", fontsize=8, loc="left")
    ax[0].set_title("every case", fontsize=8, loc="left")
    ax[1].set_title("every case", fontsize=8, loc="left")
    ax[1].hist(raw_med, bins=30, color=COLOURS["raw"], alpha=0.8, label="as stored")
    ax[1].hist([m_ - GLOBAL_DELAY * 1000 for m_ in raw_med], bins=30, histtype="step", lw=1.5, color=COLOURS["cor"],
               label=f"step 2: minus the global {GLOBAL_DELAY * 1000:.0f} ms")
    ax[1].set_xlabel("case median ECG → finger PAT (ms)")
    ax[1].set_ylabel("cases")
    ax[1].legend(frameon=False, fontsize=7)
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{len(ok)} cases with ECG lead II and PLETH, 60 min each; {beyond['raw']:,} of {len(dev['raw']):,} "
                 "windows more than 40 ms from their case's median are not shown", fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "pat_cohort.png", dpi=140)
    plt.close(fig)
    print(json.dumps({"cases": len(ok), "offset_ms_median": float(np.median(off)),
                      "offset_ms_p10_p90": [float(np.percentile(off, 10)), float(np.percentile(off, 90))],
                      "within_sd_ms_median": {k: float(np.median([v["within_case_sd_ms"][k] for v in ok])) for k in ("raw", "saw")},
                      "pooled_window_sd_ms": {k: robust_sd(dev[k]) for k in dev},
                      "finger_minus_art_sd_ms": {k: robust_sd(gap[k]) if gap[k] else None for k in gap}}))


ADULT_RANGE_MS = (200, 285)   # awake adults, shared-clock studies (atlas: ppg-offset-reference); to be verified


def distribution_figure(out: Path) -> None:
    """Actual PAT values per 30 s window, as stored and after both steps, with the arterial-line PAT for reference.

    Negative corrected values (windows where the correction failed) are left out and counted, not drawn at zero.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = json.loads((out / "summary.json").read_text())
    ok = [v for v in s.values() if usable(v)]
    acc: dict[str, list] = {"raw": [], "cor": [], "art": []}
    n = {k: 0 for k in acc}
    for v in ok:
        z = np.load(out / "windows" / f"{v['case']}.npz")
        for k in acc:
            if k in z.files and len(z[k]):
                acc[k].append(z[k] * 1000)
                n[k] += 1
    x = {k: np.concatenate(v) for k, v in acc.items() if v}
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.axvspan(*ADULT_RANGE_MS, color="#e5e5e5", zorder=0)
    ax.text(ADULT_RANGE_MS[0] - 5, 0.0215, f"awake adults,\nshared-clock studies\n{ADULT_RANGE_MS[0]}–{ADULT_RANGE_MS[1]} ms"
            "\n(to be verified)", ha="right", va="top", fontsize=7.5, color="#555")
    bins = np.arange(0, 901, 10)
    spec = (("art", COLOURS["art"], f"arterial line, ECG → radial upstroke ({n['art']} cases)", dict(histtype="stepfilled", alpha=.35)),
            ("raw", COLOURS["raw"], f"finger PPG as stored ({n['raw']} cases)", dict(histtype="step", lw=2)),
            ("cor", COLOURS["cor"], f"finger PPG after steps 1 and 2 ({n['cor']} cases)", dict(histtype="step", lw=2)))
    for k, col, lab, kw in spec:
        if k in x:
            y = x[k]
            ax.hist(y[(y >= 0) & (y <= 900)], bins=bins, density=True, color=col, label=lab, **kw)
    ax.set(xlim=(0, 900), xlabel="pulse arrival time (ms), 30 s windows", ylabel="share of windows (per ms)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "pat_distribution.png", dpi=150)
    plt.close(fig)
    lo, hi = ADULT_RANGE_MS
    print(json.dumps({k: {"windows": int(len(y)), "p5_p50_p95_ms": np.percentile(y, [5, 50, 95]).round(0).tolist(),
                          "outside_range": round(float(((y < lo) | (y > hi)).mean()), 3),
                          "negative": round(float((y < 0).mean()), 3)} for k, y in x.items()}))


def scatter_figure(out: Path) -> None:
    """Finger PAT against arterial-line PAT, window by window, as stored and after both steps; both axes from zero."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = json.loads((out / "summary.json").read_text())
    pts: dict[str, list] = {"raw": [], "cor": []}
    cases = 0
    for v in s.values():
        if "minus_art_sd_ms" not in v or not usable(v):
            continue
        z = np.load(out / "windows" / f"{v['case']}.npz")
        if "art_k" not in z.files or not len(z["art"]):
            continue
        cases += 1
        for k in pts:
            _, iu, iv = np.intersect1d(z[k + "_k"], z["art_k"], return_indices=True)
            pts[k].append(np.c_[z["art"][iv], z[k][iu]] * 1000)
    xy = {k: np.vstack(v) for k, v in pts.items()}
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    for k, lab in (("raw", "finger PPG as stored"), ("cor", "finger PPG after steps 1 and 2")):
        d = xy[k][xy[k][:, 1] >= 0]
        ax.scatter(d[:, 0], d[:, 1], s=4, alpha=.25, lw=0, color=COLOURS[k], label=f"{lab} (median {np.median(d[:, 1]):.0f} ms)")
    lim = 900
    ax.plot([0, lim], [0, lim], color="#555", lw=.8, ls="--", label="equal PAT")
    ax.set(xlim=(0, 450), ylim=(0, lim), xlabel="PAT, ECG → radial arterial line (ms)",
           ylabel="PAT, ECG → finger PPG (ms)", title=f"{len(xy['raw']):,} paired 30 s windows, {cases} cases", )
    ax.title.set_fontsize(9)
    ax.legend(frameon=False, fontsize=8, loc="upper left", markerscale=4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "pat_scatter.png", dpi=150)
    plt.close(fig)
    print(json.dumps({k: {"pairs": int(len(v)), "finger_minus_art_median_ms": round(float(np.median(v[:, 1] - v[:, 0])), 1)}
                      for k, v in xy.items()}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path)
    ap.add_argument("--cases", nargs="*", type=int, default=[])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--figure", nargs="*", type=int, default=[])
    ap.add_argument("--cohort-figure", action="store_true")
    ap.add_argument("--time-budget", type=float, default=150)
    a = ap.parse_args()
    (a.out / "windows").mkdir(parents=True, exist_ok=True)
    sf = a.out / "summary.json"
    done = json.loads(sf.read_text()) if sf.exists() else {}
    import time
    t_start = time.time()
    for c in a.cases:
        if str(c) in done and c not in a.figure:
            continue
        if time.time() - t_start > a.time_budget:
            break
        res, series = one_case(a.root, c, a.minutes)
        done[str(c)] = res
        if series is not None:
            np.savez(a.out / "windows" / f"{c}.npz", **{k: v[2] for k, v in series["win"].items()},
                     **{k + "_k": v[0] for k, v in series["win"].items()})
            if c in a.figure:
                figure(c, series, a.out / f"pat_{c}.png")
        sf.write_text(json.dumps(done, indent=1))
    if a.cohort_figure:
        cohort_figure(a.out)
        distribution_figure(a.out)
        scatter_figure(a.out)
    print(len(done), "cases in the summary")


if __name__ == "__main__":
    main()
