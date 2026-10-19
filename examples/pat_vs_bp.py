# SPDX-License-Identifier: MIT
"""Worked example: pulse arrival time against blood pressure, before and after synchronisation.

    python examples/pat_vs_bp.py scan --root R --cases 1,2,3 --out part.jsonl [--time-budget 170] [--minutes 60]
    python examples/pat_vs_bp.py agg part*.jsonl --out DIR

Per recording with ECG lead II, PLETH and a pulsatile arterial line (SNUADC, one clock), on up to 60 minutes
starting 10 minutes into their overlap (as examples/pulse_arrival_time.py):

PAT, ECG R peak to the steepest PLETH upstroke, under five timings:
  naive     each R peak paired with the first PLETH upstroke 0.1 s or more after it (one beat out whenever the
            stored delay, about 0.7 s, is longer than the RR interval);
  own       each R peak paired with its own pulse: the upstroke within 0.25 s of the recording's typical R-to-PLETH
            delay (the mode of all upstrokes 0.1-1.3 s after an R peak);
  saw       own, with the PLETH sawtooth removed (physiosync.fit_ecg_ppg_delay, step 1 of the PAT example);
  cor       saw, less the global device delay (step 2); a constant, so every correlation must equal saw's;
  art       ECG R peak to the arterial upstroke: no PLETH delay and no sawtooth, the reference.
Blood pressure per beat from the arterial waveform: systolic (peak after the arterial upstroke) and mean (foot to
foot); heart rate from the RR interval. The monitor's ART_SBP numeric at each beat, as stamped and moved 2.1 s
earlier for its delay, as a second BP source.

Correlations (Pearson r) per recording at two scales, beat to beat and over 30 s windows (window medians; PAT also
by maximum likelihood as in the PAT example): PAT-SBP, PAT-MAP, PAT-HR, and PAT-SBP with HR held constant
(partial r). Between patients: the correlation of recording medians. scan writes one JSON line per recording
(aggregates of that recording; stays on this machine); agg writes DIR/summary.json and DIR/pat_vs_bp.png.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pulse_arrival_time as P  # noqa: E402

warnings.filterwarnings("ignore")
FS, WIN = P.FS, 30.0
CONDS = ("naive", "own", "saw", "cor", "art")
NUM = "Solar8000/ART_SBP"


def first_upstroke(t, x, r_t, rr):
    """The first PLETH upstroke (a local maximum of the slope) at least 0.1 s after each R peak, before the next R
    peak plus 0.1 s; refined to the sub-sample peak as in the PAT example."""
    sos = signal.butter(2, (0.5, 10), btype="bandpass", fs=FS, output="sos")
    d = np.gradient(signal.sosfiltfilt(sos, x))
    pk, _ = signal.find_peaks(d, distance=int(0.2 * FS), prominence=np.percentile(d, 99) * 0.3)
    up = t[pk]
    out = np.full(r_t.size, np.nan)
    for k in range(r_t.size):
        hi = min(0.1 + (rr[k] if np.isfinite(rr[k]) else 1.2), 1.3)
        j = np.searchsorted(up, r_t[k] + 0.1)
        if j < up.size and up[j] < r_t[k] + hi:
            m = pk[j]
            den = d[m - 1] - 2 * d[m] + d[m + 1] if 0 < m < d.size - 1 else 0
            out[k] = t[m] + (float(np.clip(0.5 * (d[m - 1] - d[m + 1]) / den, -0.5, 0.5)) / FS if den else 0.0)
    return out


def typical_delay(t, x, r_t):
    """Mode of R-to-upstroke delays 0.1-1.3 s (every upstroke in that range after every R peak)."""
    sos = signal.butter(2, (0.5, 10), btype="bandpass", fs=FS, output="sos")
    d = np.gradient(signal.sosfiltfilt(sos, x))
    pk, _ = signal.find_peaks(d, distance=int(0.25 * FS), prominence=np.percentile(d, 99) * 0.3)
    up = t[pk]
    lags = []
    for rt in r_t[:: max(1, r_t.size // 3000)]:
        a, b = np.searchsorted(up, rt + 0.1), np.searchsorted(up, rt + 1.3)
        lags += list(up[a:b] - rt)
    h, e = np.histogram(lags, bins=np.arange(0.1, 1.31, 0.01))
    h = np.convolve(h, np.ones(5) / 5, "same")
    return float(e[np.argmax(h)] + 0.005)


def beat_bp(ta, xa, a_t):
    """Systolic peak within 0.35 s after each arterial upstroke, and mean pressure from this foot to the next."""
    sbp, mbp = np.full(a_t.size, np.nan), np.full(a_t.size, np.nan)
    cs = np.r_[0, np.cumsum(xa)]
    for k, u in enumerate(a_t):
        if not np.isfinite(u):
            continue
        i = np.searchsorted(ta, u)
        j = np.searchsorted(ta, u + 0.35)
        if j >= xa.size or j - i < 5:
            continue
        sbp[k] = xa[i:j].max()
        nxt = a_t[k + 1] if k + 1 < a_t.size and np.isfinite(a_t[k + 1]) else np.nan
        if np.isfinite(nxt) and 0.3 < nxt - u < 2.0:
            i0, i1 = np.searchsorted(ta, u - 0.05), np.searchsorted(ta, nxt - 0.05)
            mbp[k] = (cs[i1] - cs[i0]) / max(i1 - i0, 1)
    return sbp, mbp


def corr(x, y, z=None):
    m = np.isfinite(x) & np.isfinite(y) & (np.isfinite(z) if z is not None else True)
    if m.sum() < 20 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return None
    if z is None:
        return float(np.corrcoef(x[m], y[m])[0, 1])
    rxy, rxz, ryz = (np.corrcoef(a, b)[0, 1] for a, b in ((x[m], y[m]), (x[m], z[m]), (y[m], z[m])))
    den = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return float((rxy - rxz * ryz) / den) if den > 0 else None


def windows(t, y, t0):
    k = np.floor((t - t0) / WIN).astype(int)
    ok = np.isfinite(y)
    out = {}
    for w in np.unique(k[ok]):
        m = ok & (k == w)
        if m.sum() >= 15:
            out[int(w)] = float(np.median(y[m]))
    return out


def wcorr(a: dict, b: dict, c: dict | None = None):
    keys = sorted(set(a) & set(b) & (set(c) if c is not None else set(a)))
    if len(keys) < 15:
        return None
    x, y = np.array([a[k] for k in keys]), np.array([b[k] for k in keys])
    return corr(x, y, np.array([c[k] for k in keys]) if c is not None else None)


def slope(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(y[m], x[m], 1)[0] * 1000) if m.sum() >= 20 and np.std(y[m]) > 0 else None   # ms per mmHg


def scan_case(root, c, minutes):
    from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
    v = read_vital(Path(root) / "vital_files" / f"{c:04d}.vital", track_names={P.ECG, P.PPG, P.ART, NUM})
    tr = {k: v.tracks.get(k) for k in (P.ECG, P.PPG, P.ART)}
    if any(tr[k] is None or not tr[k].record_times.size or tr[k].srate != FS for k in tr):
        return {"case": c, "skip": "no 500 Hz ECG, PLETH and arterial line"}
    ts = {k: P.sample_times(tr[k]) for k in tr}
    t0 = max(ts[k][0] for k in tr) + 600
    t1 = min(min(ts[k][-1] for k in tr), t0 + 60 * minutes)
    if t1 - t0 < 600:
        return {"case": c, "skip": "less than 10 minutes of overlap"}
    seg = {}
    for k in tr:
        a, b = np.searchsorted(ts[k], t0), np.searchsorted(ts[k], t1)
        seg[k] = (ts[k][a:b], np.nan_to_num(tr[k].physical()[a:b].astype(float)))
    te, xe = seg[P.ECG]
    r = te[P.r_peaks(xe, FS)]
    if r.size < 600:
        return {"case": c, "skip": "too few beats"}
    rr = np.r_[np.diff(r), np.nan]
    hr = 60 / np.r_[np.nan, np.diff(r)]                                 # HR from the RR interval ending at the beat
    a_t = P.upstrokes(*seg[P.ART], r, 0.05, 0.45)[0]
    if np.mean(np.isfinite(a_t)) < 0.7:
        return {"case": c, "skip": "arterial line not pulsatile"}
    D = typical_delay(*seg[P.PPG], r)
    own = P.upstrokes(*seg[P.PPG], r, D - 0.25, D + 0.25)[0]
    naive = first_upstroke(*seg[P.PPG], r, rr)
    sbp, mbp = beat_bp(*seg[P.ART], a_t)
    ok = np.isfinite(own) & np.isfinite(a_t)
    m = P.fit_ecg_ppg_delay(r[ok], own[ok], adopted_pat_s=P.ADOPTED, frame_s=1 / 30, matched=True, ref_t=a_t[ok])
    saw = np.full(r.size, np.nan)
    saw[ok] = m.shift(own[ok]) - m.offset_s
    pat = {"naive": naive - r, "own": own - r, "saw": own - saw - r, "cor": own - saw - r - P.GLOBAL_DELAY, "art": a_t - r}
    good = (sbp > 40) & (sbp < 250) & (hr > 30) & (hr < 180)
    for k in ("naive", "own", "saw", "art"):
        pat[k] = np.where(good & (pat[k] > 0.05) & (pat[k] < 1.5), pat[k], np.nan)
    pat["cor"] = pat["saw"] - P.GLOBAL_DELAY            # the same beats as saw, one constant less
    num = None
    nt = v.tracks.get(NUM)
    if nt is not None and nt.record_times.size > 10:
        tn, xn = nt.record_times.astype(float), nt.physical().astype(float)
        okn = (xn > 40) & (xn < 250)
        num = {"as_stamped": np.interp(r, tn[okn], xn[okn], left=np.nan, right=np.nan),
               "moved_2_1s": np.interp(r, tn[okn] - 2.1, xn[okn], left=np.nan, right=np.nan)}
    res = {"case": c, "minutes": round((t1 - t0) / 60, 1), "beats": int(r.size), "typical_delay_s": round(D, 3),
           "naive_one_beat_out": round(float(np.nanmean(np.abs(pat["naive"] - pat["own"]) > 0.15)), 4),
           "hr_median": round(float(np.nanmedian(hr)), 1), "slips": len(m.slips_t),
           "sbp_sd": round(float(np.nanstd(sbp[good])), 2), "medians": {}, "beat": {}, "win30": {}, "slope_ms_per_mmhg": {}}
    W = {"sbp": windows(r, np.where(good, sbp, np.nan), t0), "mbp": windows(r, np.where(good, mbp, np.nan), t0),
         "hr": windows(r, np.where(good, hr, np.nan), t0)}
    dens = None
    for k, y in pat.items():
        res["medians"][k] = float(np.nanmedian(y)) if np.isfinite(y).any() else None
        res["beat"][k] = {"sbp": corr(y, sbp), "mbp": corr(y, mbp), "hr": corr(y, hr), "sbp_given_hr": corr(y, sbp, hr)}
        wy = windows(r, y, t0)
        res["win30"][k] = {"sbp": wcorr(wy, W["sbp"]), "mbp": wcorr(wy, W["mbp"]), "hr": wcorr(wy, W["hr"]),
                           "sbp_given_hr": wcorr(wy, W["sbp"], W["hr"])}
        res["slope_ms_per_mmhg"][k] = slope(np.array([wy[q] for q in sorted(set(wy) & set(W["sbp"]))]),
                                            np.array([W["sbp"][q] for q in sorted(set(wy) & set(W["sbp"]))]))
    if num is not None:
        res["beat_numeric"] = {s: corr(pat["saw"], x) for s, x in num.items()}
        res["beat_numeric_vs_wave_sbp"] = {s: corr(sbp, x) for s, x in num.items()}
    res["medians"]["sbp"] = float(np.nanmedian(sbp[good]))
    res["medians"]["hr"] = float(np.nanmedian(hr[good]))
    return res


def scan(a) -> None:
    out = Path(a.out)
    done = {json.loads(l)["case"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    t0 = time.time()
    with open(out, "a") as fh:
        for c in [int(x) for x in a.cases.split(",") if x]:
            if c in done:
                continue
            if a.time_budget and time.time() - t0 > a.time_budget:
                break
            t1 = time.time()
            try:
                r = scan_case(a.root, c, a.minutes)
            except Exception as err:  # noqa: BLE001
                r = {"case": c, "error": f"{type(err).__name__}: {err}"[:300]}
            r["secs"] = round(time.time() - t1, 1)
            fh.write(json.dumps(r) + "\n")
            fh.flush()


def agg(a) -> None:
    rows = [json.loads(l) for p in a.parts for l in Path(p).read_text().splitlines() if l.strip()]
    rows = list({r["case"]: r for r in rows}.values())
    ok = [r for r in rows if "beat" in r and r["sbp_sd"] >= 3]          # some BP variation to correlate with
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def q(vals):
        x = np.array([v for v in vals if v is not None and np.isfinite(v)], float)
        return {"median": round(float(np.median(x)), 3), "iqr": [round(float(np.percentile(x, 25)), 3), round(float(np.percentile(x, 75)), 3)],
                "n": int(x.size)} if x.size else None

    summ = {"recordings_read": len(rows), "used": len(ok),
            "skipped": {k: sum(r.get("skip") == k for r in rows) for k in {r.get("skip") for r in rows if r.get("skip")}},
            "errors": sum("error" in r for r in rows),
            "naive_one_beat_out_share": q([r["naive_one_beat_out"] for r in ok]),
            "hr_median": q([r["hr_median"] for r in ok]), "within_scale": {}}
    for scale in ("beat", "win30"):
        summ["within_scale"][scale] = {k: {m: q([r[scale][k][m] for r in ok]) for m in ("sbp", "mbp", "hr", "sbp_given_hr")} for k in CONDS}
    summ["slope_ms_per_mmhg"] = {k: q([r["slope_ms_per_mmhg"][k] for r in ok]) for k in CONDS}
    # change per recording, saw against own and against naive
    for scale in ("beat", "win30"):
        for a_, b_ in (("own", "naive"), ("saw", "own")):
            summ[f"{scale}_change_{a_}_minus_{b_}_sbp"] = q([(r[scale][a_]["sbp"] - r[scale][b_]["sbp"]) for r in ok
                                                           if r[scale][a_]["sbp"] is not None and r[scale][b_]["sbp"] is not None])
            summ[f"{scale}_change_{a_}_minus_{b_}_hr"] = q([(r[scale][a_]["hr"] - r[scale][b_]["hr"]) for r in ok
                                                          if r[scale][a_]["hr"] is not None and r[scale][b_]["hr"] is not None])
    # as Ruffolo et al. 2025 (Physiol Meas 46:115006, appendix B): Pearson r of beat-paired PAT and SBP over a 60 min
    # window per patient, summarised by the mean r and the share of windows with r < 0
    summ["ruffolo_comparison"] = {}
    for scale in ("beat", "win30"):
        for k in CONDS:
            x = np.array([r[scale][k]["sbp"] for r in ok if r[scale][k]["sbp"] is not None])
            summ["ruffolo_comparison"][f"{scale}_{k}"] = {"mean_r": round(float(x.mean()), 3), "share_negative": round(float(np.mean(x < 0)), 3),
                                                          "n": int(x.size)}
    fast = [r for r in ok if r["naive_one_beat_out"] > 0.2]
    summ["fast_hr_recordings"] = {"n": len(fast), "beat_hr_naive": q([r["beat"]["naive"]["hr"] for r in fast]),
                                  "beat_hr_own": q([r["beat"]["own"]["hr"] for r in fast]),
                                  "beat_sbp_naive": q([r["beat"]["naive"]["sbp"] for r in fast]),
                                  "beat_sbp_own": q([r["beat"]["own"]["sbp"] for r in fast])}
    between = {}
    sb = np.array([r["medians"]["sbp"] for r in ok]); hh = np.array([r["medians"]["hr"] for r in ok])
    for k in CONDS:
        x = np.array([r["medians"][k] if r["medians"][k] is not None else np.nan for r in ok])
        between[k] = {"sbp": corr(x, sb), "hr": corr(x, hh), "sbp_given_hr": corr(x, sb, hh)}
    summ["between_patients"] = between
    nb = [r for r in ok if r.get("beat_numeric")]
    if nb:
        summ["beat_numeric_sbp"] = {s: q([r["beat_numeric"][s] for r in nb]) for s in ("as_stamped", "moved_2_1s")}
        summ["beat_numeric_vs_wave_sbp"] = {s: q([r["beat_numeric_vs_wave_sbp"][s] for r in nb]) for s in ("as_stamped", "moved_2_1s")}
    summ["secs_per_recording_median"] = float(np.median([r.get("secs", 0) for r in rows]))
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    figure(ok, summ, out)


def figure(ok, summ, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lab = {"naive": "naive pairing", "own": "own beat", "saw": "+ sawtooth removed", "cor": "+ global delay", "art": "arterial PAT (reference)"}
    col = {"naive": "#9e9e9e", "own": "#7986cb", "saw": "#3f51b5", "cor": "#e65100", "art": "#2e7d32"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, (scale, title) in enumerate((("beat", "A. Beat to beat, within patients"), ("win30", "B. 30 s windows, within patients"))):
        for j, mname in enumerate(("sbp", "hr", "sbp_given_hr")):
            for k_i, k in enumerate(CONDS):
                s = summ["within_scale"][scale][k][mname]
                if s:
                    x = j + (k_i - 2) * 0.15
                    ax[i].plot([x, x], s["iqr"], color=col[k], lw=1.2)
                    ax[i].plot(x, s["median"], "o", color=col[k], ms=4, label=lab[k] if j == 0 else None)
        ax[i].axhline(0, color="0.6", lw=0.8)
        ax[i].set_xticks(range(3), ["PAT-SBP", "PAT-HR", "PAT-SBP | HR"])
        ax[i].set_ylabel("r per recording (median, IQR)")
        ax[i].set_title(title, fontsize=9, loc="left")
    ax[1].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"PAT against blood pressure within patients, five timings: {summ['used']:,} recordings", fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "pat_vs_bp.png", dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    for k in ("--root", "--cases", "--out"):
        s.add_argument(k, required=True)
    s.add_argument("--time-budget", type=float, default=0)
    s.add_argument("--minutes", type=float, default=60)
    g = sub.add_parser("agg")
    g.add_argument("parts", nargs="+")
    g.add_argument("--out", required=True)
    a = ap.parse_args()
    scan(a) if a.cmd == "scan" else agg(a)


if __name__ == "__main__":
    main()
