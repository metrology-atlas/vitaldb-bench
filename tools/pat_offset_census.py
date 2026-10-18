# SPDX-License-Identifier: MIT
"""Census of the stored ECG-to-PLETH delay: is there one global offset, a few, or one per setup? This machine only.

    python tools/pat_offset_census.py scan --root R --cases 1,2,3 --out part.jsonl [--time-budget 150]
    python tools/pat_offset_census.py agg part*.jsonl --ports ports.jsonl [--assigned assigned.csv] --out DIR

scan: per case with SNUADC ECG lead II and PLETH at 500 Hz, 8 segments of 2 minutes spread evenly over their
overlap (from 10 minutes in). In each: R peaks and one steepest upstroke per pulse in PLETH (and SNUADC/ART where recorded),
paired by the segment's modal delay (0.10-1.20 s for PLETH, 0.05-0.60 s for ART), because the stored PLETH delay
often exceeds one RR interval and a nearest-after search would pair one beat out. Writes per segment the median finger PAT, arterial PAT, finger minus arterial PAT
(paired beat by beat), the median RR interval and beat counts. No sawtooth fit and no offset correction: these are
the stored delays. Resumable: cases already in --out are skipped; stops at --time-budget seconds.

agg: per case medians over its good segments (one clear delay mode: at least half of all R-to-upstroke pairs
within 20 ms of it, and an IQR of at most 30 ms; at least 3 such segments); the PC key (ports of SNUADC, Solar8000, Primus, BIS; from setups.py's
ports.jsonl, or setup_assign.py's assigned.csv for incomplete keys); summaries by key, by SNUADC port and by case
order; DIR/summary.json (aggregates only), DIR/cases.csv (per case; stays local) and DIR/pat_offset_census.png.

Why finger minus arterial: it is the stored device delay plus the finger-to-radial transit (tens of ms, varying
little between patients), so its spread between cases shows the device delay's structure far more sharply than the
finger PAT alone, which also carries the patient's own PAT.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

ECG, PPG, ART = "SNUADC/ECG_II", "SNUADC/PLETH", "SNUADC/ART"
FS = 500.0
SEGMENTS, SEG_S = 8, 120.0


def upstroke_times(t, x, fs=FS):
    """One steepest-upstroke time per pulse: peaks of the derivative of the 0.5-10 Hz band-passed signal."""
    from scipy import signal  # noqa: PLC0415

    sos = signal.butter(2, (0.5, 10), btype="bandpass", fs=fs, output="sos")
    d = np.gradient(signal.sosfiltfilt(sos, x))
    pk, _ = signal.find_peaks(d, distance=int(0.3 * fs), height=np.percentile(d, 90) * 0.3)
    return t[pk]


def mode_paired(r, u, lo, hi, tol=0.12):
    """Delay of upstrokes u after R peaks r by the mode, robust to delays longer than one beat.

    All (u - r) within lo..hi are histogrammed (5 ms bins, smoothed); a constant delay makes one sharp peak while
    pairings one beat out move with the RR interval and smear. Each R peak then takes the upstroke nearest to
    r + mode (within tol). Returns (mode s, per-beat delays with NaN where none, share of all pairs at the mode).
    """
    if r.size < 30 or u.size < 30:
        return np.nan, np.full(r.size, np.nan), np.nan
    i0 = np.searchsorted(u, r + lo)
    lags = []
    for k in range(r.size):
        j = i0[k]
        while j < u.size and u[j] - r[k] <= hi:
            lags.append(u[j] - r[k])
            j += 1
    if len(lags) < 30:
        return np.nan, np.full(r.size, np.nan), np.nan
    lags = np.asarray(lags)
    edges = np.arange(lo, hi + 0.005, 0.005)
    h, _ = np.histogram(lags, bins=edges)
    h = np.convolve(h, np.ones(3), mode="same")
    m = (edges[np.argmax(h)] + edges[np.argmax(h) + 1]) / 2
    j = np.clip(np.searchsorted(u, r + m), 1, u.size - 1)
    near = np.where(np.abs(u[j - 1] - r - m) <= np.abs(u[j] - r - m), u[j - 1], u[j])
    d = near - r
    d[np.abs(d - m) > tol] = np.nan
    return m, d, float(np.mean(np.abs(lags - m) <= 0.02))


def scan_case(root: Path, case: int) -> dict:
    from pat import r_peaks, sample_times  # noqa: PLC0415
    from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital  # noqa: PLC0415

    v = read_vital(root / "vital_files" / f"{case:04d}.vital", track_names={ECG, PPG, ART})
    tr = {k: v.tracks.get(k) for k in (ECG, PPG, ART)}
    if any(tr[k] is None or not tr[k].record_times.size or tr[k].srate != FS for k in (ECG, PPG)):
        return {"case": case, "skip": "no 500 Hz ECG II and PLETH"}
    has_art = tr[ART] is not None and tr[ART].record_times.size and tr[ART].srate == FS
    use = [ECG, PPG] + ([ART] if has_art else [])
    ts = {k: sample_times(tr[k]) for k in use}
    xs = {k: tr[k].physical() for k in use}
    t0 = max(ts[k][0] for k in (ECG, PPG)) + 600
    t1 = min(ts[k][-1] for k in (ECG, PPG))
    if t1 - t0 < SEG_S * 2:
        return {"case": case, "skip": "overlap too short"}
    starts = np.linspace(t0, t1 - SEG_S, SEGMENTS) if t1 - t0 > SEGMENTS * SEG_S else np.arange(t0, t1 - SEG_S, SEG_S)
    segs = []
    for s0 in starts:
        cut = {}
        for k in use:
            a, b = np.searchsorted(ts[k], s0), np.searchsorted(ts[k], s0 + SEG_S)
            cut[k] = (ts[k][a:b], np.nan_to_num(xs[k][a:b].astype(float)))
        te, xe = cut[ECG]
        if te.size < FS * 60:
            continue
        r = te[r_peaks(xe, FS)]
        if r.size < 60:
            continue
        rr = np.diff(r)
        seg = {"t_rel_s": round(float(s0 - t0), 0), "beats": int(r.size), "rr_ms": round(float(np.median(rr)) * 1000, 1)}
        mp, dp, sharp = mode_paired(r, upstroke_times(*cut[PPG]), 0.10, 1.20) if cut[PPG][0].size > FS * 60 else (np.nan, np.full(r.size, np.nan), np.nan)
        seg["ppg_n"] = int(np.isfinite(dp).sum())
        if seg["ppg_n"] >= 30:
            seg["ppg_ms"] = round(float(np.nanmedian(dp)) * 1000, 1)
            seg["ppg_iqr_ms"] = round(float(np.subtract(*np.nanpercentile(dp, [75, 25]))) * 1000, 1)
            seg["ppg_mode_share"] = round(sharp, 3)
        if has_art and cut[ART][0].size > FS * 60:
            ma, da, _ = mode_paired(r, upstroke_times(*cut[ART]), 0.05, 0.60)
            if np.isfinite(da).sum() >= 30:
                seg["art_ms"] = round(float(np.nanmedian(da)) * 1000, 1)
                dd = (dp - da)[np.isfinite(dp) & np.isfinite(da)]
                if dd.size >= 30:
                    seg["ppg_minus_art_ms"] = round(float(np.median(dd)) * 1000, 1)
                    seg["ppg_minus_art_iqr_ms"] = round(float(np.subtract(*np.percentile(dd, [75, 25]))) * 1000, 1)
        segs.append(seg)
    return {"case": case, "art": bool(has_art), "segments": segs}


def scan(a) -> None:
    out = Path(a.out)
    done = {json.loads(l)["case"] for l in open(out)} if out.exists() else set()
    t_start = time.time()
    for c in [int(x) for x in a.cases.split(",") if x]:
        if c in done:
            continue
        if time.time() - t_start > a.time_budget:
            break
        try:
            r = scan_case(Path(a.root), c)
        except Exception as e:  # noqa: BLE001
            r = {"case": c, "error": repr(e)[:200]}
        with open(out, "a") as fh:
            fh.write(json.dumps(r) + "\n")


def agg(a) -> None:
    import csv

    import pandas as pd  # noqa: F811

    rows = {}
    for p in a.inputs:
        for l in open(p):
            r = json.loads(l)
            if r.get("segments"):
                rows[r["case"]] = r
    ports = {}
    for l in open(a.ports):
        r = json.loads(l)
        ports[r["case"]] = r["ports"]
    assigned = {}
    if a.assigned:
        for r in csv.DictReader(open(a.assigned)):
            if r["key"] and float(r["posterior"] or 0) >= 0.9:
                assigned[int(r["case"])] = r["key"]
    recs = []
    for c, r in rows.items():
        sg = pd.DataFrame(r["segments"])
        # a segment counts only if its PLETH delay has one clear mode and a tight spread; noisy ECG (R peaks doubled,
        # RR ~0.33 s) or a lost PLETH gives no mode and is dropped
        if "ppg_mode_share" not in sg:
            continue
        sg = sg[(sg.ppg_mode_share >= 0.5) & (sg.ppg_iqr_ms <= 30)]
        if len(sg) < 3:
            continue
        pr = ports.get(c, {})
        core = [pr.get(k) for k in ("SNUADC", "Solar8000", "Primus", "BIS")]
        key = " · ".join(core) if all(core) else assigned.get(c)
        rec = {"case": c, "key": key, "snuadc_port": pr.get("SNUADC"), "art": r["art"], "segments": len(sg),
               "rr_ms": sg.rr_ms.median()}
        for col in ("ppg_ms", "art_ms", "ppg_minus_art_ms"):
            if col in sg and sg[col].notna().sum() >= 3:
                rec[col] = sg[col].median()
                rec[col + "_seg_range"] = sg[col].max() - sg[col].min()
        recs.append(rec)
    df = pd.DataFrame(recs)
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    df.to_csv(od / "cases.csv", index=False)

    def q(x):
        x = pd.Series(x).dropna()
        return {"n": int(len(x)), **{f"p{p}": round(float(x.quantile(p / 100)), 1) for p in (5, 25, 50, 75, 95)}} if len(x) else {}

    s = {"cases": int(len(df)), "with_art": int(df.art.sum()),
         "ppg_ms": q(df.get("ppg_ms")), "art_ms": q(df.get("art_ms")), "ppg_minus_art_ms": q(df.get("ppg_minus_art_ms")),
         "within_case_segment_range_ms": {c: q(df.get(c + "_seg_range")) for c in ("ppg_ms", "ppg_minus_art_ms")}}
    by = {}
    for k, g in df.groupby("key"):
        if len(g) >= 10:
            by[k] = {"n": int(len(g)), "ppg_ms_median": round(float(g.ppg_ms.median()), 1),
                     "ppg_ms_iqr": round(float(g.ppg_ms.quantile(.75) - g.ppg_ms.quantile(.25)), 1)}
            if g.get("ppg_minus_art_ms") is not None and g.ppg_minus_art_ms.notna().sum() >= 5:
                by[k]["minus_art_median"] = round(float(g.ppg_minus_art_ms.median()), 1)
                by[k]["minus_art_iqr"] = round(float(g.ppg_minus_art_ms.quantile(.75) - g.ppg_minus_art_ms.quantile(.25)), 1)
    s["by_key"] = by
    # how much of the between-case spread the key explains: share within 15 ms of the key's median vs of the global median
    if "ppg_minus_art_ms" in df:
        d = df.dropna(subset=["ppg_minus_art_ms"])
        km = d.groupby("key").ppg_minus_art_ms.transform("median")
        s["minus_art_within_15ms"] = {"global": round(float(((d.ppg_minus_art_ms - d.ppg_minus_art_ms.median()).abs() <= 15).mean()), 3),
                                     "by_key": round(float(((d.ppg_minus_art_ms - km).abs() <= 15).dropna().mean()), 3)}
    (od / "summary.json").write_text(json.dumps(s, indent=1))
    figure(df, od / "pat_offset_census.png")
    print(json.dumps({k: v for k, v in s.items() if k != "by_key"}, indent=1))
    print(f"{len(by)} keys with >= 10 cases")


def figure(df, out: Path) -> None:
    """Left: per-case stored delays (finger PAT, finger minus arterial). Middle: finger minus arterial by PC key.
    Right: by case order (bins of 400 cases), as a check for epochs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.1, 1.3, 1]})
    a = df.dropna(subset=["ppg_minus_art_ms"])
    b = np.arange(380, 821, 5)
    ax[0].hist(df.ppg_ms.dropna(), bins=b, color="#9e9e9e", alpha=.8, label=f"ECG → finger PPG, as stored ({df.ppg_ms.notna().sum():,} cases)")
    ax[0].hist(a.ppg_minus_art_ms, bins=b, histtype="step", lw=2, color="#08519c",
               label=f"finger minus arterial PAT ({len(a):,} cases with a line)")
    for v, col in ((df.ppg_ms.median(), "#555"), (a.ppg_minus_art_ms.median(), "#08519c")):
        ax[0].axvline(v, color=col, lw=.8, ls="--")
        ax[0].text(v, ax[0].get_ylim()[1] * .62, f" {v:.0f} ms", color=col, fontsize=8, va="top")
    ax[0].set(xlabel="per-case median (ms)", ylabel="cases", title="One peak each, no second cluster")
    ax[0].legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, 1.0))
    ax[0].set_ylim(0, ax[0].get_ylim()[1] * 1.25)

    g = a.groupby("key").ppg_minus_art_ms
    st = pd.DataFrame({"n": g.size(), "med": g.median(), "q1": g.quantile(.25), "q3": g.quantile(.75)})
    st = st[st.n >= 20].sort_values("med")
    y = np.arange(len(st))
    ax[1].hlines(y, st.q1, st.q3, color="#9ecae1", lw=3)
    ax[1].plot(st.med, y, "o", color="#08519c", ms=4)
    gm = a.ppg_minus_art_ms.median()
    ax[1].axvline(gm, color="#d62728", lw=1)
    ax[1].axvspan(gm - 15, gm + 15, color="#d62728", alpha=.08)
    ax[1].set_yticks([])
    ax[1].set(xlabel="finger minus arterial PAT (ms): median and IQR per setup", ylabel=f"{len(st)} PC keys with ≥ 20 cases",
              title=f"Setups all sit at the global value ({gm:.0f} ms ± 15 shaded)", xlim=(440, 560))

    import pandas as _pd  # noqa: F401
    df = df.sort_values("case")
    bins = (df.case - 1) // 400
    t = df.groupby(bins).agg(c=("case", "median"), ppg=("ppg_ms", "median"), ma=("ppg_minus_art_ms", "median"))
    ax[2].plot(t.c, t.ppg - df.ppg_ms.median(), "s-", color="#9e9e9e", label="finger PPG, as stored")
    ax[2].plot(t.c, t.ma - a.ppg_minus_art_ms.median(), "o-", color="#08519c", label="finger minus arterial")
    ax[2].axhline(0, color="#555", lw=.6)
    ax[2].set(xlabel="case id (bins of 400)", ylabel="bin median minus overall median (ms)", ylim=(-20, 20),
              title="No shift along the case order")
    ax[2].legend(frameon=False, fontsize=8)
    for x in ax:
        x.title.set_fontsize(9)
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["scan", "agg"])
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--root")
    ap.add_argument("--cases")
    ap.add_argument("--out")
    ap.add_argument("--time-budget", type=float, default=1e9)
    ap.add_argument("--ports")
    ap.add_argument("--assigned")
    a = ap.parse_args()
    scan(a) if a.mode == "scan" else agg(a)


if __name__ == "__main__":
    main()
