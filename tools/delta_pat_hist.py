# SPDX-License-Identifier: MIT
"""Beat-to-beat change in pulse arrival time (delta PAT), on the 2 ms sample grid, across the release.

    python tools/delta_pat_hist.py scan --root R --cases 1,2,3 --out part.jsonl [--minutes 60]
    python tools/delta_pat_hist.py agg part*.jsonl --out DIR

Per recording with SNUADC ECG II and PLETH at 500 Hz, on up to 60 minutes starting 10 minutes into their overlap
(as examples/pat_vs_bp.py): R peaks (tools/pat.py r_peaks) and PLETH upstrokes (local maxima of the slope of the
0.5-10 Hz band-passed PLETH), both kept at whole samples (no sub-sample refinement), so every PAT and every delta PAT
is a whole number of 2 ms samples. Each pulse is matched to its own beat, as in examples/pat_vs_bp.py ("own"): the
steepest upstroke within 0.25 s of the recording's typical R-to-upstroke delay.
  own       delta PAT of the finger: PAT of a beat minus PAT of the beat before (consecutive R peaks, RR 0.3-2 s);
  own_art   the same delta of the finger minus the delta of the arterial PAT (R peak to the steepest arterial
            upstroke 0.05-0.45 s after it) for the same pair of beats, in recordings with a pulsatile SNUADC arterial
            line: the ECG and the beat-to-beat physiology cancel, the oximeter path (frames, slips) remains;
  art       the arterial delta alone (no oximeter), for comparison.
Naive pairing (the first upstroke 0.1 s or more after each R peak) is kept in summary.json only: it adds jumps of a
whole RR interval whenever it pairs a pulse with the wrong beat.
The ECG and PLETH sample times are checked to lie on one 2 ms grid (their offset is reported).

scan writes one JSON line per recording: sparse histograms of delta PAT in samples (stays on this machine).
agg writes DIR/summary.json (the summed histograms, the largest local peaks) and DIR/delta_pat.png.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pat import r_peaks, sample_times  # noqa: E402

ECG, PPG, ART = "SNUADC/ECG_II", "SNUADC/PLETH", "SNUADC/ART"
FS = 500.0
MAXLAG = 750          # samples (1.5 s) kept in the histograms


def steepest(d, rp, lo, hi):
    """Whole-sample index offset of the steepest upstroke between lo and hi samples after each R peak; -1 if none."""
    out = np.full(rp.size, -1)
    for k in range(rp.size):
        a, b = rp[k] + lo, rp[k] + hi
        if a < 2 or b > d.size - 2:
            continue
        m = a + int(np.argmax(d[a:b]))
        if d[m] > 0:
            out[k] = m - rp[k]
    return out


def slope(x):
    sos = signal.butter(2, (0.5, 10), btype="bandpass", fs=FS, output="sos")
    return np.gradient(signal.sosfiltfilt(sos, x))


def slope_peaks(x):
    sos = signal.butter(2, (0.5, 10), btype="bandpass", fs=FS, output="sos")
    d = np.gradient(signal.sosfiltfilt(sos, x))
    pk, _ = signal.find_peaks(d, distance=int(0.2 * FS), prominence=np.percentile(d, 99) * 0.3)
    return d, pk


def typical_delay(pk, r):
    lags = []
    for ri in r[:: max(1, r.size // 3000)]:
        a, b = np.searchsorted(pk, ri + 0.1 * FS), np.searchsorted(pk, ri + 1.3 * FS)
        lags += list(pk[a:b] - ri)
    h, e = np.histogram(np.array(lags) / FS, bins=np.arange(0.1, 1.31, 0.01))
    h = np.convolve(h, np.ones(5) / 5, "same")
    return float(e[np.argmax(h)] + 0.005)


def scan_case(root, c, minutes):
    from vitaldb_reader.vital_file import read_vital
    v = read_vital(Path(root) / "vital_files" / f"{c:04d}.vital", track_names={ECG, PPG, ART})
    tr = {k: v.tracks.get(k) for k in (ECG, PPG)}
    if any(tr[k] is None or not tr[k].record_times.size or tr[k].srate != FS for k in tr):
        return {"case": c, "skip": "no 500 Hz ECG II and PLETH"}
    at = v.tracks.get(ART)
    if at is not None and at.record_times.size and at.srate == FS:
        tr[ART] = at
    ts = {k: sample_times(tr[k]) for k in tr}
    t0 = max(ts[k][0] for k in tr) + 600
    t1 = min(min(ts[k][-1] for k in tr), t0 + 60 * minutes)
    if t1 - t0 < 600:
        return {"case": c, "skip": "less than 10 minutes of overlap"}
    seg = {}
    for k in tr:
        a, b = np.searchsorted(ts[k], t0), np.searchsorted(ts[k], t1)
        seg[k] = (ts[k][a:b], np.nan_to_num(tr[k].physical()[a:b].astype(float)))
    te, xe = seg[ECG]
    tp, xp = seg[PPG]
    ri = r_peaks(xe, FS)
    if ri.size < 600:
        return {"case": c, "skip": "too few beats"}
    # one 2 ms grid: map each R peak time to the PLETH sample index at the same time
    rp = np.searchsorted(tp, te[ri] - 0.5 / FS)
    ok = (rp > 0) & (rp < tp.size)
    grid_off = np.abs(tp[np.clip(rp, 0, tp.size - 1)][ok] - te[ri][ok])
    d, pk = slope_peaks(xp)
    rr = np.r_[np.diff(rp), -1]
    naive = np.full(rp.size, -1)
    for k in range(rp.size):
        hi = min(0.1 * FS + (rr[k] if rr[k] > 0 else 1.2 * FS), 1.3 * FS)
        j = np.searchsorted(pk, rp[k] + 0.1 * FS)
        if j < pk.size and pk[j] < rp[k] + hi:
            naive[k] = pk[j] - rp[k]
    D = typical_delay(pk, rp)
    own = steepest(d, rp, int(round((D - 0.25) * FS)), int(round((D + 0.25) * FS)))
    art = None
    if ART in seg:
        ta, xa = seg[ART]
        ra = np.searchsorted(ta, te[ri] - 0.5 / FS)
        if ta.size > FS * 600:     # R peaks mapped to the arterial samples by time; a constant offset cancels in a delta
            a_ = steepest(slope(xa), ra, int(0.05 * FS), int(0.45 * FS))
            if np.mean(a_ >= 0) > 0.7:                                   # a pulsatile line
                art = a_
    rr_prev = np.r_[-1, np.diff(rp)]
    good_rr = (rr_prev >= 0.3 * FS) & (rr_prev <= 2.0 * FS)
    res = {"case": c, "minutes": round((t1 - t0) / 60, 1), "beats": int(rp.size), "typical_delay_s": round(D, 3),
           "grid_offset_max_ms": round(float(grid_off.max()) * 1000, 3) if grid_off.size else None,
           "arterial": art is not None, "hist": {}}

    def delta(p):
        ok = (p >= 0) & (np.r_[-1, p[:-1]] >= 0) & good_rr
        return np.where(ok, p - np.r_[0, p[:-1]], np.iinfo(np.int64).min), ok

    series = {"naive": delta(naive), "own": delta(own)}
    if art is not None:
        da, oka = delta(art)
        do, oko = series["own"]
        series["art"] = (da, oka)
        series["own_art"] = (np.where(oko & oka, do - da, 0), oko & oka)
    for name, (dl, ok) in series.items():
        dl = dl[ok]
        dl = dl[np.abs(dl) <= MAXLAG]
        res["hist"][name] = {str(int(k)): int(n) for k, n in Counter(dl.tolist()).items()}
        res[f"{name}_pairs"] = int(ok.sum())
    return res


def scan(a) -> None:
    out = Path(a.out)
    done = {json.loads(l)["case"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    with open(out, "a") as fh:
        for c in [int(x) for x in a.cases.split(",") if x]:
            if c in done:
                continue
            t1 = time.time()
            try:
                r = scan_case(a.root, c, a.minutes)
            except (OSError, ValueError) as err:   # an unreadable file or a degenerate signal skips the case
                r = {"case": c, "error": f"{type(err).__name__}: {err}"[:300]}
            r["secs"] = round(time.time() - t1, 1)
            fh.write(json.dumps(r) + "\n")
            fh.flush()


def local_peaks(lags, counts, n=12, span=8):
    """The n largest local maxima (a bin larger than every bin within +-span samples), as (ms, count)."""
    out = []
    for i in range(counts.size):
        a, b = max(0, i - span), min(counts.size, i + span + 1)
        if counts[i] > 0 and counts[i] == counts[a:b].max() and np.argmax(counts[a:b]) == i - a:
            out.append((float(lags[i] * 2), int(counts[i])))
    return sorted(out, key=lambda x: -x[1])[:n]


def agg(a) -> None:
    rows = {}
    for f in a.parts:
        for l in open(f):
            r = json.loads(l)
            rows[r["case"]] = r
    ok = [r for r in rows.values() if "hist" in r]
    lags = np.arange(-MAXLAG, MAXLAG + 1)
    H = {}
    for name in ("naive", "own", "own_art", "art"):
        h = np.zeros(lags.size, np.int64)
        for r in ok:
            for k, n in r["hist"].get(name, {}).items():
                h[int(k) + MAXLAG] += n
        H[name] = h
    offs = [r["grid_offset_max_ms"] for r in ok if r.get("grid_offset_max_ms") is not None]
    summ = {"recordings_read": len(rows), "used": len(ok),
            "skipped": dict(Counter(r["skip"] for r in rows.values() if "skip" in r)),
            "errors": sum("error" in r for r in rows.values()),
            "grid_offset_max_ms_max": max(offs) if offs else None,
            "with_arterial": sum(bool(r.get("arterial")) for r in ok), "bin_ms": 2.0}
    for name, h in H.items():
        core = np.abs(lags) <= 100
        summ[name] = {"pairs": int(h.sum()), "pairs_within_200ms": int(h[core].sum()),
                      "share_zero": round(float(h[MAXLAG] / h.sum()), 4),
                      "peaks_within_200ms_ms_count": local_peaks(lags[core], h[core], n=16, span=4),
                      "peaks_all_ms_count": local_peaks(lags, h, n=12, span=25),
                      "counts_by_sample": h.tolist()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    figure(lags, H, len(ok), summ["with_arterial"], out / "delta_pat.png")
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "counts_by_sample"})
                      for k, v in summ.items()}, indent=1))


def figure(lags, H, n, n_art, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ms = lags * 2.0
    rows = (("own", f"finger, own beat ({n:,} recordings)"),
            ("own_art", f"finger minus arterial, same beats ({n_art:,} with an arterial line)"),
            ("art", f"arterial alone, no oximeter ({n_art:,})"))
    fig, ax = plt.subplots(len(rows), 2, figsize=(15, 11), gridspec_kw={"width_ratios": [1.6, 1]})
    for i, (name, lab) in enumerate(rows):
        h = H[name]
        a = ax[i, 0]
        core = np.abs(ms) <= 120
        a.bar(ms[core], h[core], width=2.0, color="#3f51b5", lw=0)
        a.set_yscale("log")
        for k in range(-3, 4):
            if k:
                a.axvline(k * 1000 / 30, color="#e65100", lw=.8, ls="--", alpha=.7)
        a.set(xlim=(-121, 121), ylabel="beat pairs (log)", title=f"{lab}: within ±120 ms (dashed: multiples of 1/30 s)")
        b = ax[i, 1]
        b.bar(ms, h, width=2.0, color="#3f51b5", lw=0)
        b.set_yscale("log")
        b.set(xlim=(-600, 600), title="±600 ms")
        for s in (a, b):
            s.spines[["top", "right"]].set_visible(False)
    ax[-1, 0].set_xlabel("delta PAT, this beat minus the beat before (ms; 2 ms bins)")
    ax[-1, 1].set_xlabel("delta PAT (ms; 2 ms bins)")
    fig.suptitle("Beat-to-beat change in PAT, R peak to the steepest upstroke, whole samples", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    for k in ("--root", "--cases", "--out"):
        s.add_argument(k, required=True)
    s.add_argument("--minutes", type=float, default=60)
    g = sub.add_parser("agg")
    g.add_argument("parts", nargs="+")
    g.add_argument("--out", required=True)
    a = ap.parse_args()
    scan(a) if a.cmd == "scan" else agg(a)


if __name__ == "__main__":
    main()
