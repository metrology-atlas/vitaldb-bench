# SPDX-License-Identifier: MIT
"""Worked example: intraoperative hypotension per patient, from the monitor's numerics and from the waveform.

    python examples/hypotension_burden.py scan --root R --cases 1,2,3 --out part.jsonl [--time-budget 170]
    python examples/hypotension_burden.py agg part*.jsonl --out DIR

Hypotension burden is usually computed from the patient monitor's mean arterial pressure numeric: Solar8000
ART_MBP, one integer mmHg value every 2 s, a moving average over several seconds, about 1.7 s late. The same
pressure is in the SNUADC arterial waveform (500 Hz), from which each beat's mean pressure can be computed.
Per recording, over the time both are present and plausible:

  numeric   minutes with ART_MBP < 65 mmHg (each value counts 2 s), with < 65 and <= 65 (integer values);
  waveform  minutes with the beat mean (pressure averaged from one diastolic foot to the next) < 65 mmHg;
  and the area below 65 (mmHg min), the number of dips below 65 lasting under 1 minute and at least 1 minute, and
  per-patient labels: any 1 minute continuously below 65, and 10 minutes or more in total;
  wav_cal   the waveform moved onto the numeric's level (its median offset in that recording), which separates
            the level difference between the two routes from the dynamics (averaging, short dips).

A 30 s plausibility screen applies to both (mean 20-200 mmHg; waveform pulse pressure 10-150 mmHg; no flat line)
so that flushes and zeroing are left out of both. scan writes one JSON line per recording (aggregates of that
recording; stays on this machine); agg writes DIR/summary.json (aggregates only) and DIR/hypotension_burden.png.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import signal

warnings.filterwarnings("ignore")
LIM = 65.0
TR = {"num": "Solar8000/ART_MBP", "wav": "SNUADC/ART"}


def runs(mask: np.ndarray, t: np.ndarray, dt: np.ndarray):
    """Durations (s) of runs where mask is True; t are sample times, dt each sample's duration."""
    if not mask.any():
        return np.empty(0)
    m = np.r_[False, mask, False].astype(np.int8)
    starts, ends = np.flatnonzero(np.diff(m) == 1), np.flatnonzero(np.diff(m) == -1)
    out = []
    for s, e in zip(starts, ends):
        # a run is broken by a gap in the data longer than 10 s
        cut = np.flatnonzero(np.diff(t[s:e]) > 10) + s + 1
        for a, b in zip(np.r_[s, cut], np.r_[cut, e]):
            out.append(dt[a:b].sum())
    return np.array(out)


def summary(v, t, dt):
    below = v < LIM
    r = runs(below, t, dt)
    return {"minutes_below": float(dt[below].sum() / 60), "area_mmhg_min": float(np.sum((LIM - v[below]) * dt[below]) / 60),
            "dips_under_1min": int((r < 60).sum()), "dips_1min_plus": int((r >= 60).sum()),
            "longest_dip_s": float(r.max()) if r.size else 0.0,
            "label_1min": bool((r >= 60).any()), "label_10min_total": bool(dt[below].sum() >= 600),
            "minutes_total": float(dt.sum() / 60)}


def beat_means(x, fs):
    """Beat mean pressure and the beat's mid time (sample index), from diastolic foot to foot."""
    ok = np.isfinite(x)
    xf = np.where(ok, x, np.nanmedian(x))
    sm = signal.sosfiltfilt(signal.butter(2, 8, fs=fs, output="sos"), xf)
    pk, _ = signal.find_peaks(sm, distance=int(0.3 * fs), prominence=10)
    if pk.size < 10:
        return np.empty(0), np.empty(0), np.empty(0)
    feet = np.array([a + int(np.argmin(sm[a:b])) for a, b in zip(pk[:-1], pk[1:])])
    cs = np.r_[0, np.cumsum(xf)]
    a, b = feet[:-1], feet[1:]
    mean = (cs[b] - cs[a]) / (b - a)
    pp = np.array([xf[i:j].max() - xf[i:j].min() for i, j in zip(a, b)])
    return mean, (a + b) // 2, pp


def plausible(t, v, extra_ok=None, win=30.0):
    """Mask of samples in 30 s windows whose values are all plausible (20-200 mmHg) and not flat."""
    ok = (v > 20) & (v < 200) if extra_ok is None else (v > 20) & (v < 200) & extra_ok
    w = np.floor((t - t[0]) / win).astype(np.int64)
    bad = np.bincount(w, ~ok, minlength=w.max() + 1) > 0
    sd = np.zeros(w.max() + 1)
    for k in np.unique(w):
        sd[k] = np.std(v[w == k]) if (w == k).sum() > 2 else 1.0
    return ~bad[w] & (sd[w] > 0.3)


def scan_case(c, root):
    from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
    v = read_vital(Path(root) / "vital_files" / f"{c:04d}.vital", track_names=set(TR.values()))
    num, wav = v.tracks.get(TR["num"]), v.tracks.get(TR["wav"])
    if num is None or wav is None or not num.record_times.size or not wav.record_times.size:
        return {"case": c, "skip": "no ART numeric or waveform"}
    tn, vn = num.record_times.astype(float) - 1.7, num.physical().astype(float)   # numeric delay, parameter art-mbp-delay
    fs = float(wav.srate)
    rec = np.repeat(np.arange(wav.record_times.size), wav.record_sizes)
    tw_all = wav.record_times[rec] + (np.arange(rec.size) - wav.record_offsets[rec]) / fs
    m, mid, pp = beat_means(wav.physical().astype(float), fs)
    if m.size < 600:
        return {"case": c, "skip": "fewer than 600 beats"}
    tb = tw_all[mid]
    order = np.argsort(tb)
    tb, m, pp = tb[order], m[order], pp[order]
    db = np.clip(np.diff(np.r_[tb, tb[-1] + np.median(np.diff(tb))]), 0, 3)
    okw = plausible(tb, m, (pp > 10) & (pp < 150))
    okn = np.isfinite(vn) & plausible(tn, np.nan_to_num(vn))
    # common time: both plausible; the numeric's 2 s grid decides
    lo, hi = max(tn[0], tb[0]), min(tn[-1], tb[-1])
    gw = np.zeros(int((hi - lo) / 2) + 1, bool)
    i = ((tb[okw] - lo) / 2).astype(int); gw[i[(i >= 0) & (i < gw.size)]] = True
    j = ((tn - lo) / 2).astype(int)
    keepn = okn & (j >= 0) & (j < gw.size)
    keepn[keepn] = gw[j[keepn]]
    gn = np.zeros(gw.size, bool); gn[j[keepn]] = True
    i_all = ((tb - lo) / 2).astype(int)
    keepw = okw & (i_all >= 0) & (i_all < gw.size)
    keepw[keepw] = gn[i_all[keepw]]
    if keepn.sum() < 900:
        return {"case": c, "skip": "under 30 min of common plausible data"}
    dn = np.full(keepn.sum(), 2.0)
    res = {"case": c,
           "num": summary(vn[keepn], tn[keepn], dn),
           "num_le": summary(vn[keepn] - 1.0 * (vn[keepn] == LIM), tn[keepn], dn),   # counts == 65 as below
           "wav": summary(m[keepw], tb[keepw], db[keepw]),
           "share_num_at_65": float(np.mean(vn[keepn] == LIM)),
           "mean_bias_mmhg": float(np.median(vn[keepn]) - np.median(m[keepw]))}
    # the waveform moved onto the numeric's level (its median offset over the recording): what is left is dynamics
    res["wav_cal"] = summary(m[keepw] + res["mean_bias_mmhg"], tb[keepw], db[keepw])
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
                r = scan_case(c, a.root)
            except Exception as err:  # noqa: BLE001
                r = {"case": c, "error": f"{type(err).__name__}: {err}"[:300]}
            r["secs"] = round(time.time() - t1, 1)
            fh.write(json.dumps(r) + "\n")
            fh.flush()


def agg(a) -> None:
    rows = [json.loads(l) for p in a.parts for l in Path(p).read_text().splitlines() if l.strip()]
    ok = [r for r in rows if "num" in r]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def arr(src, k):
        return np.array([r[src][k] for r in ok], float)

    def q(x):
        return [round(float(np.percentile(x, p)), 3) for p in (10, 25, 50, 75, 90)]

    summ = {"recordings_read": len(rows), "used": len(ok), "errors": sum("error" in r for r in rows),
            "skipped": {k: sum(r.get("skip") == k for r in rows) for k in {r.get("skip") for r in rows if r.get("skip")}},
            "hours_common": round(float(arr("num", "minutes_total").sum() / 60), 1)}
    for src in ("num", "num_le", "wav", "wav_cal"):
        summ[src] = {"minutes_below_mean": round(float(arr(src, "minutes_below").mean()), 2),
                     "minutes_below_p10_25_50_75_90": q(arr(src, "minutes_below")),
                     "area_mean": round(float(arr(src, "area_mmhg_min").mean()), 1),
                     "dips_under_1min_mean": round(float(arr(src, "dips_under_1min").mean()), 2),
                     "dips_1min_plus_mean": round(float(arr(src, "dips_1min_plus").mean()), 2),
                     "label_1min_share": round(float(np.mean([r[src]["label_1min"] for r in ok])), 4),
                     "label_10min_share": round(float(np.mean([r[src]["label_10min_total"] for r in ok])), 4)}
    d = arr("num", "minutes_below") - arr("wav", "minutes_below")
    dle = arr("num_le", "minutes_below") - arr("num", "minutes_below")
    dc = arr("num", "minutes_below") - arr("wav_cal", "minutes_below")
    summ["individual"] = {
        "num_minus_wav_minutes_p10_25_50_75_90": q(d), "abs_diff_minutes_median": round(float(np.median(np.abs(d))), 2),
        "abs_diff_ge_5min_share": round(float(np.mean(np.abs(d) >= 5)), 4),
        "relative_diff_median_where_wav_ge_5min": round(float(np.median((d / arr("wav", "minutes_below"))[arr("wav", "minutes_below") >= 5])), 3),
        "label_1min_disagree": round(float(np.mean([r["num"]["label_1min"] != r["wav"]["label_1min"] for r in ok])), 4),
        "label_10min_disagree": round(float(np.mean([r["num"]["label_10min_total"] != r["wav"]["label_10min_total"] for r in ok])), 4),
        "le_minus_lt_minutes_p50_90": [round(float(np.median(dle)), 2), round(float(np.percentile(dle, 90)), 2)],
        "le_vs_lt_label_10min_disagree": round(float(np.mean([r["num"]["label_10min_total"] != r["num_le"]["label_10min_total"] for r in ok])), 4),
        "mean_bias_num_minus_wav_mmhg_p10_50_90": [round(float(np.percentile([r["mean_bias_mmhg"] for r in ok], p)), 2) for p in (10, 50, 90)],
        "share_numeric_exactly_65_median": round(float(np.median([r["share_num_at_65"] for r in ok])), 4),
        "after_level_num_minus_wav_minutes_p10_25_50_75_90": q(dc),
        "after_level_abs_diff_minutes_median": round(float(np.median(np.abs(dc))), 2),
        "after_level_label_10min_disagree": round(float(np.mean([r["num"]["label_10min_total"] != r["wav_cal"]["label_10min_total"] for r in ok])), 4),
        "after_level_label_1min_disagree": round(float(np.mean([r["num"]["label_1min"] != r["wav_cal"]["label_1min"] for r in ok])), 4)}
    summ["population"] = {"mean_minutes_num_minus_wav": round(float(d.mean()), 2),
                          "mean_minutes_num_minus_wav_cal": round(float(dc.mean()), 2),
                          "mean_minutes_num_minus_wav_ci95": [round(float(d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(d.size)), 2),
                                                              round(float(d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(d.size)), 2)]}
    summ["secs_per_recording_median"] = float(np.median([r.get("secs", 0) for r in rows]))
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    figure(ok, summ, out)


def figure(ok, summ, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.3))
    w = np.array([r["wav"]["minutes_below"] for r in ok]); n = np.array([r["num"]["minutes_below"] for r in ok])
    lim = float(np.percentile(np.r_[w, n], 98))
    ax[0].scatter(w, n, s=3, alpha=0.4, color="#3f51b5", lw=0)
    ax[0].plot([0, lim], [0, lim], color="0.5", lw=0.8)
    for f in (ax[0].axvline, ax[0].axhline):
        f(10, color="#e65100", lw=0.8, ls=":")
    ax[0].set_xlim(0, lim); ax[0].set_ylim(0, lim)
    ax[0].set_xlabel("minutes below 65, waveform beats"); ax[0].set_ylabel("minutes below 65, monitor numeric")
    ax[0].set_title("A. Per patient", fontsize=9, loc="left")
    cats = ["dips_under_1min", "dips_1min_plus"]
    x = np.arange(2)
    for k, (src, col, name) in enumerate((("wav", "#3f51b5", "waveform"), ("num", "0.4", "numeric"))):
        ax[1].bar(x + (k - 0.5) * 0.38, [summ[src][c + "_mean"] for c in cats], 0.38, color=col, label=name)
    ax[1].set_xticks(x, ["dips < 1 min", "dips >= 1 min"])
    ax[1].set_ylabel("per recording (mean)")
    ax[1].legend(fontsize=7, frameon=False)
    ax[1].set_title("B. Short dips", fontsize=9, loc="left")
    d = n - w
    wc = np.array([r["wav_cal"]["minutes_below"] for r in ok])
    ax[2].hist(np.clip(d, -15, 15), bins=np.arange(-15, 15.5, 0.5), color="#3f51b5", alpha=0.6, label="as recorded")
    ax[2].hist(np.clip(n - wc, -15, 15), bins=np.arange(-15, 15.5, 0.5), histtype="step", color="#e65100", lw=1.2,
               label="waveform on the numeric's level")
    ax[2].legend(fontsize=7, frameon=False)
    ax[2].axvline(0, color="0.3", lw=0.8)
    ax[2].set_xlabel("numeric minus waveform (minutes below 65)")
    ax[2].set_yticks([])
    ax[2].set_title(f"C. Mean {summ['population']['mean_minutes_num_minus_wav']:+.1f} min; per patient p10-p90 "
                    f"{summ['individual']['num_minus_wav_minutes_p10_25_50_75_90'][0]:+.1f} to "
                    f"{summ['individual']['num_minus_wav_minutes_p10_25_50_75_90'][4]:+.1f}", fontsize=9, loc="left")
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Hypotension (MAP < 65 mmHg): {summ['used']:,} recordings, {summ['hours_common']:,.0f} hours",
                 fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "hypotension_burden.png", dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    for k in ("--root", "--cases", "--out"):
        s.add_argument(k, required=True)
    s.add_argument("--time-budget", type=float, default=0)
    g = sub.add_parser("agg")
    g.add_argument("parts", nargs="+")
    g.add_argument("--out", required=True)
    a = ap.parse_args()
    scan(a) if a.cmd == "scan" else agg(a)


if __name__ == "__main__":
    main()
