# SPDX-License-Identifier: MIT
"""Figures ppg-correction-check and pat-timeseries (metrology-atlas study ppg-sawtooth).

    python tools/ppg_sawtooth_figures.py inputs --per-case $W/results/pat/ppg_correction_check_per_case.json \
        --out $W/ppg_correction_check
    python tools/ppg_fix_check.py --root $R --cases $W/ppg_correction_check/cases266.json \
        --out $W/ppg_correction_check/check_tierC.jsonl --adopted-pat 0.27 \
        --priors $W/ppg_correction_check/priors266.json --apply-min-tier C
    python tools/ppg_sawtooth_figures.py check --scan $W/ppg_correction_check/check_tierC.jsonl \
        --units $W/results/sync_full/units.csv --modes $W/results/sync_full/snuadc_mode_label.json \
        --examples 2156 6241 1094 --out $W/ppg_correction_check
    python tools/ppg_sawtooth_figures.py timeseries --root $R --cases 1912 2 2153 2611 476 3964 4194 5581 489 1599 1759 1858 \
        --out $W/pat_timeseries

``inputs`` writes the case list (cases266.json) and the per-case setup priors (priors266.json) of the first run,
from its saved per-case summary (whose priors file was not kept).

``check`` sums the per-case output of tools/ppg_fix_check.py:

a-c. each example's measured PAT (15-beat medians) and the corrected PAT shifted to the same median, with the
     detected slips;
d.   the PAT step at each modelled slip (15 beats after minus before), measured and corrected;
e.   the PAT slope inside the teeth (median per case), measured and corrected, for cases whose rate comes from
     their own full teeth (rate_source teeth);
f.   the 5-95% spread of 1-minute PAT medians before and after correction, coloured by the rate's source (red:
     the case's own teeth; orange: the setup prior; grey: not corrected, no slips or no rate);
g.   each case's sawtooth rate (median of its teeth's rates, cases with 3 or more kept slips and their own teeth)
     against its SNUADC drift (units.csv), coloured by setup (SNUADC port and correction mode);
h.   the rate per setup (median and IQR, setups with 2 or more such cases);
i.   the measured median PAT per case against the adopted PAT, and the combined offset removed.

With a --method rebuilt run of tools/ppg_fix_check.py (physiosync 0.3.1) the same panels show the rebuilt model:
the titles give the steps corrected and the slips placed only within an interval (not corrected); in f, orange is a
sawtooth whose tooth rests on two slips (steps corrected, creep reported) and grey a recording without a sawtooth
(steps only, if any).

``timeseries`` reads each case: PAT from each R peak (ECG II, tools/pat.py r_peaks, sub-sample) to the matched PPG
foot (intersecting tangent, as tools/pat_offset_census.py), 30 s medians; the SNUADC clock corrections are the
record-stamp steps of the ECG track (30 ms to 2 s against the sample count, as tools/shared_bursts_slopes.py;
steps within 60 s merged);
gaps of more than 1 s between ECG samples are shaded. Writes pat-timeseries.png and timeseries.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def rate_of(r):
    t = [x for x, s in zip(r.get("rates_ppm", []), r.get("sources", [])) if s == "tooth"]
    return float(np.median(t)) if t else None


def inputs(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = json.loads(Path(a.per_case).read_text())
    (out / "cases266.json").write_text(json.dumps({"cases": [r["case"] for r in d], "note": "the cases of "
                                       + Path(a.per_case).name + ", in its order"}))
    (out / "priors266.json").write_text(json.dumps({str(r["case"]): r["prior_ppm"] for r in d if r["prior_ppm"] is not None}))
    print(json.dumps({"cases": len(d), "priors": sum(r["prior_ppm"] is not None for r in d)}))


def check(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in Path(a.scan).read_text().splitlines() if x.strip()]
    ok = [r for r in rows if "series" in r]
    U = {int(r["case"]): r for r in csv.DictReader(open(a.units))}
    modes = json.loads(Path(a.modes).read_text())

    def setup(c):
        port = U.get(c, {}).get("snuadc_port") or "?"
        return f"{port} {modes.get(str(c), '?')}"

    st_raw = np.array([s for r in ok for s in r["step_at_slips_raw_ms"]])
    st_cor = np.array([s for r in ok for s in r["step_at_slips_corr_ms"]])
    full = [r for r in ok if r["tooth_slope_raw_ppm"] and r.get("rate_source") == "teeth"]   # rates from its own full teeth
    sl_raw = np.array([np.median(r["tooth_slope_raw_ppm"]) for r in full])
    sl_cor = np.array([np.median(r["tooth_slope_corr_ppm"]) for r in full])
    rebuilt = ok[0].get("method") == "rebuilt"
    grp = []
    for r in ok:
        if rebuilt:
            grp.append({"teeth": "teeth", "reported": "prior"}.get(r.get("rate_source"), "none"))
        elif not r["n_slips"] or r["p2p_raw_ms"] == r["p2p_corr_ms"]:
            grp.append("none")
        elif r.get("rate_source") == "teeth":
            grp.append("teeth")
        else:
            grp.append("prior")
    reg = []
    for r in ok:
        rt = rate_of(r)
        dr = U.get(r["case"], {}).get("snuadc_drift_ppm")
        if rt is not None and (r.get("n_kept") or 0) >= 3:
            reg.append((r["case"], setup(r["case"]), rt, float(dr) if dr not in (None, "") else np.nan))
    setups = {}
    for c, s_, rt, _ in reg:
        setups.setdefault(s_, []).append(rt)
    by_setup = {k: v for k, v in sorted(setups.items(), key=lambda kv: -len(kv[1])) if len(v) >= 2}
    tiers = {}
    for r in ok:
        for t in r.get("tiers", []):
            tiers[t] = tiers.get(t, 0) + 1
    pat = np.array([r["pat_median_raw_ms"] for r in ok])
    off = np.array([r["offset_ms"] for r in ok])
    dvals = np.array([d for *_, d in reg])
    rvals = np.array([rt for _, _, rt, _ in reg])
    fin = np.isfinite(dvals)
    S = {"scan": a.scan, "cases": len(rows), "cases_measured": len(ok),
         "apply_min_tier": ok[0].get("apply_min_tier"), "tooth_tiers": tiers,
         "slips": int(st_raw.size), "step_raw_ms_median": round(float(np.median(st_raw)), 1),
         "step_corr_ms_median": round(float(np.median(st_cor)), 1),
         "full_teeth_cases": len(full), "slope_raw_ppm_median": round(float(np.median(sl_raw)), 1),
         "slope_corr_ppm_median": round(float(np.median(sl_cor)), 1),
         "wander_groups": {g: grp.count(g) for g in ("teeth", "prior", "none")},
         "rate_cases": len(reg), "rate_vs_snuadc_drift_r": round(float(np.corrcoef(dvals[fin], rvals[fin])[0, 1]), 2) if fin.sum() > 2 else None,
         "by_setup": {k: {"n": len(v), "median_ppm": round(float(np.median(v)), 2), "iqr": [round(float(x), 2) for x in np.quantile(v, [0.25, 0.75])]} for k, v in by_setup.items()},
         "offset_ms_median_iqr": [round(float(x), 1) for x in np.quantile(off, [0.5, 0.25, 0.75])],
         "adopted_pat_ms": round(float(np.median(pat - off)), 1)}
    ex = {}
    for c in a.examples:
        r = next(x for x in ok if x["case"] == c)
        rt = rate_of(r)
        ex[str(c)] = {"kept": r["n_kept"], "rejected": r["n_rejected"], "inserted": r["n_inserted"],
                      "uncertain": r.get("n_uncertain"), "rate_ppm_median": round(rt, 1) if rt else None,
                      "tiers": r.get("tiers")}
    S["examples"] = ex
    S["method"] = "rebuilt" if rebuilt else "detect"
    if rebuilt:
        S["rate_source"] = {k: sum(1 for r in ok if r.get("rate_source") == k) for k in ("teeth", "reported", "none")}
        S["uncertain_not_corrected"] = sum(r.get("n_uncertain") or 0 for r in ok)
        S["slips_after_old_detector"] = sum(r.get("slips_after") or 0 for r in ok)
        # each corrected step by how it was found: how many leave more than 15 ms (a third of a frame or more)
        from collections import Counter
        allk, bad, badc = Counter(), Counter(), []
        for r in ok:
            for k_, sr, sc in zip(r.get("step_kinds", []), r["step_at_slips_raw_ms"], r["step_at_slips_corr_ms"]):
                allk[k_] += 1
                if abs(sc) > 15:
                    bad[k_] += 1
                    badc.append((r["case"], k_, sr, sc))
        S["corrected_steps_by_kind"] = {k_: {"steps": v, "left_over_15ms": bad[k_]} for k_, v in allk.most_common()}
        S["left_over_15ms_cases"] = Counter(c_ for c_, *_ in badc).most_common(10)
        # and by the arterial verdict at the step (slip, neither, circulation, no line)
        ka, kb = Counter(), Counter()
        for r in ok:
            for k_, a_, sc in zip(r.get("step_kinds", []), r.get("step_arterial", []), r["step_at_slips_corr_ms"]):
                ka[(k_, a_)] += 1
                kb[(k_, a_)] += int(abs(sc) > 15)
        if ka:
            S["corrected_steps_by_kind_and_arterial"] = {f"{k_} / {a_}": {"steps": v, "left_over_15ms": kb[(k_, a_)]}
                                                         for (k_, a_), v in sorted(ka.items())}
        mid = np.array([x for r in ok for x in r.get("step_mid_corr_ms", [])])
        S["control_mid_tooth"] = {"places": int(mid.size), "over_15ms": int(np.sum(np.abs(mid) > 15)),
                                  "abs_ms_median": round(float(np.median(np.abs(mid))), 1) if mid.size else None}
        sc_all = np.array([x for r in ok for x in r["step_at_slips_corr_ms"]])
        S["corrected_abs_ms_median"] = round(float(np.median(np.abs(sc_all))), 1)
    (out / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps(S, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(20, 13.5))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1])
    for k, c in enumerate(a.examples):
        axx = fig.add_subplot(gs[k, :2])
        r = next(x for x in ok if x["case"] == c)
        t, raw, cor = (np.array(v) for v in r["series"])
        cor = cor - np.median(cor) + np.median(raw)
        axx.scatter(t / 60, raw, s=3, color="C0", label="measured PAT (15-beat medians)")
        axx.scatter(t / 60, cor, s=3, color="C3", label="after sawtooth correction (shifted to the same median for comparison)")
        for x in r["slips_t"]:
            axx.axvline(x / 60, color="0.75", lw=0.8)
        e = ex[str(c)]
        tl = "".join(sorted(set(e["tiers"] or [])))
        if rebuilt:
            for x in r.get("uncertain_t", []):
                axx.axvline(x / 60, color="C1", lw=0.8, ls="--")
            axx.set_title(f"{'abc'[k]}. Case {c}: {e['kept']} steps corrected, {e['uncertain']} placed only within an "
                          f"interval (orange dashed, not corrected); creep {e['rate_ppm_median']} ppm; tiers {tl}; "
                          "grey = slips", fontsize=10)
        else:
            axx.set_title(f"{'abc'[k]}. Case {c}: {e['kept']} slips kept, {e['rejected']} rejected, {e['inserted']} inserted; "
                          f"tooth rates {e['rate_ppm_median']} ppm (median); tiers {tl}; grey = detected slips", fontsize=10)
        axx.set_ylabel("PAT (ms)")
        if k == 0:
            axx.legend(fontsize=8, loc="lower left")
        if k == len(a.examples) - 1:
            axx.set_xlabel("minutes")
    ax = fig.add_subplot(gs[0, 2])
    b = np.arange(-20, 51, 1)
    ax.hist(np.clip(st_raw, -20, 50), bins=b, color="C0", alpha=0.6, label=f"measured (median {S['step_raw_ms_median']} ms)")
    ax.hist(np.clip(st_cor, -20, 50), bins=b, color="C3", alpha=0.6, label=f"corrected (median {S['step_corr_ms_median']} ms)")
    ax.set_xlabel("PAT step at each modelled slip (ms, 15 beats after minus before)")
    ax.set_ylabel("slips")
    ax.set_title(f"d. The frame steps are removed ({st_raw.size} slips, {len(ok)} cases)")
    ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[1, 2])
    b = np.arange(-80, 64, 4)
    ax.hist(np.clip(sl_raw, -80, 60), bins=b, color="C0", alpha=0.6, label=f"measured (median {S['slope_raw_ppm_median']:.0f} ppm)")
    ax.hist(np.clip(sl_cor, -80, 60), bins=b, color="C3", alpha=0.6, label=f"corrected (median {S['slope_corr_ppm_median']:.0f} ppm)")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("PAT slope inside the teeth (ppm = us per s), median per case")
    ax.set_ylabel("cases")
    ax.set_title(f"e. The creep is removed ({len(full)} cases with full teeth)")
    ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[2, 2])
    col = {"teeth": "C3", "prior": "C1", "none": "0.6"}
    for g in ("none", "prior", "teeth"):
        sel = [i for i, x in enumerate(grp) if x == g]
        ax.scatter([ok[i]["p2p_raw_ms"] for i in sel], [ok[i]["p2p_corr_ms"] for i in sel], s=12 if g != "none" else 6, color=col[g])
    ax.plot([0, 170], [0, 170], "k", lw=0.8)
    ax.set_xlim(0, 170)
    ax.set_ylim(0, 170)
    ax.set_xlabel("wander of 1-min PAT medians, measured (5-95%, ms)")
    ax.set_ylabel("after correction (ms)")
    ax.set_title("f. Slow wander (includes physiology): red creep corrected,\norange tooth from two slips (steps only), grey no sawtooth"
                 if rebuilt else
                 "f. Slow wander (includes physiology): red full teeth,\norange setup prior, grey untouched (no slips / no rate)")
    ax = fig.add_subplot(gs[3, 0])
    names = list(by_setup)[:10]
    for j, s_ in enumerate(names):
        pts = [(d, rt) for _, ss, rt, d in reg if ss == s_]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=18, color=plt.cm.tab10(j % 10), label=f"{s_} ({len(pts)})")
    shown = [(d, rt) for _, ss, rt, d in reg if ss in names]
    off_ax = sum(1 for d, rt in shown if not (-100 <= d <= 40 and 0 <= rt <= 80))
    ax.set_xlim(-100, 40)
    ax.set_ylim(0, 80)
    ax.text(0.01, 0.97, f"{off_ax} cases outside the axes not shown", transform=ax.transAxes, fontsize=7, va="top", color="0.4")
    ax.set_xlabel("SNUADC clock drift vs PC (ppm), same case")
    ax.set_ylabel("sawtooth rate (ppm)")
    ax.set_title("g. Sawtooth rate does not follow the SNUADC drift,\nbut groups by recording setup (port, SNUADC mode)")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax = fig.add_subplot(gs[3, 1])
    for j, s_ in enumerate(names):
        v = by_setup[s_]
        q = np.quantile(v, [0.25, 0.5, 0.75])
        ax.errorbar([j], [q[1]], yerr=[[q[1] - q[0]], [q[2] - q[1]]], fmt="o", color="C0", capsize=3)
        ax.text(j, q[2] + 1.5, f"n={len(v)}", ha="center", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
    ax.set_ylim(0, 85)
    ax.set_ylabel("sawtooth rate (ppm), median and IQR")
    ax.set_title("h. Rate per recording setup (cases with >=3 kept slips)")
    ax = fig.add_subplot(gs[3, 2])
    ax.hist(pat, bins=np.arange(0, 905, 10), color="C0", alpha=0.7, label="measured median PAT per case")
    ax.axvline(S["adopted_pat_ms"], color="C3", lw=2, label=f"adopted mean PAT (here {S['adopted_pat_ms']:.0f} ms: user-supplied)")
    q = S["offset_ms_median_iqr"]
    ax.set_xlabel("PAT (ms)")
    ax.set_ylabel("cases")
    ax.set_title(f"i. Offset: every case snaps to the adopted value\n(combined offset here {q[0]:.0f} ms, IQR {q[1]:.0f}-{q[2]:.0f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "ppg-correction-check.png", dpi=100)


def timeseries(a) -> None:
    from scipy import signal
    from pat import ECG, PPG, r_peaks, sample_times
    from pat_res import ecg_subsample
    from physiosync import match_beats
    from vitaldb_reader.vital_file import read_vital
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def feet(raw, fs):
        """PPG foot per pulse: the intersecting tangent (the tangent at the steepest upstroke of the 0.5-10 Hz
        band-passed signal meets the horizontal through the preceding minimum)."""
        sos = signal.butter(2, (0.5, 10), btype="bandpass", fs=fs, output="sos")
        y = signal.sosfiltfilt(sos, raw.astype(float))
        d = np.gradient(y)
        up, _ = signal.find_peaks(d, distance=int(0.3 * fs), prominence=0.3 * (np.quantile(d, 0.99) - np.quantile(d, 0.5)))
        out_ = []
        for u in up:
            lo = max(u - int(0.4 * fs), 0)
            m = lo + int(np.argmin(y[lo:u + 1]))
            out_.append(u - (y[u] - y[m]) / d[u] if d[u] > 0 else np.nan)
        return np.array(out_)

    res = {}
    for c in a.cases:
        v = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={ECG, PPG})
        e, p = v.tracks[ECG], v.tracks[PPG]
        fs = float(e.srate)
        te, tp = sample_times(e), sample_times(p)
        t0 = float(np.floor(min(te[0], tp[0])))
        # contiguous ECG stretches; beats within each
        rt_all, pt_all = [], []
        gap = np.flatnonzero(np.abs(np.diff(te) - 1 / fs) > 0.01 / fs)
        be = np.r_[0, gap + 1, te.size]
        gaps = [(float(te[g] - t0), float(te[g + 1] - t0)) for g in gap if te[g + 1] - te[g] > 1.0]
        for s_, e_ in zip(be[:-1], be[1:]):
            if e_ - s_ < 60 * fs:
                continue
            xe = e.physical()[s_:e_].astype(float)
            rp = r_peaks(xe, fs)
            if rp.size < 10:
                continue
            rt_all.append(te[s_] + ecg_subsample(xe, rp, fs) / fs)
        gp = np.flatnonzero(np.abs(np.diff(tp) - 1 / float(p.srate)) > 0.01 / float(p.srate))
        bp = np.r_[0, gp + 1, tp.size]
        for s_, e_ in zip(bp[:-1], bp[1:]):
            if e_ - s_ < 60 * p.srate:
                continue
            f = feet(np.asarray(p.values[s_:e_]), float(p.srate))
            f = f[np.isfinite(f)]
            pt_all.append(tp[s_] + f / float(p.srate))
        rt, pt = np.concatenate(rt_all), np.concatenate(pt_all)
        i, j, _ = match_beats(rt, pt)
        tr, pat = rt[i] - t0, (pt[j] - rt[i]) * 1000
        bins = np.arange(0, tr.max() + 30, 30)
        k = np.digitize(tr, bins) - 1
        tm = np.array([np.median(pat[k == b]) if np.sum(k == b) >= 5 else np.nan for b in range(bins.size - 1)])
        rtm = e.record_times
        resid = rtm - rtm[0] - e.record_offsets[:rtm.size] / fs
        dd = np.diff(resid)
        corr = rtm[1:][(np.abs(dd) > 0.03) & (np.abs(dd) < 2)] - t0
        corr = corr[np.r_[True, np.diff(corr) > 60]] if corr.size else corr   # one correction may span records
        res[str(c)] = {"median_ms": round(float(np.median(pat)), 1), "beats": int(pat.size),
                       "t_h": np.round((bins[:-1] + 15) / 3600, 4).tolist(), "pat_ms": [None if not np.isfinite(x) else round(float(x), 1) for x in tm],
                       "corrections_h": np.round(np.array(corr) / 3600, 4).tolist(), "gaps_h": [[round(g0 / 3600, 4), round(g1 / 3600, 4)] for g0, g1 in gaps]}
        print(c, res[str(c)]["median_ms"], len(res[str(c)]["corrections_h"]))
    (out / "timeseries.json").write_text(json.dumps(res) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(a.cases)
    fig, axs = plt.subplots((n + 1) // 2, 2, figsize=(13.5, 2.75 * ((n + 1) // 2) + 0.6), squeeze=False)
    for ax, c in zip(axs.ravel(), a.cases):
        r = res[str(c)]
        y = np.array([np.nan if v is None else v for v in r["pat_ms"]], float)
        ax.scatter(r["t_h"], y, s=3, color="C0")
        for g0, g1 in r["gaps_h"]:
            ax.axvspan(g0, g1, color="0.85", lw=0)
        for x in r["corrections_h"]:
            ax.axvline(x, color="C1", ls="--", lw=0.9)
        ax.set_ylim(r["median_ms"] - 120, r["median_ms"] + 120)
        ax.set_title(f"case {c}: median PAT {r['median_ms']:.0f} ms", fontsize=10)
        ax.tick_params(labelsize=8)
    for ax in axs[:, 0]:
        ax.set_ylabel("PAT, ms (median per 30 s)", fontsize=9)
    for ax in axs[-1]:
        ax.set_xlabel("hours from case start", fontsize=9)
    for ax in axs.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle("Pulse-foot PAT through each case (blue); SNUADC clock corrections (orange dashed); grey = gaps > 1 s. "
                 "Y axis: case median ±120 ms", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out / "pat-timeseries.png", dpi=100)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inputs")
    p.add_argument("--per-case", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("check")
    p.add_argument("--scan", required=True)
    p.add_argument("--units", required=True)
    p.add_argument("--modes", required=True)
    p.add_argument("--examples", type=int, nargs="+", default=[2156, 6241, 1094])
    p.add_argument("--out", required=True)
    p = sub.add_parser("timeseries")
    p.add_argument("--root", required=True)
    p.add_argument("--cases", type=int, nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"inputs": inputs, "check": check, "timeseries": timeseries}[a.cmd](a)


if __name__ == "__main__":
    main()
