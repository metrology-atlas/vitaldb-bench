# SPDX-License-Identifier: MIT
"""Worked example: pulse pressure variation gated by ventilator breaths, and how much the clocks matter.

    python examples/ppv_by_breath.py scan --root R --cases 1,2,3 --modes MODES --ports PORTS --step-rates RATES \
        --out part.jsonl [--time-budget 170]
    python examples/ppv_by_breath.py agg part*.jsonl --out DIR

Breath onsets come from the Primus airway pressure (AWP) on the Primus's own stamps; beats and their pulse
pressure come from the SNUADC arterial pressure (ART), beats found on SNUADC ECG II. The two devices have
separate clocks ([[link:primus-vs-pc]], [[link:snuadc-step-vs-pc]]). Per recording, breaths are matched to beats
under three timings:

  recorded  SNUADC samples on their recorded stamps (the stamp sawtooth left in);
  model     SNUADC samples on the stamp model (physiosync.stamp_clock; rules as tools/sa_corrected.py model2);
  aligned   model, plus the Primus moved onto the SNUADC timeline by the AWP-to-CVP lag in each 10-min window
            (as tools/vent_sync_pilot.py), when the recording has CVP.

Per breath (onset to next onset, 2-10 s, at least 3 beats): PPV = (PPmax - PPmin) / mean(PPmax, PPmin), with
each beat assigned to a breath by the time of its systolic peak. Per recording: the median PPV, the per-breath
change between timings, the share of breaths whose PPV crosses 13%, the phase of the pulse-pressure maximum within
the breath (first-harmonic fit), and a sweep of deliberate shifts of the breath onsets (-2 to +2 s).

scan writes one JSON line per recording (aggregates of that recording; stays on this machine). agg writes
DIR/summary.json (aggregates only) and DIR/ppv_by_breath.png.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from pat import r_peaks, sample_times  # noqa: E402
from vent_sync_pilot import G, binned, breath_onsets, env_series, harmonic, shape_lag, trig_avg, xlag  # noqa: E402

warnings.filterwarnings("ignore")
SNU = {"ecg": "SNUADC/ECG_II", "art": "SNUADC/ART", "cvp": "SNUADC/CVP"}
SHIFTS = np.round(np.arange(-2.0, 2.0001, 0.1), 2)
PPV_CUT = 0.13
R2_MIN = 0.02     # first-harmonic fit of pulse pressure must explain at least this share to give a phase


def model_times(tr, port, mode, step_rates):
    from physiosync import corrected_times, stamp_clock
    st = tr.record_offsets[: tr.record_times.size]
    clk = stamp_clock(tr.record_times, st, float(tr.srate), correction_sign=-1)
    if clk["rate_source"] == "fit":
        ev = clk["corrections"]["median_event_s"]
        pr = step_rates.get(port, step_rates.get("*", 0.0))
        fall = pr if (ev is not None and ev <= -0.4) or (ev is None and mode != "small") else 0.0
        clk = stamp_clock(tr.record_times, st, float(tr.srate), rate_ppm=fall, correction_sign=-1)
    return corrected_times(clk, st, tr.record_sizes[: st.size])


def beats(xe, xa, fs):
    """Per beat: index of the systolic peak in ART, systolic and diastolic pressure (beats from ECG R peaks)."""
    ok = np.isfinite(xe)
    rp = r_peaks(np.where(ok, xe, np.nanmedian(xe)), fs)
    rows = []
    for i in rp:
        s0, s1, d0, d1 = i + int(0.1 * fs), i + int(0.5 * fs), max(i - int(0.15 * fs), 0), i + int(0.15 * fs)
        if s1 >= xa.size:
            break
        seg = xa[s0:s1]
        if not np.isfinite(seg).any() or not np.isfinite(xa[d0:d1]).any():
            continue
        j = s0 + int(np.nanargmax(seg))
        rows.append((j, xa[j], np.nanmin(xa[d0:d1])))
    b = np.array(rows, float).reshape(-1, 3)
    sbp, dbp = b[:, 1], b[:, 2]
    good = (sbp < 300) & (dbp > 10) & (sbp - dbp > 10) & (sbp - dbp < 150)
    return b[good, 0].astype(np.int64), sbp[good] - dbp[good]


def ppv_per_breath(bt, pp, on):
    """PPV of each breath k (onset k to onset k+1); NaN where the breath is not usable."""
    out = np.full(max(on.size - 1, 0), np.nan)
    if on.size < 2:
        return out
    k = np.searchsorted(on, bt, side="right") - 1
    T = np.diff(on)
    ok = (k >= 0) & (k < on.size - 1)
    k, v = k[ok], pp[ok]
    order = np.argsort(k, kind="stable")
    k, v = k[order], v[order]
    starts = np.searchsorted(k, np.arange(on.size - 1))
    ends = np.searchsorted(k, np.arange(on.size - 1), side="right")
    for b in np.flatnonzero((ends - starts >= 3) & (T > 2) & (T < 10)):
        x = v[starts[b] : ends[b]]
        hi, lo = x.max(), x.min()
        out[b] = (hi - lo) / ((hi + lo) / 2)
    return out


def cvp_lag(ta, xa, tc, xc, on):
    """AWP-to-CVP lag (s) at 10-min windows, as tools/vent_sync_pilot.py; returns (times, lags) of good windows."""
    t0 = float(np.floor(min(ta[0], tc[0])))
    n50 = int((max(ta[-1], tc[-1]) - t0) / G) + 1
    okc = np.isfinite(xc)
    cvp50, awp50 = binned(tc[okc], xc[okc], t0, G, n50), binned(ta, xa, t0, G, n50)
    cv10 = np.nanmean(np.r_[cvp50, np.full((-n50) % 5, np.nan)].reshape(-1, 5), axis=1)
    aw10 = np.nanmean(np.r_[awp50, np.full((-n50) % 5, np.nan)].reshape(-1, 5), axis=1)
    ea, ec = env_series(aw10), env_series(cv10)
    coarse = []
    if ea is not None and ec is not None:
        W = 36000
        for s in range(0, ea.size - W // 2, W // 2):
            l, r = xlag(ea[s : s + W], ec[s : s + W], 300)
            if l is not None and r >= 0.3:
                coarse.append((t0 + (s + W / 2) / 10, l / 10))
    if not coarse:
        return np.empty(0), np.empty(0)
    taus = np.arange(-4.0, 12.0, G)
    tt, ll, prev = [], [], None
    for ws in np.arange(on[0], on[-1] - 600, 300):
        o = on[(on >= ws) & (on < ws + 600)]
        if o.size < 40:
            continue
        T, mid = float(np.median(np.diff(o))), ws + 300
        centre = prev[1] if prev is not None and mid - prev[0] <= 900 else min(coarse, key=lambda x: abs(x[0] - mid))[1]
        lo, hi = centre - T / 2, centre + T / 2
        lag, r = shape_lag(trig_avg(awp50, t0, o, taus), trig_avg(cvp50, t0, o, taus), taus, lo, hi)
        h1 = shape_lag(trig_avg(awp50, t0, o[0::2], taus), trig_avg(cvp50, t0, o[0::2], taus), taus, lo, hi)[0]
        h2 = shape_lag(trig_avg(awp50, t0, o[1::2], taus), trig_avg(cvp50, t0, o[1::2], taus), taus, lo, hi)[0]
        if lag is not None and r >= 0.8 and h1 is not None and h2 is not None and abs(h1 - h2) < 0.1:
            tt.append(mid); ll.append(lag); prev = (mid, lag)
    return np.array(tt), np.array(ll)


def compare(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20:
        return None
    d = np.abs(a[m] - b[m])
    return {"n": int(m.sum()), "abs_change_median": float(np.median(d)), "abs_change_p90": float(np.percentile(d, 90)),
            "share_changed_1pt": float(np.mean(d >= 0.01)), "share_cross_13": float(np.mean((a[m] > PPV_CUT) != (b[m] > PPV_CUT)))}


def scan_case(c, a, modes, ports, step_rates):
    from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
    v = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={*SNU.values(), "Primus/AWP"})
    aw = v.tracks.get("Primus/AWP")
    if aw is None or not aw.record_times.size or any(v.tracks.get(SNU[k]) is None for k in ("ecg", "art")):
        return {"case": c, "skip": "no AWP, ART or ECG"}
    ta = sample_times(aw)
    xa_ = np.nan_to_num(aw.physical().astype(np.float64))
    on = breath_onsets(ta, xa_, float(aw.srate))
    gaps = np.diff(ta)
    res = {"case": c, "breaths": int(on.size), "hours": round(float(ta[-1] - ta[0]) / 3600, 3),
           "primus_gaps_over_5s": int((gaps > 5).sum()), "primus_gap_s": round(float(gaps[gaps > 5].sum()), 1)}
    if on.size < 100:
        res["skip"] = "fewer than 100 mechanical breaths"
        return res
    e, art = v.tracks[SNU["ecg"]], v.tracks[SNU["art"]]
    if e.srate != art.srate or e.record_times.size != art.record_times.size:
        res["skip"] = "ECG and ART not on one grid"
        return res
    fs = float(art.srate)
    bi, pp = beats(e.physical().astype(np.float64), art.physical().astype(np.float64), fs)
    port, mode = ports.get(c), modes.get(c)
    t_rec, t_mod = sample_times(art), model_times(art, port, mode, step_rates)
    times = {"recorded": (t_rec[bi], on), "model": (t_mod[bi], on)}
    cv = v.tracks.get(SNU["cvp"])
    if cv is not None and cv.record_times.size and cv.srate == art.srate and cv.record_times.size == art.record_times.size:
        tl, ll = cvp_lag(ta, xa_, t_mod, cv.physical().astype(np.float64), on)
        if tl.size >= 3:
            times["aligned"] = (t_mod[bi], on + np.interp(on, tl, ll))
            res["cvp_lag"] = {"windows": int(tl.size), "median_s": round(float(np.median(ll)), 3),
                              "range_s": round(float(ll.max() - ll.min()), 3),
                              "slope_ppm": round(float(np.polyfit(tl, ll, 1)[0] * 1e6), 1) if tl.size >= 5 else None}
    del v
    d = t_mod[bi] - t_rec[bi]
    res["beats"] = int(bi.size)
    res["model_minus_recorded_s"] = {"median": round(float(np.median(d)), 3), "p95_abs": round(float(np.percentile(np.abs(d - np.median(d)), 95)), 3)}
    ppvs, res["cond"] = {}, {}
    for name, (bt, o) in times.items():
        p = ppv_per_breath(bt, pp, o)
        ppvs[name] = p
        h = harmonic(bt, pp, o)
        res["cond"][name] = {"breaths_used": int(np.isfinite(p).sum()), "ppv_median": float(np.nanmedian(p)) if np.isfinite(p).any() else None,
                             "share_over_13": float(np.nanmean(p > PPV_CUT)) if np.isfinite(p).any() else None,
                             "pp_max_phase_s": round(h["lag_s"], 3) if h else None, "pp_harmonic_r2": round(h["r2"], 4) if h else None,
                             "pp_harmonic_amp": round(h["amp"], 3) if h else None, "breath_s": round(h["T"], 2) if h else None}
    res["recorded_vs_model"] = compare(ppvs["recorded"], ppvs["model"])
    if "aligned" in ppvs:
        res["model_vs_aligned"] = compare(ppvs["model"], ppvs["aligned"])
        res["recorded_vs_aligned"] = compare(ppvs["recorded"], ppvs["aligned"])
    ref = "aligned" if "aligned" in times else "model"
    bt, o = times[ref]
    base = ppvs[ref]
    sweep = []
    for s in SHIFTS:
        p = ppv_per_breath(bt, pp, o + s)
        m = np.isfinite(p) & np.isfinite(base)
        sweep.append([float(s), float(np.nanmedian(p)) if np.isfinite(p).any() else None,
                      float(np.median(np.abs(p[m] - base[m]))) if m.sum() >= 20 else None,
                      float(np.mean((p[m] > PPV_CUT) != (base[m] > PPV_CUT))) if m.sum() >= 20 else None])
    res["sweep_ref"], res["sweep"] = ref, sweep
    return res


def scan(a) -> None:
    modes = {int(k): v for k, v in json.loads(Path(a.modes).read_text()).items()}
    ports = {int(k): (v or {}).get("port") for k, v in json.loads(Path(a.ports).read_text()).items()}
    step_rates = json.loads(Path(a.step_rates).read_text())
    out = Path(a.out)
    done = {json.loads(l)["case"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    t_start = time.time()
    with open(out, "a") as fh:
        for c in [int(x) for x in a.cases.split(",") if x]:
            if c in done:
                continue
            if a.time_budget and time.time() - t_start > a.time_budget:
                break
            t1 = time.time()
            try:
                r = scan_case(c, a, modes, ports, step_rates)
            except Exception as err:  # noqa: BLE001
                r = {"case": c, "error": f"{type(err).__name__}: {err}"[:300]}
            r["secs"] = round(time.time() - t1, 1)
            fh.write(json.dumps(r) + "\n")
            fh.flush()


def agg(a) -> None:
    rows = [json.loads(l) for p in a.parts for l in Path(p).read_text().splitlines() if l.strip()]
    ok = [r for r in rows if "cond" in r]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def q(x, ps=(10, 50, 90)):
        x = np.array([v for v in x if v is not None and np.isfinite(v)], float)
        return [round(float(np.percentile(x, p)), 4) for p in ps] + [int(x.size)] if x.size else None

    summ = {"recordings_read": len(rows), "used": len(ok),
            "skipped": {k: sum(1 for r in rows if r.get("skip") == k) for k in {r.get("skip") for r in rows if r.get("skip")}},
            "errors": sum(1 for r in rows if "error" in r),
            "with_cvp_alignment": sum(1 for r in ok if "aligned" in r["cond"]),
            "breaths_used_total": sum(r["cond"]["model"]["breaths_used"] for r in ok),
            "primus_gaps_per_hour_median": q([r["primus_gaps_over_5s"] / r["hours"] for r in ok if r["hours"] > 0])}
    for name in ("recorded", "model", "aligned"):
        cs = [r["cond"][name] for r in ok if name in r["cond"]]
        summ[name] = {"ppv_median_p10_50_90": q([c["ppv_median"] for c in cs]),
                      "pp_max_phase_s_p10_50_90": q([c["pp_max_phase_s"] for c in cs if (c["pp_harmonic_r2"] or 0) >= R2_MIN]),
                      "pp_max_phase_iqr_s": None}
        ph = np.array([c["pp_max_phase_s"] for c in cs if c["pp_max_phase_s"] is not None and (c["pp_harmonic_r2"] or 0) >= R2_MIN])
        if ph.size:
            summ[name]["pp_max_phase_iqr_s"] = round(float(np.subtract(*np.percentile(ph, [75, 25]))), 3)
    for pair in ("recorded_vs_model", "model_vs_aligned", "recorded_vs_aligned"):
        cs = [r[pair] for r in ok if r.get(pair)]
        summ[pair] = {k: q([c[k] for c in cs]) for k in ("abs_change_median", "abs_change_p90", "share_changed_1pt", "share_cross_13")}
        summ[pair]["recordings_crossing_13_median"] = None
    for pair, (x, y) in {"recorded_vs_aligned": ("recorded", "aligned"), "recorded_vs_model": ("recorded", "model")}.items():
        both = [(r["cond"][x]["ppv_median"], r["cond"][y]["ppv_median"]) for r in ok if y in r["cond"] and r["cond"][x]["ppv_median"] is not None and r["cond"][y]["ppv_median"] is not None]
        summ[pair]["recordings_crossing_13_median"] = [sum((p > PPV_CUT) != (q_ > PPV_CUT) for p, q_ in both), len(both)]
    al = [r for r in ok if r.get("sweep_ref") == "aligned"] or ok   # the sweep is about misalignment from the best timing
    sw = np.array([[s[2] if s[2] is not None else np.nan for s in r["sweep"]] for r in al], float)
    swc = np.array([[s[3] if s[3] is not None else np.nan for s in r["sweep"]] for r in al], float)
    summ["sweep"] = {"recordings": len(al), "shift_s": SHIFTS.tolist(), "abs_change_median": np.round(np.nanmedian(sw, 0), 4).tolist(),
                     "abs_change_p25": np.round(np.nanpercentile(sw, 25, 0), 4).tolist(),
                     "abs_change_p75": np.round(np.nanpercentile(sw, 75, 0), 4).tolist(),
                     "share_cross_13_median": np.round(np.nanmedian(swc, 0), 4).tolist(),
                     "share_cross_13_p25": np.round(np.nanpercentile(swc, 25, 0), 4).tolist(),
                     "share_cross_13_p75": np.round(np.nanpercentile(swc, 75, 0), 4).tolist()}
    summ["sawtooth_p95_abs_s"] = q([r["model_minus_recorded_s"]["p95_abs"] for r in ok])
    summ["cvp_lag_slope_ppm"] = q([r["cvp_lag"]["slope_ppm"] for r in ok if r.get("cvp_lag", {}).get("slope_ppm") is not None])
    summ["cvp_lag_median_s"] = q([r["cvp_lag"]["median_s"] for r in ok if r.get("cvp_lag")])
    summ["cvp_lag_range_s"] = q([r["cvp_lag"]["range_s"] for r in ok if r.get("cvp_lag")])
    summ["secs_per_recording_median"] = q([r.get("secs") for r in rows])
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    figure(ok, summ, out)


def figure(ok, summ, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"recorded": "#9e9e9e", "model": "#3f51b5", "aligned": "#e65100"}
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.3))
    s = summ["sweep"]
    x = np.array(s["shift_s"])
    ax[0].fill_between(x, np.array(s["share_cross_13_p25"]) * 100, np.array(s["share_cross_13_p75"]) * 100, color="#3f51b5", alpha=0.2, lw=0)
    ax[0].plot(x, np.array(s["share_cross_13_median"]) * 100, color="#3f51b5")
    ax[0].set_xlabel("breath onsets moved from the aligned timing (s)")
    ax[0].set_ylabel("breaths whose PPV crosses 13% (%)")
    ax[0].set_title("A. Sensitivity to misalignment", fontsize=9, loc="left")
    bins = np.linspace(-3, 3, 61)
    for name in ("recorded", "model", "aligned"):
        ph = [r["cond"][name]["pp_max_phase_s"] for r in ok if name in r["cond"] and r["cond"][name]["pp_max_phase_s"] is not None
              and (r["cond"][name]["pp_harmonic_r2"] or 0) >= R2_MIN]
        if ph:
            ax[1].hist(ph, bins=bins, histtype="step", color=col[name], lw=1.3, density=True, label=f"{name} ({len(ph):,})")
    ax[1].set_xlabel("PP maximum after the breath onset (s)")
    ax[1].set_yticks([])
    ax[1].legend(fontsize=7, frameon=False)
    ax[1].set_title("B. Where in the breath pulse pressure peaks", fontsize=9, loc="left")
    pairs = [(r["cond"]["recorded"]["ppv_median"], r["cond"]["aligned"]["ppv_median"]) for r in ok if "aligned" in r["cond"]]
    pairs = np.array([p for p in pairs if None not in p], float) * 100
    ax[2].scatter(pairs[:, 1], pairs[:, 0], s=4, color="#3f51b5", alpha=0.5, lw=0)
    lim = [0, min(40, float(np.nanpercentile(pairs, 99.5)) + 2)]
    ax[2].plot(lim, lim, color="0.5", lw=0.8)
    for f in (ax[2].axvline, ax[2].axhline):
        f(13, color="#e65100", lw=0.8, ls=":")
    ax[2].set_xlim(lim); ax[2].set_ylim(lim)
    ax[2].set_xlabel("median PPV, Primus aligned by CVP (%)")
    ax[2].set_ylabel("median PPV, recorded stamps (%)")
    ax[2].set_title(f"C. Per recording ({len(pairs):,} with CVP)", fontsize=9, loc="left")
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Pulse pressure variation by ventilator breath: {summ['used']:,} recordings, "
                 f"{summ['breaths_used_total']:,} breaths", fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "ppv_by_breath.png", dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    for k in ("--root", "--cases", "--modes", "--ports", "--step-rates", "--out"):
        s.add_argument(k, required=True)
    s.add_argument("--time-budget", type=float, default=0)
    g = sub.add_parser("agg")
    g.add_argument("parts", nargs="+")
    g.add_argument("--out", required=True)
    a = ap.parse_args()
    scan(a) if a.cmd == "scan" else agg(a)


if __name__ == "__main__":
    main()
