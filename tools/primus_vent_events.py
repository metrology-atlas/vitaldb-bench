# SPDX-License-Identifier: MIT
"""Which breath is which, from events that are not periodic: pauses and restarts of mechanical ventilation.

    bash tools/run_tool_scan.sh primus_vent_events --modes $W/results/sync_full/snuadc_mode_label.json \
        --ports $W/results/sync_full/snuadc_mode.json --step-rates $W/results/pat/snuadc_step_rates.json
    (writes $W/primus_vent_events/part_*.jsonl; rerunning resumes)
    python tools/primus_vent_events.py agg $W/primus_vent_events/part_*.jsonl --out $W/primus_vent_events

Why. Matching breaths cannot tell one breath from the next when ventilation is regular (atlas study
primus-awp-anchoring), so the AWP-to-reference lag of tools/primus_sync_accuracy.py is known only to within a
breath. A pause in ventilation is not periodic: the airway pressure stops cycling, and the breathing modulation of
CVP and of the arterial pressure stops a lag later. Timing that stop and the restart gives the lag without the
ambiguity, to well within a breath.

Events. Breath onsets on the AWP stamps (vent_sync_pilot.breath_onsets). An event is a run of no breaths longer
than max(15 s, 3 x the local breath period) while the AWP stream continues (no AWP stamp gap over 5 s inside it),
or the start of ventilation (first breath after at least 60 s of AWP without breaths), or its end. Each event is
timed by its edges: the last breath before the pause (stop) and the first after (restart).

Lag at an event. Both signals on a 10 Hz grid; the reference (CVP or ART, SNUADC on its stamp model, as in
primus_sync_accuracy) is low-passed over 2 s to remove the heartbeat. Each is band-passed 0.1-0.7 Hz and its
envelope taken (RMS over 4 s). Within +-150 s of the event the two envelopes are cross-correlated over lags
-3..+12 s: lag = reference time minus AWP stamp, as in primus_sync_accuracy (bias = -lag). Reported with the peak
correlation and the correlation one breath period either side of the peak (a pause makes the peak unique; if the
neighbours are as high, the event did not resolve the breath).

Resolved (agg): peak r >= 0.5, at least 0.15 above the correlation one breath either side, and not at the edge of
the lag range; k is summarised only where the matched window is within 600 s of the event.

Per event it also reports the release-wide matched lag at the nearest window of the same AWP stretch (rerun here
with the same code) and k = (matched lag - event lag) / T, the number of breaths the matching is off.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage, signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ppv_by_breath as P  # noqa: E402
from pat import sample_times  # noqa: E402
from primus_sync_accuracy import lowpass_mean  # noqa: E402
from vent_sync_pilot import binned  # noqa: E402

DT = 0.1                      # envelope grid (s)
HALF_WIN_S = 150.0            # window either side of an event
LAG_RANGE = (-3.0, 12.0)
_SOS = signal.butter(2, (0.1, 0.7), btype="bandpass", fs=1 / DT, output="sos")


def envelope(y):
    ok = np.isfinite(y)
    if ok.mean() < 0.5:
        return None
    f = np.interp(np.arange(y.size), np.flatnonzero(ok), y[ok])
    f = signal.sosfiltfilt(_SOS, f)
    e = np.sqrt(ndimage.uniform_filter1d(f * f, int(4 / DT)))
    e[~ok] = np.nan
    return e


def lag_at(ea, er, i0, T):
    """Cross-correlation of two envelope pieces centred on grid index i0, over LAG_RANGE: (lag, r, r one breath
    either side of the peak)."""
    h = int(HALF_WIN_S / DT)
    lo, hi = int(LAG_RANGE[0] / DT), int(LAG_RANGE[1] / DT)
    if i0 - h + lo < 0 or i0 + h + hi >= er.size or i0 + h >= ea.size:
        return None
    a = ea[i0 - h:i0 + h]
    if not np.isfinite(a).all() or a.std() == 0:
        return None
    a = (a - a.mean()) / a.std()
    rs = np.full(hi - lo + 1, np.nan)
    for j, L in enumerate(range(lo, hi + 1)):
        b = er[i0 - h + L:i0 + h + L]
        if np.isfinite(b).mean() < 0.9 or np.nanstd(b) == 0:
            continue
        b = np.where(np.isfinite(b), b, np.nanmean(b))
        rs[j] = float(np.mean(a * (b - b.mean()) / b.std()))
    if not np.isfinite(rs).any():
        return None
    j = int(np.nanargmax(rs))
    lag = (lo + j) * DT
    nb = []
    for s in (-1, 1):
        jj = j + s * int(round(T / DT))
        nb.append(round(float(rs[jj]), 3) if 0 <= jj < rs.size and np.isfinite(rs[jj]) else None)
    return round(lag, 2), round(float(rs[j]), 3), nb


def events_of(on, ta, pauses):
    """Pauses in ventilation (no breaths) while the AWP stream runs, and the start and end of ventilation."""
    ev = []
    if on.size < 20:
        return ev
    d = np.diff(on)
    T = ndimage.median_filter(d, 15)
    for i in np.flatnonzero(d > np.maximum(15.0, 3 * T)):
        a, b = on[i], on[i + 1]
        if np.any((pauses > a) & (pauses <= b)):
            continue  # a stamp gap, not a pause in ventilation
        ev.append({"type": "pause", "stop": float(a), "restart": float(b), "len_s": float(b - a), "T": float(T[i])})
    if on[0] - ta[0] >= 60 and not np.any(pauses <= on[0]):
        ev.append({"type": "start", "stop": None, "restart": float(on[0]), "len_s": float(on[0] - ta[0]), "T": float(np.median(d[:15]))})
    if ta[-1] - on[-1] >= 60 and not np.any(pauses > on[-1]):
        ev.append({"type": "end", "stop": float(on[-1]), "restart": None, "len_s": float(ta[-1] - on[-1]), "T": float(np.median(d[-15:]))})
    return ev


def scan_case(c, a, modes, ports, step_rates):
    from vitaldb_reader.vital_file import read_vital
    v = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={"Primus/AWP", "SNUADC/CVP", "SNUADC/ART"})
    aw = v.tracks.get("Primus/AWP")
    refs = {k: v.tracks.get(f"SNUADC/{k.upper()}") for k in ("cvp", "art")}
    refs = {k: tr for k, tr in refs.items() if tr is not None and tr.record_times.size}
    if aw is None or not aw.record_times.size or not refs:
        return {"case": c, "skip": "no AWP, or neither CVP nor ART"}
    ta = sample_times(aw)
    xa = np.nan_to_num(aw.physical().astype(np.float64))[: ta.size]
    on = P.breath_onsets(ta, xa, float(aw.srate))
    if on.size < 200:
        return {"case": c, "skip": "fewer than 200 mechanical breaths"}
    z = float(ta[0])
    pauses = ta[1:][np.diff(ta) > 5]
    ev = events_of(on, ta, pauses)
    res = {"case": c, "z": z, "pauses": (pauses - z).round(1).tolist(), "events": []}
    if not ev:
        return res
    out_arms = {}
    for k, tr in refs.items():
        tc = P.model_times(tr, ports.get(c), modes.get(c), step_rates)
        xc = tr.physical().astype(np.float64)[: tc.size]
        xc = lowpass_mean(np.nan_to_num(xc, nan=float(np.nanmedian(xc))), float(tr.srate))
        t0 = float(np.floor(min(ta[0], tc[0])))
        n = int((max(ta[-1], tc[-1]) - t0) / DT) + 1
        ea, er = envelope(binned(ta, xa, t0, DT, n)), envelope(binned(tc, xc, t0, DT, n))
        tl, ll = P.cvp_lag(ta, xa, tc, xc, on)
        out_arms[k] = (t0, ea, er, tl, ll)
    for e in ev:
        row = {k: (round(e[k] - z, 1) if isinstance(e[k], float) and k in ("stop", "restart") else
                   (round(e[k], 2) if isinstance(e[k], float) else e[k])) for k in e}
        stretch_of = lambda t: int(np.searchsorted(pauses, t))  # noqa: E731
        for k, (t0, ea, er, tl, ll) in out_arms.items():
            if ea is None or er is None:
                continue
            r = {}
            for edge in ("stop", "restart"):
                if e[edge] is None:
                    continue
                got = lag_at(ea, er, int(round((e[edge] - t0) / DT)), e["T"])
                if got is None:
                    continue
                r[edge] = {"lag": got[0], "r": got[1], "r_one_breath_either_side": got[2]}
                same = [i for i in range(tl.size) if stretch_of(tl[i]) == stretch_of(e[edge])]
                if same:
                    i = min(same, key=lambda i: abs(tl[i] - e[edge]))
                    r[edge]["matched_lag"] = round(float(ll[i]), 3)
                    r[edge]["matched_dt_s"] = round(float(tl[i] - e[edge]), 1)
                    r[edge]["k"] = round(float((ll[i] - got[0]) / e["T"]), 2)
            if r:
                row[k] = r
        res["events"].append(row)
    return res


def scan(a):
    modes = {int(k): v for k, v in json.loads(Path(a.modes).read_text()).items()}
    ports = {int(k): (v or {}).get("port") for k, v in json.loads(Path(a.ports).read_text()).items()}
    step_rates = json.loads(Path(a.step_rates).read_text())
    out = Path(a.out)
    done = {json.loads(l)["case"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    t_start = time.time()
    with open(out, "a", encoding="utf-8") as fh:
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


def _bias(near):
    """Where the event and the matching agree on the breath (|k| < 0.25), event lag minus matched lag (s): a bias of
    the envelope method shows as a non-zero median. Split by arm, by edge (stop, restart), by peak correlation and by
    the time between the event and the matched window (drift)."""
    q = lambda v: [round(float(x), 3) for x in np.percentile(v, [10, 25, 50, 75, 90])] if len(v) else None
    out = {}
    for arm in ("cvp", "art"):
        A = [x for x in near if x["arm"] == arm and abs(x["k"]) < 0.25]
        d = lambda S: [x["lag"] - x["matched_lag"] for x in S]  # noqa: E731
        out[arm] = {"n": len(A), "event_minus_matched_s_p10_25_50_75_90": q(d(A)),
                    "by_edge": {e: q(d([x for x in A if x["edge"] == e])) for e in ("stop", "restart")},
                    "r_0.5_0.7": q(d([x for x in A if x["r"] < 0.7])), "r_over_0.7": q(d([x for x in A if x["r"] >= 0.7])),
                    "matched_within_120_s": q(d([x for x in A if abs(x["matched_dt_s"]) <= 120]))}
    return out


def _by_rate(good):
    """Resolved event lags per breath rate (rounded 60 / T): a mode that moves with the period would mean the
    events themselves are ambiguous by a breath; a mode that stays put is a stamp offset."""
    out = {}
    lag = np.array([x["lag"] for x in good]); rate = np.round(60 / np.array([x["T"] for x in good]))
    for r_ in (10, 12, 14, 16, 18):
        m = rate == r_
        if m.sum() >= 50:
            out[f"{r_}/min"] = {"n": int(m.sum()), "hist_0.5s_bins_from_-2": np.histogram(lag[m], bins=np.arange(-2, 12.01, 0.5))[0].tolist()}
    return out


def agg(a):
    R = [json.loads(l) for p in a.parts for l in open(p, encoding="utf-8") if l.strip()]
    rows = []
    for r in R:
        for e in r.get("events", []):
            for k in ("cvp", "art"):
                for edge, x in (e.get(k) or {}).items():
                    rows.append({"case": r["case"], "type": e["type"], "edge": edge, "arm": k, "T": e["T"], **x})
    q = lambda v: [round(float(x), 3) for x in np.percentile(v, [5, 10, 25, 50, 75, 90, 95])] if len(v) else None
    def resolved(x):
        nb = [y for y in x["r_one_breath_either_side"] if y is not None]
        return x["r"] >= 0.5 and nb and x["r"] - max(nb) >= 0.15
    good = [x for x in rows if resolved(x) and LAG_RANGE[0] < x["lag"] < LAG_RANGE[1]]
    lag = np.array([x["lag"] for x in good])
    near = [x for x in good if "k" in x and abs(x.get("matched_dt_s", 1e9)) <= 600]
    kk = np.array([x["k"] for x in near])
    summ = {"recordings_read": len(R), "errors": sum("error" in r for r in R),
            "recordings_with_events": sum(bool(r.get("events")) for r in R),
            "events": sum(len(r.get("events", [])) for r in R),
            "event_edges_measured": len(rows),
            "resolved_(r>=0.5,_0.15_above_one_breath_either_side)": len(good),
            "resolved_recordings": len({x["case"] for x in good}),
            "event_lag_s_p5_10_25_50_75_90_95": q(lag),
            "resolved_with_a_matched_window_within_600_s": len(near),
            "k_breaths_off_p5_10_25_50_75_90_95": q(kk),
            "k_histogram_bins_-3.5..3.5": np.histogram(kk, bins=np.arange(-3.5, 3.51, 1))[0].tolist() if kk.size else None,
            "share_k_within_0.25_of_0": round(float(np.mean(np.abs(kk) < 0.25)), 3) if kk.size else None,
            "share_k_within_0.25_of_1": round(float(np.mean(np.abs(kk - 1) < 0.25)), 3) if kk.size else None,
            "k_histogram_0.1_bins_from_-1.5": np.histogram(kk, bins=np.arange(-1.5, 2.01, 0.1))[0].tolist() if kk.size else None,
            "event_lag_by_set_rate": _by_rate(good),
            "bias_check": _bias(near),
            "by_arm": {arm: {"resolved": sum(x["arm"] == arm for x in good),
                             "event_lag_s_p10_25_50_75_90": [round(float(v), 2) for v in np.percentile(
                                 [x["lag"] for x in good if x["arm"] == arm] or [np.nan], [10, 25, 50, 75, 90])],
                             "event_lag_over_T_histogram_0.25_bins_from_-0.5": np.histogram(
                                 [x["lag"] / x["T"] for x in good if x["arm"] == arm], bins=np.arange(-0.5, 2.51, 0.25))[0].tolist(),
                             "k_histogram_0.1_bins_from_-1.5": np.histogram([x["k"] for x in near if x["arm"] == arm],
                                                                           bins=np.arange(-1.5, 2.01, 0.1))[0].tolist()}
                       for arm in ("cvp", "art")}}
    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    (o / "summary.json").write_text(json.dumps(summ, indent=1))
    (o / "event_edges.json").write_text(json.dumps(rows))
    print(json.dumps(summ, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scan")
    for k in ("root", "cases", "modes", "ports", "step_rates", "out"):
        s.add_argument("--" + k.replace("_", "-"), required=True)
    s.add_argument("--time-budget", type=float, default=0)
    g = sp.add_parser("agg"); g.add_argument("parts", nargs="+"); g.add_argument("--out", required=True)
    a = ap.parse_args()
    scan(a) if a.cmd == "scan" else agg(a)


if __name__ == "__main__":
    main()
